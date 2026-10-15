"""Unified quantum-data generation for the canonical GenKSR v3 protocol.

Two protocols are supported, matching the two simulation experiments in the paper:

KQD  (Section 4.1, 1D Heisenberg):
    PennyLane Hadamard-test (QSD) circuits + Pauli-6 classical-shadow measurements.
    H(x) = Σ_<i,j> x_ij (X_iX_j + Y_iY_j + Z_iZ_j),  x_ij ~ U[0, 2].
    Output: data/heisenberg/{N}qubits_pennylane/{train,test}/

    python scripts/generate_data.py --task kqd --n-qubit 15 --krylov-dim 15 \
        --num-hamiltonians-train 30 --num-hamiltonians-test 20 --shots 10000 \
        --output-dir ./data/heisenberg/15qubits_pennylane/ --seed 42

SQKD (Section 4.2, 2D J1-J2 Heisenberg):
    Qiskit AerSimulator + Trotterised power-Krylov circuits, computational-basis shots.
    H = J1 Σ_<r,r'> S·S + J2 Σ_<<r,r'>> S·S on an LxL torus,  J1 = 1, J2 ~ U[0, 1].
    Output: data/heisenberg_SQKD_2d/{N}qubits/{train,test}/

    python scripts/generate_data.py --task sqkd --n-qubit 16 --L 4 --krylov-dim 30 \
        --num-hamiltonians-train 30 --num-hamiltonians-test 20 --shots 10000 \
        --output-dir ./data/heisenberg_SQKD_2d/16qubits/ --seed 42
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from genksr.data.integrity import (
    assert_disjoint_hamiltonian_splits,
    write_dataset_manifest,
)
from genksr.protocol import (
    PROTOCOL_VERSION,
    build_sqkd_evolution_gate,
    evolution_timestep,
    kqd_shadow_sampling_seeds,
    kqd_trotter_rotation_angle,
    neel_reference_bitstring,
)
from genksr.utils.quantum import (
    sample_coupling_matrix_1d,
    sample_coupling_matrix_2d_nn,
    sample_coupling_matrix_2d_j1j2,
    build_hamiltonian_from_couplings,
    build_2d_xxz_hamiltonian,
    build_interaction_list_from_couplings,
    controlled_state_prep,
    create_trotter_evolution_circuit_pennylane,
    create_trotter_evolution_circuit_pennylane_2d,
)
from genksr.evaluation.kqd import exact_ground_state


def parse_args():
    p = argparse.ArgumentParser(description='GenKSR v3 data generation')
    p.add_argument('--task', type=str, default='sqkd', choices=['kqd', 'sqkd'])
    p.add_argument('--lattice', type=str, default='1d', choices=['1d', '2d'],
                   help='KQD lattice. SQKD is always 2d.')
    p.add_argument('--n-qubit', type=int, default=16)
    p.add_argument('--krylov-dim', type=int, default=30,
                   help='Maximum subspace size: KQD writes D0..D(K-2); '
                        'SQKD writes D0..D(K-1).')
    p.add_argument('--num-hamiltonians-train', type=int, default=30)
    p.add_argument('--num-hamiltonians-test', type=int, default=20)
    p.add_argument('--num-trotter-steps', type=int, default=6)
    p.add_argument('--shots', type=int, default=10000)
    p.add_argument('--output-dir', type=str, required=True,
                   help='Writes {output_dir}/train and {output_dir}/test.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=1,
                   help='(KQD) parallelise the per-Hamiltonian shadow generation across processes.')

    # --- 2D SQKD options ---
    p.add_argument('--L', type=int, default=4, help='2D lattice side length (L x L).')
    p.add_argument('--coupling-model', type=str, default='j1j2', choices=['j1j2', 'nn'],
                   help="'j1j2' (paper Sec 4.2: J1 fixed + J2 sampled) or 'nn' (per-bond random NN).")
    p.add_argument('--J1', type=float, default=1.0, help='(j1j2) fixed nearest-neighbour coupling.')
    p.add_argument('--J2-range', type=float, nargs=2, default=[0.0, 1.0],
                   help='(j1j2) next-nearest-neighbour coupling range.')
    p.add_argument('--J-range', type=float, nargs=2, default=[0.5, 1.5],
                   help='(nn) per-bond coupling range.')
    p.add_argument('--delta', type=float, default=1.0, help='XXZ anisotropy (1.0 = isotropic).')
    p.add_argument('--h-z', type=float, default=0.0, help='External Z field.')
    p.add_argument('--trotter-order', type=int, default=2, choices=[1, 2],
                   help='2 = canonical v3 SuzukiTrotter; 1 = legacy LieTrotter.')
    p.add_argument('--sqkd-dt', type=float, default=None,
                   help='(SQKD) Override the physical time per Krylov step. '
                        'Default: pi / sum(abs(H.coeffs)). Legacy runs may use 0.15.')
    p.add_argument('--sqkd-simulator', type=str, default='sampler',
                   choices=['sampler', 'statevector'],
                   help='(SQKD) sampler = Qiskit SamplerV2 shots; statevector = exact probabilities '
                        'sampled locally. Both use the same Trotter evolution gate.')
    return p.parse_args()


def _prepare_dirs(args, params):
    """Create {output_dir}/{train,test} and write a task-specific args.json (only relevant params)."""
    train_dir = os.path.join(args.output_dir, 'train')
    test_dir = os.path.join(args.output_dir, 'test')
    for d in (train_dir, test_dir):
        os.makedirs(d, exist_ok=True)
    for d in (train_dir, test_dir):
        existing_artifacts = [
            name for name in os.listdir(d)
            if name.endswith(('.npy', '.txt')) or name == 'manifest.json'
        ]
        if existing_artifacts:
            raise FileExistsError(
                f'{d} already contains dataset artifacts; use a new v3 output directory '
                'instead of mixing or partially overwriting data')
    for d in (train_dir, test_dir):
        with open(os.path.join(d, 'args.json'), 'w') as f:
            json.dump(params, f, indent=2)
    return train_dir, test_dir


def _finalize_dataset(train_dir, test_dir):
    assert_disjoint_hamiltonian_splits(train_dir, test_dir)
    write_dataset_manifest(train_dir)
    write_dataset_manifest(test_dir)


def _kqd_params(args):
    params = {
        'task': 'kqd',
        'lattice': args.lattice,
        'n_qubit': args.n_qubit,
        'krylov_dim': args.krylov_dim,
        'num_hamiltonians_train': args.num_hamiltonians_train,
        'num_hamiltonians_test': args.num_hamiltonians_test,
        'num_trotter_steps': args.num_trotter_steps,
        'shots': args.shots,
        'seed': args.seed,
        'protocol_version': PROTOCOL_VERSION,
        'reference_state': 'Neel',
        'time_index': 'data_Dd -> (d+1)*dt',
        'dt_mode': 'pi_over_pauli_l1_bound',
        'n_outcomes': 6,
        'measurement': 'Pauli-6 classical shadow (encoded as outcome_bit*3 + basis, values 0..5)',
        'measurement_seed': (
            'device and classical-shadow recipe seeds derived from '
            'SeedSequence([hamiltonian_seed, data_index, chunk_index])'
        ),
    }
    if args.lattice == '2d':
        params.update({
            'L': args.L,
            'coupling_model': args.coupling_model,
            'J1': args.J1,
            'J2_range': args.J2_range,
            'J_range': args.J_range,
            'coupling': '2D square lattice KQD; j1j2 uses fixed J1 and J2 ~ U[J2_low,J2_high]',
        })
    else:
        params['coupling'] = '1D chain, x_ij ~ U[0, 2]'
    return params


def _sqkd_params(args):
    params = {
        'task': 'sqkd',
        'n_qubit': args.n_qubit,
        'L': args.L,
        'krylov_dim': args.krylov_dim,
        'num_hamiltonians_train': args.num_hamiltonians_train,
        'num_hamiltonians_test': args.num_hamiltonians_test,
        'num_trotter_steps': args.num_trotter_steps,
        'trotter_order': args.trotter_order,
        'sqkd_dt': args.sqkd_dt,
        'sqkd_dt_mode': 'override' if args.sqkd_dt is not None else 'pi_over_sum_abs_coeffs',
        'sqkd_simulator': args.sqkd_simulator,
        'shots': args.shots,
        'seed': args.seed,
        'delta': args.delta,
        'h_z': args.h_z,
        'coupling_model': args.coupling_model,
        'protocol_version': PROTOCOL_VERSION,
        'reference_state': '2D checkerboard Neel',
        'time_index': 'data_Dd -> d*dt',
        'n_outcomes': 2,
        'measurement': 'computational-basis bitstrings',
    }
    if args.coupling_model == 'j1j2':
        params['J1'] = args.J1
        params['J2_range'] = args.J2_range
    else:
        params['J_range'] = args.J_range
    return params


# ---------------------------------------------------------------------------
# KQD (Section 4.1): 1D Heisenberg, PennyLane Pauli-6 classical shadows
# ---------------------------------------------------------------------------

def _generate_kqd_hamiltonian(spec):
    """Generate one Hamiltonian's KQD shadow data. Module-level so it is picklable for pools.

    spec = (save_dir, hid, n_qubit, krylov_dim, num_trotter_steps, shots, seed_int,
            lattice, L, coupling_model, J1, J2_range, J_range)
    """
    import pennylane as qml

    (save_dir, hid, n, krylov_dim, num_trotter_steps, shots, seed_int,
     lattice, L, coupling_model, J1, J2_range, J_range) = spec
    rng = np.random.default_rng(seed_int)

    if lattice == '2d':
        assert L * L == n, f'L={L} gives {L * L} qubits but --n-qubit={n}'
        if coupling_model == 'j1j2':
            coupling_matrix = sample_coupling_matrix_2d_j1j2(
                L=L, J1=J1, J2_range=tuple(J2_range), rng=rng)
        else:
            coupling_matrix = sample_coupling_matrix_2d_nn(
                L=L, J_range=tuple(J_range), rng=rng)
        H_op = build_2d_xxz_hamiltonian(N=n, J=coupling_matrix)
        gs_energy, _ = exact_ground_state(H_op, n)
        reference_state_str = neel_reference_bitstring(n, lattice_side=L)
        interaction_list = build_interaction_list_from_couplings(
            n, coupling_matrix, wire_offset=1)
    else:
        coupling_matrix = sample_coupling_matrix_1d(n_qubits=n, low=0.0, high=2.0, rng=rng)
        H_op = build_hamiltonian_from_couplings(n_qubits=n, J=coupling_matrix)
        gs_energy, _ = exact_ground_state(H_op, n)
        reference_state_str = neel_reference_bitstring(n)
        interaction_list = None

    dt = evolution_timestep(H_op)
    # A dimension-D Toeplitz KQD solve needs non-zero lags 1 .. D-1.
    krylov_angles = [
        kqd_trotter_rotation_angle(idx, dt, num_trotter_steps)
        for idx in range(krylov_dim - 1)
    ]

    # Vectorised classical shadow via qml.classical_shadow on default.qubit (random X/Y/Z basis
    # per qubit per shot — same protocol as the legacy per-shot loop, but ~600x faster; lightning
    # does not support the shadow measurement). At larger qubit counts default.qubit's shadow
    # sampling is memory-heavy (~shots * 2^(n+1)), so we draw shots in CHUNKS with a fresh device
    # per chunk to bound peak memory. Small systems use a single full-shot call.
    def _build_qnode(dev, shadow_seed):
        @qml.qnode(dev)
        def q(angle):
            control = 0
            system = list(range(1, n + 1))
            qml.Hadamard(wires=control)
            controlled_state_prep(reference_state_str, system, control)
            if lattice == '2d':
                create_trotter_evolution_circuit_pennylane_2d(
                    interaction_list, num_trotter_steps, angle)
            else:
                create_trotter_evolution_circuit_pennylane(
                    n, coupling_matrix, num_trotter_steps, angle)
            qml.X(wires=control)
            controlled_state_prep(reference_state_str, system, control)
            qml.X(wires=control)
            return qml.classical_shadow(
                wires=list(range(n + 1)), seed=shadow_seed)
        return q

    # Cap per-call shots so shots_chunk * 2^(n+1) * 16 bytes stays ~<= 2 GB.
    chunk = min(shots, max(500, int(2e9 / ((2 ** (n + 1)) * 16))))

    for i, angle in enumerate(krylov_angles):
        bits_parts, rec_parts = [], []
        got = 0
        chunk_index = 0
        while got < shots:
            this = min(chunk, shots - got)
            device_seed, shadow_seed = kqd_shadow_sampling_seeds(
                seed_int, i, chunk_index)
            dev = qml.device(
                "default.qubit", wires=n + 1, shots=this, seed=device_seed)
            b, r = _build_qnode(dev, shadow_seed)(angle)
            bits_parts.append(np.asarray(b))
            rec_parts.append(np.asarray(r))
            got += this
            chunk_index += 1
        bits = np.concatenate(bits_parts, axis=0)
        recipes = np.concatenate(rec_parts, axis=0)
        # Preserve the original Pauli-6 encoding (outcome_bit 1 <-> eigenvalue +1). PennyLane's
        # classical_shadow uses bit 0 <-> +1, so flip the bit: outcome_bit = 1 - bit.
        encoded = ((1 - bits) * 3 + recipes).astype(int)   # values 0..5
        np.save(os.path.join(save_dir, f'data_D{i}_id{hid}.npy'), encoded)

    np.save(os.path.join(save_dir, f'coupling_matrix_id{hid}.npy'), coupling_matrix)
    with open(os.path.join(save_dir, f'ground_state_energy_id{hid}.txt'), 'w') as f:
        f.write(str(gs_energy))
    return hid


def generate_kqd_data(args):
    if args.lattice == '2d':
        assert args.L * args.L == args.n_qubit, (
            f'L={args.L} gives {args.L * args.L} qubits but --n-qubit={args.n_qubit}')
    train_dir, test_dir = _prepare_dirs(args, _kqd_params(args))
    n = args.n_qubit

    # One independent, reproducible seed per Hamiltonian (works under parallelism).
    n_total = args.num_hamiltonians_train + args.num_hamiltonians_test
    child_seeds = np.random.SeedSequence(args.seed).spawn(n_total)

    specs, gi = [], 0
    for nh, save_dir in zip([args.num_hamiltonians_train, args.num_hamiltonians_test],
                            [train_dir, test_dir]):
        for hid in range(nh):
            seed_int = int(child_seeds[gi].generate_state(1)[0])
            specs.append((save_dir, hid, n, args.krylov_dim,
                          args.num_trotter_steps, args.shots, seed_int,
                          args.lattice, args.L, args.coupling_model, args.J1,
                          args.J2_range, args.J_range))
            gi += 1

    if args.num_workers and args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            list(tqdm(ex.map(_generate_kqd_hamiltonian, specs), total=len(specs),
                      desc=f'KQD (x{args.num_workers} workers)'))
    else:
        for spec in tqdm(specs, total=len(specs), desc='KQD'):
            _generate_kqd_hamiltonian(spec)

    _finalize_dataset(train_dir, test_dir)
    print('KQD data generation complete.')


# ---------------------------------------------------------------------------
# SQKD (Section 4.2): 2D J1-J2 Heisenberg, Qiskit statevector/Aer sampler
# ---------------------------------------------------------------------------

def _preprocess_counts(counts_dict):
    measurements = []
    for bitstring, count in counts_dict.items():
        measurements.extend([[int(b) for b in bitstring]] * count)
    return measurements


def generate_sqkd_data(args):
    from qiskit.circuit import QuantumCircuit, QuantumRegister

    L = args.L
    assert L * L == args.n_qubit, f'L={L} gives {L*L} qubits but --n-qubit={args.n_qubit}'

    train_dir, test_dir = _prepare_dirs(args, _sqkd_params(args))
    rng = np.random.default_rng(seed=args.seed)

    if args.sqkd_simulator == 'statevector':
        from qiskit.quantum_info import Statevector

        for nh, save_dir in zip([args.num_hamiltonians_train, args.num_hamiltonians_test],
                                [train_dir, test_dir]):
            pbar = tqdm(range(nh), total=nh, desc=f'SQKD {os.path.basename(save_dir)}')
            for hid in pbar:
                if args.coupling_model == 'j1j2':
                    coupling_matrix = sample_coupling_matrix_2d_j1j2(
                        L=L, J1=args.J1, J2_range=tuple(args.J2_range), rng=rng)
                else:
                    coupling_matrix = sample_coupling_matrix_2d_nn(
                        L=L, J_range=tuple(args.J_range), rng=rng)

                H_op = build_2d_xxz_hamiltonian(N=args.n_qubit, J=coupling_matrix,
                                                delta=args.delta, h_z=args.h_z)
                dt = evolution_timestep(H_op, override=args.sqkd_dt)

                # Prepare the canonical checkerboard Neel state. The helper returns
                # Qiskit's displayed q[N-1]...q[0] order, hence the reversal here.
                reference_state = neel_reference_bitstring(args.n_qubit, lattice_side=L)
                qc_state = QuantumCircuit(args.n_qubit)
                for qubit, bit in enumerate(reversed(reference_state)):
                    if bit == '1':
                        qc_state.x(qubit)

                evol_gate = build_sqkd_evolution_gate(
                    H_op, dt, reps=args.num_trotter_steps, order=args.trotter_order)
                qr = QuantumRegister(args.n_qubit)
                qc_evol = QuantumCircuit(qr)
                qc_evol.append(evol_gate, qargs=qr)

                # Expand PauliEvolutionGate into its synthesized product formula.
                # Applying the opaque gate as an Operator could instead use its exact
                # matrix and silently bypass the requested Trotter approximation.
                qc_evol = qc_evol.decompose()

                state = Statevector.from_instruction(qc_state)
                bit_shifts = np.arange(args.n_qubit - 1, -1, -1)
                for k in range(args.krylov_dim):
                    probs = np.asarray(state.probabilities(), dtype=float)
                    probs = probs / probs.sum()
                    basis_idx = rng.choice(probs.size, size=args.shots, replace=True, p=probs)
                    data_gen = ((basis_idx[:, None] >> bit_shifts) & 1).astype(int)
                    np.save(os.path.join(save_dir, f'data_D{k}_id{hid}.npy'), data_gen)
                    if k + 1 < args.krylov_dim:
                        state = state.evolve(qc_evol)

                np.save(os.path.join(save_dir, f'coupling_matrix_id{hid}.npy'), coupling_matrix)

        _finalize_dataset(train_dir, test_dir)
        print('SQKD data generation complete.')
        return

    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    backend = AerSimulator()
    with Session(backend=backend) as session:
        for nh, save_dir in zip([args.num_hamiltonians_train, args.num_hamiltonians_test],
                                [train_dir, test_dir]):
            pbar = tqdm(range(nh), total=nh, desc=f'SQKD {os.path.basename(save_dir)}')
            for hid in pbar:
                if args.coupling_model == 'j1j2':
                    coupling_matrix = sample_coupling_matrix_2d_j1j2(
                        L=L, J1=args.J1, J2_range=tuple(args.J2_range), rng=rng)
                else:
                    coupling_matrix = sample_coupling_matrix_2d_nn(
                        L=L, J_range=tuple(args.J_range), rng=rng)

                H_op = build_2d_xxz_hamiltonian(N=args.n_qubit, J=coupling_matrix,
                                                delta=args.delta, h_z=args.h_z)
                dt = evolution_timestep(H_op, override=args.sqkd_dt)

                reference_state = neel_reference_bitstring(args.n_qubit, lattice_side=L)
                qc_state = QuantumCircuit(args.n_qubit)
                for qubit, bit in enumerate(reversed(reference_state)):
                    if bit == '1':
                        qc_state.x(qubit)

                evol_gate = build_sqkd_evolution_gate(
                    H_op, dt, reps=args.num_trotter_steps, order=args.trotter_order)
                qr = QuantumRegister(args.n_qubit)
                qc_evol = QuantumCircuit(qr)
                qc_evol.append(evol_gate, qargs=qr)

                circuits = []
                for rep in range(args.krylov_dim):
                    circ = qc_state.copy()
                    for _ in range(rep):
                        circ.compose(other=qc_evol, inplace=True)
                    circ.measure_all()
                    circuits.append(circ)

                pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
                isa = pm.run(circuits=circuits)
                sampler = Sampler(mode=session)
                job = sampler.run(isa, shots=args.shots)

                for k in range(args.krylov_dim):
                    counts = job.result()[k].data.meas.get_counts()
                    data_gen = np.array(_preprocess_counts(counts))
                    np.save(os.path.join(save_dir, f'data_D{k}_id{hid}.npy'), data_gen)

                np.save(os.path.join(save_dir, f'coupling_matrix_id{hid}.npy'), coupling_matrix)

    _finalize_dataset(train_dir, test_dir)
    print('SQKD data generation complete.')


def main():
    args = parse_args()
    if args.task == 'kqd':
        generate_kqd_data(args)
    elif args.task == 'sqkd':
        generate_sqkd_data(args)
    else:
        raise ValueError(f'Unknown task: {args.task}')


if __name__ == '__main__':
    main()
