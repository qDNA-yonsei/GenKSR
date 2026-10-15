#!/usr/bin/env python3
"""Evaluate whether a trained 16-qubit SKQD model learned the exact distribution.

For each held-out Hamiltonian and stored SKQD data index, this script computes the
exact Trotter-state probabilities and exhaustively scores all 2**N bitstrings with
the autoregressive model.  It reports raw physical-sector mass separately from the
distribution agreement after Sz=0 post-selection, so post-selection cannot hide a
symmetry failure during Krylov-dimension extrapolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch as GraphBatch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from genksr.constants import WDEGREE_FEATURE, measurement_outcomes_for_task
from genksr.data.coupling_graphs import CouplingGraphsDataset
from genksr.data.dataset import normalize_coupling_matrices
from genksr.data.hamiltonian import validate_protocol_metadata
from genksr.data.sample_structure import get_sample_structure
from genksr.nn.common import make_std_mask
from genksr.nn.encoder import get_graph_encoder
from genksr.nn.gc_model import build_gc_model
from genksr.protocol import (
    PROTOCOL_VERSION,
    SQKD_TASK,
    build_sqkd_evolution_gate,
    evolution_timestep,
    neel_reference_bitstring,
    physical_krylov_time,
)
from genksr.utils.quantum import build_2d_xxz_hamiltonian


MODEL_STYLE = {
    "transformer": {"color": "tab:blue", "marker": "o", "label": "Transformer"},
    "mamba": {"color": "tab:green", "marker": "s", "label": "Mamba"},
}
SUPPORT_STYLE = {
    "device": {"color": "tab:red", "marker": "^", "label": "Exact simulation"},
    **MODEL_STYLE,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", choices=("compute", "plot", "both"), default="both")
    parser.add_argument("--data-root", default="data/heisenberg_SQKD_2d")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--transformer-results", required=True)
    parser.add_argument("--mamba-results", required=True)
    parser.add_argument("--output-dir", default="artifacts/v3_skqd_distribution_learning")
    parser.add_argument("--hamiltonian-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=[0, 4, 9, 14, 19, 24, 29],
        help="Stored data_D indices; plotted Krylov dimension is index+1.",
    )
    parser.add_argument("--n-qubit", type=int, default=16)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--num-trotter-steps", type=int, default=6)
    parser.add_argument("--trotter-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--h-z", type=float, default=0.0)
    parser.add_argument("--sqkd-dt", type=float, default=None)
    parser.add_argument("--snapshots", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth.tar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-distributions", action="store_true")
    parser.add_argument("--metrics-csv", default=None,
                        help="Plot mode input; defaults to OUTPUT_DIR/distribution_metrics.csv")
    parser.add_argument("--support-csv", default=None,
                        help="Plot mode input; defaults to OUTPUT_DIR/support_metrics.csv")
    return parser.parse_args(argv)


def validate_args(args):
    if args.L * args.L != args.n_qubit:
        raise ValueError("L*L must equal n-qubit")
    if args.n_qubit > 20:
        raise ValueError("Exhaustive bitstring scoring is restricted to n_qubit <= 20")
    if not args.dimensions or min(args.dimensions) < 0:
        raise ValueError("dimensions must be a non-empty list of non-negative indices")
    if len(set(args.dimensions)) != len(args.dimensions):
        raise ValueError("dimensions must not contain duplicates")
    args.dimensions = sorted(args.dimensions)
    if args.batch_size <= 0 or args.snapshots <= 0:
        raise ValueError("batch-size and snapshots must be positive")


def all_bitstrings(n_qubit):
    states = np.arange(1 << n_qubit, dtype=np.int64)
    shifts = np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    bits = ((states[:, None] >> shifts) & 1).astype(np.int64)
    return states, bits


def load_model(results_dir, checkpoint, device, n_qubit):
    results_dir = Path(results_dir)
    with (results_dir / "config.json").open() as handle:
        cfg = json.load(handle)
    data_cfg = cfg["data"]
    if data_cfg.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{results_dir} is not a canonical v3 checkpoint")
    if data_cfg.get("task") != SQKD_TASK or data_cfg.get("n_qubit") != n_qubit:
        raise ValueError(f"{results_dir} is incompatible with {n_qubit}-qubit SKQD")
    n_outcomes = measurement_outcomes_for_task(SQKD_TASK)
    if data_cfg.get("n_outcomes") != n_outcomes:
        raise ValueError(f"{results_dir} has an incompatible output vocabulary")

    enc = cfg["encoder"]
    dec = cfg["decoder"]
    structure = get_sample_structure(data_cfg["sample_structure"])
    in_node_dim = 1 if enc["features"] == WDEGREE_FEATURE else n_qubit
    encoder = get_graph_encoder(
        enc["arch"], in_node_dim=in_node_dim, gcn_dim=enc["gcn_dim"],
        gcn_layers=enc["gcn_layers"], d_model=dec["d_model"], qubits=n_qubit,
    )
    model = build_gc_model(
        decoder_type=dec["type"], n_outcomes=n_outcomes, encoder=encoder,
        n_layers=dec["n_layers"], d_model=dec["d_model"],
        d_ff=4 * dec["d_model"], n_heads=dec.get("n_heads", 4),
        dropout=dec["dropout"], pad_token=structure.pad_token,
        start_token=structure.start_token, end_token=structure.end_token,
        token_shift=structure.token_shift, d_state=dec.get("d_state", 16),
        d_conv=dec.get("d_conv", 4), expand=dec.get("expand", 2),
    )
    checkpoint_path = results_dir / "checkpoints" / checkpoint
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, cfg


def build_graph(coupling, hamiltonian_id, feature_type, device, seed):
    normalized = normalize_coupling_matrices(coupling[None, ...])
    dataset = CouplingGraphsDataset(
        normalized, [hamiltonian_id], np.random.default_rng(seed),
        device=device, feature_type=feature_type,
    )
    graph, _, _ = dataset[0]
    return GraphBatch.from_data_list([graph]).to(device)


@torch.inference_mode()
def score_all_bitstrings(model, graph, bits, noise_factor, batch_size, device):
    """Return normalized q(z) in displayed bitstring / Qiskit basis order."""
    graph_embedding = model.encoder(graph)
    log_probabilities = []
    valid_start = model.token_shift
    valid_stop = valid_start + model.n_outcomes

    for start in range(0, len(bits), batch_size):
        batch_bits_np = bits[start:start + batch_size]
        batch_bits = torch.as_tensor(batch_bits_np, dtype=torch.long, device=device)
        start_column = torch.full(
            (len(batch_bits), 1), model.start_token, dtype=torch.long, device=device
        )
        target_input = torch.cat(
            [start_column, batch_bits[:, :-1] + model.token_shift], dim=1
        )
        mask = (
            make_std_mask(target_input, pad=model.pad_token)
            if model.decoder.needs_mask else None
        )
        embedded = model.tgt_embed(target_input)
        times = torch.full(
            (len(batch_bits),), float(noise_factor), dtype=torch.float32, device=device
        )
        time_embedding = model.noise_embed(times.unsqueeze(-1)).unsqueeze(1)
        conditioned = embedded + graph_embedding + time_embedding
        decoded = model.decoder(conditioned, mask)

        # Sampling excludes special tokens and renormalizes over the physical
        # outcomes, so exhaustive likelihood scoring must use the same conditional.
        valid_log_probs = model.generator(decoded)[..., valid_start:valid_stop]
        valid_log_probs = torch.log_softmax(valid_log_probs, dim=-1)
        selected = torch.gather(valid_log_probs, -1, batch_bits.unsqueeze(-1)).squeeze(-1)
        log_probabilities.append(selected.sum(dim=1).double().cpu().numpy())

    log_q = np.concatenate(log_probabilities)
    q = np.exp(log_q)
    raw_normalization = float(q.sum())
    if not np.isfinite(raw_normalization) or raw_normalization <= 0:
        raise RuntimeError("Model distribution has invalid normalization")
    if abs(raw_normalization - 1.0) > 5e-4:
        raise RuntimeError(
            f"Autoregressive distribution normalization={raw_normalization}, expected 1"
        )
    return q / raw_normalization, raw_normalization


def exact_probabilities_for_dimensions(
        coupling, n_qubit, lattice_side, dimensions, num_trotter_steps,
        trotter_order, delta, h_z, sqkd_dt):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    hamiltonian = build_2d_xxz_hamiltonian(
        N=n_qubit, J=coupling, delta=delta, h_z=h_z
    )
    dt = evolution_timestep(hamiltonian, override=sqkd_dt)
    reference = neel_reference_bitstring(n_qubit, lattice_side=lattice_side)
    prepare = QuantumCircuit(n_qubit)
    for qubit, bit in enumerate(reversed(reference)):
        if bit == "1":
            prepare.x(qubit)

    evolution = QuantumCircuit(n_qubit)
    evolution.append(
        build_sqkd_evolution_gate(
            hamiltonian, dt, reps=num_trotter_steps, order=trotter_order
        ),
        qargs=range(n_qubit),
    )
    evolution = evolution.decompose()

    selected = set(dimensions)
    state = Statevector.from_instruction(prepare)
    probabilities = {}
    for dimension in range(max(dimensions) + 1):
        if dimension in selected:
            p = np.asarray(state.probabilities(), dtype=float)
            probabilities[dimension] = p / p.sum()
        if dimension < max(dimensions):
            state = state.evolve(evolution)
    return probabilities, dt


def distribution_metrics(p, q, bits, coupling, dimension, hamiltonian_id, model_name,
                         raw_normalization):
    sector_mask = bits.sum(axis=1) == bits.shape[1] // 2
    p_sector_mass = float(p[sector_mask].sum())
    q_sector_mass = float(q[sector_mask].sum())
    if p_sector_mass < 1 - 1e-8:
        raise RuntimeError(f"Exact state left Sz=0: mass={p_sector_mass}")
    if q_sector_mass <= 0:
        raise RuntimeError("Model assigns zero probability to Sz=0")

    p_sector = p[sector_mask] / p_sector_mass
    q_sector = q[sector_mask] / q_sector_mass
    positive = p_sector > 0
    kl_sector = float(np.sum(
        p_sector[positive] * (
            np.log(p_sector[positive]) - np.log(q_sector[positive])
        )
    ))
    positive_full = p > 0
    kl_raw = float(np.sum(
        p[positive_full] * (np.log(p[positive_full]) - np.log(q[positive_full]))
    ))
    bhattacharyya = float(np.sum(np.sqrt(p_sector * q_sector)))
    fidelity_sector = float(np.clip(bhattacharyya ** 2, 0.0, 1.0))
    hellinger_sector = float(np.sqrt(max(0.0, 1.0 - bhattacharyya)))

    z = 1.0 - 2.0 * bits[sector_mask].astype(float)
    p_zi = p_sector @ z
    q_zi = q_sector @ z
    p_zz = z.T @ (p_sector[:, None] * z)
    q_zz = z.T @ (q_sector[:, None] * z)
    upper = np.triu_indices(bits.shape[1], k=1)
    edge_mask = np.abs(coupling[upper]) > 0
    edge_zz_mae = float(np.mean(np.abs((p_zz - q_zz)[upper][edge_mask])))

    return {
        "hamiltonian_id": int(hamiltonian_id),
        "data_dimension": int(dimension),
        "krylov_dimension": int(dimension + 1),
        "model": model_name,
        "exact_sector_mass": p_sector_mass,
        "model_sector_mass": q_sector_mass,
        "raw_normalization": raw_normalization,
        "kl_raw": kl_raw,
        "kl_raw_per_qubit": kl_raw / bits.shape[1],
        "kl_sector": kl_sector,
        "kl_sector_per_qubit": kl_sector / bits.shape[1],
        "fidelity_sector": fidelity_sector,
        "hellinger_sector": hellinger_sector,
        "zi_mae": float(np.mean(np.abs(p_zi - q_zi))),
        "zz_mae": float(np.mean(np.abs((p_zz - q_zz)[upper]))),
        "edge_zz_mae": edge_zz_mae,
    }


def samples_to_unique_states(samples, n_qubit):
    samples = np.asarray(samples)
    if samples.shape[1] != n_qubit or not np.all((samples == 0) | (samples == 1)):
        raise ValueError("Sample array is not a binary n-qubit array")
    samples = samples[samples.sum(axis=1) == n_qubit // 2]
    if len(samples) == 0:
        return np.empty(0, dtype=np.int64)
    powers = 1 << np.arange(n_qubit - 1, -1, -1, dtype=np.int64)
    return np.unique(samples.astype(np.int64) @ powers)


def support_metrics(p, data_dir, model_results, hamiltonian_id, dimension,
                    snapshots, n_qubit):
    rows = []
    sample_paths = {
        "device": data_dir / f"data_D{dimension}_id{hamiltonian_id}.npy",
        **{
            model: Path(results_dir) / "samples" / data_dir.name / "model"
            / f"model_samples_D{dimension}_{snapshots}_id{hamiltonian_id}.npy"
            for model, results_dir in model_results.items()
        },
    }
    # Model results use samples/<split>/model; data_dir.name is the split name.
    for method, path in sample_paths.items():
        if not path.exists():
            continue
        samples = np.load(path, allow_pickle=False)
        basis = samples_to_unique_states(samples, n_qubit)
        rows.append({
            "hamiltonian_id": int(hamiltonian_id),
            "data_dimension": int(dimension),
            "krylov_dimension": int(dimension + 1),
            "method": method,
            "input_samples": int(len(samples)),
            "postselected_samples": int(np.count_nonzero(
                samples.sum(axis=1) == n_qubit // 2
            )),
            "unique_sector_basis": int(len(basis)),
            "captured_exact_mass": float(p[basis].sum()),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data_dir = Path(args.data_root) / f"{args.n_qubit}qubits" / args.split
    validate_protocol_metadata(
        data_dir, task=SQKD_TASK, lattice="2d", n_qubit=args.n_qubit,
        delta=args.delta, h_z=args.h_z, sqkd_dt=args.sqkd_dt,
        num_trotter_steps=args.num_trotter_steps,
    )
    available_ids = sorted(
        int(path.stem.split("id")[-1]) for path in data_dir.glob("coupling_matrix_id*.npy")
    )
    hamiltonian_ids = available_ids if args.hamiltonian_ids is None else args.hamiltonian_ids
    missing_ids = sorted(set(hamiltonian_ids) - set(available_ids))
    if missing_ids:
        raise ValueError(f"Unknown Hamiltonian ids: {missing_ids}")

    model_paths = {
        "transformer": args.transformer_results,
        "mamba": args.mamba_results,
    }
    models = {}
    configs = {}
    for model_name, path in model_paths.items():
        models[model_name], configs[model_name] = load_model(
            path, args.checkpoint, device, args.n_qubit
        )

    states, bits = all_bitstrings(args.n_qubit)
    if not np.array_equal(states, bits @ (1 << np.arange(args.n_qubit - 1, -1, -1))):
        raise RuntimeError("Bitstring/integer ordering mismatch")

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "distributions"
    if args.save_distributions:
        cache_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    support_rows = []

    for hamiltonian_id in tqdm(hamiltonian_ids, desc="Hamiltonians"):
        coupling = np.load(data_dir / f"coupling_matrix_id{hamiltonian_id}.npy")
        exact_by_dim, dt = exact_probabilities_for_dimensions(
            coupling, args.n_qubit, args.L, args.dimensions,
            args.num_trotter_steps, args.trotter_order,
            args.delta, args.h_z, args.sqkd_dt,
        )
        graphs = {
            model_name: build_graph(
                coupling, hamiltonian_id, configs[model_name]["encoder"]["features"],
                device, args.seed,
            )
            for model_name in models
        }

        for dimension in args.dimensions:
            p = exact_by_dim[dimension]
            support_rows.extend(support_metrics(
                p, data_dir, model_paths, hamiltonian_id, dimension,
                args.snapshots, args.n_qubit,
            ))
            if args.save_distributions:
                np.save(cache_dir / f"exact_D{dimension}_id{hamiltonian_id}.npy", p)

            noise_factor = physical_krylov_time(SQKD_TASK, dimension, dt)
            for model_name, model in models.items():
                q, raw_normalization = score_all_bitstrings(
                    model, graphs[model_name], bits, noise_factor,
                    args.batch_size, device,
                )
                metric_rows.append(distribution_metrics(
                    p, q, bits, coupling, dimension, hamiltonian_id,
                    model_name, raw_normalization,
                ))
                if args.save_distributions:
                    np.save(cache_dir / f"{model_name}_D{dimension}_id{hamiltonian_id}.npy", q)

    write_csv(output_dir / "distribution_metrics.csv", metric_rows)
    write_csv(output_dir / "support_metrics.csv", support_rows)
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "skqd_distribution_learning",
        "n_qubit": args.n_qubit,
        "L": args.L,
        "split": args.split,
        "hamiltonian_ids": hamiltonian_ids,
        "data_dimensions": args.dimensions,
        "krylov_dimensions": [dimension + 1 for dimension in args.dimensions],
        "trained_data_dimensions": [0, 14],
        "num_trotter_steps": args.num_trotter_steps,
        "trotter_order": args.trotter_order,
        "sqkd_dt": args.sqkd_dt,
        "snapshots": args.snapshots,
        "models": {name: os.path.abspath(path) for name, path in model_paths.items()},
        "seed": args.seed,
        "exhaustive_state_count": 1 << args.n_qubit,
        "sector_size": math.comb(args.n_qubit, args.n_qubit // 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir / "distribution_metrics.csv", output_dir / "support_metrics.csv"


def aggregate(rows, group_key, value_key):
    grouped = {}
    for row in rows:
        key = (row[group_key], int(row["krylov_dimension"]))
        grouped.setdefault(key, []).append(float(row[value_key]))
    return grouped


def write_aggregate_tables(output_dir, metrics, support):
    metric_fields = [
        "kl_raw_per_qubit", "kl_sector_per_qubit", "fidelity_sector",
        "model_sector_mass", "zi_mae", "zz_mae", "edge_zz_mae",
    ]
    support_fields = [
        "postselected_samples", "unique_sector_basis", "captured_exact_mass",
    ]

    def summarize(rows, method_key, fields):
        records = []
        methods = sorted({row[method_key] for row in rows})
        dimensions = sorted({int(row["krylov_dimension"]) for row in rows})
        for method in methods:
            for dimension in dimensions:
                selected = [
                    row for row in rows
                    if row[method_key] == method
                    and int(row["krylov_dimension"]) == dimension
                ]
                if not selected:
                    continue
                record = {
                    method_key: method,
                    "krylov_dimension": dimension,
                    "n_hamiltonians": len(selected),
                }
                for field in fields:
                    values = np.asarray([float(row[field]) for row in selected])
                    record[f"{field}_mean"] = float(np.mean(values))
                    record[f"{field}_median"] = float(np.median(values))
                    record[f"{field}_q25"] = float(np.quantile(values, 0.25))
                    record[f"{field}_q75"] = float(np.quantile(values, 0.75))
                records.append(record)
        return records

    write_csv(
        output_dir / "aggregate_distribution_metrics.csv",
        summarize(metrics, "model", metric_fields),
    )
    write_csv(
        output_dir / "aggregate_support_metrics.csv",
        summarize(support, "method", support_fields),
    )


def coupling_j2(coupling):
    values = coupling[np.triu_indices(coupling.shape[0], k=1)]
    candidates = values[(values > 1e-12) & (~np.isclose(values, 1.0))]
    if len(candidates) == 0:
        return 0.0
    if not np.allclose(candidates, candidates[0]):
        raise ValueError("Coupling matrix does not have a single J2 value")
    return float(candidates[0])


def plot_j2_heatmaps(args, metrics, output_dir, dimensions):
    data_dir = Path(args.data_root) / f"{args.n_qubit}qubits" / args.split
    hamiltonian_ids = sorted({int(row["hamiltonian_id"]) for row in metrics})
    j2_by_id = {
        hamiltonian_id: coupling_j2(
            np.load(data_dir / f"coupling_matrix_id{hamiltonian_id}.npy")
        )
        for hamiltonian_id in hamiltonian_ids
    }
    sorted_ids = sorted(hamiltonian_ids, key=j2_by_id.get)
    row_lookup = {
        (row["model"], int(row["hamiltonian_id"]), int(row["krylov_dimension"])): row
        for row in metrics
    }
    panels = [
        ("fidelity_sector", "Sector fidelity"),
        ("model_sector_mass", r"Raw $S_z=0$ mass"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for row_index, model_name in enumerate(("transformer", "mamba")):
        for column_index, (field, title) in enumerate(panels):
            ax = axes[row_index, column_index]
            values = np.asarray([
                [float(row_lookup[(model_name, hamiltonian_id, dimension)][field])
                 for dimension in dimensions]
                for hamiltonian_id in sorted_ids
            ])
            image = ax.imshow(values, aspect="auto", origin="lower", vmin=0, vmax=1,
                              cmap="viridis")
            ax.set_title(f"{MODEL_STYLE[model_name]['label']}: {title}")
            ax.set_xticks(range(len(dimensions)), dimensions)
            tick_rows = np.linspace(0, len(sorted_ids) - 1, min(6, len(sorted_ids)), dtype=int)
            ax.set_yticks(tick_rows)
            ax.set_yticklabels([f"{j2_by_id[sorted_ids[index]]:.3f}" for index in tick_rows])
            if 15 in dimensions and any(dimension > 15 for dimension in dimensions):
                boundary = dimensions.index(15) + 0.5
                ax.axvline(boundary, color="white", linestyle=":", linewidth=1.5)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel(r"Test Hamiltonians sorted by $J_2$")
    axes[1, 0].set_ylabel(r"$J_2$")
    axes[1, 0].set_xlabel("Krylov dimension D")
    axes[1, 1].set_xlabel("Krylov dimension D")
    fig.suptitle(r"Distribution learning across held-out $J_2$ values")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "distribution_learning_j2_heatmap.png", dpi=250)
    fig.savefig(output_dir / "distribution_learning_j2_heatmap.pdf")
    plt.close(fig)


def write_validation_report(output_dir, metrics, support):
    support_lookup = {
        (row["method"], int(row["hamiltonian_id"]), int(row["krylov_dimension"])): row
        for row in support
    }
    sampling_differences = {model_name: [] for model_name in MODEL_STYLE}
    for row in metrics:
        model_name = row["model"]
        key = (
            model_name,
            int(row["hamiltonian_id"]),
            int(row["krylov_dimension"]),
        )
        support_row = support_lookup[key]
        empirical_rate = (
            float(support_row["postselected_samples"])
            / float(support_row["input_samples"])
        )
        sampling_differences[model_name].append(
            empirical_rate - float(row["model_sector_mass"])
        )

    numeric_fields = (
        "exact_sector_mass", "model_sector_mass", "raw_normalization",
        "kl_raw_per_qubit", "kl_sector_per_qubit", "fidelity_sector",
        "zi_mae", "zz_mae", "edge_zz_mae",
    )
    all_finite = all(
        np.isfinite(float(row[field]))
        for row in metrics for field in numeric_fields
    )
    exact_sector_masses = np.asarray([
        float(row["exact_sector_mass"]) for row in metrics
    ])
    normalizations = np.asarray([
        float(row["raw_normalization"]) for row in metrics
    ])
    sector_masses = np.asarray([
        float(row["model_sector_mass"]) for row in metrics
    ])
    fidelities = np.asarray([
        float(row["fidelity_sector"]) for row in metrics
    ])
    sampling_consistency = {}
    for model_name, differences in sampling_differences.items():
        values = np.asarray(differences)
        sampling_consistency[model_name] = {
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "max_absolute_error": float(np.max(np.abs(values))),
        }

    hamiltonian_count = len({row["hamiltonian_id"] for row in metrics})
    dimension_count = len({row["krylov_dimension"] for row in metrics})
    report = {
        "distribution_row_count": len(metrics),
        "expected_distribution_row_count": hamiltonian_count * dimension_count * 2,
        "support_row_count": len(support),
        "expected_support_row_count": hamiltonian_count * dimension_count * 3,
        "hamiltonian_count": hamiltonian_count,
        "dimension_count": dimension_count,
        "all_metrics_finite": bool(all_finite),
        "exact_sector_mass_min": float(np.min(exact_sector_masses)),
        "exact_sector_mass_max": float(np.max(exact_sector_masses)),
        "max_autoregressive_normalization_error": float(
            np.max(np.abs(normalizations - 1.0))
        ),
        "model_sector_mass_range": [
            float(np.min(sector_masses)), float(np.max(sector_masses))
        ],
        "sector_fidelity_range": [
            float(np.min(fidelities)), float(np.max(fidelities))
        ],
        "sampling_vs_exhaustive_sector_mass": sampling_consistency,
    }
    report["passed"] = bool(
        len(metrics) == report["expected_distribution_row_count"]
        and len(support) == report["expected_support_row_count"]
        and all_finite
        and report["exact_sector_mass_min"] >= 1 - 1e-8
        and report["max_autoregressive_normalization_error"] < 5e-4
        and 0 <= report["model_sector_mass_range"][0]
        and report["model_sector_mass_range"][1] <= 1 + 1e-8
        and 0 <= report["sector_fidelity_range"][0]
        and report["sector_fidelity_range"][1] <= 1 + 1e-8
        and all(value["max_absolute_error"] < 0.03
                for value in sampling_consistency.values())
    )
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def add_training_boundary(ax, max_dimension):
    ax.axvline(15.5, color="black", linestyle=":", linewidth=1)
    if max_dimension > 15:
        ax.axvspan(15.5, max_dimension + 0.5, color="gray", alpha=0.10)
        ax.text(15.8, 0.97, "Extrapolation", transform=ax.get_xaxis_transform(),
                va="top", fontsize=9, color="dimgray")


def plot(args, metrics_path, support_path):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(metrics_path)
    support = read_csv(support_path)
    dimensions = sorted({int(row["krylov_dimension"]) for row in metrics})
    max_dimension = max(dimensions)
    write_aggregate_tables(output_dir, metrics, support)
    write_validation_report(output_dir, metrics, support)

    panels = [
        ("kl_raw_per_qubit", r"$D_{KL}(p\Vert q)$ per qubit", True),
        ("fidelity_sector", r"Fidelity in $S_z=0$ sector", False),
        ("model_sector_mass", r"Raw probability mass in $S_z=0$", False),
        ("edge_zz_mae", r"Edge $\langle Z_i Z_j\rangle$ MAE", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (field, ylabel, log_scale) in zip(axes.flat, panels):
        grouped = aggregate(metrics, "model", field)
        for model_name, style in MODEL_STYLE.items():
            x = np.asarray(dimensions)
            values = [np.asarray(grouped[(model_name, dimension)]) for dimension in dimensions]
            median = np.asarray([np.median(value) for value in values])
            q25 = np.asarray([np.quantile(value, 0.25) for value in values])
            q75 = np.asarray([np.quantile(value, 0.75) for value in values])
            ax.plot(x, median, color=style["color"], marker=style["marker"],
                    label=style["label"])
            ax.fill_between(x, q25, q75, color=style["color"], alpha=0.18)
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
        add_training_boundary(ax, max_dimension)
    axes[1, 0].set_xlabel("Krylov dimension D")
    axes[1, 1].set_xlabel("Krylov dimension D")
    axes[0, 0].legend()
    fig.suptitle("Exact full-distribution learning on held-out 4x4 J1-J2 Hamiltonians")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "distribution_learning.png", dpi=250)
    fig.savefig(output_dir / "distribution_learning.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    support_panels = [
        ("captured_exact_mass", "Exact probability mass captured"),
        ("unique_sector_basis", r"Unique $S_z=0$ basis states"),
    ]
    for ax, (field, ylabel) in zip(axes, support_panels):
        grouped = aggregate(support, "method", field)
        for method, style in SUPPORT_STYLE.items():
            present_dimensions = [
                dimension for dimension in dimensions if (method, dimension) in grouped
            ]
            if not present_dimensions:
                continue
            values = [np.asarray(grouped[(method, dimension)])
                      for dimension in present_dimensions]
            median = np.asarray([np.median(value) for value in values])
            q25 = np.asarray([np.quantile(value, 0.25) for value in values])
            q75 = np.asarray([np.quantile(value, 0.75) for value in values])
            ax.plot(present_dimensions, median, color=style["color"],
                    marker=style["marker"], label=style["label"])
            ax.fill_between(present_dimensions, q25, q75,
                            color=style["color"], alpha=0.18)
        ax.set_xlabel("Krylov dimension D")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        add_training_boundary(ax, max_dimension)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend()
    fig.suptitle(f"Usefulness of {args.snapshots:,} generated samples per dimension")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "sample_support_learning.png", dpi=250)
    fig.savefig(output_dir / "sample_support_learning.pdf")
    plt.close(fig)

    plot_j2_heatmaps(args, metrics, output_dir, dimensions)

    print(f"Saved figures and metrics to {output_dir}")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    metrics_path = Path(args.metrics_csv or Path(args.output_dir) / "distribution_metrics.csv")
    support_path = Path(args.support_csv or Path(args.output_dir) / "support_metrics.csv")
    if args.mode in ("compute", "both"):
        metrics_path, support_path = compute(args)
    if args.mode in ("plot", "both"):
        plot(args, metrics_path, support_path)


if __name__ == "__main__":
    main()
