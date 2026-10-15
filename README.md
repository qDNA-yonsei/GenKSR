# GenKSR

**Generative Krylov Subspace Representations** for quantum eigensolver experiments.

GenKSR learns measurement distributions conditioned on a Hamiltonian and its
Krylov evolution time. A graph encoder represents the Hamiltonian, and an
autoregressive Transformer or Mamba decoder generates measurement samples for
classical energy estimation. The code includes quantum simulation, dataset
generation, training, sampling, evaluation, and plotting utilities.

This repository contains the source and configurations needed to regenerate the
simulation experiments. No external dataset download or quantum hardware
account is needed for the simulation workflows below.

## Implemented experiments

| Experiment | Configuration / entry point | Measurement representation |
| --- | --- | --- |
| 1D Heisenberg KQD | `configs/experiment/kqd_1d_{transformer,mamba}.yaml` | Pauli-6 classical shadows, including an ancilla |
| 2D Heisenberg KQD | `configs/experiment/kqd_2d_{transformer,mamba}.yaml` | Pauli-6 classical shadows, including an ancilla |
| 2D J1–J2 Heisenberg SKQD | `configs/experiment/skqd_2d_{transformer,mamba}.yaml` | Computational-basis bitstrings |
| 1D XXZ sparsity and scaling | `configs/experiment/skqd_1d_xxz_sparsity_n{12,16,20}_{transformer,mamba}.yaml` | Exact simulated Krylov-state bitstrings |
| Pauli cross-family generalization | `configs/experiment/pauli_*.yaml`, `scripts/experiments/*pauli*.py` | Separate Pauli-term-conditioned experiment pipeline |
| ViT neural quantum state baseline | `scripts/baselines/vit_nqs.py` | Variational Monte Carlo |

The sample-based method is called **SKQD** here; its command-line task name and
configuration value are **`sqkd`**. Use `--task sqkd`, not `--task skqd`.

## Installation

The package declares Python **3.10 or newer**. Use a dedicated environment and
run all commands from the repository root. This example uses Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[quantum]"
```

Alternatively, use the included `uv.lock` for the locked dependency resolution:

```bash
uv sync --locked --extra quantum --python 3.11
source .venv/bin/activate
```

| Installation | Purpose |
| --- | --- |
| `pip install -e .` | Core package: PyTorch, PyTorch Geometric, NumPy, SciPy, plotting, and training utilities |
| `pip install -e ".[quantum]"` | Adds PennyLane, Qiskit/Aer, and the SQD add-on for simulation workflows |
| `pip install -e ".[quantum,mamba]"` | Adds the Mamba decoder and its Transformers dependency constraint |
| `pip install -e ".[quantum,vit]"` | Adds NetKet, JAX, Flax, and Optax for the ViT baseline |

The Transformer smoke test can run on a CPU. Larger simulations and training
runs require substantially more memory and compute. Mamba requires a compatible
CUDA GPU and a `mamba-ssm` build matching the installed PyTorch/CUDA environment;
build notes are included in `pyproject.toml`. The ViT baseline uses JAX
independently of PyTorch.

The training and sampling commands select CUDA automatically when it is
available. To restrict them to one physical GPU, set its index before running
the commands; for example, to use GPU 2:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
```

Use `export CUDA_VISIBLE_DEVICES=""` for CPU-only Transformer runs.

## Quick start: a small SKQD experiment

This example generates a 2×2 lattice dataset, trains a small Transformer for two
epochs, generates held-out samples, and evaluates ground-state energies. It is
an execution check, not a converged scientific result. Its output directories
are separate from the full experiment directories.

### 1. Generate simulated measurements

```bash
python scripts/generate_data.py \
  --task sqkd --n-qubit 4 --L 2 --krylov-dim 3 \
  --num-hamiltonians-train 2 --num-hamiltonians-test 2 \
  --shots 64 --num-trotter-steps 2 --sqkd-simulator statevector \
  --output-dir data/smoke_skqd/4qubits --seed 0
```

Generation creates `train/` and `test/` splits with protocol metadata, coupling
matrices, measurement arrays, and integrity manifests. The generator refuses to
mix new data with an existing populated dataset; use a fresh output directory
when repeating this step.

### 2. Train a Transformer

```bash
python scripts/train.py \
  --config configs/experiment/skqd_2d_transformer.yaml \
  --data.n_qubit 4 --data.krylov_dim 3 \
  --data.data_root ./data/smoke_skqd/ --data.num_trotter_steps 2 \
  --decoder.d_model 32 --decoder.n_layers 1 \
  --results_dir ./results/smoke_skqd --seed 0 --debug
```

`--debug` uses 20 training measurements per condition, a batch size of 20, and
two training iterations (epochs with the default `epoch_mode: true`). Each run
saves its resolved `config.json`, Hamiltonian IDs, logs, and checkpoints.

The trainer prints the full run directory. For this exact command, set:

```bash
RUN=results/smoke_skqd/conditional_heisenberg_SQKD_2d_4qubit/gcn_proj_3_16-transformer_l1_d32_h4_featone_hot_D_conditional/ns20/iter2_lr0.001_wd0.0_bs20_dropout0.1_samplestruct2_lrschedulewarmup_cosine_seed0
```

### 3. Generate samples from the trained model

```bash
python scripts/sample.py \
  --results-dir "$RUN" --checkpoint checkpoint_final.pth.tar \
  --snapshots 64 --krylov-dim-range 3 --splits test --seed 0
```

Samples are written under `$RUN/samples/test/model/`. Sampling also records the
checkpoint hash, random seed, physical times, and input/output paths.

### 4. Evaluate the generated and simulated samples

```bash
python scripts/evaluate.py \
  --task sqkd --n-qubit 4 --max-krylov-dim 3 \
  --num-trotter-steps 2 \
  --snapshots 64 --num-samples 64 \
  --data-root ./data/smoke_skqd/ --transformer-results "$RUN" \
  --models device transformer --splits test \
  --output-dir artifacts/smoke_skqd/evaluation --seed 0
```

The evaluator compares both sample sources against exact diagonalization and
writes `artifacts/smoke_skqd/evaluation/all_results.pkl`. Here, `device` means
simulated measurement data. The default `--skqd-solver fast` builds the projected
XXZ Hamiltonian directly; `--skqd-solver addon` selects the SQD add-on.
The SKQD plotting scripts use a 4×4 normalization and should be used with the
16-qubit experiment below, rather than this 4-qubit smoke test.

## Reproducing larger experiments

### 2D J1–J2 SKQD

Generate a 4×4 dataset with 30 stored Krylov indices, then train on the first 15:

```bash
python scripts/generate_data.py \
  --task sqkd --n-qubit 16 --L 4 --krylov-dim 30 \
  --num-hamiltonians-train 30 --num-hamiltonians-test 20 \
  --shots 10000 --coupling-model j1j2 --J1 1.0 --J2-range 0.0 1.0 \
  --output-dir data/heisenberg_SQKD_2d/16qubits --seed 42

python scripts/train.py --config configs/experiment/skqd_2d_transformer.yaml
```

Set `RUN` to the directory printed by this training run, then sample and evaluate:

```bash
python scripts/sample.py --results-dir "$RUN" \
  --snapshots 10000 --krylov-dim-range 30 --splits test --seed 0

python scripts/evaluate.py \
  --task sqkd --n-qubit 16 --max-krylov-dim 30 \
  --snapshots 10000 --num-samples 1000 \
  --data-root ./data/heisenberg_SQKD_2d/ --transformer-results "$RUN" \
  --models device transformer --splits test \
  --output-dir artifacts/skqd_2d/evaluation_1000 --seed 0

python scripts/figures/fig3_table2_fig8_skqd.py \
  --mode fig3 --results artifacts/skqd_2d/evaluation_1000/all_results.pkl \
  --output-dir figures/skqd_2d
```

For Mamba, train with `skqd_2d_mamba.yaml`, sample its run directory, and pass it
as `--mamba-results` with `--models device transformer mamba` during evaluation.
To compare shot budgets, repeat evaluation with `--num-samples 5000` and `10000`
and distinct output directories. `--snapshots` must continue to match the sample
count used when generating the model sample files.

For a fixed-coupling comparison, generate a separate dataset with
`--J2-range 0.5 0.5 --num-hamiltonians-train 0 --num-hamiltonians-test 1` and a
new output directory. Point sampling to its parent data root with `--data-root`
and use `--output-results-dir` to preserve the original samples. Evaluate that
same data root and those new sample directories before plotting.

The supplied 2D SKQD configs use a 128-dimensional, four-layer decoder and a
16-dimensional GCN. The larger historical architecture can be selected with
`--decoder.d_model 256 --decoder.n_layers 8 --decoder.n_heads 8
--encoder.arch gcn_encoder_3_256 --encoder.gcn_dim 256` during training.

### 1D KQD

Generate Pauli-6 measurements and train a Transformer on the first five stored
Krylov indices:

```bash
python scripts/generate_data.py \
  --task kqd --n-qubit 5 --krylov-dim 15 \
  --num-hamiltonians-train 30 --num-hamiltonians-test 20 --shots 10000 \
  --output-dir data/heisenberg/5qubits_pennylane --seed 42

python scripts/train.py --config configs/experiment/kqd_1d_transformer.yaml \
  --data.n_qubit 5
```

Set `KQD_RUN` to the run directory printed by training, then generate samples
for indices `D0` through `D13` and evaluate subspace sizes 1 through 15:

```bash
python scripts/sample.py --results-dir "$KQD_RUN" \
  --snapshots 10000 --krylov-dim-range 14 --splits test --seed 0

python scripts/evaluate.py \
  --task kqd --n-qubit 5 --max-krylov-dim 15 \
  --snapshots 10000 --num-samples 10000 --data-root ./data/heisenberg/ \
  --transformer-results "$KQD_RUN" --models ideal shadow transformer \
  --output-dir artifacts/kqd_1d/data_comparison_5qubit_10000samples --seed 0

python scripts/figures/fig2_table1_kqd.py \
  --results artifacts/kqd_1d/data_comparison_5qubit_10000samples/all_results.pkl \
  --data-comparison-dirs artifacts/kqd_1d/data_comparison_5qubit_10000samples \
  --output-dir figures/kqd_1d
```

KQD samples retain the ancilla column and the six measurement outcomes `0..5`.
The `data_comparison_5qubit_10000samples` directory name lets the table generator
identify the qubit count and evaluation shot budget. Update both numbers when
changing those settings.
For Mamba, use `kqd_1d_mamba.yaml` with the same `--data.n_qubit 5` override,
sample its run directory, and add `--mamba-results` with
`--models ideal shadow transformer mamba` during evaluation while retaining
`--transformer-results`.
To run only the ideal and classical-shadow baselines, omit training and sampling
and evaluate with `--models ideal shadow`.

Change the qubit count consistently for other sizes. For 2D KQD, generate with
`--lattice 2d --L L --n-qubit N`, where `N = L * L`, under a separate data root.
Train with a `kqd_2d_*.yaml` configuration and matching `data.n_qubit` and
`data.data_root` values. Sampling reads the lattice from the saved configuration;
evaluation needs `--lattice 2d` and the matching `--n-qubit` and `--data-root`.

### 1D XXZ sparsity and scaling

The XXZ workflow uses its own exact-state data generator and a frozen manifest
shared across system sizes:

```bash
python scripts/experiments/generate_xxz_sparsity_manifest.py \
  --output data/xxz_sparsity_1d/ratio_manifest.csv --seed 20260715

python scripts/experiments/generate_xxz_sparsity_data.py \
  --n-qubit 12 --depth 15 --shots 1000 \
  --manifest data/xxz_sparsity_1d/ratio_manifest.csv \
  --output-root data/xxz_sparsity_1d

python scripts/train.py \
  --config configs/experiment/skqd_1d_xxz_sparsity_n12_transformer.yaml
```

Repeat data generation and training with matching 16- and 20-qubit configs, or
select a Mamba config. These datasets preserve raw coupling strengths
(`coupling_normalization: none`), so use the XXZ configs.

The specialized equal-budget evaluator requires **both Transformer and Mamba**
samples for each requested size. Set `XXZ_RUN` to a trained run directory and
sample its best validation checkpoint. For the 12-qubit Transformer:

```bash
python scripts/sample.py --results-dir "$XXZ_RUN" \
  --checkpoint model_best.pth.tar \
  --output-results-dir artifacts/xxz_sparsity/sampling/n12_transformer \
  --snapshots 1000 --krylov-dim-range 15 --splits test --seed 20260716
```

Repeat for each size and decoder, naming the output folders
`n12_transformer`, `n12_mamba`, `n16_transformer`, `n16_mamba`,
`n20_transformer`, and `n20_mamba`. Keep the source checkpoints available:
evaluation verifies their hashes. The evaluator expects sample filenames with
`1000` shots. After generating all six sample sets:

```bash
python scripts/experiments/evaluate_xxz_sparsity.py \
  --sampling-root artifacts/xxz_sparsity/sampling \
  --data-root data/xxz_sparsity_1d --sizes 12 16 20 \
  --depth 15 --budgets 3000 15000 \
  --output-dir artifacts/xxz_sparsity/evaluation

python scripts/experiments/summarize_xxz_sparsity.py \
  --evaluation-dir artifacts/xxz_sparsity/evaluation

python scripts/figures/xxz_sparsity_paper_figures.py \
  --evaluation-dir artifacts/xxz_sparsity/evaluation
```

The summary and plotting commands require the full 12/16/20-qubit evaluation
with the default test sets and budgets. Training-budget sweeps have separate
`run_xxz_training_budget*.py` launchers with `--dry-run` support. These launchers
set GPU visibility themselves; pass `--gpus 2` to restrict their workers to
physical GPU 2.

### Pauli cross-family experiments

This workflow has a separate configuration loader, dataset format, and model.
Use its dedicated entry points instead of `scripts/train.py`. A six-qubit CPU
smoke run is:

```bash
python scripts/experiments/generate_pauli_cross_family.py \
  --config configs/experiment/pauli_cross_family_n8.yaml \
  --output-root data/pauli_smoke --workers 1 --smoke

python scripts/experiments/run_pauli_cross_family.py \
  --data-root data/pauli_smoke --results-root results/pauli_smoke \
  --evaluation-root artifacts/pauli_smoke --seeds 0 --devices cpu

python scripts/experiments/summarize_pauli_cross_family.py \
  --evaluation-dirs artifacts/pauli_smoke/seed0 \
  --output-dir artifacts/pauli_smoke/summary
```

For full runs, omit `--smoke` and use fresh output roots. Other `pauli_*.yaml`
files define cross-family and target-trained comparisons.

### Additional analyses

| Analysis | Entry point |
| --- | --- |
| Ground-state sparsity from exact diagonalization | `scripts/figures/fig5_table3_sparsity.py` |
| Training and inference timing | `scripts/figures/fig7_timing.py` |
| ViT variational baseline | `scripts/baselines/vit_nqs.py` |
| SKQD sample and result validation | `scripts/validate_skqd_samples.py`, `scripts/validate_skqd_results.py` |
| SKQD distribution learning and sample-budget studies | `scripts/experiments/skqd_*.py` |

For example, ground-state sparsity requires no trained model or saved dataset:

```bash
python scripts/figures/fig5_table3_sparsity.py \
  --L 4 --sizes 3 4 --output-dir figures/sparsity
```

Some shell launchers encode historical run-directory names, checkpoint choices,
and GPU assignments. Inspect those values before using them on another machine;
the individual Python commands above expose the inputs directly. The XXZ shell
launchers accept `PYTHON_BIN` to select an environment, and the other pipeline
launchers accept `PY`.

## Configuration and protocol details

`scripts/train.py` layers `configs/defaults.yaml`, the selected experiment YAML,
and dotted command-line overrides in that order. For example,
`--training.lr 0.0005 --data.train_samples 1000 --seed 1` overrides those values.
The resolved configuration is saved with each run.

- Canonical KQD/SKQD data and checkpoints use **protocol version 3**. Older
  datasets and checkpoints are incompatible. Pauli cross-family experiments
  carry their own protocol metadata.
- Generation with `--krylov-dim K` stores KQD indices `D0` through `D(K-2)` and
  SKQD indices `D0` through `D(K-1)`. Training's `data.krylov_dim` is the number
  of stored indices consumed, starting at zero. Sampling's
  `--krylov-dim-range K` generates indices `0, ..., K-1`.
- KQD index `d` corresponds to physical time `(d + 1) * dt`; SKQD index `d`
  corresponds to `d * dt`. The default timestep is
  `dt = pi / sum(abs(Pauli coefficients))`.
- The number of Trotter steps controls circuit approximation, not physical
  time. Keep timestep, Trotter, Hamiltonian, and lattice settings consistent
  across generation, training, sampling, and evaluation.
- Seeds control dataset construction, model initialization, and sampling.
  Numerical results can still vary with hardware and dependency versions.
- Evaluation files use pickle, and checkpoint loaders may deserialize Python
  objects. Load only artifacts you generated or otherwise trust.

## Repository layout

```text
.
├── README.md
├── pyproject.toml              # Package metadata and optional dependencies
├── uv.lock                     # Dependency resolution
├── configs/                    # Shared defaults and experiment configurations
├── genksr/
│   ├── data/                   # Dataset loading, integrity, and Hamiltonians
│   ├── nn/                     # Graph encoders and autoregressive decoders
│   ├── training/               # Trainer and learning-rate schedules
│   ├── sampling/               # Conditional sample generation
│   ├── evaluation/             # KQD/SKQD solvers and sparsity metrics
│   ├── experiments/            # Pauli cross-family implementation
│   ├── protocol.py             # Canonical time and measurement conventions
│   └── utils/                  # Quantum and I/O helpers
├── scripts/                    # Data, training, sampling, and evaluation CLIs
│   ├── experiments/            # Extended experiment workflows
│   ├── baselines/              # ViT baseline
│   └── figures/                # Plotting source code
└── tests/                      # Protocol, model, data, and evaluation checks
```

## Tests

Install pytest in the selected environment and run the suite from the root:

```bash
python -m pip install pytest
python -m pytest tests/ -q
```

Install the `quantum` extra for quantum-dependent tests. Tests covering optional
backends may be skipped when their dependencies or hardware are unavailable.
