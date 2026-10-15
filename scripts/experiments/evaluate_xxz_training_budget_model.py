#!/usr/bin/env python3
"""Score and sample one model in the XXZ training-budget study.

The likelihood and downstream sample evaluations use the same fixed test
Hamiltonians for every training-budget cell.  Likelihood is evaluated under
the physical two-outcome distribution used by autoregressive sampling (special
tokens are excluded and the two physical logits are renormalized).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch as GraphBatch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genksr.constants import WDEGREE_FEATURE, measurement_outcomes_for_task  # noqa: E402
from genksr.data.coupling_graphs import CouplingGraphsDataset  # noqa: E402
from genksr.data.sample_structure import get_sample_structure  # noqa: E402
from genksr.data.xxz_sparsity import (  # noqa: E402
    HAMILTONIAN_FAMILY,
    read_instance_rows,
    validate_xxz_sparsity_metadata,
)
from genksr.nn.common import make_std_mask  # noqa: E402
from genksr.nn.encoder import get_graph_encoder  # noqa: E402
from genksr.nn.gc_model import build_gc_model  # noqa: E402
from genksr.protocol import PROTOCOL_VERSION, SQKD_TASK  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/xxz_sparsity_1d"))
    parser.add_argument("--checkpoint", default="checkpoint_final.pth.tar")
    parser.add_argument("--n-qubit", type=int, required=True)
    parser.add_argument("--num-train-hamiltonians", type=int, required=True)
    parser.add_argument("--train-shots", type=int, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--fit-samples", type=int, required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(range(15)))
    parser.add_argument("--likelihood-shots", type=int, default=1000)
    parser.add_argument("--sample-shots", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=2026080300)
    parser.add_argument("--likelihood-batch-size", type=int, default=1000)
    parser.add_argument("--hamiltonian-ids", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_sample_seed(base_seed, n_qubit, hamiltonian_id, dimension):
    payload = (
        f"xxz-training-budget-sampling|{base_seed}|{n_qubit}|"
        f"{hamiltonian_id}|{dimension}"
    ).encode()
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False
    ) % (2**63 - 1)


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_csv_atomic(path, rows):
    require(rows, "cannot write an empty likelihood table")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_model(results_dir, checkpoint, device, expected_n_qubit):
    config_path = Path(results_dir) / "config.json"
    cfg = json.loads(config_path.read_text())
    data_cfg = cfg["data"]
    require(data_cfg.get("protocol_version") == PROTOCOL_VERSION,
            f"{results_dir}: not a canonical v3 checkpoint")
    require(data_cfg.get("task") == SQKD_TASK,
            f"{results_dir}: expected SQKD")
    require(data_cfg.get("hamiltonian_family") == HAMILTONIAN_FAMILY,
            f"{results_dir}: expected the frozen XXZ family")
    require(data_cfg.get("coupling_normalization") == "none",
            f"{results_dir}: XXZ edge scale must remain unnormalized")
    require(int(data_cfg.get("n_qubit")) == expected_n_qubit,
            f"{results_dir}: system-size mismatch")

    n_outcomes = measurement_outcomes_for_task(SQKD_TASK)
    require(data_cfg.get("n_outcomes") == n_outcomes,
            f"{results_dir}: incompatible output vocabulary")
    enc = cfg["encoder"]
    dec = cfg["decoder"]
    structure = get_sample_structure(data_cfg["sample_structure"])
    in_node_dim = 1 if enc["features"] == WDEGREE_FEATURE else expected_n_qubit
    encoder = get_graph_encoder(
        enc["arch"], in_node_dim=in_node_dim, gcn_dim=enc["gcn_dim"],
        gcn_layers=enc["gcn_layers"], d_model=dec["d_model"],
        qubits=expected_n_qubit,
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
    checkpoint_path = Path(results_dir) / "checkpoints" / checkpoint
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    require(all(torch.isfinite(parameter).all().item()
                for parameter in model.parameters()),
            f"{checkpoint_path}: checkpoint contains a non-finite parameter")
    return model, cfg, saved, checkpoint_path


def build_graph(coupling, hamiltonian_id, feature_type, device):
    dataset = CouplingGraphsDataset(
        coupling_matrices=np.asarray(coupling)[None, ...],
        coupling_matrices_ids=[hamiltonian_id],
        rng=np.random.default_rng(0),
        device=device,
        feature_type=feature_type,
    )
    graph, returned_id, _ = dataset[0]
    require(int(returned_id) == int(hamiltonian_id), "graph ID mismatch")
    return GraphBatch.from_data_list([graph]).to(device)


@torch.inference_mode()
def physical_nll_per_sample(model, graph, samples, condition_time, batch_size,
                            device):
    """Return per-qubit NLL under the normalized physical-outcome model."""
    samples = np.asarray(samples)
    require(samples.ndim == 2, f"expected a sample matrix, got {samples.shape}")
    require(np.all((samples == 0) | (samples == 1)),
            "held-out SQKD samples must be binary")
    valid_start = model.token_shift
    valid_stop = valid_start + model.n_outcomes
    graph_embedding = model.encoder(graph)
    output = []

    for start in range(0, len(samples), batch_size):
        # mmap-backed arrays are read-only; copy before constructing a tensor so
        # PyTorch never observes non-writable storage.
        bits = torch.as_tensor(
            np.array(samples[start:start + batch_size], copy=True),
            dtype=torch.long, device=device)
        start_column = torch.full(
            (len(bits), 1), model.start_token, dtype=torch.long, device=device)
        target_input = torch.cat(
            [start_column, bits[:, :-1] + model.token_shift], dim=1)
        mask = (
            make_std_mask(target_input, pad=model.pad_token)
            if model.decoder.needs_mask else None
        )
        embedded = model.tgt_embed(target_input)
        times = torch.full(
            (len(bits),), float(condition_time), dtype=torch.float32,
            device=device)
        time_embedding = model.noise_embed(times.unsqueeze(-1)).unsqueeze(1)
        decoded = model.decoder(embedded + graph_embedding + time_embedding, mask)
        physical_logits = model.generator(decoded)[..., valid_start:valid_stop]
        physical_log_probs = torch.log_softmax(physical_logits, dim=-1)
        selected = torch.gather(
            physical_log_probs, -1, bits.unsqueeze(-1)).squeeze(-1)
        output.append((-selected.mean(dim=1)).double().cpu().numpy())
    return np.concatenate(output)


def valid_saved_samples(path, expected_shape):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        samples = np.load(path, allow_pickle=False, mmap_mode="r")
        return (
            samples.shape == expected_shape
            and samples.dtype == np.uint8
            and np.all((samples == 0) | (samples == 1))
        )
    except (OSError, ValueError):
        return False


def main():
    args = parse_args()
    require(args.n_qubit > 0 and args.n_qubit % 2 == 0,
            "n-qubit must be a positive even integer")
    require(args.likelihood_shots > 0 and args.sample_shots > 0,
            "evaluation shot counts must be positive")
    require(args.likelihood_batch_size > 0, "batch size must be positive")
    require(args.dimensions and min(args.dimensions) >= 0,
            "dimensions must be non-empty and non-negative")
    require(len(args.dimensions) == len(set(args.dimensions)),
            "dimensions must be unique")
    require(args.device == "cpu" or torch.cuda.is_available(),
            f"requested device {args.device!r}, but CUDA is unavailable")

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples" / "test" / "model"
    sample_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.sample_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.sample_seed)

    model, cfg, checkpoint, checkpoint_path = load_model(
        args.results_dir, args.checkpoint, device, args.n_qubit)
    data_cfg = cfg["data"]
    require(int(data_cfg.get("hamiltonians")) == args.num_train_hamiltonians,
            "training Hamiltonian count does not match the job manifest")
    require(int(data_cfg.get("train_samples")) == args.train_shots,
            "training shots do not match the job manifest")
    require(int(cfg.get("seed")) == args.train_seed,
            "training seed does not match the job manifest")

    test_dir = args.data_root / f"{args.n_qubit}qubits" / "test"
    metadata = validate_xxz_sparsity_metadata(test_dir)
    require(int(metadata["n_qubit"]) == args.n_qubit,
            "test system-size mismatch")
    require(max(args.dimensions) < int(metadata["krylov_dim"]),
            "requested dimension is outside the saved test data")
    require(max(args.likelihood_shots, args.sample_shots) <= int(metadata["shots"]),
            "evaluation requires more shots than the frozen test set stores")

    instance_rows = sorted(
        read_instance_rows(test_dir), key=lambda row: int(row["split_id"]))
    instances = {int(row["split_id"]): row for row in instance_rows}
    if args.hamiltonian_ids is None:
        hamiltonian_ids = sorted(instances)
    else:
        hamiltonian_ids = sorted(set(args.hamiltonian_ids))
        require(set(hamiltonian_ids) <= set(instances),
                "requested held-out Hamiltonian ID is unavailable")

    likelihood_rows = []
    for position, hamiltonian_id in enumerate(hamiltonian_ids, start=1):
        row = instances[hamiltonian_id]
        coupling_path = test_dir / f"coupling_matrix_id{hamiltonian_id}.npy"
        coupling = np.load(coupling_path, allow_pickle=False)
        graph = build_graph(
            coupling, hamiltonian_id, cfg["encoder"]["features"], device)
        dt = float(row["dt"])

        for dimension in sorted(args.dimensions):
            exact_path = test_dir / f"data_D{dimension}_id{hamiltonian_id}.npy"
            exact_samples = np.load(
                exact_path, allow_pickle=False, mmap_mode="r")[:args.likelihood_shots]
            require(len(exact_samples) == args.likelihood_shots,
                    f"insufficient likelihood samples in {exact_path}")
            condition_time = dimension * dt
            per_sample_nll = physical_nll_per_sample(
                model, graph, exact_samples, condition_time,
                args.likelihood_batch_size, device)
            require(np.all(np.isfinite(per_sample_nll)),
                    "held-out likelihood contains a non-finite value")
            likelihood_rows.append({
                "job_name": args.job_name,
                "N": int(args.n_qubit),
                "num_train_hamiltonians": int(args.num_train_hamiltonians),
                "train_shots_per_condition": int(args.train_shots),
                "fit_samples": int(args.fit_samples),
                "train_seed": int(args.train_seed),
                "hamiltonian_id": int(hamiltonian_id),
                "paired_key": row["paired_key"],
                "regime": row["regime"],
                "J_xy": float(row["J_xy"]),
                "dimension": int(dimension),
                "condition_time": float(condition_time),
                "likelihood_shots": int(args.likelihood_shots),
                "nll_nats_per_qubit": float(np.mean(per_sample_nll)),
                "nll_sample_std": float(
                    np.std(per_sample_nll, ddof=1)
                    if len(per_sample_nll) > 1 else 0.0),
            })

            generated_path = sample_dir / (
                f"model_samples_D{dimension}_{args.sample_shots}_"
                f"id{hamiltonian_id}.npy")
            if not valid_saved_samples(
                    generated_path, (args.sample_shots, args.n_qubit)):
                # A context-specific common random seed makes every cell paired
                # and makes interrupted/resumed evaluation bitwise independent
                # of which earlier files were already present.
                sample_context_seed = context_sample_seed(
                    args.sample_seed, args.n_qubit, hamiltonian_id, dimension)
                torch.manual_seed(sample_context_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(sample_context_seed)
                generated = model.sample(
                    samples=args.sample_shots,
                    coupling_graph=graph,
                    noise_factor=condition_time,
                    qubits=args.n_qubit,
                    batch_size=args.sample_shots,
                    print_progress=False,
                )
                generated = np.asarray(generated)
                require(generated.shape == (args.sample_shots, args.n_qubit),
                        f"generated sample shape mismatch: {generated.shape}")
                require(np.all((generated == 0) | (generated == 1)),
                        "model generated a non-binary SQKD outcome")
                temporary = generated_path.with_suffix(".npy.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, generated.astype(np.uint8, copy=False))
                temporary.replace(generated_path)

        print(
            f"{args.job_name}: held-out Hamiltonian "
            f"{position}/{len(hamiltonian_ids)} complete",
            flush=True,
        )

    likelihood_path = args.output_dir / "heldout_nll_context.csv"
    write_csv_atomic(likelihood_path, likelihood_rows)
    protocol = {
        "status": "complete",
        "job_name": args.job_name,
        "source_results_dir": str(args.results_dir.resolve()),
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "N": args.n_qubit,
        "num_train_hamiltonians": args.num_train_hamiltonians,
        "train_shots_per_condition": args.train_shots,
        "fit_samples": args.fit_samples,
        "train_seed": args.train_seed,
        "data_root": str(args.data_root.resolve()),
        "dimensions": sorted(args.dimensions),
        "heldout_hamiltonian_ids": hamiltonian_ids,
        "likelihood_shots": args.likelihood_shots,
        "likelihood_definition": (
            "mean negative log probability per qubit after renormalizing each "
            "autoregressive conditional over the two physical SQKD outcomes"),
        "sample_shots_per_condition": args.sample_shots,
        "sample_seed": args.sample_seed,
        "sample_context_seed_definition": (
            "SHA256(base seed, N, held-out Hamiltonian ID, dimension); the "
            "same context seed is used for every training-budget cell"),
        "sample_repeats": 1,
        "wall_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(args.output_dir / "protocol.json", protocol)
    print(
        f"COMPLETE {args.job_name}: {len(likelihood_rows)} likelihood contexts, "
        f"{protocol['wall_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
