#!/usr/bin/env python
"""Reproduce GenKSR paper Appendix B.4, Figure 7.

Training and inference wall-clock time vs. number of qubits, for the
Transformer and Mamba GenKSR backbones.

This script builds the *actual* GenKSR v2 models (paper Table 4 configuration:
d_model=128, 4 decoder layers, transformer with 4 heads, GCN encoder
gcn_proj_3_16, Mamba with d_state=16/d_conv=4/expand=2) and times them on
synthetic coupling graphs + measurement batches. No trained checkpoints or
real datasets are required: the timing only depends on tensor shapes and the
model architecture, so synthetic inputs reproduce the scaling curves faithfully.

For each qubit count and backbone we measure
  (a) mean wall-clock per *training step* (forward + KL loss + backward + step)
      on a small synthetic batch, and
  (b) mean *inference time* (autoregressive sampling of `--shots` shots),
averaged over `--repeats` timed runs after a warmup run.

Outputs (written to --output-dir):
  fig7_timing.pdf, fig7_timing.png  : two log-log panels (train, inference)
  timing.csv                        : raw per-(backbone, qubits) measurements

Mamba requires `mamba-ssm` + a CUDA build; if it is not importable the Mamba
backbone is skipped gracefully (a clear message is logged) and the Transformer
curve is still produced. The device (CUDA if available, else CPU) is detected
automatically and recorded in both the CSV and the figure.

Example
-------
    python scripts/figures/fig7_timing.py \
        --qubits 10 20 40 80 \
        --backbones transformer mamba \
        --repeats 5 --shots 256 \
        --output-dir figures/
"""
import argparse
import csv
import os
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
# Heavy imports are done lazily inside main() so that --help works even when
# torch / torch_geometric are not installed in the active interpreter.
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce GenKSR Appendix B.4 Figure 7 (training/inference "
                    "time vs number of qubits) for Transformer and Mamba backbones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qubits", type=int, nargs="+", default=[10, 20, 40, 80],
                   help="Qubit counts (system sizes) to benchmark.")
    p.add_argument("--backbones", type=str, nargs="+", default=["transformer", "mamba"],
                   choices=["transformer", "mamba"],
                   help="Decoder backbones to benchmark (mamba is skipped if "
                        "mamba-ssm is not importable).")
    p.add_argument("--d-model", type=int, default=128, help="Model dimension (Table 4).")
    p.add_argument("--n-layers", type=int, default=4, help="Number of decoder layers (Table 4).")
    p.add_argument("--n-heads", type=int, default=4, help="Transformer attention heads (Table 4).")
    p.add_argument("--gcn-dim", type=int, default=16, help="GCN encoder node dim (Table 4: gcn_proj_3_16).")
    p.add_argument("--gcn-layers", type=int, default=3, help="GCN encoder layers (Table 4).")
    p.add_argument("--d-state", type=int, default=16, help="Mamba state dim (Table 4).")
    p.add_argument("--d-conv", type=int, default=4, help="Mamba conv width (Table 4).")
    p.add_argument("--expand", type=int, default=2, help="Mamba expansion factor (Table 4).")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout (Table 4).")
    p.add_argument("--train-batch-size", type=int, default=32,
                   help="Synthetic batch size used for the training-step timing.")
    p.add_argument("--repeats", type=int, default=5,
                   help="Number of timed repeats (a separate warmup run is always done first).")
    p.add_argument("--shots", type=int, default=256,
                   help="Number of shots sampled for the inference timing.")
    p.add_argument("--sample-batch-size", type=int, default=256,
                   help="Per-call batch size used inside autoregressive sampling.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic inputs.")
    p.add_argument("--output-dir", type=str, default="figures/",
                   help="Directory to write fig7_timing.{pdf,png} and timing.csv.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


def log(msg):
    print(f"[fig7_timing] {msg}", flush=True)


def mamba_available():
    """True only if mamba-ssm is importable AND exposes a usable Mamba class.

    The v2 mamba decoder module imports fine even without mamba-ssm (it falls
    back to ``Mamba = None``), so we must check the symbol itself, not merely
    that the import succeeds.
    """
    try:
        from mamba_ssm import Mamba  # noqa: F401
        return Mamba is not None
    except Exception:
        return False


def make_coupling_matrix(n_qubit, rng):
    """Synthetic 1D-chain-like symmetric coupling matrix with random weights.

    A nearest-neighbour ring guarantees a connected graph (non-empty edge set)
    so the GCN encoder produces a well-defined embedding for every qubit count.
    """
    mat = np.zeros((n_qubit, n_qubit), dtype=np.float32)
    for i in range(n_qubit):
        j = (i + 1) % n_qubit
        w = float(rng.uniform(0.5, 1.5))
        mat[i, j] = w
        mat[j, i] = w
    return mat


def build_graph_batch(n_qubit, rng, device, features):
    """Build a single-graph torch_geometric Batch on `device`."""
    from torch_geometric.data import Batch as GraphBatch
    from genksr.data.coupling_graphs import CouplingGraphsDataset
    from genksr.data.dataset import normalize_coupling_matrices

    coupling = make_coupling_matrix(n_qubit, rng)[None, ...]  # (1, nq, nq)
    coupling = normalize_coupling_matrices(coupling)

    ds = CouplingGraphsDataset(
        coupling_matrices=coupling,
        coupling_matrices_ids=[0],
        rng=rng,
        device=device,
        feature_type=features,
    )
    data = ds[0][0]
    return GraphBatch.from_data_list([data]).to(device)


def build_model(backbone, n_qubit, args, sample_struct, device):
    """Construct the Table-4 GenKSR v2 model for one backbone + qubit count."""
    from genksr.constants import NUM_MMT_OUTCOMES, ONE_HOT_FEATURE
    from genksr.nn.encoder import get_graph_encoder
    from genksr.nn.gc_model import build_gc_model

    # one_hot node features -> in_node_dim == n_qubit (matches scripts/train.py)
    features = ONE_HOT_FEATURE
    in_node_dim = n_qubit

    encoder = get_graph_encoder(
        f"gcn_proj_{args.gcn_layers}_{args.gcn_dim}",
        in_node_dim=in_node_dim,
        gcn_dim=args.gcn_dim,
        gcn_layers=args.gcn_layers,
        d_model=args.d_model,
        qubits=n_qubit,
    )

    model = build_gc_model(
        decoder_type=backbone,
        n_outcomes=NUM_MMT_OUTCOMES,
        encoder=encoder,
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_ff=4 * args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        pad_token=sample_struct.pad_token,
        start_token=sample_struct.start_token,
        end_token=sample_struct.end_token,
        token_shift=sample_struct.token_shift,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
    )
    model.to(device)
    return model, features


def make_training_batch(model, n_qubit, batch_size, sample_struct, device, rng):
    """Build a synthetic (tgt, tgt_y, tgt_mask, noise_factor) training batch.

    Mirrors genksr.data.dataloading.TgtBatch / DatasetGCModel: a sequence is
    [start_token, m_1, ..., m_qubits] with measurement tokens shifted by
    token_shift into [token_shift, token_shift + NUM_MMT_OUTCOMES).
    """
    import torch
    from genksr.constants import NUM_MMT_OUTCOMES
    from genksr.data.dataloading import TgtBatch

    shift = sample_struct.token_shift
    # measurement outcomes in {0..5} -> tokens in {shift .. shift+5}
    meas = rng.integers(0, NUM_MMT_OUTCOMES, size=(batch_size, n_qubit)) + shift
    starts = np.full((batch_size, 1), sample_struct.start_token, dtype=np.int64)
    seq = np.concatenate([starts, meas.astype(np.int64)], axis=1)

    tgt_full = torch.from_numpy(seq).to(device)
    batch = TgtBatch(tgt_full, pad=sample_struct.pad_token)
    batch.tgt = batch.tgt.to(device)
    batch.tgt_y = batch.tgt_y.to(device)
    batch.tgt_mask = batch.tgt_mask.to(device)

    noise_factor = torch.rand(batch_size, device=device, dtype=torch.float32)
    return batch, noise_factor


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_training_step(model, graph_batch, n_qubit, args, sample_struct, device, rng):
    """Mean wall-clock per training step (forward + KL loss + backward + step)."""
    import torch
    from genksr.nn.common import KLDivLoss

    criterion = KLDivLoss(padding_idx=model.pad_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    batch, noise_factor = make_training_batch(
        model, n_qubit, args.train_batch_size, sample_struct, device, rng
    )
    norm = batch.ntokens.item()
    if norm == 0:
        norm = 1.0

    def one_step():
        _, out = model.forward(
            batch.tgt, batch.tgt_mask,
            coupling_graph=graph_batch, noise_factor=noise_factor,
        )
        log_probs = model.generator(out)
        loss = criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)),
            batch.tgt_y.contiguous().view(-1),
        ) / norm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # warmup
    one_step()
    sync(device)

    times = []
    for _ in range(args.repeats):
        sync(device)
        t0 = time.perf_counter()
        one_step()
        sync(device)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), float(np.std(times))


def time_inference(model, graph_batch, n_qubit, args, device):
    """Mean wall-clock for autoregressive sampling of `--shots` shots."""
    import torch

    model.eval()

    def one_sample():
        with torch.no_grad():
            model.sample(
                samples=args.shots,
                coupling_graph=graph_batch,
                noise_factor=0.5,
                qubits=n_qubit,
                batch_size=args.sample_batch_size,
                print_progress=False,
            )

    # warmup
    one_sample()
    sync(device)

    times = []
    for _ in range(args.repeats):
        sync(device)
        t0 = time.perf_counter()
        one_sample()
        sync(device)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), float(np.std(times))


def main():
    args = parse_args()

    # --- heavy imports (give a clear message if the env lacks torch) -------- #
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception as e:
        log("FATAL: could not import torch / torch_geometric in this interpreter.")
        log(f"        {type(e).__name__}: {e}")
        log("        Run with the project environment, e.g. ./.venv/bin/python")
        sys.exit(1)

    from genksr.data.sample_structure import get_sample_structure

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    device_name = "cpu"
    if device.type == "cuda":
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "cuda"
    log(f"Using device: {device.type} ({device_name})")

    # --- resolve which backbones we can actually run ----------------------- #
    backbones = list(dict.fromkeys(args.backbones))  # de-dup, keep order
    have_mamba = mamba_available()
    if "mamba" in backbones and not have_mamba:
        log("Mamba backbone requested but `mamba-ssm` (with a CUDA build) is not "
            "importable -> SKIPPING Mamba. Install with `pip install mamba-ssm` "
            "on a CUDA machine to enable it. Transformer curve will still be produced.")
        backbones = [b for b in backbones if b != "mamba"]
    if "mamba" in backbones and device.type != "cuda":
        log("Mamba requires CUDA but no GPU is available -> SKIPPING Mamba.")
        backbones = [b for b in backbones if b != "mamba"]

    if not backbones:
        log("FATAL: no runnable backbones remain. Nothing to do.")
        sys.exit(1)

    sample_struct = get_sample_structure(version=2)  # pad=-1, start=0, end=-1, shift=1
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # results[backbone] = list of dicts
    results = {b: [] for b in backbones}

    for backbone in backbones:
        for n_qubit in sorted(args.qubits):
            row = {
                "backbone": backbone,
                "qubits": n_qubit,
                "device": device.type,
                "device_name": device_name,
                "train_time_s": float("nan"),
                "train_time_std_s": float("nan"),
                "infer_time_s": float("nan"),
                "infer_time_std_s": float("nan"),
                "shots": args.shots,
                "train_batch_size": args.train_batch_size,
                "repeats": args.repeats,
            }

            # Build model + synthetic graph for this (backbone, qubits).
            try:
                model, features = build_model(backbone, n_qubit, args, sample_struct, device)
                graph_batch = build_graph_batch(n_qubit, rng, device, features)
            except Exception as e:
                log(f"[{backbone} | {n_qubit}q] build FAILED: {type(e).__name__}: {e}")
                results[backbone].append(row)
                continue

            # Training-step timing.
            try:
                t_mean, t_std = time_training_step(
                    model, graph_batch, n_qubit, args, sample_struct, device, rng
                )
                row["train_time_s"] = t_mean
                row["train_time_std_s"] = t_std
                log(f"[{backbone} | {n_qubit}q] train step: {t_mean*1e3:.2f} ms "
                    f"(+/- {t_std*1e3:.2f} ms)")
            except Exception as e:
                log(f"[{backbone} | {n_qubit}q] train timing FAILED: {type(e).__name__}: {e}")

            # Inference timing.
            try:
                i_mean, i_std = time_inference(model, graph_batch, n_qubit, args, device)
                row["infer_time_s"] = i_mean
                row["infer_time_std_s"] = i_std
                log(f"[{backbone} | {n_qubit}q] inference ({args.shots} shots): "
                    f"{i_mean:.3f} s (+/- {i_std:.3f} s)")
            except Exception as e:
                log(f"[{backbone} | {n_qubit}q] inference timing FAILED: {type(e).__name__}: {e}")

            results[backbone].append(row)

            # Free GPU memory between configurations.
            del model, graph_batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- write CSV --------------------------------------------------------- #
    csv_path = os.path.join(args.output_dir, "timing.csv")
    fieldnames = [
        "backbone", "qubits", "device", "device_name",
        "train_time_s", "train_time_std_s",
        "infer_time_s", "infer_time_std_s",
        "shots", "train_batch_size", "repeats",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for backbone in backbones:
            for row in results[backbone]:
                writer.writerow(row)
    log(f"Wrote {csv_path}")

    # --- plot -------------------------------------------------------------- #
    try:
        plot_path = make_plot(results, backbones, device, device_name, args)
        log(f"Wrote {plot_path}.pdf and {plot_path}.png")
    except Exception as e:
        log(f"plotting FAILED: {type(e).__name__}: {e}")

    log("Done.")


def make_plot(results, backbones, device, device_name, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"transformer": "tab:blue", "mamba": "tab:red"}
    markers = {"transformer": "o", "mamba": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (axes[0], "train_time_s", "train_time_std_s",
         f"Training time / step\n(batch={args.train_batch_size})", "training time [s]"),
        (axes[1], "infer_time_s", "infer_time_std_s",
         f"Inference time\n({args.shots} shots)", "inference time [s]"),
    ]

    for ax, key, ekey, title, ylabel in panels:
        plotted = False
        for backbone in backbones:
            rows = results[backbone]
            xs = np.array([r["qubits"] for r in rows], dtype=float)
            ys = np.array([r[key] for r in rows], dtype=float)
            es = np.array([r[ekey] for r in rows], dtype=float)
            mask = np.isfinite(ys) & (ys > 0)
            if not mask.any():
                continue
            ax.errorbar(
                xs[mask], ys[mask], yerr=es[mask],
                marker=markers.get(backbone, "o"),
                color=colors.get(backbone, None),
                capsize=3, label=backbone, linewidth=1.8, markersize=6,
            )
            plotted = True
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of qubits")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if plotted:
            ax.legend()

    fig.suptitle(
        f"GenKSR Fig. 7 - timing vs. system size  "
        f"(device: {device.type} / {device_name})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    base = os.path.join(args.output_dir, "fig7_timing")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=150)
    plt.close(fig)
    return base


if __name__ == "__main__":
    main()
