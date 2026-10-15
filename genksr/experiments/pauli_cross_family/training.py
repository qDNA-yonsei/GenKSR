"""Standalone trainer for the isolated Pauli cross-family model."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genksr.training.lr_schedulers import warm_up_cosine_lr_scheduler

from .dataset import PauliSampleDataset, load_protocol
from .model import PauliConditionalARModel, build_model, parameter_count


def resolve_device(requested: str | None = None) -> torch.device:
    requested = "auto" if requested is None else str(requested)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable")
    return device


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_or_verify_json(path: Path, payload: Mapping) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError(f"existing run metadata differs: {path}")
    if not path.exists():
        path.write_bytes(encoded)


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict:
    return {
        name: tensor.to(device, non_blocking=device.type == "cuda")
        for name, tensor in batch.items()
    }


def train_model(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    device: str = "auto",
    resume: bool = False,
) -> dict:
    """Train one model seed using only the frozen source_train split."""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(dataset_root)
    config = protocol["config"]
    training_config = config["training"]
    model_config = config["model"]
    n_qubit = int(config["data"]["n_qubit"])
    seed = int(seed)
    resolved_device = resolve_device(device)
    seed_everything(seed)
    if resolved_device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    dataset = PauliSampleDataset(dataset_root)
    batch_size = int(training_config["batch_size"])
    epochs = int(training_config["epochs"])
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        num_workers=int(training_config.get("data_loader_workers", 0)),
        pin_memory=resolved_device.type == "cuda",
        generator=data_generator,
        drop_last=False,
    )
    model = build_model(model_config, n_qubit=n_qubit).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["lr"]),
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    total_steps = len(loader) * epochs
    warmup_steps = int(float(training_config.get("warmup_fraction", 0.05)) * total_steps)
    scheduler = warm_up_cosine_lr_scheduler(
        optimizer,
        epochs=total_steps,
        warm_up_epochs=warmup_steps,
        eta_min=float(training_config.get("final_lr", 1e-7)),
    )
    target_family_hamiltonians = sum(
        row["family"] == protocol["target_family"] for row in dataset.rows
    )
    target_family_samples_used = (
        target_family_hamiltonians
        * int(config["data"]["depth"])
        * int(config["data"]["train_shots_per_condition"])
    )

    run_metadata = {
        "experiment_name": protocol["experiment_name"],
        "protocol_sha256": protocol["protocol_sha256"],
        "dataset_root": str(dataset_root.resolve()),
        "seed": seed,
        "requested_device": device,
        "resolved_device": str(resolved_device),
        "model_config": model.model_config,
        "training_config": training_config,
        "num_parameters": parameter_count(model),
        "num_training_samples": len(dataset),
        "num_training_hamiltonians": len(dataset.rows),
        "total_optimizer_steps": total_steps,
        "target_family_samples_used": target_family_samples_used,
        "checkpoint_selection": "fixed final epoch; no source/target validation selection",
    }
    _write_or_verify_json(output_dir / "run.json", run_metadata)

    start_epoch = 1
    global_step = 0
    checkpoint_path = output_dir / "checkpoint_last.pt"
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=resolved_device, weights_only=False
        )
        if checkpoint["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("resume checkpoint protocol mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    history_path = output_dir / "history.jsonl"
    if start_epoch == 1 and history_path.exists():
        raise FileExistsError(
            f"history already exists; use --resume or a new output directory: {history_path}"
        )

    wall_start = time.perf_counter()
    checkpoint_every = int(training_config.get("checkpoint_every", 5))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        sample_count = 0
        for batch in loader:
            batch = _move_batch(batch, resolved_device)
            logits = model(
                batch["bits"],
                pauli_codes=batch["pauli_codes"],
                coefficients=batch["coefficients"],
                term_mask=batch["term_mask"],
                tau=batch["tau"],
            )
            loss = F.cross_entropy(
                logits.reshape(-1, model.n_outcomes), batch["bits"].reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_clip = training_config.get("gradient_clip")
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            scheduler.step()
            batch_samples = int(batch["bits"].shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_samples
            sample_count += batch_samples
            global_step += 1

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_nll_per_qubit": loss_sum / sample_count,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        print(
            f"[seed={seed} epoch={epoch}/{epochs}] "
            f"nll/q={epoch_record['train_nll_per_qubit']:.6f} "
            f"lr={epoch_record['learning_rate']:.3e} "
            f"seconds={epoch_record['epoch_seconds']:.1f}",
            flush=True,
        )

        if epoch % checkpoint_every == 0 or epoch == epochs:
            checkpoint_payload = {
                "protocol_sha256": protocol["protocol_sha256"],
                "epoch": epoch,
                "global_step": global_step,
                "seed": seed,
                "model_config": model.model_config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch_record": epoch_record,
            }
            _atomic_torch_save(checkpoint_payload, checkpoint_path)

    final_checkpoint = output_dir / "model_final.pt"
    final_payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "epoch": epochs,
        "global_step": global_step,
        "seed": seed,
        "model_config": model.model_config,
        "model_state_dict": model.state_dict(),
    }
    _atomic_torch_save(final_payload, final_checkpoint)
    summary = {
        **run_metadata,
        "epochs_completed": epochs,
        "global_steps_completed": global_step,
        "training_wall_seconds": time.perf_counter() - wall_start,
        "final_checkpoint": str(final_checkpoint.resolve()),
    }
    _write_or_verify_json(output_dir / "training_complete.json", summary)
    return summary


def load_trained_model(
    checkpoint_path: str | Path,
    *,
    protocol: Mapping,
    device: str | torch.device = "cpu",
) -> tuple[PauliConditionalARModel, dict]:
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("checkpoint and dataset protocol do not match")
    model = PauliConditionalARModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
