#!/usr/bin/env python3
"""Parallel held-out evaluation launcher for the XXZ training-budget grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_RUN = ROOT / "artifacts" / "v3_xxz_training_budget_20260801"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "v3_xxz_training_budget_20260801" /
    "formal_evaluation_seed0"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-dir", type=Path,
                        default=DEFAULT_TRAINING_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data" / "xxz_sparsity_1d")
    parser.add_argument("--python",
                        default=sys.executable)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--likelihood-shots", type=int, default=1000)
    parser.add_argument("--sample-shots", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=2026080300)
    parser.add_argument("--likelihood-batch-size", type=int, default=1000)
    parser.add_argument("--primary-nll-gap", type=float, default=0.01)
    parser.add_argument("--primary-energy-gap", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def evaluation_complete(output_dir, job):
    protocol_path = Path(output_dir) / "protocol.json"
    nll_path = Path(output_dir) / "heldout_nll_context.csv"
    if not protocol_path.is_file() or not nll_path.is_file():
        return False
    try:
        protocol = json.loads(protocol_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected_files = 30 * int(job["krylov_conditions"])
    sample_dir = Path(output_dir) / "samples" / "test" / "model"
    return (
        protocol.get("status") == "complete"
        and protocol.get("job_name") == job["name"]
        and int(protocol.get("sample_shots_per_condition", -1)) > 0
        and len(list(sample_dir.glob("model_samples_D*_*.npy"))) == expected_files
    )


def build_command(args, job, training_run_dir, output_dir):
    return [
        str(args.python),
        "scripts/experiments/evaluate_xxz_training_budget_model.py",
        "--job-name", job["name"],
        "--results-dir", str(training_run_dir),
        "--output-dir", str(output_dir),
        "--data-root", str(args.data_root),
        "--n-qubit", str(job["n_qubit"]),
        "--num-train-hamiltonians", str(job["num_hamiltonians"]),
        "--train-shots", str(job["train_shots"]),
        "--train-seed", str(job["seed"]),
        "--fit-samples", str(job["fit_samples"]),
        "--likelihood-shots", str(args.likelihood_shots),
        "--sample-shots", str(args.sample_shots),
        "--sample-seed", str(args.sample_seed),
        "--likelihood-batch-size", str(args.likelihood_batch_size),
        "--device", "cuda",
    ]


def main():
    args = parse_args()
    if args.workers_per_gpu <= 0 or args.omp_threads <= 0:
        raise ValueError("worker counts and thread counts must be positive")
    if args.likelihood_shots <= 0 or args.sample_shots <= 0:
        raise ValueError("evaluation shot counts must be positive")
    if args.primary_nll_gap < 0 or args.primary_energy_gap < 0:
        raise ValueError("accuracy thresholds must be non-negative")
    if not Path(args.python).is_file():
        raise FileNotFoundError(args.python)

    jobs_path = args.training_run_dir / "jobs.json"
    training_status_path = args.training_run_dir / "status.json"
    evaluation_plan_path = args.training_run_dir / "evaluation_plan.json"
    evaluation_plan_hash_path = args.training_run_dir / "evaluation_plan.sha256"
    jobs = json.loads(jobs_path.read_text())
    training_status = json.loads(training_status_path.read_text())
    evaluation_plan = json.loads(evaluation_plan_path.read_text())
    registered_plan_hash = evaluation_plan_hash_path.read_text().split()[0]
    if sha256_file(evaluation_plan_path) != registered_plan_hash:
        raise ValueError("registered evaluation plan hash mismatch")
    if len(jobs) != 48 or {job["seed"] for job in jobs} != {0}:
        raise ValueError(
            "formal single-seed evaluation expects the 48-cell seed-0 grid")
    registered_criterion = evaluation_plan["primary_accuracy_criterion"]
    if not math.isclose(
            args.primary_nll_gap,
            float(registered_criterion[
                "nll_gap_to_same_size_NH45_M1000_anchor_nats_per_qubit_max"]),
            rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("NLL threshold differs from the registered plan")
    if not math.isclose(
            args.primary_energy_gap,
            float(registered_criterion[
                "mean_absolute_normalized_energy_difference_vs_direct_max"]),
            rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("energy threshold differs from the registered plan")
    if args.sample_seed != int(
            evaluation_plan["heldout_protocol"]["downstream_sampling_seed"]):
        raise ValueError("sample seed differs from the registered plan")
    if args.likelihood_shots != int(
            evaluation_plan["heldout_protocol"]
            ["likelihood_reference_shots_per_condition"]):
        raise ValueError("likelihood shots differ from the registered plan")
    if args.sample_shots != int(
            evaluation_plan["heldout_protocol"]
            ["downstream_generated_shots_per_condition"]):
        raise ValueError("downstream shots differ from the registered plan")

    evaluation_jobs = []
    for job in jobs:
        state = training_status.get(job["name"], {})
        if state.get("state") != "completed":
            raise RuntimeError(
                f"training is not complete for {job['name']}: "
                f"{state.get('state')!r}")
        training_run_dir = Path(state["run_dir"])
        checkpoint = training_run_dir / "checkpoints" / "checkpoint_final.pth.tar"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output_dir = args.output_dir / "models" / job["name"]
        evaluation_jobs.append({
            **job,
            "training_run_dir": str(training_run_dir),
            "output_dir": str(output_dir),
            "evaluation_command": build_command(
                args, job, training_run_dir, output_dir),
        })
    evaluation_jobs.sort(
        key=lambda job: (-job["n_qubit"], -job["fit_samples"], job["name"]))

    if args.dry_run:
        print(json.dumps(evaluation_jobs, indent=2))
        return 0

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"evaluation output exists; inspect it and pass --resume: "
            f"{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)

    protocol_path = args.output_dir / "protocol.json"
    if not protocol_path.exists():
        protocol = {
            "status": "running",
            "study": "single-training-seed finite-size training-budget evidence",
            "resource_interpretation": (
                "Noiseless simulator samples; counts are training samples and "
                "QPU-shot-equivalent acquisition budgets, not measured QPU use."),
            "budget_scope": (
                "Held-out likelihood and direct-reference samples are "
                "evaluation-only and excluded from the fit budget."),
            "training_seed_repeats": 1,
            "training_seeds": [0],
            "sizes": sorted({job["n_qubit"] for job in jobs}),
            "num_train_hamiltonians": sorted(
                {job["num_hamiltonians"] for job in jobs}),
            "train_shots_per_condition": sorted(
                {job["train_shots"] for job in jobs}),
            "D_train": 15,
            "D_eval": 15,
            "heldout_hamiltonians_per_size": 30,
            "likelihood_shots_per_condition": args.likelihood_shots,
            "downstream_sample_shots_per_condition": args.sample_shots,
            "downstream_sample_seed": args.sample_seed,
            "downstream_sampling_repeats": 1,
            "primary_accuracy_criterion": {
                "nll_gap_to_same_N_full_budget_anchor_nats_per_qubit_max": (
                    args.primary_nll_gap),
                "mean_absolute_normalized_energy_difference_vs_direct_max": (
                    args.primary_energy_gap),
                "combination": "both conditions must hold",
            },
            "accuracy_criterion_registration": (
                "written by the launcher before held-out evaluation outputs existed"),
            "interpretation_limit": (
                "No training-seed uncertainty, asymptotic exponent, or "
                "polynomial-versus-exponential conclusion is supported."),
            "training_run_dir": str(args.training_run_dir.resolve()),
            "data_root": str(args.data_root.resolve()),
            "ratio_manifest_sha256": sha256_file(
                args.data_root / "ratio_manifest.csv"),
            "jobs_sha256": sha256_file(jobs_path),
            "evaluation_plan": str(evaluation_plan_path.resolve()),
            "evaluation_plan_sha256": sha256_file(evaluation_plan_path),
            "model_evaluator_sha256": sha256_file(
                ROOT / "scripts" / "experiments" /
                "evaluate_xxz_training_budget_model.py"),
            "launch_unix_time": time.time(),
        }
        write_json(protocol_path, protocol)
    else:
        protocol = json.loads(protocol_path.read_text())

    status_path = args.output_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    pending = []
    for job in evaluation_jobs:
        if evaluation_complete(job["output_dir"], job):
            status[job["name"]] = {
                "state": "completed",
                "output_dir": job["output_dir"],
                "recovered_on_resume": True,
            }
        else:
            pending.append(job)
            status[job["name"]] = {"state": "pending"}
    write_json(status_path, status)

    slots = [gpu for gpu in args.gpus for _ in range(args.workers_per_gpu)]
    active = {}
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}; terminating active workers", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while pending or active:
        free_slots = list(slots)
        for entry in active.values():
            free_slots.remove(entry["gpu"])

        while pending and free_slots and not stop_requested:
            job = pending.pop(0)
            gpu = free_slots.pop(0)
            Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / "logs" / f"{job['name']}.log"
            log_handle = log_path.open("w")
            environment = os.environ.copy()
            environment.update({
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": str(args.omp_threads),
                "MKL_NUM_THREADS": str(args.omp_threads),
                "OPENBLAS_NUM_THREADS": str(args.omp_threads),
            })
            started = time.time()
            process = subprocess.Popen(
                job["evaluation_command"], cwd=ROOT, env=environment,
                stdout=log_handle, stderr=subprocess.STDOUT)
            active[process.pid] = {
                "process": process,
                "job": job,
                "gpu": gpu,
                "started": started,
                "log_handle": log_handle,
                "log_path": str(log_path),
            }
            status[job["name"]] = {
                "state": "running", "gpu": gpu, "pid": process.pid,
                "started_unix_time": started, "log": str(log_path),
            }
            print(
                f"START {job['name']} gpu={gpu} pid={process.pid} "
                f"pending={len(pending)} active={len(active)}", flush=True)
            write_json(status_path, status)

        if stop_requested:
            for entry in active.values():
                entry["process"].terminate()
        if active:
            time.sleep(args.poll_seconds)

        for pid, entry in list(active.items()):
            return_code = entry["process"].poll()
            if return_code is None:
                continue
            entry["log_handle"].close()
            job = entry["job"]
            complete = evaluation_complete(job["output_dir"], job)
            state = "completed" if return_code == 0 and complete else "failed"
            status[job["name"]] = {
                "state": state,
                "gpu": entry["gpu"],
                "pid": pid,
                "return_code": return_code,
                "elapsed_seconds": time.time() - entry["started"],
                "log": entry["log_path"],
                "output_dir": job["output_dir"],
            }
            print(
                f"{state.upper()} {job['name']} gpu={entry['gpu']} "
                f"pending={len(pending)}", flush=True)
            del active[pid]
            write_json(status_path, status)

        if stop_requested and not active:
            break

    counts = {
        state: sum(item.get("state") == state for item in status.values())
        for state in ("completed", "failed", "pending", "running")
    }
    write_json(args.output_dir / "model_evaluation_summary.json", {
        "finish_unix_time": time.time(), "counts": counts,
        "total_jobs": len(evaluation_jobs), "stopped": stop_requested,
    })
    if stop_requested or counts["failed"]:
        return 1

    if not args.skip_energy:
        energy_command = [
            str(args.python),
            "scripts/experiments/evaluate_xxz_training_budget_energy.py",
            "--training-run-dir", str(args.training_run_dir),
            "--evaluation-dir", str(args.output_dir),
            "--data-root", str(args.data_root),
        ]
        subprocess.run(energy_command, cwd=ROOT, check=True)
        subprocess.run([
            str(args.python),
            "scripts/experiments/plot_xxz_training_budget.py",
            "--evaluation-dir", str(args.output_dir),
        ], cwd=ROOT, check=True)

    protocol = json.loads(protocol_path.read_text())
    protocol["status"] = "complete"
    protocol["finish_unix_time"] = time.time()
    write_json(protocol_path, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
