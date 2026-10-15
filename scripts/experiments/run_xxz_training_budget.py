#!/usr/bin/env python3
"""Parallel launcher for the finite-size XXZ training-budget study.

Each result root is unique to one (n, N_H, M_train, seed) cell.  The launcher
uses longest-processing-time ordering and a fixed number of workers per GPU so
the expensive cells start first and the tail remains short.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "v3_xxz_training_budget_20260801"
DEFAULT_RUN_DIR = ROOT / "artifacts" / "v3_xxz_training_budget_20260801"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument(
        "--hamiltonians", type=int, nargs="+", default=[9, 18, 36, 45])
    parser.add_argument(
        "--train-shots", type=int, nargs="+", default=[125, 250, 500, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--python", default=sys.executable)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_name(n_qubit, num_hamiltonians, train_shots, seed):
    return (
        f"n{n_qubit}_nh{num_hamiltonians}_m{train_shots}_seed{seed}"
    )


def build_jobs(args):
    jobs = []
    for n_qubit in args.sizes:
        config = ROOT / "configs" / "experiment" / (
            f"skqd_1d_xxz_sparsity_n{n_qubit}_transformer.yaml")
        if not config.is_file():
            raise FileNotFoundError(config)
        for num_hamiltonians in args.hamiltonians:
            for train_shots in args.train_shots:
                for seed in args.seeds:
                    name = job_name(
                        n_qubit, num_hamiltonians, train_shots, seed)
                    results_root = args.results_root / name
                    subset_seed = 2026080100 + seed
                    measurement_seed = 2026080200 + seed
                    command = [
                        str(args.python),
                        "scripts/train.py",
                        "--config", str(config.relative_to(ROOT)),
                        "--seed", str(seed),
                        "--data.hamiltonians", str(num_hamiltonians),
                        "--data.train_samples", str(train_shots),
                        "--data.hamiltonian_subset_mode", "stratified_nested",
                        "--data.hamiltonian_subset_seed", str(subset_seed),
                        "--data.measurement_subset_seed", str(measurement_seed),
                        "--data.eval_validation", "false",
                        "--data.eval_test", "false",
                        "--training.iterations", str(args.iterations),
                        "--training.eval_every", str(args.iterations),
                        "--training.checkpoint_every",
                        str(args.checkpoint_every),
                        "--results_dir", str(results_root),
                    ]
                    jobs.append({
                        "name": name,
                        "n_qubit": n_qubit,
                        "num_hamiltonians": num_hamiltonians,
                        "train_shots": train_shots,
                        "krylov_conditions": 15,
                        "fit_samples": (
                            num_hamiltonians * 15 * train_shots),
                        "seed": seed,
                        "hamiltonian_subset_seed": subset_seed,
                        "measurement_subset_seed": measurement_seed,
                        "config": str(config.relative_to(ROOT)),
                        "results_root": str(results_root),
                        "command": command,
                        # LPT proxy. Sequence length adds a modest n-dependent cost.
                        "work_score": (
                            num_hamiltonians * train_shots * n_qubit),
                    })
    return sorted(
        jobs,
        key=lambda job: (
            -job["work_score"], -job["n_qubit"], job["name"]),
    )


def find_run_dir(results_root):
    configs = list(Path(results_root).rglob("config.json"))
    if len(configs) != 1:
        return None
    return configs[0].parent


def completed_result(results_root):
    run_dir = find_run_dir(results_root)
    if run_dir is None:
        return None
    checkpoint = run_dir / "checkpoints" / "checkpoint_final.pth.tar"
    log = run_dir / "train_out.txt"
    if checkpoint.is_file() and log.is_file() and "Training complete:" in log.read_text():
        return run_dir
    return None


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    args = parse_args()
    if args.iterations <= 0 or args.checkpoint_every <= 0:
        raise ValueError("iterations and checkpoint-every must be positive")
    if args.workers_per_gpu <= 0 or args.omp_threads <= 0:
        raise ValueError("workers-per-gpu and omp-threads must be positive")
    if not Path(args.python).is_file():
        raise FileNotFoundError(args.python)

    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0

    if args.run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Run directory exists; pass --resume after inspection: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)

    provenance_files = [
        ROOT / "data" / "xxz_sparsity_1d" / "ratio_manifest.csv",
        ROOT / "scripts" / "train.py",
        Path(__file__),
        ROOT / "genksr" / "data" / "xxz_sparsity.py",
        ROOT / "genksr" / "training" / "trainer.py",
    ]
    provenance = {
        "launch_unix_time": time.time(),
        "root": str(ROOT),
        "python": str(args.python),
        "sizes": args.sizes,
        "hamiltonians": args.hamiltonians,
        "train_shots": args.train_shots,
        "seeds": args.seeds,
        "iterations": args.iterations,
        "checkpoint_every": args.checkpoint_every,
        "gpus": args.gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "omp_threads": args.omp_threads,
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in provenance_files
        },
    }
    write_json(args.run_dir / "provenance.json", provenance)
    write_json(args.run_dir / "jobs.json", jobs)

    status_path = args.run_dir / "status.json"
    if status_path.exists() and args.resume:
        status = json.loads(status_path.read_text())
    else:
        status = {}

    pending = []
    for job in jobs:
        finished = completed_result(job["results_root"])
        if finished is not None:
            status[job["name"]] = {
                "state": "completed",
                "run_dir": str(finished),
                "recovered_on_resume": True,
            }
        elif Path(job["results_root"]).exists() and any(
                Path(job["results_root"]).iterdir()):
            raise RuntimeError(
                f"Incomplete nonempty result requires manual inspection: "
                f"{job['results_root']}")
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
        used_slots = [entry["gpu"] for entry in active.values()]
        free_slots = list(slots)
        for used_gpu in used_slots:
            free_slots.remove(used_gpu)

        while pending and free_slots and not stop_requested:
            job = pending.pop(0)
            gpu = free_slots.pop(0)
            log_path = args.run_dir / "logs" / f"{job['name']}.log"
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
                job["command"], cwd=ROOT, env=environment,
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
            active[process.pid] = {
                "process": process,
                "job": job,
                "gpu": gpu,
                "started": started,
                "log_handle": log_handle,
                "log_path": str(log_path),
            }
            status[job["name"]] = {
                "state": "running",
                "gpu": gpu,
                "pid": process.pid,
                "started_unix_time": started,
                "log": str(log_path),
            }
            print(
                f"START {job['name']} gpu={gpu} pid={process.pid} "
                f"pending={len(pending)} active={len(active)}",
                flush=True,
            )
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
            elapsed = time.time() - entry["started"]
            run_dir = completed_result(job["results_root"])
            state = "completed" if return_code == 0 and run_dir else "failed"
            status[job["name"]] = {
                "state": state,
                "gpu": entry["gpu"],
                "pid": pid,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "log": entry["log_path"],
                "run_dir": None if run_dir is None else str(run_dir),
            }
            print(
                f"{state.upper()} {job['name']} gpu={entry['gpu']} "
                f"elapsed={elapsed:.1f}s pending={len(pending)}",
                flush=True,
            )
            del active[pid]
            write_json(status_path, status)

        if stop_requested and not active:
            break

    counts = {
        state: sum(item.get("state") == state for item in status.values())
        for state in ("completed", "failed", "pending", "running")
    }
    summary = {
        "finish_unix_time": time.time(),
        "counts": counts,
        "total_jobs": len(jobs),
        "stopped": stop_requested,
    }
    write_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(stop_requested or counts["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
