#!/usr/bin/env python3
"""Launch independent model seeds, optionally assigning one device per process."""

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["auto"],
        help="Device strings assigned round-robin, e.g. cuda:0 cuda:1 cuda:2 or cpu",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_script = root / "scripts/experiments/train_pauli_cross_family.py"
    eval_script = root / "scripts/experiments/evaluate_pauli_cross_family.py"
    jobs = []
    for position, seed in enumerate(args.seeds):
        device = args.devices[position % len(args.devices)]
        train_dir = Path(args.results_root) / f"seed{seed}"
        eval_dir = Path(args.evaluation_root) / f"seed{seed}"
        command = [
            sys.executable,
            str(train_script),
            "--data-root",
            args.data_root,
            "--output-dir",
            str(train_dir),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
        if args.resume:
            command.append("--resume")
        jobs.append((seed, device, train_dir, eval_dir, command))

    if args.parallel:
        running = []
        for seed, device, train_dir, eval_dir, command in jobs:
            train_dir.mkdir(parents=True, exist_ok=True)
            log = (train_dir / "train.log").open("w")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            running.append((seed, device, train_dir, eval_dir, process, log))
        failures = []
        for seed, device, train_dir, eval_dir, process, log in running:
            return_code = process.wait()
            log.close()
            if return_code:
                failures.append((seed, return_code))
        if failures:
            raise RuntimeError(f"training failures: {failures}")
    else:
        for seed, device, train_dir, eval_dir, command in jobs:
            _run(command, train_dir / "train.log")

    for seed, device, train_dir, eval_dir, _ in jobs:
        eval_command = [
            sys.executable,
            str(eval_script),
            "--data-root",
            args.data_root,
            "--checkpoint",
            str(train_dir / "model_final.pt"),
            "--output-dir",
            str(eval_dir),
            "--device",
            device,
        ]
        _run(eval_command, eval_dir / "evaluation.log")
        print(f"completed seed={seed} device={device}", flush=True)


if __name__ == "__main__":
    main()

