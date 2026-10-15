#!/usr/bin/env python3
"""Train one Pauli cross-family model seed."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from genksr.experiments.pauli_cross_family.training import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_model(
        args.data_root,
        args.output_dir,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
