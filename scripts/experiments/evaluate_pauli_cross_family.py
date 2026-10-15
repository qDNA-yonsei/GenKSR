#!/usr/bin/env python3
"""Evaluate one trained Pauli cross-family model seed."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from genksr.experiments.pauli_cross_family.evaluation import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.data_root,
        args.checkpoint,
        args.output_dir,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
