#!/usr/bin/env python3
"""Joint evaluation of zero-shot, target-trained, direct, and random SKQD."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from genksr.experiments.pauli_cross_family.comparison import (
    evaluate_joint_comparison,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-data-root", required=True)
    parser.add_argument("--cross-checkpoint", required=True)
    parser.add_argument(
        "--cross-checkpoint-data-root",
        default=None,
        help=(
            "Original dataset protocol bound to the source checkpoint. Supply this "
            "when reusing one source-only model for a new held-out target manifest."
        ),
    )
    parser.add_argument("--target-data-root", required=True)
    parser.add_argument("--target-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = evaluate_joint_comparison(
        args.cross_data_root,
        args.cross_checkpoint,
        args.target_data_root,
        args.target_checkpoint,
        args.output_dir,
        device=args.device,
        cross_checkpoint_dataset_root=args.cross_checkpoint_data_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
