#!/usr/bin/env python3
"""Generate the frozen exact Pauli cross-family dataset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from genksr.experiments.pauli_cross_family.dataset import generate_dataset
from genksr.experiments.pauli_cross_family.protocol import (
    build_protocol,
    load_experiment_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config, smoke=args.smoke)
    protocol = build_protocol(config)
    report = generate_dataset(protocol, args.output_root, workers=args.workers)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
