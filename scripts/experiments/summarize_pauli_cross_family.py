#!/usr/bin/env python3
"""Aggregate one or more Pauli cross-family model seeds."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from genksr.experiments.pauli_cross_family.summary import summarize_evaluations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = summarize_evaluations(args.evaluation_dirs, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
