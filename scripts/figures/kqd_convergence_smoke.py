#!/usr/bin/env python3
"""Plot exact-simulation and classical-shadow KQD convergence from evaluate.py output."""

import argparse
import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Plot v3 KQD energy convergence smoke test')
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--comparison-evaluation-dir', default=None,
                        help='Optional second evaluation directory for a shot-count comparison.')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--output', default=None, help='PNG output path')
    parser.add_argument('--shots', type=int, default=None,
                        help='Shot count used in --evaluation-dir (figure label only).')
    parser.add_argument('--comparison-shots', type=int, default=None,
                        help='Shot count used in --comparison-evaluation-dir (figure label only).')
    return parser.parse_args()


def _rmse_curves(results, split):
    split_results = results[split]
    ideal = split_results['ideal']
    shadow = split_results['shadow']
    hamiltonian_ids = sorted(set(ideal) & set(shadow))
    if not hamiltonian_ids:
        raise ValueError(f'No Hamiltonians have both ideal and shadow results in {split!r}')

    max_dim = min(
        min(len(ideal[hid]['energies']) for hid in hamiltonian_ids),
        min(len(shadow[hid]['energies']) for hid in hamiltonian_ids),
    )
    errors = {'ideal': [], 'shadow': []}
    for hid in hamiltonian_ids:
        exact_energy = ideal[hid]['exact_energy']
        for method in errors:
            energies = np.asarray(split_results[method][hid]['energies'][:max_dim], dtype=float)
            errors[method].append(energies - exact_energy)

    curves = {
        method: np.sqrt(np.nanmean(np.square(np.asarray(method_errors)), axis=0))
        for method, method_errors in errors.items()
    }
    return np.arange(1, max_dim + 1), curves, len(hamiltonian_ids)


def _load_curves(evaluation_dir, split):
    results_path = os.path.join(evaluation_dir, 'all_results.pkl')
    with open(results_path, 'rb') as f:
        return _rmse_curves(pickle.load(f), split)


def _shadow_label(shots):
    if shots is None:
        return 'Classical shadow KQD'
    return f'Classical shadow KQD ({shots:,} shots)'


def main():
    args = parse_args()
    dimensions, curves, num_hamiltonians = _load_curves(
        args.evaluation_dir, args.split)
    comparison = None
    if args.comparison_evaluation_dir is not None:
        comparison = _load_curves(args.comparison_evaluation_dir, args.split)
        comparison_dimensions, comparison_curves, comparison_num_hamiltonians = comparison
        if not np.array_equal(dimensions, comparison_dimensions):
            raise ValueError('The two evaluations use different Krylov dimensions')
        if num_hamiltonians != comparison_num_hamiltonians:
            raise ValueError('The two evaluations contain different numbers of Hamiltonians')
        if not np.allclose(curves['ideal'], comparison_curves['ideal'], rtol=0, atol=1e-10):
            raise ValueError(
                'Exact-simulation curves differ; shot-count comparisons require matching Hamiltonians')

    output_path = args.output or os.path.join(
        args.evaluation_dir, f'{args.split}_kqd_convergence.png')
    csv_path = os.path.splitext(output_path)[0] + '.csv'
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(dimensions, curves['ideal'], color='black', marker='o', linewidth=2.0,
            label='Exact simulation KQD')
    ax.plot(dimensions, curves['shadow'], color='tab:blue', marker='s', linewidth=2.0,
            label=_shadow_label(args.shots))
    if comparison is not None:
        comparison_curves = comparison[1]
        ax.plot(dimensions, comparison_curves['shadow'], color='tab:orange', marker='^',
                linewidth=2.0, label=_shadow_label(args.comparison_shots))
    ax.set_xlabel('Krylov dimension $D$')
    ax.set_ylabel(r'Ground-energy RMSE $\sqrt{\langle(E_D-E_0)^2\rangle}$')
    ax.set_title(f'GenKSR v3 KQD convergence ({num_hamiltonians} held-out Hamiltonians)')
    ax.set_xticks(dimensions)
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['krylov_dimension', 'exact_simulation_rmse']
        first_shadow_column = (
            f'classical_shadow_{args.shots}_shots_rmse'
            if args.shots is not None else 'classical_shadow_rmse')
        header.append(first_shadow_column)
        columns = [dimensions, curves['ideal'], curves['shadow']]
        if comparison is not None:
            second_shadow_column = (
                f'classical_shadow_{args.comparison_shots}_shots_rmse'
                if args.comparison_shots is not None else 'classical_shadow_comparison_rmse')
            header.append(second_shadow_column)
            columns.append(comparison[1]['shadow'])
        writer.writerow(header)
        writer.writerows(zip(*columns))

    print(f'ideal: D=1 {curves["ideal"][0]:.8f} '
          f'-> D={dimensions[-1]} {curves["ideal"][-1]:.8f}')
    print(f'{_shadow_label(args.shots)}: D=1 {curves["shadow"][0]:.8f} '
          f'-> D={dimensions[-1]} {curves["shadow"][-1]:.8f}')
    if comparison is not None:
        comparison_shadow = comparison[1]['shadow']
        print(f'{_shadow_label(args.comparison_shots)}: D=1 {comparison_shadow[0]:.8f} '
              f'-> D={dimensions[-1]} {comparison_shadow[-1]:.8f}')
    print(f'figure: {output_path}')
    print(f'csv: {csv_path}')


if __name__ == '__main__':
    main()
