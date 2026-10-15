#!/usr/bin/env python3
"""Create simple paper-ready figures for the XXZ sparsity experiment."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/genksr_matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REGIMES = ('sparse', 'middle', 'dense')
REGIME_COLORS = {
    'sparse': '#0072B2',
    'middle': '#E69F00',
    'dense': '#009E73',
}
METHODS_MAIN = ('exact_state', 'transformer', 'mamba')
METHODS_ALL = (
    'exact_state', 'transformer', 'mamba',
    'uniform_full_raw', 'uniform_sector')
METHOD_LABELS = {
    'exact_state': 'Exact samples',
    'transformer': 'Transformer',
    'mamba': 'Mamba',
    'knn5_train': 'kNN-5',
    'uniform_full_raw': 'Full uniform',
    'uniform_sector': 'Sector uniform',
}
METHOD_COLORS = {
    'exact_state': '#000000',
    'transformer': '#0072B2',
    'mamba': '#E69F00',
    'knn5_train': '#009E73',
    'uniform_full_raw': '#D55E00',
    'uniform_sector': '#CC79A7',
}
METHOD_MARKERS = {
    'exact_state': 'o',
    'transformer': 's',
    'mamba': 'D',
    'knn5_train': '^',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--evaluation-dir', type=Path,
        default=Path('artifacts/v3_xxz_sparsity_scaling/'
                     'formal_evaluation_single_sampling_20260716'))
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def save_figure(fig, output_dir, stem):
    fig.savefig(output_dir / f'{stem}.png', dpi=240, bbox_inches='tight')
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def make_main_scaling_figure(ground_rows, summary_rows, output_dir):
    ground = {
        (int(row['N']), row['regime']): row for row in ground_rows}
    summary = {
        (int(row['N']), row['regime'], row['method'], int(row['budget'])): row
        for row in summary_rows}
    sizes = (12, 16, 20)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    axes = axes.ravel()

    # (a) Manipulation check: ground-state sparsity.
    axis = axes[0]
    for regime in REGIMES:
        means = [float(ground[(n, regime)]['f95_mean']) for n in sizes]
        stds = [float(ground[(n, regime)]['f95_std']) for n in sizes]
        axis.errorbar(
            sizes, means, yerr=stds, color=REGIME_COLORS[regime],
            marker='o', markersize=7, linewidth=2.2, capsize=3,
            label=regime.capitalize())
    sparse_n20 = float(ground[(20, 'sparse')]['f95_mean'])
    dense_n20 = float(ground[(20, 'dense')]['f95_mean'])
    axis.annotate(
        '', xy=(20.35, dense_n20), xytext=(20.35, sparse_n20),
        arrowprops={'arrowstyle': '<->', 'color': '0.25', 'lw': 1.4})
    axis.text(
        20.48, np.sqrt(sparse_n20 * dense_n20),
        f'{dense_n20 / sparse_n20:.1f}x', va='center', fontsize=10)
    axis.set_yscale('log')
    axis.set_xlim(11.5, 21.2)
    axis.set_xticks(sizes)
    axis.set_xlabel('System size $N$')
    axis.set_ylabel(r'Ground-state support fraction $f_{95}$')
    axis.set_title('(a) Sparsity separation')
    axis.grid(True, which='both', alpha=0.25)
    axis.legend(frameon=False)

    # (b-d) Fixed-budget physical consequence.
    for axis, regime, panel in zip(axes[1:], REGIMES, ('b', 'c', 'd')):
        for method in METHODS_MAIN:
            means = np.asarray([
                float(summary[(n, regime, method, 15000)][
                    'normalized_delta_e_mean']) for n in sizes])
            stds = np.asarray([
                float(summary[(n, regime, method, 15000)][
                    'normalized_delta_e_std']) for n in sizes])
            axis.errorbar(
                sizes, means, yerr=stds,
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                markersize=6, linewidth=2.0, capsize=2.5,
                label=METHOD_LABELS[method])
        axis.set_yscale('log')
        axis.set_ylim(3e-4, 1.2e-1)
        axis.set_xticks(sizes)
        axis.set_xlabel('System size $N$')
        axis.set_ylabel('Normalized energy error')
        axis.set_title(f'({panel}) {regime.capitalize()} regime')
        axis.grid(True, which='both', alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center', ncol=3,
        frameon=False, bbox_to_anchor=(0.68, -0.005))
    fig.suptitle(
        r'Fixed-budget XXZ scaling ($B=15{,}000$ raw draws)',
        fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, output_dir, 'main_sparsity_scaling')


def make_size_baseline_figure(
        n_qubit, ground_rows, summary_rows, output_dir):
    ground = {
        (int(row['N']), row['regime']): row for row in ground_rows}
    summary = {
        (int(row['N']), row['regime'], row['method'], int(row['budget'])): row
        for row in summary_rows}

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8), sharey=True)
    x = np.arange(len(METHODS_ALL))
    tick_labels = [METHOD_LABELS[method] for method in METHODS_ALL]
    display_floor = 3.5e-5

    for axis, regime in zip(axes, REGIMES):
        raw_means = np.asarray([
            float(summary[(n_qubit, regime, method, 15000)][
                'normalized_delta_e_mean'])
            for method in METHODS_ALL])
        means = np.maximum(raw_means, display_floor)
        stds = np.asarray([
            float(summary[(n_qubit, regime, method, 15000)][
                'normalized_delta_e_std'])
            for method in METHODS_ALL])
        bars = axis.bar(
            x, means, yerr=stds, capsize=2.5,
            color=[METHOD_COLORS[method] for method in METHODS_ALL],
            edgecolor='white', linewidth=0.6)
        axis.set_yscale('log')
        axis.set_ylim(2.5e-5, 6e-1)
        axis.set_xticks(x)
        axis.set_xticklabels(tick_labels, rotation=28, ha='right', fontsize=8.5)
        axis.grid(True, axis='y', which='both', alpha=0.25)
        f95 = float(ground[(n_qubit, regime)]['f95_mean'])
        axis.set_title(f'{regime.capitalize()}  ($f_{{95}}={f95:.3g}$)')
        for bar, raw_value, display_value in zip(bars, raw_means, means):
            if raw_value <= 1e-12:
                label = r'$\approx 0$' + '\n(full sector)'
            elif raw_value < 0.01:
                label = f'{raw_value:.1e}'
            else:
                label = f'{raw_value:.2f}'
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                display_value * 1.18, label,
                ha='center', va='bottom', rotation=90, fontsize=7.5)
    axes[0].set_ylabel('Normalized energy error')
    fig.suptitle(
        rf'$N={n_qubit}$, $B=15{{,}}000$: exact, model, and random sampling',
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output_dir, f'n{n_qubit}_baseline_comparison')


def main():
    args = parse_args()
    ground_rows = read_csv(args.evaluation_dir / 'ground_summary.csv')
    summary_rows = read_csv(args.evaluation_dir / 'summary.csv')
    make_main_scaling_figure(ground_rows, summary_rows, args.evaluation_dir)
    for n_qubit in (12, 16, 20):
        make_size_baseline_figure(
            n_qubit, ground_rows, summary_rows, args.evaluation_dir)
    print(f'Wrote paper figures to {args.evaluation_dir}', flush=True)


if __name__ == '__main__':
    main()
