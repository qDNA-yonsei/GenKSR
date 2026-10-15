#!/usr/bin/env python3
"""Validate and summarize the formal XXZ sparsity evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/genksr_matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


METHODS = (
    'exact_state', 'transformer', 'mamba', 'knn5_train',
    'uniform_full_raw', 'uniform_sector')
METHOD_LABELS = {
    'exact_state': 'Exact-state samples',
    'transformer': 'GenKSR Transformer',
    'mamba': 'GenKSR Mamba',
    'knn5_train': 'kNN-5 train reuse',
    'uniform_full_raw': 'Uniform full + postselect',
    'uniform_sector': 'Uniform sector',
}
COLORS = {
    'exact_state': 'black',
    'transformer': 'tab:blue',
    'mamba': 'tab:orange',
    'knn5_train': 'tab:green',
    'uniform_full_raw': 'tab:red',
    'uniform_sector': 'tab:purple',
}
REGIMES = ('sparse', 'middle', 'dense')


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


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(evaluation_dir, ground, rows, summary, paired, mapping, protocol):
    require(protocol['status'] == 'complete', 'protocol is not complete')
    require(len(ground) == 90, f'expected 90 ground rows, got {len(ground)}')
    require(len(rows) == 1080, f'expected 1080 result rows, got {len(rows)}')
    require(len(summary) == 108, f'expected 108 summary rows, got {len(summary)}')
    require(len(paired) == 360, f'expected 360 paired rows, got {len(paired)}')
    require(len(mapping) == 6750, f'expected 6750 kNN mappings, got {len(mapping)}')

    ground_lookup = {
        (int(row['N']), int(row['hamiltonian_id'])): row for row in ground}
    require(len(ground_lookup) == 90, 'ground rows have duplicate keys')
    counts = Counter(
        (int(row['N']), row['method'], int(row['budget'])) for row in rows)
    require(set(counts.values()) == {30}, 'method/budget cells are incomplete')

    lookup = {}
    for row in rows:
        key = (
            int(row['N']), int(row['hamiltonian_id']),
            row['method'], int(row['budget']))
        require(key not in lookup, f'duplicate result key: {key}')
        lookup[key] = row
        budget = int(row['budget'])
        raw = int(row['raw_draws'])
        accepted = int(row['accepted_draws'])
        require(budget == int(row['D']) * int(row['M']) == raw,
                f'budget identity failed: {key}')
        require(0 <= accepted <= raw, f'accepted count invalid: {key}')
        require(int(row['rejected_draws']) == raw - accepted,
                f'rejected count invalid: {key}')
        require(abs(float(row['acceptance_rate']) - accepted / raw) < 1e-14,
                f'acceptance rate invalid: {key}')
        coefficient_l1 = float(row['coefficient_l1'])
        energy = float(row['energy'])
        exact = float(row['exact_energy'])
        delta = float(row['delta_e'])
        require(energy + 1e-7 * coefficient_l1 >= exact,
                f'variational bound failed: {key}')
        require(abs(delta - max(energy - exact, 0.0)) < 1e-9,
                f'energy error mismatch: {key}')
        require(abs(float(row['normalized_delta_e']) - delta / coefficient_l1) < 1e-12,
                f'normalized error mismatch: {key}')
        require(0 <= float(row['captured_ground_mass']) <= 1 + 1e-10,
                f'ground mass invalid: {key}')
        require(0 <= float(row['captured_doublet_mass']) <= 1 + 1e-10,
                f'doublet mass invalid: {key}')
        ground_row = ground_lookup[(key[0], key[1])]
        require(abs(exact - float(ground_row['E0'])) < 1e-10,
                f'exact energy mismatch: {key}')

    nonworsening_violations = []
    for n_qubit, hid, method in sorted({
            (int(row['N']), int(row['hamiltonian_id']), row['method'])
            for row in rows}):
        low = lookup[(n_qubit, hid, method, 3000)]
        high = lookup[(n_qubit, hid, method, 15000)]
        if int(high['basis_size']) < int(low['basis_size']):
            nonworsening_violations.append((n_qubit, hid, method, 'basis'))
        if (float(high['captured_ground_mass']) + 1e-10
                < float(low['captured_ground_mass'])):
            nonworsening_violations.append((n_qubit, hid, method, 'ground_mass'))
        if (float(high['captured_doublet_mass']) + 1e-10
                < float(low['captured_doublet_mass'])):
            nonworsening_violations.append((n_qubit, hid, method, 'doublet_mass'))
        if float(high['energy']) > float(low['energy']) + 1e-8:
            nonworsening_violations.append((n_qubit, hid, method, 'energy'))
    require(not nonworsening_violations,
            f'nested-budget violations: {nonworsening_violations[:5]}')

    report = {
        'status': 'pass',
        'evaluation_dir': str(evaluation_dir.resolve()),
        'row_counts': {
            'ground_metrics': len(ground),
            'evaluation_metrics': len(rows),
            'summary': len(summary),
            'paired_model_vs_exact': len(paired),
            'knn_source_mapping': len(mapping),
        },
        'method_budget_cell_count': len(counts),
        'nested_method_hamiltonian_pairs': 90 * len(METHODS),
        'nested_budget_violations': 0,
        'variational_bound_violations': 0,
    }
    (evaluation_dir / 'validation_report.json').write_text(
        json.dumps(report, indent=2))
    return report


def ground_summary_rows(ground):
    output = []
    for n_qubit in (12, 16, 20):
        for regime in REGIMES:
            group = [
                row for row in ground
                if int(row['N']) == n_qubit and row['regime'] == regime]
            record = {
                'N': n_qubit,
                'regime': regime,
                'num_hamiltonians': len(group),
            }
            for metric in (
                    'f95', 'doublet_f95', 'normalized_gap',
                    'effective_support_fraction'):
                values = np.asarray([float(row[metric]) for row in group])
                record[f'{metric}_mean'] = float(values.mean())
                record[f'{metric}_std'] = float(values.std(ddof=1))
            output.append(record)
    return output


def comparison_rows(results):
    lookup = {
        (int(row['N']), int(row['hamiltonian_id']), int(row['budget']), row['method']): row
        for row in results}
    output = []
    for n_qubit in (12, 16, 20):
        for regime, ids in zip(REGIMES, (range(10), range(10, 20), range(20, 30))):
            for budget in (3000, 15000):
                for model in ('transformer', 'mamba'):
                    for comparator in ('exact_state', 'knn5_train'):
                        differences = np.asarray([
                            float(lookup[(n_qubit, hid, budget, model)]['normalized_delta_e'])
                            - float(lookup[(n_qubit, hid, budget, comparator)]['normalized_delta_e'])
                            for hid in ids
                        ])
                        output.append({
                            'N': n_qubit,
                            'regime': regime,
                            'budget': budget,
                            'model': model,
                            'comparator': comparator,
                            'normalized_error_difference_mean': float(differences.mean()),
                            'normalized_error_difference_std': float(differences.std(ddof=1)),
                            'model_win_count': int(np.sum(differences < 0)),
                            'tie_count': int(np.sum(np.isclose(
                                differences, 0.0, rtol=0.0, atol=1e-12))),
                            'num_hamiltonians': len(differences),
                        })
    return output


def architecture_rows(results):
    lookup = {
        (int(row['N']), int(row['hamiltonian_id']), int(row['budget']), row['method']): row
        for row in results}
    output = []
    for n_qubit in (12, 16, 20):
        for regime, ids in zip(REGIMES, (range(10), range(10, 20), range(20, 30))):
            for budget in (3000, 15000):
                differences = np.asarray([
                    float(lookup[(n_qubit, hid, budget, 'transformer')]['normalized_delta_e'])
                    - float(lookup[(n_qubit, hid, budget, 'mamba')]['normalized_delta_e'])
                    for hid in ids
                ])
                output.append({
                    'N': n_qubit,
                    'regime': regime,
                    'budget': budget,
                    'difference': 'transformer_minus_mamba_normalized_error',
                    'difference_mean': float(differences.mean()),
                    'difference_std': float(differences.std(ddof=1)),
                    'transformer_win_count': int(np.sum(differences < 0)),
                    'mamba_win_count': int(np.sum(differences > 0)),
                    'tie_count': int(np.sum(np.isclose(
                        differences, 0.0, rtol=0.0, atol=1e-12))),
                    'num_hamiltonians': len(differences),
                })
    return output


def correlation_rows(ground, results):
    ground_lookup = {
        (int(row['N']), int(row['hamiltonian_id'])): row for row in ground}
    output = []
    for n_qubit in (12, 16, 20):
        for budget in (3000, 15000):
            for method in ('exact_state', 'transformer', 'mamba', 'knn5_train'):
                group = [
                    row for row in results
                    if int(row['N']) == n_qubit
                    and int(row['budget']) == budget
                    and row['method'] == method]
                errors = np.asarray([float(row['normalized_delta_e']) for row in group])
                f95 = np.asarray([
                    float(ground_lookup[(n_qubit, int(row['hamiltonian_id']))]['f95'])
                    for row in group])
                gaps = np.asarray([
                    float(ground_lookup[(n_qubit, int(row['hamiltonian_id']))]['normalized_gap'])
                    for row in group])
                f95_result = spearmanr(f95, errors)
                gap_result = spearmanr(gaps, errors)
                output.append({
                    'N': n_qubit,
                    'budget': budget,
                    'method': method,
                    'num_hamiltonians': len(group),
                    'spearman_f95_vs_error': float(f95_result.statistic),
                    'p_f95_vs_error': float(f95_result.pvalue),
                    'spearman_gap_vs_error': float(gap_result.statistic),
                    'p_gap_vs_error': float(gap_result.pvalue),
                })
    return output


def make_ground_figure(ground_summary, output_dir):
    fig, axis = plt.subplots(figsize=(6.2, 4.6))
    for regime in REGIMES:
        group = [row for row in ground_summary if row['regime'] == regime]
        axis.errorbar(
            [row['N'] for row in group],
            [row['f95_mean'] for row in group],
            yerr=[row['f95_std'] for row in group],
            marker='o', capsize=3, label=regime)
    axis.set_yscale('log')
    axis.set_xlabel('System size N')
    axis.set_ylabel(r'$f_{95}=L_{95}/\binom{N}{N/2}$')
    axis.set_title('Ground-state computational-basis sparsity')
    axis.grid(True, which='both', alpha=0.3)
    axis.legend()
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output_dir / f'ground_f95_scaling.{extension}', dpi=180)
    plt.close(fig)


def make_energy_figure(summary, output_dir):
    lookup = {
        (int(row['N']), row['regime'], row['method'], int(row['budget'])): row
        for row in summary}
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4), sharex=True)
    for row_index, budget in enumerate((3000, 15000)):
        for column_index, regime in enumerate(REGIMES):
            axis = axes[row_index, column_index]
            for method in METHODS:
                means = np.asarray([
                    max(float(lookup[(n, regime, method, budget)][
                        'normalized_delta_e_mean']), 1e-16)
                    for n in (12, 16, 20)])
                axis.plot(
                    (12, 16, 20), means, marker='o',
                    color=COLORS[method], label=METHOD_LABELS[method])
            axis.set_yscale('log')
            axis.grid(True, which='both', alpha=0.25)
            axis.set_title(f'{regime}, B={budget:,}')
            if row_index == 1:
                axis.set_xlabel('System size N')
            if column_index == 0:
                axis.set_ylabel('Normalized energy error')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for extension in ('png', 'pdf'):
        fig.savefig(output_dir / f'energy_scaling.{extension}', dpi=180)
    plt.close(fig)


def format_energy_table(summary, budget):
    lookup = {
        (int(row['N']), row['regime'], row['method'], int(row['budget'])): row
        for row in summary}
    lines = [
        f'### Raw budget B={budget:,}',
        '',
        '| N | regime | exact | Transformer | Mamba | kNN-5 | full uniform | sector uniform |',
        '|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    for n_qubit in (12, 16, 20):
        for regime in REGIMES:
            values = [
                float(lookup[(n_qubit, regime, method, budget)][
                    'normalized_delta_e_mean'])
                for method in METHODS]
            lines.append(
                f'| {n_qubit} | {regime} | '
                + ' | '.join(f'{value:.3e}' for value in values) + ' |')
    return lines


def write_analysis(
        evaluation_dir, protocol, ground_summary, summary,
        comparisons, architectures, validation):
    ground_lookup = {(row['N'], row['regime']): row for row in ground_summary}
    result_lines = [
        '# XXZ sparsity formal evaluation',
        '',
        '## Protocol',
        '',
        '- test Hamiltonians: 30 per size, 10 per sparsity regime',
        '- sizes: `N=12,16,20`',
        '- raw budgets: `B=3,000` and `B=15,000`',
        f"- model training seed (recorded): {protocol.get('model_training_seed', 'not recorded')}",
        f"- model sampling seed: {protocol.get('model_sampling_seed', 'not recorded')}",
        f"- sampling realizations: {protocol.get('sampling_repeats', 'not recorded')}",
        '- Hamiltonian: pure open-chain XXZ with `J_zz=1`',
        '- all methods use equal raw draws before postselection',
        '',
        '## Validation',
        '',
        f"- evaluation rows: {validation['row_counts']['evaluation_metrics']}",
        f"- variational-bound violations: {validation['variational_bound_violations']}",
        f"- nested-budget violations: {validation['nested_budget_violations']} "
        f"across {validation['nested_method_hamiltonian_pairs']} method/Hamiltonian pairs",
        f"- evaluator wall time: {protocol['wall_seconds']:.1f} s",
        '',
        '## Ground-state sparsity',
        '',
        '| N | regime | mean f95 | std | doublet f95 | normalized gap |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for n_qubit in (12, 16, 20):
        for regime in REGIMES:
            row = ground_lookup[(n_qubit, regime)]
            result_lines.append(
                f"| {n_qubit} | {regime} | {row['f95_mean']:.3e} | "
                f"{row['f95_std']:.1e} | {row['doublet_f95_mean']:.3e} | "
                f"{row['normalized_gap_mean']:.3e} |")
    ratios = []
    for n_qubit in (12, 16, 20):
        ratio = (ground_lookup[(n_qubit, 'dense')]['f95_mean']
                 / ground_lookup[(n_qubit, 'sparse')]['f95_mean'])
        ratios.append(f'N={n_qubit}: {ratio:.2f}')
    result_lines.extend([
        '',
        'Dense/sparse mean f95 ratios: ' + '; '.join(ratios) + '.',
        '',
        '## Equal-budget normalized energy error',
        '',
    ])
    result_lines.extend(format_energy_table(summary, 3000))
    result_lines.append('')
    result_lines.extend(format_energy_table(summary, 15000))

    comparison_lookup = defaultdict(list)
    for row in comparisons:
        comparison_lookup[(row['budget'], row['model'], row['comparator'])].append(row)
    architecture_lookup = defaultdict(list)
    for row in architectures:
        architecture_lookup[row['budget']].append(row)

    result_lines.extend([
        '',
        '## Paired observations',
        '',
    ])
    for budget in (3000, 15000):
        for model in ('transformer', 'mamba'):
            group = comparison_lookup[(budget, model, 'knn5_train')]
            wins = sum(int(row['model_win_count']) for row in group)
            total = sum(int(row['num_hamiltonians']) for row in group)
            result_lines.append(
                f"- B={budget:,}: {model} beats kNN-5 on {wins}/{total} paired "
                'test Hamiltonians.')
        architecture_group = architecture_lookup[budget]
        transformer_wins = sum(
            int(row['transformer_win_count']) for row in architecture_group)
        mamba_wins = sum(
            int(row['mamba_win_count']) for row in architecture_group)
        total = sum(int(row['num_hamiltonians']) for row in architecture_group)
        result_lines.append(
            f'- B={budget:,}: Transformer/Mamba paired wins are '
            f'{transformer_wins}/{mamba_wins} out of {total}.')

    result_lines.extend([
        '',
        '## Interpretation limits',
        '',
        '- A single sampling realization does not support sampling-seed error bars.',
        '- A single training seed does not establish architecture superiority.',
        '- Changing J_xy can change both computational-basis sparsity and the '
        'spectral gap. Inspect correlations.csv before interpreting these associations.',
        '- Captured ground mass and doublet-averaged captured mass must be read '
        'together in the sparse near-doublet regime.',
        '',
        '## Figures',
        '',
        '- `ground_f95_scaling.png/pdf`',
        '- `energy_scaling.png/pdf`',
        '',
    ])
    (evaluation_dir / 'ANALYSIS.md').write_text('\n'.join(result_lines))


def main():
    args = parse_args()
    root = args.evaluation_dir
    ground = read_csv(root / 'ground_metrics.csv')
    results = read_csv(root / 'evaluation_metrics.csv')
    summary = read_csv(root / 'summary.csv')
    paired = read_csv(root / 'paired_model_vs_exact.csv')
    mapping = read_csv(root / 'knn_source_mapping.csv')
    protocol = json.loads((root / 'protocol.json').read_text())

    validation = validate(root, ground, results, summary, paired, mapping, protocol)
    ground_summary = ground_summary_rows(ground)
    comparisons = comparison_rows(results)
    architectures = architecture_rows(results)
    correlations = correlation_rows(ground, results)
    write_csv(root / 'ground_summary.csv', ground_summary)
    write_csv(root / 'model_comparisons.csv', comparisons)
    write_csv(root / 'architecture_comparison.csv', architectures)
    write_csv(root / 'correlations.csv', correlations)
    make_ground_figure(ground_summary, root)
    make_energy_figure(summary, root)
    write_analysis(
        root, protocol, ground_summary, summary,
        comparisons, architectures, validation)
    print(f'Wrote validated summary to {root}', flush=True)


if __name__ == '__main__':
    main()
