#!/usr/bin/env python3
"""Generate manuscript tables, macros, and panel figures from revision-3 results."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.special import logit
from sklearn.metrics import roc_auc_score

RESULTS = ROOT / 'results/revision3'
GENERATED = ROOT / 'paper/generated'
FIGURES = ROOT / 'paper/figures'
NAMES = {'rna': 'Transcriptomic', 'clinical': 'Clinical anchor', 'mri': 'MRI',
         'mean': 'Available-expert mean', 'stack': 'Probability stacking',
         'no_mri_stack': 'No-MRI stacking', 'mask_stack': 'Mask-augmented stacking',
         'carf': 'CARF (proposed)', 'carf_no_robust': 'CARF: no robust loss',
         'carf_no_context': 'CARF: constant basis', 'carf_unbounded': 'CARF: unbounded residual',
         'carf_no_mri': 'CARF: no MRI'}


def table(path, header, rows, alignment):
    body = ['\\begin{tabular}{' + alignment + '}', '\\toprule',
            ' & '.join(header) + r'\\', '\\midrule']
    body += [' & '.join(row) + r'\\' for row in rows]
    body += ['\\bottomrule', '\\end{tabular}']
    path.write_text('\n'.join(body) + '\n')


# LNCS text width is approximately 4.8 inches. Set typography at that size so
# embedding at linewidth does not shrink labels from a large plotting canvas.
FIGURE_STYLE = {
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
}


def save_figure(fig, name):
    # Keep identical physical dimensions in the manuscript and PNG preview.
    fig.savefig(FIGURES / f'{name}.pdf')
    fig.savefig(FIGURES / f'{name}.png', dpi=220)
    plt.close(fig)


def make_stress_figure(stress, budgets):
    agg = stress.groupby(['scenario', 'model']).auroc.agg(['mean', 'std'])
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.85),
                                 gridspec_kw={'width_ratios': [1.15, 1]})
        fig.subplots_adjust(left=.12, right=.98, bottom=.40, top=.85, wspace=.48)
        view = ['intact', 'mri_missing', 'mri_noisy', 'mri_permuted']
        styles = [('clinical', '#555555', 'o', 'Clinical'),
                  ('stack', '#cc6677', 's', 'Stacking'),
                  ('carf', '#0077bb', 'D', 'CARF'),
                  ('carf_no_robust', '#228833', 'v', 'CARF: no robust loss')]
        for name, color, marker, label in styles:
            means = [agg.loc[(scenario, name), 'mean'] for scenario in view]
            # Single deterministic evaluations have no across-realization SD.
            sd = [np.nan_to_num(agg.loc[(scenario, name), 'std']) for scenario in view]
            axes[0].errorbar(range(4), means, yerr=sd, marker=marker,
                             markersize=4, linewidth=1.3, color=color,
                             label=label, capsize=2)
        axes[0].set_xticks(range(4), ['Intact', 'Absent', 'Noise', 'Shuffle'],
                          rotation=40, ha='right')
        # Panel title and caption identify the MRI conditions.
        axes[0].set_ylabel('External AUROC')
        axes[0].set_ylim(.69, .91)
        axes[0].set_yticks([.70, .80, .90])
        axes[0].set_title('(a) Discrimination', loc='left', fontweight='bold', pad=10)
        sub = stress[stress.scenario.isin(['intact', 'mri_noisy', 'mri_permuted'])]
        disp = sub.groupby('model').max_anchor_logit_displacement.max()
        chosen = ['stack', 'mask_stack', 'carf']
        axes[1].bar(range(3), [disp[m] for m in chosen],
                    color=['#cc6677', '#66ccee', '#0077bb'], width=.65)
        axes[1].axhline(budgets.sum(), color='black', linestyle='--', linewidth=1.2,
                        label='CARF uniform bound')
        axes[1].set_xticks(range(3), ['Stack', 'Mask', 'CARF'],
                          rotation=40, ha='right')
        # Model names are shown directly on the ticks.
        axes[1].set_ylabel('Max. logit displacement')
        axes[1].set_title('(b) Stability', loc='left', fontweight='bold', pad=10)
        for ax in axes:
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        bound_handles, bound_labels = axes[1].get_legend_handles_labels()
        fig.legend(handles + bound_handles, labels + bound_labels, loc='lower center',
                   bbox_to_anchor=(.5, .005), ncol=2, frameon=False,
                   columnspacing=1.1, handlelength=1.8)
        save_figure(fig, 'carf_stress')


def make_architecture_figure():
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.85))
        fig.subplots_adjust(left=.015, right=.985, top=.84, bottom=.02, wspace=.13)
        panels = [
            ('(a) Prediction', [
                'Clinical logit anchor',
                'RNA / MRI residuals\nMasks + bounded context',
                'Anchor + bounded correction\nSigmoid → pCR probability']),
            ('(b) Gate training', [
                'Inner out-of-fold\nexpert predictions',
                'Nonnegative budgets\nConvex robust objective',
                'Freeze experts and gate\nPredict held-out patients']),
        ]
        for ax, (title, labels) in zip(axes, panels):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(title, loc='left', fontweight='bold', pad=9)
            for y, label in zip([.73, .40, .07], labels):
                ax.add_patch(FancyBboxPatch((.03, y), .94, .22,
                    boxstyle='round,pad=0.015', facecolor='#e8f2f8',
                    edgecolor='#21618c', linewidth=1))
                ax.text(.5, y + .11, label, ha='center', va='center', fontsize=9)
            for top, bottom in [(.71, .64), (.38, .31)]:
                ax.annotate('', xy=(.5, bottom), xytext=(.5, top),
                            arrowprops={'arrowstyle': '->', 'lw': 1.3})
        save_figure(fig, 'carf_architecture')


def run():
    if not (RESULTS / 'COMPLETE.json').exists():
        raise RuntimeError('Complete the evaluation before generating paper assets')
    from scripts.run_revision3_analysis import provenance
    complete = json.loads((RESULTS / 'COMPLETE.json').read_text())
    recorded = json.loads((RESULTS / 'provenance.json').read_text())
    if complete['internal_folds'] != 25 or complete['signature'] != recorded['cache_signature'] or recorded['hashes'] != provenance():
        raise RuntimeError('Paper requires current source hashes and the complete 25-fold run')
    GENERATED.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ext = pd.read_csv(RESULTS / 'external_metrics.csv').set_index('model')
    internal = pd.read_csv(RESULTS / 'internal_cv.csv')
    cv = internal.groupby('model').auroc.agg(['mean', 'std'])
    rows = []
    for name, label in NAMES.items():
        r = ext.loc[name]
        rows.append([label, f'{cv.loc[name, "mean"]:.3f}', f'{r.auroc:.3f}',
                     f'{r.auprc:.3f}', f'{r.brier:.3f}'])
    table(GENERATED / 'carf_comparison.tex', ['Model', 'CV AUC', 'Ext. AUC', 'Ext. AP', 'Brier'], rows, 'lrrrr')
    boot = pd.read_csv(RESULTS / 'paired_bootstrap.csv')
    rows = []
    for name, values in boot.groupby('comparator', sort=False):
        lo, hi = values.delta_auroc.quantile([.025, .975])
        delta = ext.loc['carf', 'auroc'] - ext.loc[name, 'auroc']
        rows.append([NAMES[name], f'{delta:+.3f}', f'[{lo:+.3f}, {hi:+.3f}]'])
    table(GENERATED / 'carf_paired.tex', ['Comparator', '$\\Delta$AUROC', '95\\% interval'], rows, 'lrr')
    stress = pd.read_csv(RESULTS / 'stress.csv')
    agg = stress.groupby(['scenario', 'model']).auroc.agg(['mean', 'std'])
    scenarios = ['intact', 'rna_missing', 'mri_missing', 'both_missing', 'rna_permuted', 'mri_permuted', 'rna_noisy', 'mri_noisy']
    models = ['clinical', 'stack', 'mask_stack', 'carf', 'carf_no_robust']
    rows = [[s.replace('_', ' ')] + [f'{agg.loc[(s, m), "mean"]:.3f}' for m in models] for s in scenarios]
    table(GENERATED / 'carf_stress.tex', ['Condition', 'Clinical', 'Stack', 'Mask', 'CARF', 'No robust'], rows, 'lrrrrr')
    coefficients = json.loads((RESULTS / 'fitted_gates.json').read_text())['carf']['coef']
    budgets = np.asarray(coefficients).reshape(2, 3).sum(axis=1)
    cohorts = pd.read_csv(RESULTS / 'cohorts.csv')
    pred = pd.read_csv(RESULTS / 'external_predictions.csv')
    z = logit(np.clip(pred.clinical.to_numpy(), 1e-7, 1 - 1e-7))
    pairs = np.abs(z[:, None] - z[None, :])[np.triu_indices(len(z), 1)]
    certified = np.mean(pairs > 2 * budgets.sum())
    rppa = pd.read_csv(RESULTS / 'rppa_internal.csv').groupby('model').auroc.mean()
    values = {'ClinicalDeltaLow': boot[boot.comparator == 'clinical'].delta_auroc.quantile(.025),
              'ClinicalDeltaHigh': boot[boot.comparator == 'clinical'].delta_auroc.quantile(.975),
              'CarfAuc': ext.loc['carf', 'auroc'], 'AnchorAuc': ext.loc['clinical', 'auroc'],
              'NoRobustAuc': ext.loc['carf_no_robust', 'auroc'], 'StackAuc': ext.loc['stack', 'auroc'],
              'CarfInternal': cv.loc['carf', 'mean'], 'StackInternal': cv.loc['stack', 'mean'],
              'RnaBudget': budgets[0], 'MriBudget': budgets[1], 'TotalBudget': budgets.sum(),
              'RnaPermCarf': agg.loc[('rna_permuted', 'carf'), 'mean'],
              'RnaPermNoRobust': agg.loc[('rna_permuted', 'carf_no_robust'), 'mean'],
              'MriPermCarf': agg.loc[('mri_permuted', 'carf'), 'mean'],
              'MriPermNoRobust': agg.loc[('mri_permuted', 'carf_no_robust'), 'mean'],
              'CertifiedPairs': 100 * certified, 'RppaThreeAuc': rppa['three_acquisition'],
              'RppaFourAuc': rppa['three_plus_rppa']}
    lines = [f'\\newcommand{{\\{k}}}{{{v:.3f}}}' for k, v in values.items()]
    lines += [f'\\newcommand{{\\CommonGenes}}{{{int(cohorts.common_genes.iloc[0]):,}}}']
    (GENERATED / 'carf_macros.tex').write_text('\n'.join(lines) + '\n')
    # Label-free ranking certificate: optional inputs cannot reverse these anchor pairs.
    (RESULTS / 'ranking_certificate.json').write_text(json.dumps({'total_pairs': len(pairs),
        'certified_fraction': float(certified), 'rna_logit_budget': float(budgets[0]),
        'mri_logit_budget': float(budgets[1]), 'note': 'Stability of anchor order, not correctness'}, indent=2))
    make_stress_figure(stress, budgets)
    make_architecture_figure()
    # A concise, automatically generated result summary for source/artifact review.
    (ROOT / 'revision3/RESULTS.md').write_text('# Revision 3 results\n\n' +
        'Exploratory reuse of I-SPY1; these are not fresh confirmatory estimates.\n\n' +
        '```csv\n' + ext.to_csv() + '```' + '\n\nMean internal fold AUROCs:\n\n' + '```csv\n' + cv.to_csv() + '```' +
        '\n\nLearned logit budgets: RNA {:.6f}; MRI {:.6f}.\n'.format(*budgets))


if __name__ == '__main__':
    run()
