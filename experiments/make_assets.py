"""Regenerate all LaTeX assets (macros, tables, figures) for the CARF paper.

Idempotent: reads experiments/results/** (read-only) and writes
  latex/generated/{macros,tab_main,tab_stress,tab_design_stress,tab_sweep}.tex
  latex/figures/{fig_architecture,fig_tradeoff,fig_reliability}.{pdf,png}
Missing later results (a5/a6/a7/a8) degrade gracefully: macros become ??,
table cells become --, figure elements are skipped.

Run: mike2026/env/bin/python experiments/make_assets.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'experiments' / 'results'
GEN = ROOT / 'latex' / 'generated'
FIG = ROOT / 'latex' / 'figures'
COHORTS = ROOT / 'mike2026' / 'results' / 'revision3' / 'cohorts.csv'
ISPY2_META = ROOT / 'mike2026' / 'data' / 'revision2' / 'processed' / 'ISPY2_meta.csv'
EPS = 1e-7
B_CLIP = 1.0  # fixed residual clip B used by the fitted CARF gate
MISSING = '??'
DASH = '--'

# ----------------------------------------------------------------------------- labels / colours
LABEL = {
    'clinical': 'Clinical', 'rna': 'RNA', 'mri': 'MRI', 'mean': 'Mean', 'stack': 'Stacking',
    'mask_stack': 'Mask-stacking', 'no_mri_stack': 'Stacking (no MRI)',
    'offset_lr': 'Offset LR', 'offset_lr_masked': 'Offset LR (masks)', 'mlp_gate': 'MLP gate',
    'moddrop_mlp': 'ModDrop MLP', 'carf': 'CARF', 'carf_platt': '+ Platt scaling',
    'carf_no_robust': 'No robust term', 'carf_no_context': 'Constant basis',
    'robust_offset_lr': 'Robust offset LR',
    'carf_unbounded': 'Unbounded', 'carf_no_mri': 'No MRI',
}
# Okabe-Ito based, fixed per model across all figures.
COLOR = {
    'clinical': '#666666', 'stack': '#E69F00', 'mask_stack': '#009E73',
    'offset_lr': '#CC79A7', 'moddrop_mlp': '#8C6D1F', 'mlp_gate': '#8C6D1F',
    'carf_no_robust': '#0072B2', 'carf': '#D55E00', 'carf_platt': '#000000', 'robust_offset_lr': '#882255',
}
MACRO_SUFFIX = {
    'carf': 'Carf', 'clinical': 'Clin', 'stack': 'Stack', 'mask_stack': 'MaskStack',
    'offset_lr': 'Offset', 'carf_no_robust': 'NoRobust', 'moddrop_mlp': 'ModDrop',
    'robust_offset_lr': 'RobOffset',
}


# ----------------------------------------------------------------------------- helpers
def read(path, **kw):
    path = Path(path)
    return pd.read_csv(path, **kw) if path.exists() else None


def f3(x):
    return MISSING if x is None or pd.isna(x) else f'{x:.3f}'


def f3s(x):
    """Signed, math-safe."""
    if x is None or pd.isna(x):
        return MISSING
    s = f'{x:.3f}'
    return rf'\ensuremath{{-{s[1:]}}}' if s.startswith('-') and s != '-0.000' else s.lstrip('-')


def cell(x, signed=False):
    if x is None or pd.isna(x):
        return DASH
    return f3s(x) if signed else f3(x)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------- data loading
def load():
    D = {}
    D['ext'] = pd.read_csv(RES / 'external.csv').set_index('model')
    D['int'] = pd.read_csv(RES / 'internal.csv')
    D['pred'] = pd.read_csv(RES / 'external_predictions.csv')
    D['stress'] = pd.read_csv(RES / 'stress.csv')
    D['boot'] = pd.read_csv(RES / 'paired_bootstrap.csv').set_index('comparator')
    D['gates'] = json.loads((RES / 'gates.json').read_text())
    D['sweep'] = pd.read_csv(RES / 'sweep.csv')
    D['cohorts'] = pd.read_csv(COHORTS).set_index('cohort')

    a7 = RES / 'a7'
    e7, i7, s7, b7 = (read(a7 / f) for f in ('external.csv', 'internal.csv', 'stress.csv', 'bootstrap.csv'))
    if e7 is not None:
        D['ext'] = pd.concat([D['ext'], e7.set_index('model')])
    if i7 is not None:
        D['int'] = pd.concat([D['int'], i7[D['int'].columns.intersection(i7.columns)]])
    if s7 is not None:
        D['stress'] = pd.concat([D['stress'], s7[['scenario', 'repeat', 'model', 'auroc', 'max_disp']]])
    if b7 is not None:
        b7 = b7.rename(columns={b7.columns[0]: 'comparator'}).set_index('comparator')
        b7.columns = [str(c) for c in b7.columns]
        D['boot'] = pd.concat([D['boot'], b7[['0.025', '0.5', '0.975']]])
    D['a8'] = read(RES / 'a8' / 'input_stress_summary.csv')
    D['a6'] = read(RES / 'a6' / 'metrics.csv')
    D['a6_counts'] = (json.loads((RES / 'a6' / 'counts.json').read_text())
                      if (RES / 'a6' / 'counts.json').exists() else None)
    D['a5'] = read(RES / 'a5' / 'pvalues.csv')
    D['e1_anchor'] = read(RES / 'e1' / 'anchor_failure_summary.csv')
    D['e1_boot'] = read(RES / 'e1' / 'bootstrap_deltas.csv')
    D['e1_cv'] = read(RES / 'e1' / 'budget_cv.csv')
    D['e2'] = read(RES / 'e2' / 'summary.csv', dtype={'level': float})
    D['e2_gaps'] = read(RES / 'e2' / 'anchor_sweep_gaps.csv')
    D['e4_cal'] = read(RES / 'e4' / 'calibration.csv')
    # E5: ModDrop input-level stress (a8 format) and robust offset LR (model key robust_offset_lr).
    e5 = RES / 'e5'
    for f in sorted(e5.glob('*input_stress_summary.csv')) if e5.exists() else []:
        extra = pd.read_csv(f)
        D['a8'] = extra if D['a8'] is None else pd.concat(
            [D['a8'], extra[~extra.model.isin(set(D['a8'].model))]], ignore_index=True)
    ro = {k: read(e5 / f'robust_offset_{k}.csv') for k in ('internal_folds', 'external', 'stress_raw', 'bootstrap')}
    ro['internal'], ro['stress'] = ro.pop('internal_folds'), ro.pop('stress_raw')
    for k in ('internal', 'external', 'stress'):  # e5 files repeat reference models; keep only the new key
        if ro[k] is not None:
            ro[k] = ro[k][ro[k].model == 'robust_offset_lr']
            if ro[k].empty:
                ro[k] = None
    if ro['external'] is not None:
        D['ext'] = pd.concat([D['ext'], ro['external'].set_index('model')])
    if ro['internal'] is not None:
        D['int'] = pd.concat([D['int'], ro['internal'][D['int'].columns.intersection(ro['internal'].columns)]])
    if ro['stress'] is not None:
        D['stress'] = pd.concat([D['stress'], ro['stress'][[c for c in ('scenario', 'repeat', 'model', 'auroc', 'max_disp')
                                                             if c in ro['stress'].columns]]])
    rb = ro['bootstrap']
    if rb is not None and 'comparator' in rb.columns:
        rb = rb[rb.comparator == 'robust_offset_lr']
    if rb is not None and len(rb):
        if {'q025', 'q975', 'comparator'} <= set(rb.columns):  # e1 format (cohort/scenario/.../q025/q975)
            rb = rb.copy()
            if 'cohort' not in rb.columns:
                rb['cohort'] = 'external_112'
            if 'scenario' not in rb.columns:
                rb['scenario'] = 'intact'
            D['e1_boot'] = rb if D['e1_boot'] is None else pd.concat([D['e1_boot'], rb], ignore_index=True)
            it = rb[rb.scenario == 'intact'].set_index('comparator')
            if len(it):
                D['boot'] = pd.concat([D['boot'], it.rename(columns={'q025': '0.025', 'q50': '0.5', 'q975': '0.975'})
                                       [['0.025', '0.5', '0.975']]])
        else:  # a7 format (index comparator, quantile columns); optional 'scenario' column
            rb = rb.rename(columns={rb.columns[0]: 'comparator'})
            rb.columns = [str(c) for c in rb.columns]
            if 'scenario' in rb.columns:
                perm = rb[rb.scenario == 'rna_permuted']
                add = pd.DataFrame({'cohort': 'external_112', 'scenario': 'rna_permuted', 'comparator': perm.comparator,
                                    'q025': perm['0.025'], 'q50': perm['0.5'], 'q975': perm['0.975']})
                D['e1_boot'] = pd.concat([D['e1_boot'], add], ignore_index=True)
                rb = rb[rb.scenario == 'intact']
            D['boot'] = pd.concat([D['boot'], rb.set_index('comparator')[['0.025', '0.5', '0.975']]])

    D['boot'] = D['boot'][~D['boot'].index.duplicated(keep='first')]
    I = D['int']
    I = I.assign(repeat=I.fold // 5)
    rep = I.groupby(['model', 'repeat']).auroc.mean().unstack()
    D['int_mean'] = I.groupby('model').auroc.mean()
    D['int_sd'] = I.groupby('model').auroc.std(ddof=1)      # SD over the 25 outer folds (as Table S6)
    D['int_sd_rep'] = rep.std(axis=1, ddof=1)                  # SD of the 5 repeat means
    fa = I.pivot_table(index='fold', columns='model', values='auroc')
    D['int_diff_stack'] = (fa['carf'] - fa['stack']).dropna() if {'carf', 'stack'} <= set(fa.columns) else None
    D['int_ece'] = I.groupby('model').ece.mean()
    S = D['stress']
    D['stress_mean'] = S.groupby(['scenario', 'model']).auroc.mean()
    D['stress_sd'] = S.groupby(['scenario', 'model']).auroc.std(ddof=1)
    D['max_disp'] = S.groupby('model').max_disp.max()
    return D


def get(series, *key):
    try:
        v = series.loc[key if len(key) > 1 else key[0]]
        return float(v)
    except (KeyError, TypeError):
        return None


# ----------------------------------------------------------------------------- macros
def macros(D):
    M = {}  # name -> (value, comment)
    ext, co = D['ext'], D['cohorts']
    M['NDev'] = str(int(co.loc['ISPY2', 'n']))
    # cohorts.csv is the analysed complete-case cohort (authoritative); ISPY2_meta.csv is only cross-checked.
    pos_meta = int(co.loc['ISPY2', 'positive'])
    if ISPY2_META.exists():
        meta = pd.read_csv(ISPY2_META)
        if len(meta) != int(co.loc['ISPY2', 'n']) or int(meta.pcr.sum()) != pos_meta:
            print(f'NOTE: ISPY2_meta.csv has n={len(meta)}, pCR={int(meta.pcr.sum())}; '
                  f'using cohorts.csv n={int(co.loc["ISPY2", "n"])}, pCR={pos_meta}')
    M['NDevPos'] = str(pos_meta)
    M['NExt'] = str(int(co.loc['ISPY1', 'n']))
    M['NExtPos'] = str(int(co.loc['ISPY1', 'positive']))
    M['NGenes'] = f"{int(co.loc['ISPY2', 'common_genes']):,}".replace(',', '{,}')
    M['NMriFeat'] = str(int(co.loc['ISPY2', 'mri_features']))
    M['NDevNeg'] = str(int(co.loc['ISPY2', 'n']) - pos_meta)
    M['PctDevPos'] = f"{100 * pos_meta / co.loc['ISPY2', 'n']:.1f}"
    M['PctExtPos'] = f"{100 * co.loc['ISPY1', 'positive'] / co.loc['ISPY1', 'n']:.1f}"

    ext_names = {'Carf': 'carf', 'Clin': 'clinical', 'Rna': 'rna', 'Mri': 'mri', 'Mean': 'mean',
                 'Stack': 'stack', 'MaskStack': 'mask_stack', 'Offset': 'offset_lr',
                 'OffsetMasked': 'offset_lr_masked', 'NoRobust': 'carf_no_robust',
                 'NoContext': 'carf_no_context', 'Unbounded': 'carf_unbounded', 'NoMri': 'carf_no_mri',
                 'Platt': 'carf_platt', 'Mlp': 'mlp_gate', 'ModDrop': 'moddrop_mlp',
                 'NoMriStack': 'no_mri_stack', 'RobOffset': 'robust_offset_lr'}
    for suf, m in ext_names.items():
        M['Ext' + suf] = f3(get(ext.auroc, m))
        M['ExtAp' + suf] = f3(get(ext.auprc, m))

    int_names = {'Carf': 'carf', 'Clin': 'clinical', 'Stack': 'stack', 'Mean': 'mean',
                 'MaskStack': 'mask_stack', 'Offset': 'offset_lr', 'NoRobust': 'carf_no_robust',
                 'ModDrop': 'moddrop_mlp', 'Rna': 'rna', 'Mri': 'mri', 'Mlp': 'mlp_gate',
                 'NoContext': 'carf_no_context', 'Unbounded': 'carf_unbounded', 'NoMri': 'carf_no_mri',
                 'RobOffset': 'robust_offset_lr', 'OffsetMasked': 'offset_lr_masked', 'Platt': 'carf_platt'}
    for suf, m in int_names.items():
        M['Int' + suf] = f3(get(D['int_mean'], m))
        M['Int' + suf + 'Sd'] = f3(get(D['int_sd'], m))
        M['Int' + suf + 'SdRep'] = f3(get(D['int_sd_rep'], m))
    dd = D['int_diff_stack']
    M['IntDiffStackMean'] = f3s(dd.mean()) if dd is not None else MISSING
    M['IntDiffStackSd'] = f3(dd.std(ddof=1)) if dd is not None else MISSING
    M['IntFoldsWorse'] = str(int((dd < 0).sum())) if dd is not None else MISSING
    M['IntNFolds'] = str(len(dd)) if dd is not None else MISSING

    B = D['boot']
    for suf, m in {'Clin': 'clinical', 'Stack': 'stack', 'Offset': 'offset_lr', 'NoRobust': 'carf_no_robust',
                   'ModDrop': 'moddrop_mlp', 'OffsetMasked': 'offset_lr_masked', 'Mlp': 'mlp_gate',
                   'RobOffset': 'robust_offset_lr'}.items():
        M[f'D{suf}Lo'] = f3s(get(B['0.025'], m))
        M[f'D{suf}Hi'] = f3s(get(B['0.975'], m))
        M[f'D{suf}Mid'] = f3s(get(B['0.5'], m))

    for suf, m in {'Carf': 'carf', 'Stack': 'stack', 'Clin': 'clinical', 'Platt': 'carf_platt',
                   'Offset': 'offset_lr', 'MaskStack': 'mask_stack', 'NoRobust': 'carf_no_robust',
                   'ModDrop': 'moddrop_mlp'}.items():
        M['Brier' + suf] = f3(get(ext.brier, m))
        M['Ece' + suf] = f3(get(ext.ece, m))
    for suf, m in {'Carf': 'carf', 'Platt': 'carf_platt', 'Stack': 'stack', 'Clin': 'clinical',
                   'Offset': 'offset_lr'}.items():
        M['IntEce' + suf] = f3(get(D['int_ece'], m))

    sm, sd = D['stress_mean'], D['stress_sd']
    clin0 = get(ext.auroc, 'clinical')
    for m, suf in dict(MACRO_SUFFIX, carf_no_context='NoContext', carf_unbounded='Unbounded', carf_no_mri='NoMri').items():
        for scen, name in [('rna_permuted', 'RnaPerm'), ('mri_permuted', 'MriPerm'),
                           ('rna_noisy', 'RnaNoise'), ('mri_noisy', 'MriNoise')]:
            M[name + suf] = f3(get(sm, scen, m))
            M[name + 'Sd' + suf] = f3(get(sd, scen, m))
        v = get(sm, 'rna_permuted', m)
        M['RnaPermLoss' + suf] = f3s(None if v is None else v - clin0)
        v = get(sm, 'mri_permuted', m)
        M['MriPermLoss' + suf] = f3s(None if v is None else v - clin0)
        M['MaxDisp' + suf] = f3(get(D['max_disp'], m)) if m != 'carf' else None

    g = D['gates']
    rb, mb = g['carf']['budgets']
    M['RnaBudget'], M['MriBudget'], M['TotalBudget'] = f3(rb), f3(mb), f3(rb + mb)
    M['NoRobustRnaBudget'], M['NoRobustMriBudget'] = (f3(x) for x in g['carf_no_robust']['budgets'])
    M['MaxDispCarf'] = f3(get(D['max_disp'], 'carf'))
    M = {k: v for k, v in M.items() if v is not None}
    a = logit(D['pred'].clinical.to_numpy())
    thr = 2 * B_CLIP * (rb + mb)
    diff = np.abs(a[:, None] - a[None, :])[np.triu_indices(len(a), 1)]
    M['CertifiedPairs'] = f'{100 * np.mean(diff > thr):.1f}'
    M['NCertPairs'] = f'{int(np.sum(diff > thr)):,}'.replace(',', '{,}')
    M['NTotalPairs'] = f'{len(diff):,}'.replace(',', '{,}')
    # Sharp budget: |e^{-|d|} clip(d,-B,B)| <= kappa(B) = B e^{-B} (B<=1) or e^{-1} (B>1).
    kappa = B_CLIP * np.exp(-B_CLIP) if B_CLIP <= 1 else np.exp(-1)
    th = np.asarray(g['carf']['coef']).reshape(-1, 3)
    sharp = B_CLIP * th[:, 0] + kappa * th[:, 1] + B_CLIP * th[:, 2]
    M['SharpBudgetRna'], M['SharpBudgetMri'], M['SharpBudget'] = f3(sharp[0]), f3(sharp[1]), f3(sharp.sum())
    th_nr = np.asarray(g['carf_no_robust']['coef']).reshape(-1, 3)
    sharp_nr = B_CLIP * th_nr[:, 0] + kappa * th_nr[:, 1] + B_CLIP * th_nr[:, 2]
    M['SharpBudgetRnaNoRobust'], M['SharpBudgetMriNoRobust'] = f3(sharp_nr[0]), f3(sharp_nr[1])
    M['SharpBudgetNoRobust'] = f3(sharp_nr.sum())
    M['NoContextCoefSum'] = f3(float(np.sum(g['carf_no_context']['coef'])))
    M['CertifiedPairsSharp'] = f'{100 * np.mean(diff > 2 * sharp.sum()):.1f}'
    # Max displacement over output- and input-level scenarios.
    a8 = read(RES / 'a8/input_stress_summary.csv')
    md = get(D['max_disp'], 'carf')
    if a8 is not None:
        md = max(md, a8[a8.model == 'carf'].max_shift_max.max())
    M['MaxDispCarfAll'] = f3(md)
    # Max displacement over output- and input-level scenarios for the other models (suffix All).
    for m, suf in MACRO_SUFFIX.items():
        v = get(D['max_disp'], m)
        if D['a8'] is not None and m in set(D['a8'].model):
            w = D['a8'][D['a8'].model == m].max_shift_max.max()
            v = w if v is None else max(v, w)
        if m != 'carf':
            M['MaxDisp' + suf + 'All'] = f3(v)
    # Second domain (CMU-MOSI).
    c3, s3 = read(RES / 'e3/clean.csv'), read(RES / 'e3/stress.csv')
    if c3 is not None:
        au = c3.set_index('model').auroc
        M.update(MosiN=str(int(c3.n.iloc[0])), MosiText=f3(au['text']), MosiAudio=f3(au['audio']),
                 MosiVisual=f3(au['visual']), MosiCarf=f3(au['carf']), MosiStack=f3(au['stack']),
                 MosiMaskStack=f3(au['mask_stack']), MosiOffset=f3(au['offset_lr_masked']),
                 MosiModDrop=f3(au['moddrop_mlp']))
        base = ['stack', 'mask_stack', 'offset_lr', 'offset_lr_masked', 'moddrop_mlp', 'mlp_gate', 'mean']
        M['MosiGap'] = f3(max(au[m] for m in base) - au['carf'])
        M['MosiOffsetGain'] = f3(au['offset_lr_masked'] - au['text'])
        worst = s3.groupby(['scenario', 'model']).auroc.mean().unstack().min()
        wb = [worst[m] for m in ['stack', 'mask_stack', 'mean', 'offset_lr_masked', 'moddrop_mlp']]
        M['MosiWorstCarf'], M['MosiWorstBaseLo'], M['MosiWorstBaseHi'] = f3(worst['carf']), f3(min(wb)), f3(max(wb))
        M['MosiNScen'] = str(s3.scenario.nunique() - 1)

    sw = D['sweep'].set_index(['budget', 'robust_weight'])
    def swv(b, gm, col):
        try:
            return f3(sw.loc[(b, gm), col])
        except KeyError:
            return MISSING
    M['SweepFourZeroRna'] = swv(4.0, 0.0, 'ext_rna_permuted')
    M['SweepFourQuarterRna'] = swv(4.0, 0.25, 'ext_rna_permuted')
    M['SweepFourZeroClean'] = swv(4.0, 0.0, 'external_auroc')
    M['SweepFourQuarterClean'] = swv(4.0, 0.25, 'external_auroc')
    M['SweepQuarterClean'] = swv(0.25, 0.25, 'external_auroc')
    M['SweepTwoQuarterClean'] = swv(2.0, 0.25, 'external_auroc')
    M['SweepOneQuarterClean'] = swv(1.0, 0.25, 'external_auroc')
    M['SweepOneQuarterRna'] = swv(1.0, 0.25, 'ext_rna_permuted')
    M['SweepOneQuarterMri'] = swv(1.0, 0.25, 'ext_mri_permuted')
    M['SweepHalfZeroClean'] = swv(0.5, 0.0, 'external_auroc')
    M['SweepHalfZeroRna'] = swv(0.5, 0.0, 'ext_rna_permuted')
    M['SweepHalfZeroMri'] = swv(0.5, 0.0, 'ext_mri_permuted')

    # A8 input-level corruption
    a8 = D['a8']
    a8m = a8.set_index(['scenario', 'model']).auroc_mean if a8 is not None else None
    def a8v(scen, m):
        return f3(get(a8m, scen, m)) if a8m is not None else MISSING
    for suf, m in [('Carf', 'carf'), ('Stack', 'stack'), ('Offset', 'offset_lr'), ('Clin', 'clinical'),
                   ('NoRobust', 'carf_no_robust'), ('MaskStack', 'mask_stack'),
                   ('ModDrop', 'moddrop_mlp'), ('RobOffset', 'robust_offset_lr')]:
        M['GeneDropFifty' + suf] = a8v('rna_dropout50', m)
        M['BlockShift' + suf] = a8v(STRONG_SHIFT, m)
        M['MriInNoise' + suf] = a8v(STRONG_MRI_NOISE, m)
        M['MriInDrop' + suf] = a8v(STRONG_MRI_DROP, m)

    # A6 natural missingness
    a6, cnt = D['a6'], D['a6_counts'] or {}
    a6m = a6.set_index(['subgroup', 'model']) if a6 is not None else None
    for suf, g in [('RnaOnly', 'rna_only'), ('MriOnly', 'mri_only'), ('ClinOnly', 'clinical_only'),
                   ('All', 'all_available'), ('Complete', 'complete_112')]:
        if a6m is not None and g in a6m.index.get_level_values(0):
            sub = a6m.loc[g].iloc[0]
            M[f'Nat{suf}N'], M[f'Nat{suf}Pos'] = str(int(sub.n)), str(int(sub.pcr_pos))
        elif g in cnt:
            M[f'Nat{suf}N'], M[f'Nat{suf}Pos'] = str(cnt[g]['n']), str(cnt[g]['pcr_pos'])
        else:
            M[f'Nat{suf}N'] = M[f'Nat{suf}Pos'] = MISSING
    for suf, m in [('Carf', 'carf'), ('Clin', 'clinical'), ('NoRobust', 'carf_no_robust'),
                   ('MaskStack', 'mask_stack'), ('Stack', 'stack'),
                   ('Offset', 'offset_lr_masked'), ('Mean', 'mean')]:
        for pre, g in [('NatAll', 'all_available'), ('NatRna', 'rna_only'), ('NatMri', 'mri_only')]:
            M[pre + suf] = f3(get(a6m.auroc, g, m)) if a6m is not None else MISSING
            M[pre + 'Brier' + suf] = f3(get(a6m.brier, g, m)) if a6m is not None else MISSING

    if a6m is not None:
        for suf, g6 in [('Rna', 'rna_only'), ('Mri', 'mri_only'), ('All', 'all_available'), ('Complete', 'complete_112')]:
            c, k = get(a6m.auroc, g6, 'carf'), get(a6m.auroc, g6, 'clinical')
            M[f'Nat{suf}DeltaCarf'] = f3s(None if c is None or k is None else c - k)
            try:
                sub = a6m.loc[g6].iloc[0]
                M[f'NatRate{suf}'] = f'{100 * sub.pcr_pos / sub.n:.1f}'
            except KeyError:
                M[f'NatRate{suf}'] = MISSING
    else:
        for k in 'NatRnaDeltaCarf NatMriDeltaCarf NatRateComplete NatRateRna NatRateMri'.split():
            M[k] = MISSING

    # E4 calibration (Platt-scaled external predictions)
    cal = D['e4_cal']
    for suf, m in [('Carf', 'carf'), ('Stack', 'stack'), ('MaskStack', 'mask_stack'), ('Offset', 'offset_lr'),
                   ('NoRobust', 'carf_no_robust'), ('Clin', 'clinical')]:
        r = cal[(cal.model == m) & (cal.variant == 'platt')] if cal is not None else []
        M['PlattBrier' + suf] = f3(r.brier.iloc[0]) if len(r) else MISSING
        M['PlattEce' + suf] = f3(r.ece10.iloc[0]) if len(r) else MISSING
        M['PlattSlope' + suf] = f'{r.cal_slope.iloc[0]:.2f}' if len(r) else MISSING
        M['PlattIntercept' + suf] = (f3s(r.cal_intercept.iloc[0]) if len(r) else MISSING)
        rr = cal[(cal.model == m) & (cal.variant == 'raw')] if cal is not None else []
        M['RawSlope' + suf] = f'{rr.cal_slope.iloc[0]:.2f}' if len(rr) else MISSING

    # A5 permutation
    a5 = D['a5']
    ok5 = a5 is not None and 'carf' in set(a5.model)
    a5m = a5.set_index('model') if a5 is not None else None
    M['PermPCarf'] = f"{get(a5m.p_value, 'carf'):.3f}" if ok5 else MISSING
    M['PermNullCarf'] = f3(get(a5m.null_mean, 'carf')) if ok5 else MISSING
    M['PermNullCarfSd'] = f3(get(a5m.null_sd, 'carf')) if ok5 else MISSING
    M['PermNullCarfMax'] = f3(get(a5m.null_max, 'carf')) if ok5 else MISSING
    M['NPerm'] = str(int(get(a5m.n_perm, 'carf'))) if ok5 else MISSING
    M['PermMinP'] = f"{1 / (1 + int(get(a5m.n_perm, 'carf'))):.3f}" if ok5 else MISSING

    # E1: bootstrap deltas under RNA permutation / natural missingness
    eb = D['e1_boot']
    def ebv(cohort, scen, comp, q):
        if eb is None:
            return MISSING
        r = eb[(eb.cohort == cohort) & (eb.scenario == scen) & (eb.comparator == comp)]
        return f3s(r[q].iloc[0]) if len(r) else MISSING
    for suf, m in [('Stack', 'stack'), ('MaskStack', 'mask_stack'), ('Offset', 'offset_lr'),
                   ('NoRobust', 'carf_no_robust'), ('ModDrop', 'moddrop_mlp'), ('RobOffset', 'robust_offset_lr')]:
        M[f'DPerm{suf}Lo'] = ebv('external_112', 'rna_permuted', m, 'q025')
        M[f'DPerm{suf}Hi'] = ebv('external_112', 'rna_permuted', m, 'q975')
        M[f'DPerm{suf}Mid'] = ebv('external_112', 'rna_permuted', m, 'q50')
    for suf, m in [('Clin', 'clinical'), ('Stack', 'stack'), ('MaskStack', 'mask_stack'),
                   ('Offset', 'offset_lr_masked')]:
        for sub, pre in [('all_available', 'DNat'), ('rna_only', 'DNatRna'), ('mri_only', 'DNatMri')]:
            M[f'{pre}{suf}Lo'] = ebv(f'natural_missing_{sub}', 'natural', m, 'q025')
            M[f'{pre}{suf}Hi'] = ebv(f'natural_missing_{sub}', 'natural', m, 'q975')
            M[f'{pre}{suf}Mid'] = ebv(f'natural_missing_{sub}', 'natural', m, 'q50')

    # E1: budget chosen by inner CV
    cv = D['e1_cv']
    if cv is not None:
        folds = cv[cv.split.str.startswith('fold_')]
        vc = folds.chosen_B.value_counts()
        M['CvBMode'], M['CvBCount'], M['CvBFolds'] = f'{vc.index[0]:g}', str(int(vc.iloc[0])), str(len(folds))
        ext_cv = cv[cv.split == 'external'].iloc[0]
        M['CvBExt'] = f'{ext_cv.chosen_B:g}'
        M['CvExt'], M['CvRnaPerm'] = f3(ext_cv.auroc_cv), f3(ext_cv.rna_perm_mean_cv)
        M['CvRnaPermSd'], M['CvRnaPermFixed'] = f3(ext_cv.rna_perm_sd_cv), f3(ext_cv.rna_perm_mean_fixed1)
        M['CvInt'], M['CvIntFixed'] = f3(folds.auroc_cv.mean()), f3(folds.auroc_fixed1.mean())
        M['CvExtFixed'] = f3(ext_cv.auroc_fixed1)
    else:
        for k in 'CvBMode CvBCount CvBFolds CvExt CvRnaPerm CvInt'.split():
            M[k] = MISSING

    # E1: anchor (clinical) failure
    an = D['e1_anchor']
    anm = an.set_index(['scenario', 'model'])['mean'] if an is not None else None
    ans = an.set_index(['scenario', 'model'])['std'] if an is not None else None
    for suf, m in [('Carf', 'carf'), ('Stack', 'stack'), ('MaskStack', 'mask_stack'), ('Offset', 'offset_lr'),
                   ('Clin', 'clinical'), ('NoRobust', 'carf_no_robust'), ('Rna', 'rna')]:
        for pre, sc in [('HrFlip', 'hr_flip25'), ('HerFlip', 'her2_flip25'), ('ClinPerm', 'clinical_output_permuted'),
                        ('HrFlipTen', 'hr_flip10'), ('HerFlipTen', 'her2_flip10'), ('AgeNoise', 'age_noise_sd10')]:
            M[pre + suf] = f3(get(anm, sc, m)) if anm is not None else MISSING
            M[pre + 'Sd' + suf] = f3(get(ans, sc, m)) if ans is not None else MISSING

    # E2: synthetic anchor sweep (CARF - stacking test AUROC gaps)
    gp = D['e2_gaps']
    if gp is not None:
        def gap(scen, lvl, target):
            r = gp[(gp.scenario == scen) & np.isclose(gp.level, lvl) & np.isclose(gp.anchor_target, target)]
            return float(r.carf_minus_stack_mean.iloc[0]) if len(r) else None
        M['SynIntactGap'] = f3s(gap('intact', 0.0, 0.80))
        M['SynSilentGap'] = f3s(gap('silent', 1.0, 0.80))
        for name, scen, lvl in [('Intact', 'intact', 0.0), ('Silent', 'silent', 1.0)]:
            r = gp[(gp.scenario == scen) & np.isclose(gp.level, lvl) & (gp.anchor_target >= 0.70 - 1e-9)]
            a = r.carf_minus_stack_mean.abs()
            lo, hi = f'{a.min():.2f}', f'{a.max():.2f}'
            M[f'Syn{name}Range'] = lo if lo == hi else f'{lo}--{hi}'
            M[f'Syn{name}RangeLo'], M[f'Syn{name}RangeHi'] = f3s(r.carf_minus_stack_mean.min()), f3s(r.carf_minus_stack_mean.max())
    else:
        for k in 'SynIntactGap SynSilentGap SynIntactRange SynSilentRange'.split():
            M[k] = MISSING

    # Platt recalibration of the external CARF fit: logit(p_platt) = alpha * z_carf + beta (exact)
    P = D['pred']
    alpha, beta = np.polyfit(logit(P.carf), logit(P.carf_platt), 1)
    M['PlattAlpha'], M['PlattBeta'] = f'{alpha:.2f}', (rf'\ensuremath{{-{abs(beta):.2f}}}' if beta < 0 else f'{beta:.2f}')

    required = ('NDev NDevPos NExt NExtPos NGenes NMriFeat ExtCarf ExtClin ExtRna ExtMri ExtMean ExtStack '
                'ExtMaskStack ExtOffset ExtOffsetMasked ExtNoRobust ExtNoContext ExtUnbounded ExtNoMri '
                'ExtPlatt ExtMlp ExtModDrop IntCarf IntCarfSd IntClin IntStack IntMean IntMaskStack IntOffset '
                'IntNoRobust IntModDrop DClinLo DClinHi DClinMid DStackLo DStackHi DOffsetLo DOffsetHi '
                'DNoRobustLo DNoRobustHi DModDropLo DModDropHi BrierCarf BrierStack BrierClin '
                'BrierPlatt BrierOffset EceCarf EceStack EceClin EcePlatt EceOffset IntEceCarf IntEcePlatt '
                'IntEceStack RnaBudget MriBudget TotalBudget NoRobustRnaBudget NoRobustMriBudget MaxDispCarf '
                'CertifiedPairs SweepFourZeroRna SweepFourQuarterRna SweepFourZeroClean SweepFourQuarterClean '
                'SweepQuarterClean SweepTwoQuarterClean GeneDropFiftyCarf GeneDropFiftyStack GeneDropFiftyOffset '
                'GeneDropFiftyClin BlockShiftCarf BlockShiftStack BlockShiftOffset NatRnaOnlyN NatMriOnlyN NatAllN '
                'NatAllCarf NatAllMaskStack NatAllStack NatAllClin PermPCarf PermNullCarf PermNullCarfSd NPerm '
                'PlattAlpha PlattBeta NatRnaOnlyPos NatMriOnlyPos NatClinOnlyN CvBMode CvBCount CvBFolds CvExt '
                'CvRnaPerm CvInt SynIntactGap SynSilentGap SynIntactRange SynSilentRange IntDiffStackMean '
                'IntDiffStackSd IntFoldsWorse ExtRobOffset IntRobOffset RnaPermRobOffset MriPermRobOffset '
                'DPermRobOffsetLo DPermRobOffsetHi MaxDispRobOffset PlattSlopeCarf SweepHalfZeroClean '
                'SweepHalfZeroRna SweepHalfZeroMri SweepOneQuarterMri SharpBudgetRnaNoRobust NoContextCoefSum '
                'NatRnaDeltaCarf NatMriDeltaCarf NatRateComplete NatRateRna NatRateMri').split()
    required += [f'Platt{k}{s}' for k in ('Brier', 'Ece')
                 for s in 'Carf Stack MaskStack Offset NoRobust Clin'.split()]
    required += [f'DPerm{s}{e}' for s in 'Stack MaskStack Offset NoRobust ModDrop'.split() for e in ('Lo', 'Hi')]
    required += [f'DNat{s}{e}' for s in 'Clin Stack MaskStack Offset'.split() for e in ('Lo', 'Hi')]
    required += [f'{p}{s}' for p in ('HrFlip', 'HerFlip', 'ClinPerm')
                 for s in 'Carf Stack MaskStack Offset Clin NoRobust'.split()]
    required += [f'{p}{s}' for p in ('RnaPerm', 'MriPerm', 'RnaNoise', 'MriNoise', 'RnaPermLoss')
                 for s in MACRO_SUFFIX.values()]
    for k in required:
        M.setdefault(k, MISSING)
    extra = [k for k in M if k not in required]
    missing = sorted(k for k, v in M.items() if v == MISSING)

    lines = ['% AUTO-GENERATED by experiments/make_assets.py -- do not edit by hand.',
             '% Macros use \\providecommand + \\renewcommand, so this file may be input after',
             '% older macro files without "already defined" errors (values here win).',
             '% Signed quantities (D*Lo/Hi/Mid, *Loss*) are wrapped in \\ensuremath when negative.',
             '% \\CertifiedPairs is a percentage WITHOUT the % sign (write \\CertifiedPairs\\%).',
             '% ?? = input result file not yet available.',
             '%',
             '% Extra macros beyond the requested list:',
             '%   NDevNeg PctDevPos PctExtPos (cohort); ExtAp<Model> (external AP);',
             '%   ExtNoMriStack; Int<Model>Sd for every Int<Model>; IntRna IntMri IntMlp IntNoContext',
             '%   IntUnbounded IntNoMri; D{OffsetMasked,Mlp}{Lo,Mid,Hi}, D*Mid for all comparators;',
             '%   Brier/Ece for MaskStack NoRobust ModDrop; IntEce{Clin,Offset};',
             '%   {Rna,Mri}{Perm,Noise}Sd<Model> (SD over 30 realisations); MriPermLoss<Model>;',
             '%   MaxDisp<Model> (max |z-a| over all output-level scenarios, baselines);',
             '%   NCertPairs NTotalPairs; SweepOneQuarter{Clean,Rna};',
             '%   MriInNoise/MriInDrop/GeneDropFifty/BlockShift also for Clin (and MaskStack NoRobust);',
             '%   Nat{RnaOnly,MriOnly,ClinOnly,All,Complete}{N,Pos}; Nat{All,Rna,Mri}Brier<Model>;',
             '%   Nat*Offset = offset LR with masks; PermNullCarfMax PermMinP (=1/(1+NPerm)).',
             '%   PlattAlpha/PlattBeta recovered exactly from external predictions (logit-linear map).',
             '%   E1: DPerm<M>Mid; DNat{,Rna,Mri}<M>{Lo,Mid,Hi} (DNat = all 199); CvBExt CvRnaPermSd',
             '%   CvRnaPermFixed CvIntFixed CvExtFixed; {HrFlip,HerFlip,ClinPerm}Sd<M>; HrFlipTen HerFlipTen',
             '%   AgeNoise (+Sd) for all anchor models; anchor macros also for Rna.',
             '%   Int*Sd = SD over the 25 outer folds; Int*SdRep = SD of the 5 repeat means; IntNFolds;',
             '%   IntDiffStack{Mean,Sd}: per-fold paired CARF - stacking internal AUROC; IntFoldsWorse: #folds CARF<stack.',
             '%   SharpBudget{Mri,}NoRobust; PlattIntercept<M>, PlattSlope<M>, RawSlope<M> (E4);',
             '%   Nat{All,Complete}DeltaCarf, NatRateAll (percent, no % sign); DPerm/DPermOffsetMasked.',
             '%   E2: Syn{Intact,Silent}Range{Lo,Hi} (signed min/max CARF-stack gap, anchor >= 0.70);',
             '%   Syn*Range are |gap| ranges at 2 decimals, e.g. 0.03--0.04 (text mode).',
             f'%   BlockShift* uses the strongest A8 shift scenario: {STRONG_SHIFT}.',
             '%', f'% Currently missing ({len(missing)}): ' + ' '.join(missing), '']
    for k, v in M.items():
        lines.append(f'\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}')
    (GEN / 'macros.tex').write_text('\n'.join(lines) + '\n')
    return M, missing, extra


STRONG_SHIFT = 'rna_block25_shift+2'
STRONG_MRI_NOISE = 'mri_noise_sd1'
STRONG_MRI_DROP = 'mri_dropout60'


# ----------------------------------------------------------------------------- tables
MAIN_COMPACT_DROP = {'offset_lr_masked', 'mlp_gate', 'carf_no_mri'}


def tab_main(D, full=True):
    """full=True -> tab_main_full.tex (supplement); full=False -> compact tab_main.tex (main paper)."""
    ext, B = D['ext'], D['boot']
    groups = [('Single experts', ['clinical', 'rna', 'mri']),
              ('Fusion baselines', ['mean', 'stack', 'mask_stack', 'offset_lr', 'offset_lr_masked',
                                    'robust_offset_lr', 'mlp_gate', 'moddrop_mlp']),
              ('CARF and ablations', ['carf', 'carf_platt', 'carf_no_robust', 'carf_no_context',
                                      'carf_unbounded', 'carf_no_mri'])]
    if not full:
        groups = [(g, [m for m in ms if m not in MAIN_COMPACT_DROP | {'carf_platt'}]) for g, ms in groups]
        return tab_main_compact(D, groups)
    cols = ('auroc', 'auprc', 'brier', 'ece') if full else ('auroc', 'brier', 'ece')
    nc = 3 + len(cols)
    out = [r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}l' + 'c' * (nc - 1) + r'@{}}', r'\toprule',
           rf' & Internal & \multicolumn{{{len(cols)}}}{{c}}{{External (I-SPY1)}} & $\Delta$AUROC \\',
           rf'\cmidrule(lr){{2-2}}\cmidrule(lr){{3-{2 + len(cols)}}}\cmidrule(l){{{nc}-{nc}}}',
           'Model & AUROC & ' + ' & '.join({'auroc': 'AUROC', 'auprc': 'AP', 'brier': 'Brier', 'ece': 'ECE'}[c]
                                            for c in cols) + r' & 95\% CI \\', r'\midrule']
    for gi, (gname, models) in enumerate(groups):
        if gi:
            out.append(r'\midrule')
        out.append(rf'\multicolumn{{{nc}}}{{@{{}}l}}{{\emph{{{gname}}}}} \\')
        for m in models:
            im, isd = get(D['int_mean'], m), get(D['int_sd'], m)
            if im is None and m not in ext.index:
                continue  # later result not available yet
            intc = DASH if im is None else f'{im:.3f} $\\pm$ {isd:.3f}'
            e = [cell(get(ext[c], m)) for c in cols]
            eb = D['e1_boot']
            eb_row = (eb[(eb.cohort == 'external_112') & (eb.scenario == 'intact') & (eb.comparator == m)]
                      if eb is not None else [])
            if m in B.index:
                d = f"[{f3s(B.loc[m, '0.025'])}, {f3s(B.loc[m, '0.975'])}]"
            elif len(eb_row):  # same paired bootstrap (results/e1), covers mask-stacking
                d = f"[{f3s(eb_row.q025.iloc[0])}, {f3s(eb_row.q975.iloc[0])}]"
            else:
                d = DASH
            out.append(f'\\hspace{{0.5em}}{LABEL[m]} & {intc} & ' + ' & '.join(e) + f' & {d} \\\\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / ('tab_main_full.tex' if full else 'tab_main.tex')).write_text('% AUTO-GENERATED by experiments/make_assets.py\n'
                                      '% Internal: mean over 25 folds $\\pm$ SD of the 5 repeat means. '
                                      'Delta: [2.5, 97.5] percentiles of paired bootstrap (CARF minus model).\n'
                                      '\\small\n' + '\n'.join(out) + '\n')



def bold_best(rows, cols, mode):
    """rows: list of [label, cells(list of str), in_set(bool)]; bold best value (ties included) per column
    in `cols` among rows with in_set; mode[c] is 'max' or 'min'."""
    for c in cols:
        vals = []
        for r in rows:
            if r[2]:
                try:
                    vals.append(round(float(r[1][c]), 3))
                except ValueError:
                    pass
        if not vals:
            continue
        best = max(vals) if mode[c] == 'max' else min(vals)
        for r in rows:
            if not r[2]:
                continue
            try:
                if round(float(r[1][c]), 3) == best:
                    r[1][c] = r'\textbf{' + r[1][c] + '}'
            except ValueError:
                pass
    return rows


def tab_main_compact(D, groups):
    """Main-paper Table 1: clean discrimination next to robustness and the displacement guarantee."""
    ext, B = D['ext'], D['boot']
    sm = D['stress_mean']
    a8 = D['a8']

    def perm(m):
        v = get(sm, 'rna_permuted', m)
        return DASH if v is None else f'{v:.3f}'

    def disp(m):
        v = get(D['max_disp'], m)
        if a8 is not None and m in set(a8.model):
            w = a8[a8.model == m].max_shift_max.max()
            v = w if v is None else max(v, w)
        return DASH if v is None else f'{v:.2f}'

    out = [r'\setlength{\tabcolsep}{1.9pt}', r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
           r' & Internal & \multicolumn{4}{c}{External (I-SPY1)} \\',
           r'\cmidrule(lr){2-2}\cmidrule(l){3-6}',
           r'Model & AUROC & AUROC & $\Delta$AUROC & RNA perm. & $\max|z{-}a|\downarrow$ \\', r'\midrule']
    body = []
    for gi, (gname, models) in enumerate(groups):
        body.append(('rule', gi, gname))
        for m in models:
            im, isd = get(D['int_mean'], m), get(D['int_sd'], m)
            if im is None and m not in ext.index:
                continue
            intc = DASH if im is None else f'{im:.3f} $\\pm$ {isd:.3f}'
            eb = D['e1_boot']
            eb_row = (eb[(eb.cohort == 'external_112') & (eb.scenario == 'intact') & (eb.comparator == m)]
                      if eb is not None else [])
            if m in B.index:
                d = f"[{f3s(B.loc[m, '0.025'])}, {f3s(B.loc[m, '0.975'])}]"
            elif len(eb_row):
                d = f"[{f3s(eb_row.q025.iloc[0])}, {f3s(eb_row.q975.iloc[0])}]"
            else:
                d = DASH
            rp = DASH if m in ('rna', 'mri') else perm(m)
            dz = DASH if m in ('rna', 'mri') else disp(m)
            in_set = gname == 'Fusion baselines' or m == 'carf'
            body.append(['row', LABEL[m], [intc, cell(get(ext.auroc, m)), d, rp, dz], in_set])
    rows = [b for b in body if b[0] == 'row']
    bold_best([[r[1], r[2], r[3]] for r in rows], [], {})
    tmp = [[r[1], r[2], r[3]] for r in rows]
    for r in tmp:  # internal AUROC cell is 'mean $\\pm$ sd': bold the mean part only
        r[1].append(r[1][0].split(' ')[0])
    bold_best(tmp, [5, 1, 3], {5: 'max', 1: 'max', 3: 'max'})
    bold_best(tmp, [4], {4: 'min'})
    for r in tmp:
        mean_cell = r[1].pop()
        if mean_cell.startswith(r'\textbf'):
            r[1][0] = mean_cell + r[1][0][len(mean_cell) - len(r'\textbf{}'):]
    it = iter(tmp)
    for b in body:
        if b[0] == 'rule':
            if b[1]:
                out.append(r'\midrule')
            out.append(rf'\multicolumn{{6}}{{@{{}}l}}{{\emph{{{b[2]}}}}} \\')
        else:
            label, cells, _ = next(it)
            out.append(f'\\hspace{{0.5em}}{label} & ' + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_main.tex').write_text('% AUTO-GENERATED by experiments/make_assets.py\n'
                                      '% Internal: mean +- SD over 25 folds. Delta: CARF minus model, paired bootstrap 95% interval. '
                                      'RNA perm.: external AUROC with RNA output permuted (mean of 30). max|z-a|: largest |fused - clinical logit| over all scenarios.\n'
                                      '\\small\n' + '\n'.join(out) + '\n')


STRESS_ROWS = ['mean', 'stack', 'mask_stack', 'offset_lr', 'robust_offset_lr', 'moddrop_mlp',
               'carf_no_robust', 'carf']
STRESS_LABEL = dict(LABEL)


def tab_stress_full(D):
    sm, a8 = D['stress_mean'], D['a8']
    a8m = a8.set_index(['scenario', 'model']).auroc_mean if a8 is not None else None
    out = [r'\setlength{\tabcolsep}{2.5pt}',
           r'\begin{tabular}{@{}lcccc' + ('cccc' if a8m is not None else '') + r'c@{}}', r'\toprule']
    if a8m is not None:
        out += [r' & \multicolumn{4}{c}{Output-level} & '
                r'\multicolumn{4}{c}{Input-level} & \\',
                r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
                r' & \multicolumn{2}{c}{RNA} & \multicolumn{2}{c}{MRI} & \multicolumn{2}{c}{RNA} '
                r'& \multicolumn{2}{c}{MRI} & max \\',
                r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
                r'Model & perm. & noise & perm. & noise & drop & shift & noise & drop & $|z-a|$ \\']
    else:
        out += [r' & \multicolumn{4}{c}{Output-level} & \\',
                r'\cmidrule(lr){2-5}',
                r' & \multicolumn{2}{c}{RNA} & \multicolumn{2}{c}{MRI} & max \\',
                r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
                r'Model & perm. & noise & perm. & noise & $|z-a|$ \\']
    out.append(r'\midrule')
    for m in STRESS_ROWS:
        if m not in set(D['stress'].model):
            continue
        vals = [cell(get(sm, s, m)) for s in ('rna_permuted', 'rna_noisy', 'mri_permuted', 'mri_noisy')]
        if a8m is not None:
            vals += [cell(get(a8m, s, m)) for s in ('rna_dropout50', STRONG_SHIFT, STRONG_MRI_NOISE, STRONG_MRI_DROP)]
        md = get(D['max_disp'], m)
        vals.append(DASH if md is None else (f'{md:.2f}' if md < 100 else f'{md:.0f}'))
        out.append(f'{STRESS_LABEL[m]} & ' + ' & '.join(vals) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    note = ('% AUTO-GENERATED by experiments/make_assets.py\n'
            '% External (I-SPY1) AUROC, mean over 30 realisations. Input-level from results/a8 '
            f'(RNA 50% gene dropout; {STRONG_SHIFT} = 25% of genes shifted by +2; MRI feature noise SD 1; '
            f'{STRONG_MRI_DROP} = 60% MRI feature dropout).\n'
            '% max |z-a|: largest logit displacement from the clinical anchor over all output-level scenarios.\n')
    (GEN / 'tab_stress_full.tex').write_text(note + '\\small\n' + '\n'.join(out) + '\n')


def tab_stress(D):
    """Compact main-paper stress table: permutations, input-level corruptions, CARF-model delta under RNA perm."""
    sm, a8, eb = D['stress_mean'], D['a8'], D['e1_boot']
    a8m = a8.set_index(['scenario', 'model']).auroc_mean if a8 is not None else None
    in_scen = [('rna_dropout50', 'RNA', 'drop'), (STRONG_SHIFT, 'RNA', 'shift'),
               (STRONG_MRI_NOISE, 'MRI', 'noise'), (STRONG_MRI_DROP, 'MRI', 'drop')] if a8m is not None else []
    n_in = len(in_scen)
    out = [r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}l' + 'cc' + 'c' * n_in + r'c@{}}', r'\toprule']
    hdr1 = r' & \multicolumn{2}{c}{Permuted}'
    rule = r'\cmidrule(lr){2-3}'
    if n_in:
        hdr1 += r' & \multicolumn{4}{c}{Input-level}'
        rule += r'\cmidrule(lr){4-7}'
    last = 4 + n_in
    out += [hdr1 + r' & $\Delta$ RNA perm. \\', rule + rf'\cmidrule(l){{{last}-{last}}}']
    if n_in:
        out += [r' & & & \multicolumn{2}{c}{RNA} & \multicolumn{2}{c}{MRI} & \\',
                r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}']
    out += ['Model & RNA & MRI' + ''.join(f' & {b}' for _, a, b in in_scen) + r' & CARF$-$model \\', r'\midrule']
    order = [m for m in STRESS_ROWS if m not in ('carf', 'carf_no_robust')] + ['carf', 'carf_no_robust']
    rows = []
    for m in order:
        if m not in set(D['stress'].model):
            continue
        vals = [cell(get(sm, s, m)) for s in ('rna_permuted', 'mri_permuted')]
        vals += [cell(get(a8m, s, m)) for s, _, _ in in_scen]
        d = DASH
        if eb is not None and m != 'carf':
            r = eb[(eb.cohort == 'external_112') & (eb.scenario == 'rna_permuted') & (eb.comparator == m)]
            if len(r):
                d = f'[{f3s(r.q025.iloc[0])}, {f3s(r.q975.iloc[0])}]'
        vals.append(d)
        rows.append([m, vals, m != 'carf_no_robust'])
    ncol = 2 + n_in
    bold_best(rows, list(range(ncol)), {c: 'max' for c in range(ncol)})
    nfull = 2 + ncol
    out.append(rf'\multicolumn{{{nfull}}}{{@{{}}l}}{{\emph{{Fusion baselines}}}} \\')
    for m, vals, _ in rows:
        if m == 'carf':
            out.append(r'\midrule')
            out.append(rf'\multicolumn{{{nfull}}}{{@{{}}l}}{{\emph{{CARF and ablation}}}} \\')
        out.append(f'\\hspace{{0.5em}}{STRESS_LABEL[m]} & ' + ' & '.join(vals) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    note = ('% AUTO-GENERATED by experiments/make_assets.py\n'
            '% External (I-SPY1) AUROC, mean over 30 realisations. Input-level (results/a8): RNA 50% gene dropout; '
            f'{STRONG_SHIFT} (25% of genes, +2); MRI feature noise SD 1; 60% MRI feature dropout.\n'
            '% Delta: CARF minus model under RNA permutation, [2.5, 97.5] bootstrap percentiles (results/e1).\n'
            '% Full version with noise columns and max |z-a|: tab_stress_full.tex.\n')
    (GEN / 'tab_stress.tex').write_text(note + '\\small\n' + '\n'.join(out) + '\n')


def tab_design_stress(D):
    sm = D['stress_mean']
    scen = ['intact', 'rna_low', 'rna_high', 'rna_neutral', 'rna_missing',
            'mri_low', 'mri_high', 'mri_neutral', 'mri_missing', 'both_missing']
    out = [r'\setlength{\tabcolsep}{2.7pt}', r'\begin{tabular}{@{}l' + 'c' * len(scen) + r'@{}}', r'\toprule',
           r' & & \multicolumn{4}{c}{RNA} & \multicolumn{4}{c}{MRI} & Both \\',
           r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(l){11-11}',
           r'Model & Intact & 0.01 & 0.99 & 0.5 & miss. & 0.01 & 0.99 & 0.5 & miss. & miss. \\', r'\midrule']
    for m in STRESS_ROWS:
        if m not in set(D['stress'].model):
            continue
        out.append(f'{STRESS_LABEL[m]} & ' + ' & '.join(cell(get(sm, s, m)) for s in scen) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_design_stress.tex').write_text(
        '% AUTO-GENERATED by experiments/make_assets.py\n% External AUROC when one optional expert '
        'output is set to 0.01/0.99/0.5 or marked missing (training-design environments).\n'
        '\\small\n' + '\n'.join(out) + '\n')


def tab_sweep(D):
    sw = D['sweep'].sort_values(['budget', 'robust_weight'])
    out = [r'\setlength{\tabcolsep}{5pt}', r'\begin{tabular}{@{}cccccccc@{}}', r'\toprule',
           r' & & Internal & \multicolumn{3}{c}{External AUROC} & \multicolumn{2}{c}{Budget} \\',
           r'\cmidrule(lr){3-3}\cmidrule(lr){4-6}\cmidrule(l){7-8}',
           r'$B$ & $\gamma$ & AUROC & intact & RNA perm. & MRI perm. & RNA & MRI \\', r'\midrule']
    prev = None
    for _, r in sw.iterrows():
        if prev is not None and r.budget != prev:
            out.append(r'\addlinespace[2pt]')
        prev = r.budget
        star = r'$^\ast$' if (r.budget == 1.0 and r.robust_weight == 0.25) else ''
        out.append(f'{r.budget:g} & {r.robust_weight:g}{star} & {r.internal_auroc:.3f} $\\pm$ {r.internal_sd:.3f} & '
                   f'{r.external_auroc:.3f} & {r.ext_rna_permuted:.3f} & {r.ext_mri_permuted:.3f} & '
                   f'{r.rna_budget:.3f} & {r.mri_budget:.3f} \\\\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_sweep.tex').write_text(
        '% AUTO-GENERATED by experiments/make_assets.py\n% $^\\ast$ = fixed setting used for CARF '
        '(B=1, gamma=0.25). Internal: mean $\\pm$ SD over folds as reported by sweep.csv.\n'
        '\\small\n' + '\n'.join(out) + '\n')


def tab_perm(D):
    a5 = D['a5']
    if a5 is None:
        return
    order = ['clinical', 'rna', 'mri', 'stack', 'mask_stack', 'offset_lr', 'carf_no_robust', 'carf']
    a5 = a5.set_index('model')
    out = [r'\setlength{\tabcolsep}{5pt}', r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
           r' & Observed & \multicolumn{3}{c}{Permutation null} & & \\',
           r'\cmidrule(lr){3-5}',
           r'Model & AUROC & mean & SD & max & $n_{\geq}$ & $p$ \\', r'\midrule']
    for m in [m for m in order if m in a5.index] + [m for m in a5.index if m not in order]:
        r = a5.loc[m]
        lab = STRESS_LABEL.get(m, m)
        out.append(f'{lab} & {r.observed:.3f} & {r.null_mean:.3f} & {r.null_sd:.3f} & {r.null_max:.3f} & '
                   f'{int(r.n_ge)}/{int(r.n_perm)} & {r.p_value:.3f} \\\\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_perm.tex').write_text(
        '% AUTO-GENERATED by experiments/make_assets.py\n% Whole-pipeline label-permutation test (results/a5): '
        'external AUROC vs. null from refitting on permuted I-SPY2 labels; p = (1 + n_ge)/(1 + n_perm).\n'
        '\\small\n' + '\n'.join(out) + '\n')


def tab_natural(D):
    a6 = D['a6']
    if a6 is None:
        return
    a6 = a6.set_index(['subgroup', 'model'])
    groups = [('rna_only', 'RNA only'), ('mri_only', 'MRI only'), ('all_available', 'All available')]
    rows = ['clinical', 'mean', 'stack', 'mask_stack', 'offset_lr_masked', 'carf_no_robust', 'carf']
    lab = dict(STRESS_LABEL)
    hdr, hdr2 = [], []
    for g, name in groups:
        sub = a6.loc[g].iloc[0]
        hdr.append(rf'\multicolumn{{2}}{{c}}{{{name}}}')
        hdr2.append(rf'\multicolumn{{2}}{{c}}{{($n={int(sub.n)}$, {int(sub.pcr_pos)} pCR)}}')
    out = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule', ' & ' + ' & '.join(hdr) + r' \\',
           ' & ' + ' & '.join(hdr2) + r' \\',
           r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}',
           'Model' + ' & AUROC & Brier' * 3 + r' \\', r'\midrule']
    for m in rows:
        vals = []
        for g, _ in groups:
            vals += [cell(get(a6.auroc, g, m)), cell(get(a6.brier, g, m))]
        out.append(f'{lab[m]} & ' + ' & '.join(vals) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_natural.tex').write_text(
        '% AUTO-GENERATED by experiments/make_assets.py\n% Natural missingness on I-SPY1 (results/a6): '
        'frozen models evaluated with true availability masks.\n\\small\n'
        '\\setlength{\\tabcolsep}{4pt}\n' + '\n'.join(out) + '\n')


def tab_anchor(D):
    an = D['e1_anchor']
    if an is None:
        return
    mean = an.set_index(['scenario', 'model'])['mean']
    sd = an.set_index(['scenario', 'model'])['std']
    models = [m for m in ['clinical', 'stack', 'mask_stack', 'offset_lr', 'carf_no_robust', 'carf']
              if m in set(an.model)]
    hdr = {'clinical': 'Clinical', 'rna': 'RNA', 'stack': 'Stacking',
           'mask_stack': r'\shortstack{Mask-\\stacking}', 'offset_lr': r'\shortstack{Offset\\LR}',
           'carf_no_robust': r'\shortstack{No robust\\term}', 'carf': 'CARF'}
    scen_lab = {'intact': 'Intact', 'clinical_output_permuted': 'Clin.\\ output perm.',
                'hr_flip10': r'HR flip 10\%', 'hr_flip25': r'HR flip 25\%',
                'her2_flip10': r'HER2 flip 10\%', 'her2_flip25': r'HER2 flip 25\%',
                'age_noise_sd10': 'Age noise (SD 10 y)'}
    order = [s for s in scen_lab if s in set(an.scenario)] + sorted(set(an.scenario) - set(scen_lab))
    out = [r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}l' + 'c' * len(models) + r'@{}}', r'\toprule',
           'Scenario & ' + ' & '.join(hdr[m] for m in models) + r' \\', r'\midrule']
    for i, sc in enumerate(order):
        if i:
            out.append(r'\addlinespace[1pt]')
        out.append(scen_lab.get(sc, sc.replace('_', r'\_')) + ' & ' +
                   ' & '.join(cell(get(mean, sc, m)) for m in models) + r' \\')
        sds = [get(sd, sc, m) for m in models]
        if any(v is not None and v > 0 for v in sds):
            out.append(' & ' + ' & '.join('' if (v is None or v == 0) else rf'{{\scriptsize$\pm${v:.3f}}}' for v in sds) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_anchor.tex').write_text(
        '% AUTO-GENERATED by experiments/make_assets.py\n% Anchor (clinical) failure on I-SPY1 (results/e1): '
        'external AUROC, mean with SD over realisations in the line below. RNA-only expert omitted (unaffected by anchor corruption).\n\\footnotesize\n' + '\n'.join(out) + '\n')


SYNTH_SCEN = [('intact', None, 0.0, 'Intact'),
              ('silent', 'R', 0.5, r'R silent, $q=0.5$'),
              ('silent', 'R', 1.0, r'R silent, $q=1$'),
              ('flip', 'R', 1.0, r'R sign-flipped, $q=1$'),
              ('missing', 'R', 1.0, r'R missing, $q=1$'),
              ('shift', 'R', 2.0, r'R shift, $\delta=2$'),
              ('cost', 'R', 2.0, r'Cost ($2\times\beta_R$)')]
SYNTH_MODELS = [('clinical', 'Anchor'), ('stack', 'Stack'), ('mask_stack', r'\shortstack{Mask-\\stack}'),
                ('offset_lr', r'\shortstack{Offset\\LR}'), ('moddrop_mlp', r'\shortstack{Mod-\\Drop}'),
                ('carf_no_robust', r'\shortstack{No rob.\\term}'), ('carf', 'CARF'),
                ('carf_B0.5', r'\shortstack{CARF\\$B{=}0.5$}')]


def tab_synth(D):
    e2 = D['e2']
    if e2 is None:
        return
    e2 = e2[e2.setting == 'base']
    out = [r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{@{}l' + 'c' * len(SYNTH_MODELS) + r'@{}}',
           r'\toprule', 'Scenario & ' + ' & '.join(h for _, h in SYNTH_MODELS) + r' \\', r'\midrule']
    for scen, mod, lvl, lab in SYNTH_SCEN:
        r = e2[(e2.scenario == scen) & np.isclose(e2.level, lvl)]
        if mod is not None:
            r = r[r.modality == mod]
        r = r.set_index('model').auroc_mean
        out.append(lab + ' & ' + ' & '.join(cell(r.get(m)) for m, _ in SYNTH_MODELS) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_synth.tex').write_text(
        '% AUTO-GENERATED by experiments/make_assets.py\n% Synthetic study (results/e2, setting=base): test AUROC, '
        'mean over replicates.\n\\small\n' + '\n'.join(out) + '\n')


def copy_e2_figures():
    import shutil
    for f in ('fig_synthetic', 'fig_anchor_sweep'):
        for ext in ('pdf', 'png'):
            src = RES / 'e2' / f'{f}.{ext}'
            if src.exists():
                shutil.copyfile(src, FIG / f'{f}.{ext}')


def tab_calib(D):
    cal = D['e4_cal']
    if cal is None:
        return
    order = ['clinical', 'stack', 'mask_stack', 'offset_lr', 'robust_offset_lr', 'carf_no_robust', 'carf']
    models = [m for m in order if m in set(cal.model)] + sorted(set(cal.model) - set(order))

    def f2s(x):
        s = f'{x:.2f}'
        return rf'$-${s[1:]}' if s.startswith('-') and s != '-0.00' else s.lstrip('-')

    out = [r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
           r' & \multicolumn{2}{c}{Raw} & \multicolumn{4}{c}{Platt-scaled} \\',
           r'\cmidrule(lr){2-3}\cmidrule(l){4-7}',
           r'Model & Brier & ECE & Brier & ECE & Intercept [95\% CI] & Slope [95\% CI] \\', r'\midrule']
    for m in models:
        raw = cal[(cal.model == m) & (cal.variant == 'raw')]
        pl = cal[(cal.model == m) & (cal.variant == 'platt')]
        v = [cell(raw.brier.iloc[0]) if len(raw) else DASH, cell(raw.ece10.iloc[0]) if len(raw) else DASH]
        if len(pl):
            r = pl.iloc[0]
            v += [cell(r.brier), cell(r.ece10),
                  f'{f2s(r.cal_intercept)} [{f2s(r.int_lo)}, {f2s(r.int_hi)}]',
                  f'{f2s(r.cal_slope)} [{f2s(r.slope_lo)}, {f2s(r.slope_hi)}]']
        else:
            v += [DASH] * 4
        out.append(f'{STRESS_LABEL.get(m, m)} & ' + ' & '.join(v) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_calib.tex').write_text(
        '% AUTO-GENERATED by experiments/make_assets.py\n% External (I-SPY1) calibration (results/e4): Brier, '
        'ECE (10 equal-width bins); Platt scaling fitted on inner out-of-fold development predictions; '
        'calibration intercept and slope with 95% CI.\n\\small\n' + '\n'.join(out) + '\n')


# ----------------------------------------------------------------------------- figures
CM = 1 / 2.54
TEXTW = 12.2 * CM


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Roboto', 'Arial', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Roboto', 'mathtext.it': 'Roboto:italic',
        'mathtext.bf': 'Roboto:bold', 'mathtext.fallback': 'cm', 'legend.frameon': False,
    })


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=200)
    plt.close(fig)


def fig_tradeoff(D):
    sw = D['sweep']
    clin = get(D['ext'].auroc, 'clinical')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 5.2 * CM), gridspec_kw={'width_ratios': [1, 1.25]},
                             layout='constrained')
    gcol = {0.0: '#0072B2', 0.25: '#D55E00', 1.0: '#999999'}
    for gm, c in gcol.items():
        s = sw[sw.robust_weight == gm].sort_values('budget')
        a1.plot(s.budget, s.external_auroc, '-o', color=c, lw=1.4, ms=3.5, label=f'$\\gamma={gm:g}$')
        a1.plot(s.budget, s.ext_rna_permuted, '--o', color=c, lw=1.4, ms=3.5, mfc='white')
    a1.axhline(clin, color='#000000', lw=0.8, ls=':', zorder=0)
    a1.set_xscale('log', base=2)
    a1.set_xticks([0.25, 0.5, 1, 2, 4])
    a1.set_xticklabels(['0.25', '0.5', '1', '2', '4'])
    a1.set_xlim(0.2, 5.2)
    a1.set_xlabel('Budget $B$ (log scale)')
    a1.set_ylabel('External AUROC')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=1.4, label=f'$\\gamma={gm:g}$') for gm, c in gcol.items()]
    a1.legend(handles=h, loc='lower left', ncol=1, handlelength=1.8, fontsize=7, borderaxespad=0.1,
              labelspacing=0.3)
    top = max(sw.external_auroc.max(), clin)
    a1.set_ylim(min(sw.ext_rna_permuted.min(), clin) - 0.01, top + 0.025)
    a1.text(0.23, top + 0.012, 'solid: intact', fontsize=7, va='center')
    a1.text(5.1, 0.76, 'dashed:\nRNA permuted', fontsize=7, va='center', ha='right', linespacing=1.0)
    a1.text(5.1, clin + 0.004, 'clinical only', fontsize=7, va='bottom', ha='right')
    a1.set_title('(a) Budget and robustness weight', loc='left', fontsize=8)

    S = D['stress']
    models = [m for m in ['stack', 'mask_stack', 'offset_lr', 'robust_offset_lr', 'moddrop_mlp',
                          'carf_no_robust', 'carf']
              if m in set(S.model)]
    y = np.arange(len(models))[::-1]
    for i, m in enumerate(models):
        for scen, mk, dy in [('rna_permuted', 'o', 0.16), ('mri_permuted', 's', -0.16)]:
            v = S[(S.scenario == scen) & (S.model == m)].auroc - clin
            a2.errorbar(v.mean(), y[i] + dy, xerr=v.std(ddof=1), fmt=mk, color=COLOR[m], ms=4,
                        mfc=COLOR[m] if mk == 'o' else 'white', mew=1.1, elinewidth=1.0, capsize=1.5)
    a2.axvline(0, color='#666666', lw=0.8, ls=':')
    a2.set_yticks(y)
    a2.set_yticklabels([STRESS_LABEL[m] for m in models])
    a2.tick_params(axis='y', length=0)
    a2.spines['left'].set_visible(False)
    a2.set_xlabel('AUROC $-$ clinical-only AUROC')
    a2.set_ylim(-0.6, len(models) - 0.4)
    h = [Line2D([], [], color='#333333', marker='o', ls='', ms=4, label='RNA permuted'),
         Line2D([], [], color='#333333', marker='s', mfc='white', ls='', ms=4, label='MRI permuted')]
    a2.legend(handles=h, loc='upper center', bbox_to_anchor=(0.4, -0.3), fontsize=7, handletextpad=0.2,
              borderaxespad=0, ncol=2, columnspacing=1.0)
    a2.set_title('(b) Permutation vs. clinical-only', loc='left', fontsize=8)
    for lab, m in zip(a2.get_yticklabels(), models):
        lab.set_color('#000000')
    save(fig, 'fig_tradeoff')


def fig_reliability(D):
    P = D['pred']
    y = P.pcr.to_numpy()
    fig, ax = plt.subplots(figsize=(6 * CM, 5.6 * CM), layout='constrained')
    ax.plot([0, 1], [0, 1], color='#999999', lw=0.8, ls=':', zorder=0)
    for m, mk in [('clinical', 'o'), ('stack', 's'), ('carf', 'D'), ('carf_platt', '^')]:
        p = P[m].to_numpy()
        order = np.argsort(p, kind='mergesort')
        bins = np.array_split(order, 5)
        mp = [p[b].mean() for b in bins]
        oy = [y[b].mean() for b in bins]
        ax.plot(mp, oy, '-', marker=mk, color=COLOR[m], lw=1.2, ms=3.8, label=LABEL[m] if m != 'carf_platt'
                else 'CARF + Platt')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed pCR rate')
    ax.legend(loc='upper left', fontsize=7, handlelength=1.6, borderaxespad=0.2)
    save(fig, 'fig_reliability')


def fig_architecture(D):
    # Printed at 0.95\linewidth (11.6 cm): 8 pt here -> 7.6 pt printed; smallest text 7.5 pt -> 7.1 pt.
    W, H = 12.2, 4.5
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    fs, fs2 = 8, 7.5

    def box(x, y, w, h, text='', fc='#FFFFFF', ec='#333333', lw=0.7, fs=fs, ls='-', **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.1',
                                    fc=fc, ec=ec, lw=lw, ls=ls))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, **kw)

    def arrow(p, q, color='#333333', ls='-', lw=0.8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=6.5, color=color, lw=lw,
                                     ls=ls, shrinkA=0, shrinkB=0))

    CLIN, RNA, MRI = '#E8E8E8', '#DCEBF7', '#FBE3D2'
    yC, yR, yM = 3.88, 3.0, 2.12  # row centres
    bh = 0.56
    x1, w1 = 0.03, 1.62         # experts
    x2, w2 = 2.05, 2.45         # anchor / residual
    x3, w3 = 4.8, 0.95          # clip
    x4, w4 = 6.05, 2.15         # weighted sum
    for yy, fc, t in [(yC, CLIN, 'Clinical'), (yR, RNA, 'RNA (opt.)'), (yM, MRI, 'MRI (opt.)')]:
        box(x1, yy - bh / 2, w1, bh, t, fc=fc)
    box(x2, yC - bh / 2, w2, bh, r'$a=\mathrm{logit}\,p_C$', fc=CLIN, lw=1.1)
    ax.text(x2 + w2 / 2, yC + bh / 2 + 0.03, 'anchor (fixed)', ha='center', va='bottom', fontsize=fs2)
    box(x2, yR - bh / 2, w2, bh, r'$d_R=\mathrm{logit}\,p_R-a$', fc=RNA)
    box(x2, yM - bh / 2, w2, bh, r'$d_M=\mathrm{logit}\,p_M-a$', fc=MRI)
    for yy, lab in [(yC, '$p_C$'), (yR, '$p_R$'), (yM, '$p_M$')]:
        arrow((x1 + w1, yy), (x2, yy))
        ax.text(x1 + w1 + 0.2, yy + 0.03, lab, ha='center', va='bottom', fontsize=fs2)
    for y0, y1 in [(yC - bh / 2, yR + bh / 2), (yR - bh / 2, yM + bh / 2)]:
        ax.plot([x2 + w2 - 0.2] * 2, [y0, y1], color='#777', lw=0.6, ls=(0, (2, 1.5)))
    for yy, fc in [(yR, RNA), (yM, MRI)]:
        box(x3, yy - bh / 2, w3, bh, fc=fc)
        ix, iy, sz = x3 + 0.1, yy - 0.17, 0.34
        ax.plot([ix, ix + sz * .3, ix + sz * .7, ix + sz], [iy, iy, iy + sz, iy + sz], color='#333', lw=0.8)
        ax.text(x3 + 0.72, yy, r'$r_j$', ha='center', va='center', fontsize=fs)
        arrow((x2 + w2, yy), (x3, yy))
        arrow((x3 + w3, yy), (x4, yy))
    ax.text(x3 + w3 / 2, yR + bh / 2 + 0.03, r'clip $\pm B$', ha='center', va='bottom', fontsize=fs2)
    y4a, y4b = yM - bh / 2, yR + bh / 2
    box(x4, y4a, w4, y4b - y4a, lw=0.9)
    cx = x4 + w4 / 2
    ax.text(cx, (y4a + y4b) / 2 + 0.05, r'$\sum_{j,k}\theta_{jk}\,m_j\,h_{jk}\,r_j$', ha='center',
            va='center', fontsize=8.5)
    ax.add_patch(Circle((cx, yC), 0.19, fc='white', ec='#333', lw=0.8))
    ax.text(cx, yC - 0.01, '+', ha='center', va='center', fontsize=8.5)
    arrow((x2 + w2, yC), (cx - 0.19, yC))
    arrow((cx, y4b), (cx, yC - 0.19))
    box(cx + 0.5, yC - 0.22, 0.44, 0.44, r'$\sigma$')
    arrow((cx + 0.19, yC), (cx + 0.5, yC))
    ax.text(cx + 0.345, yC + 0.04, '$z$', ha='center', va='bottom', fontsize=fs2)
    arrow((cx + 0.94, yC), (cx + 1.22, yC))
    ax.text(cx + 1.27, yC, r'$\hat p$', ha='left', va='center', fontsize=8.5)
    # context functions and budget, right of the sum box
    tx = x4 + w4 + 0.22
    ax.text(tx, y4b + 0.36, 'Context functions $h_{jk}$', ha='left', va='top', fontsize=fs, fontweight='bold')
    for i, t in enumerate([r'$h_{j0}=1$', r'$h_{j1}=e^{-|d_j|}$', r'$h_{j2}=1-H(p_j)/\log 2$']):
        ax.text(tx + 0.1, y4b + 0.02 - 0.34 * i, t, ha='left', va='top', fontsize=fs)
    ax.text(tx, y4a - 0.1, r'$\theta\geq 0,\ \sum\theta\leq 1\ \Rightarrow\ |z-a|\leq B$', ha='left',
            va='bottom', fontsize=fs)

    # gate-training strip along the bottom
    yg0, yg1 = 0.02, 1.52
    box(0.03, yg0, W - 0.06, yg1 - yg0, fc='#F5F5F5', ec='#888', lw=0.7, ls=(0, (3, 2)))
    gx, gy, dy = 0.18, yg1 - 0.07, 0.355
    ax.text(gx, gy, 'Gate training (convex)', ha='left', va='top', fontsize=fs, fontweight='bold')
    ax.text(W - 0.18, gy, f'fixed: $B={B_CLIP:g}$, $\\gamma=0.25$, $\\tau=0.05$, $\\lambda=0.01$',
            ha='right', va='top', fontsize=fs, color='#444')
    ax.text(gx, gy - dy, 'Inner out-of-fold expert outputs under 4 availability masks (C, C+R, C+M, C+R+M)',
            ha='left', va='top', fontsize=fs)
    ax.text(gx, gy - 2 * dy, 'and 4 corruptions (R or M set to 0.01 or 0.99)', ha='left', va='top', fontsize=fs)
    ax.text(gx, gy - 3 * dy, r'Loss: mean log loss (masks) $+\ \gamma\cdot$smooth-max (corruptions) '
            r'$+\ \lambda\|\theta\|^2$', ha='left', va='top', fontsize=fs)
    arrow((cx, yg1), (cx, y4a), color='#888', ls=(0, (3, 2)))
    ax.text(cx + 0.08, (yg1 + y4a) / 2, r'$\theta$', ha='left', va='center', fontsize=8.5, color='#555')
    save(fig, 'fig_architecture')


# ----------------------------------------------------------------------------- main
def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    D = load()
    M, missing, extra = macros(D)
    tab_main(D, full=True)
    tab_main(D, full=False)
    tab_stress_full(D)
    tab_stress(D)
    tab_design_stress(D)
    tab_sweep(D)
    tab_perm(D)
    tab_natural(D)
    tab_anchor(D)
    tab_synth(D)
    tab_calib(D)
    copy_e2_figures()
    fig_architecture(D)
    fig_tradeoff(D)
    fig_reliability(D)
    print(f'{len(M)} macros written; {len(missing)} missing: {" ".join(missing)}')
    print('a7:', (RES / 'a7' / 'external.csv').exists(), 'a8:', D['a8'] is not None,
          'a6:', D['a6'] is not None, 'a5:', D['a5'] is not None)


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------- supplement: all input-level levels
def tab_input_full():
    d = read(RES / 'a8/input_stress_summary.csv')
    if d is None:
        return
    scen = [('rna_dropout10', 'RNA drop 10\\%'), ('rna_dropout30', 'RNA drop 30\\%'), ('rna_dropout50', 'RNA drop 50\\%'),
            ('rna_block10_shift+1', 'RNA shift 10\\%/+1'), ('rna_block10_shift+2', 'RNA shift 10\\%/+2'),
            ('rna_block25_shift+1', 'RNA shift 25\\%/+1'), ('rna_block25_shift+2', 'RNA shift 25\\%/+2'),
            ('mri_noise_sd0.5', 'MRI noise 0.5'), ('mri_noise_sd1', 'MRI noise 1.0'),
            ('mri_dropout30', 'MRI drop 30\\%'), ('mri_dropout60', 'MRI drop 60\\%')]
    models = [('clinical', 'Clin.'), ('stack', 'Stack'), ('mask_stack', 'Mask'),
              ('offset_lr', 'Offset'), ('carf_no_robust', 'No rob.'), ('carf', 'CARF')]
    piv = d.pivot(index='scenario', columns='model', values='auroc_mean')
    out = [r'\begin{tabular}{@{}l' + 'c' * len(models) + '@{}}', r'\toprule',
           'Scenario & ' + ' & '.join(l for _, l in models) + r' \\', r'\midrule']
    for s, lab in scen:
        out.append(lab + ' & ' + ' & '.join(f'{piv.loc[s, m]:.3f}' for m, _ in models) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_input_full.tex').write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    tab_input_full()


def tab_mosi():
    c, s = read(RES / 'e3/clean.csv'), read(RES / 'e3/stress.csv')
    if c is None or s is None:
        return
    c = c.set_index('model')
    sm = s.groupby(['scenario', 'model']).auroc.mean().unstack()
    rows = [('text', 'Text (anchor)'), ('mean', 'Mean'), ('stack', 'Stacking'),
            ('mask_stack', 'Mask-stacking'), ('offset_lr_masked', 'Offset LR (masks)'),
            ('moddrop_mlp', 'ModDrop MLP'), ('carf_no_robust', 'CARF (no robust)'), ('carf', 'CARF')]
    out = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
           r' & \multicolumn{3}{c}{Intact} & \multicolumn{2}{c}{Permuted} & Worst \\',
           r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}',
           r'Model & AUROC & Acc. & ECE & audio & visual & case \\', r'\midrule']
    for m, lab in rows:
        worst = sm[m].drop('intact').min()
        out.append(f"{lab} & {c.loc[m, 'auroc']:.3f} & {c.loc[m, 'accuracy']:.3f} & {c.loc[m, 'ece']:.3f} & "
                   f"{sm.loc['audio_permuted', m]:.3f} & {sm.loc['visual_permuted', m]:.3f} & {worst:.3f} \\\\")
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_mosi.tex').write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    tab_mosi()


def robust_g1_macros():
    b = read(RES / 'e5/robust_offset_bootstrap.csv'); st = read(RES / 'e5/robust_offset_stress.csv')
    if b is None or st is None:
        return
    r = b[(b.scenario == 'rna_permuted') & (b.comparator == 'robust_offset_lr_g1')].iloc[0]
    g = st[st.model == 'robust_offset_lr_g1'].set_index('scenario')
    M = {'RnaPermRobOffsetGOne': f"{g.loc['rna_permuted', 'auroc_mean']:.3f}",
         'ExtRobOffsetGOne': f"{g.loc['intact', 'auroc_mean']:.3f}",
         'DPermRobOffsetGOneLo': f"{r.q025:.3f}".replace('-', '\\ensuremath{-}'),
         'DPermRobOffsetGOneHi': f"{r.q975:.3f}"}
    lines = [f'\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}' for k, v in M.items()]
    with open(GEN / 'macros.tex', 'a') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    robust_g1_macros()
