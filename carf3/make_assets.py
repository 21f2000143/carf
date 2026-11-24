#!/usr/bin/env python3
"""Tables, macros, and figures for latex2/ from carf2/results (no hand-typed numbers)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
TEX = HERE.parent / 'latex2'
GEN, FIG = TEX / 'generated', TEX / 'figures'
SEED = 20260923
CM = 1 / 2.54

NAMES = {'clinical': 'Clinical', 'rna': 'RNA', 'mri': 'MRI', 'rppa': 'RPPA',
         'mean': 'Mean', 'stack': 'Stacking', 'mask_stack': 'Mask-stacking', 'offset_lr': 'Offset LR',
         'robust_offset_lr': 'Robust offset LR', 'moddrop_mlp': 'ModDrop MLP',
         'carf_v1': 'Base CARF', 'carf2': 'CARF',
         'carf2_no_gate': 'no anchor gate', 'carf2_no_robust': 'no robust term ($\\gamma=0$)',
         'carf2_no_swap': 'no swap environments', 'carf2_B1': 'fixed $B=1$', 'carf2_const': 'constant basis only',
         'carf2_unbounded': 'unbounded (no clip, no budget)'}
BASE = ['mean', 'stack', 'mask_stack', 'offset_lr', 'robust_offset_lr', 'moddrop_mlp']
GATED = ['gated_' + m for m in BASE]
NAMES.update({'gated_' + m: 'Gated ' + (NAMES[m] if m in ('mean',) else NAMES[m][0].lower() + NAMES[m][1:])
              for m in BASE})
NAMES.update({'gated_offset_lr': 'Gated offset LR', 'gated_robust_offset_lr': 'Gated robust offset LR',
              'gated_moddrop_mlp': 'Gated ModDrop MLP', 'gated_mean': 'Gated mean'})
FUSION = BASE + GATED
MIX = ['offset', 'batchswap', 'subtypeswap']
ABL = ['carf2_no_gate', 'carf2_no_robust', 'carf2_no_swap', 'carf2_B1', 'carf2_const', 'carf2_unbounded']
INT_FAIL = ['rna_missing', 'rna_swap', 'mri_missing', 'mri_swap', 'rppa_missing', 'rppa_swap']
EXT_FAIL = ['rna_missing', 'rna_swap', 'mri_missing', 'mri_swap']
MACROS = {}


def mac(name, value, fmt='{:.3f}'):
    v = fmt.format(value) if isinstance(value, (float, np.floating, int, np.integer)) and fmt else str(value)
    if isinstance(value, (float, np.floating)) and value < 0 and fmt:
        v = f'\\ensuremath{{{v}}}'
    MACROS[name] = v
    return v


def f3(x):
    return f'{x:.3f}' if x >= 0 else f'\\ensuremath{{{x:.3f}}}'


DIG = dict(zip('0123456789', ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']))


def camel(s):
    s = ''.join(DIG.get(ch, ch) if ch.isdigit() else ch for ch in s.replace('.', '_'))
    return ''.join(w[:1].upper() + w[1:] for w in s.split('_'))


# ------------------------------------------------------------------ loading
def load():
    I = pd.read_csv(RES / 'main/internal.csv')
    X = pd.read_csv(RES / 'main/external.csv')
    P = pd.read_csv(RES / 'main/predictions.csv.gz', dtype={'patient_id': str})
    G = json.loads((RES / 'main/gates.json').read_text())
    return I, X, P, G


def fold_table(I, design, scen_list):
    d = I[(I.design == design) & I.scenario.isin(['intact'] + scen_list)]
    f = d.groupby(['split', 'model', 'scenario']).auroc.mean().unstack()
    f['fail'] = f[scen_list].mean(1)
    f['worst'] = f[['intact'] + scen_list].min(1)
    return f


def repeat_sd(series):
    """SD of the 5 repeat means (folds 5r..5r+4)."""
    s = series.copy(); s.index = [int(i.split('_')[1]) // 5 for i in s.index]
    return s.groupby(level=0).mean().std(ddof=1)


def corrected_p(diff, ratio=0.25):
    k = len(diff); t = diff.mean() / np.sqrt((1 / k + ratio) * diff.var(ddof=1))
    return 2 * stats.t.sf(abs(t), k - 1)


def corrected_ci(x, ratio=0.25):
    """Mean and 95% interval over the outer folds with the corrected resampled variance."""
    x = np.asarray(x, float); k = len(x)
    h = stats.t.ppf(.975, k - 1) * np.sqrt((1 / k + ratio) * x.var(ddof=1))
    return x.mean() - h, x.mean() + h


def fast_auc(pred, y):
    """AUROC along the last axis (average ranks for ties)."""
    r = stats.rankdata(pred, axis=-1)
    n1 = y.sum(); n0 = len(y) - n1
    return ((r[..., y == 1]).sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def ext_boot(models, n=2000):
    """Patient bootstrap of the external failure average and worst case (realizations held fixed)."""
    z = np.load(RES / 'main/external_scenarios.npz')
    pred, y, names, scen = z['pred'], z['y'].astype(int), list(z['models']), z['scenario']
    mi = [names.index(m) for m in models]
    keep = np.isin(scen, ['intact'] + EXT_FAIL)
    pred, scen = pred[keep][:, mi], scen[keep]
    groups = ['intact'] + EXT_FAIL
    gidx = [np.where(scen == g)[0] for g in groups]
    rng = np.random.default_rng(SEED); fail, worst = [], []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        a = fast_auc(pred[:, :, ix], y[ix])                      # scenario-rep x model
        g = np.stack([a[i].mean(0) for i in gidx])               # scenario x model
        fail.append(g[1:].mean(0)); worst.append(g.min(0))
    return {'fail': np.array(fail), 'worst': np.array(worst), 'models': models}


def ext_table(X, design, scen_list):
    d = X[(X.design == design) & X.scenario.isin(['intact'] + scen_list)]
    t = d.groupby(['model', 'scenario']).auroc.mean().unstack()
    t['fail'] = t[scen_list].mean(1)
    t['worst'] = t[['intact'] + scen_list].min(1)
    return t


def bootstrap(P, design, comps, n=2000):
    e = P[(P.design == design) & (P.split == 'external')]
    y = e.pcr.to_numpy(); rng = np.random.default_rng(SEED); out = {c: [] for c in comps}
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        base = roc_auc_score(y[ix], e.carf2.to_numpy()[ix])
        for c in comps:
            out[c].append(base - roc_auc_score(y[ix], e[c].to_numpy()[ix]))
    return {c: np.quantile(v, [.025, .975]) for c, v in out.items()}


def bold_best(vals, higher=True, nd=3):
    ok = {k: v for k, v in vals.items() if np.isfinite(v)}
    best = max(ok.values()) if higher else min(ok.values())
    return {k: (f'\\textbf{{{v:.{nd}f}}}' if np.isfinite(v) and round(v, nd) == round(best, nd) else (f'{v:.{nd}f}' if np.isfinite(v) else '--'))
            for k, v in vals.items()}


# ------------------------------------------------------------------ main table
def tab_main(I, X, P, G, extra):
    fi = fold_table(I, 'carf2', INT_FAIL)
    xe = ext_table(X, 'carf2', EXT_FAIL)
    disp = X[(X.design == 'carf2')].groupby('model').max_disp.max()
    models = FUSION + ['carf2']
    ci = bootstrap(P, 'carf2', ['clinical'] + FUSION)
    inp = extra.get('input')
    cols = {}
    cols['int'] = {m: fi['intact'].unstack()[m].mean() for m in models}
    cols['intfail'] = {m: fi['fail'].unstack()[m].mean() for m in models}
    cols['intworst'] = {m: fi['worst'].unstack()[m].mean() for m in models}
    cols['ext'] = {m: xe.loc[m, 'intact'] for m in models}
    cols['extfail'] = {m: xe.loc[m, 'fail'] for m in models}
    cols['extworst'] = {m: xe.loc[m, 'worst'] for m in models}
    B = {k: bold_best(v) for k, v in cols.items()}
    dispB = bold_best({m: disp[m] for m in models}, higher=False, nd=2)
    lines = [r'\begin{tabular}{@{}lccccccc@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{Internal (I-SPY2, 25 folds)} & \multicolumn{4}{c}{External (I-SPY1, $n=\NExt$)} \\',
             r'\cmidrule(lr){2-4}\cmidrule(l){5-8}',
             r'Model & Intact & Fail.\ avg & Worst & Intact & Fail.\ avg & Worst & $\max|z{-}a|\downarrow$ \\', r'\midrule',
             r'\multicolumn{8}{@{}l}{\emph{Single experts}} \\']
    ints = fi['intact'].unstack()
    for m in ['clinical', 'rna', 'mri']:
        lines.append(f'\\hspace{{0.5em}}{NAMES[m]} & {ints[m].mean():.3f} & -- & -- & {xe.loc[m, "intact"]:.3f} & -- & -- & -- \\\\')
    lines.append(f'\\hspace{{0.5em}}RPPA & {extra["rppa_int"]:.3f} & -- & -- & n/a & -- & -- & -- \\\\')
    lines += [r'\midrule', r'\multicolumn{8}{@{}l}{\emph{Fusion baselines}} \\']
    eb = ext_boot(models + ['carf2_no_gate'])
    k_c = eb['models'].index('carf2')
    CI = {'intfail': corrected_ci(fi['fail'].unstack()['carf2']), 'intworst': corrected_ci(fi['worst'].unstack()['carf2']),
          'extfail': np.quantile(eb['fail'][:, k_c], [.025, .975]), 'extworst': np.quantile(eb['worst'][:, k_c], [.025, .975])}
    for m in models:
        if m == 'gated_mean':
            lines += [r'\midrule', r'\multicolumn{8}{@{}l}{\emph{Fusion baselines with the verification gate}} \\']
        if m == 'carf2':
            lines += [r'\midrule']
        dstr = dispB[m] if m in disp else '--'
        lines.append(f'\\hspace{{0.5em}}{NAMES[m]} & {B["int"][m]} & {B["intfail"][m]} & {B["intworst"][m]} & '
                     f'{B["ext"][m]} & {B["extfail"][m]} & {B["extworst"][m]} & {dstr} \\\\')
    cis = {k: f'{{\\scriptsize[{v[0]:.3f}, {v[1]:.3f}]}}' for k, v in CI.items()}
    lines.append(f'\\hspace{{0.5em}}\\quad 95\\% CI & & {cis["intfail"]} & {cis["intworst"]} & & {cis["extfail"]} & {cis["extworst"]} & \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    for k, (lo, hi) in CI.items():
        nm = {'intfail': 'IntFail', 'intworst': 'IntWorst', 'extfail': 'ExtFail', 'extworst': 'ExtWorst'}[k]
        mac(f'{nm}CarfTwoLo', lo); mac(f'{nm}CarfTwoHi', hi)
    # paired differences CARF minus model for the failure average and the worst case
    ci_rows = []
    for m in FUSION + ['carf2_no_gate']:
        r = {'model': m}
        for col in ['fail', 'worst']:
            t = fi[col].unstack(); dif = (t['carf2'] - t[m]).to_numpy()
            lo, hi = corrected_ci(dif); r[f'int_{col}'] = (dif.mean(), lo, hi)
            mac(f'DInt{camel(col)}{camel(m)}Lo', lo); mac(f'DInt{camel(col)}{camel(m)}Hi', hi)
            de = eb[col][:, k_c] - eb[col][:, eb['models'].index(m)]
            pt = xe.loc['carf2', col] - xe.loc[m, col]
            lo, hi = np.quantile(de, [.025, .975]); r[f'ext_{col}'] = (pt, lo, hi)
            mac(f'DExt{camel(col)}{camel(m)}', pt); mac(f'DExt{camel(col)}{camel(m)}Lo', lo); mac(f'DExt{camel(col)}{camel(m)}Hi', hi)
        ci_rows.append(r)
    fmt = lambda v: '--' if v is None else f'{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]'.replace('-', '$-$')
    _write('tab_supp_ci.tex', r' & \multicolumn{2}{c}{Internal (corrected $t$)} & \multicolumn{2}{c}{External (bootstrap)} \\'
           r' \cmidrule(lr){2-3}\cmidrule(l){4-5} Comparator & Fail.\ avg & Worst & Fail.\ avg & Worst',
           [f'{NAMES[r["model"]].replace(" (ours)", "")} & ' + ' & '.join(fmt(r.get(c)) for c in ['int_fail', 'int_worst', 'ext_fail', 'ext_worst']) + r' \\'
            for r in ci_rows], 'lcccc')
    (GEN / 'tab_main.tex').write_text('% AUTO-GENERATED by carf2/make_assets.py\n' + '\n'.join(lines) + '\n')
    # macros
    for m in models + ['clinical', 'rna', 'mri']:
        k = camel(m)
        mac(f'Int{k}', ints[m].mean()); mac(f'Int{k}Sd', repeat_sd(ints[m]))
        mac(f'Ext{k}', xe.loc[m, 'intact'])
        if m in models:
            mac(f'IntFail{k}', cols['intfail'][m]); mac(f'IntWorst{k}', cols['intworst'][m])
            mac(f'ExtFail{k}', cols['extfail'][m]); mac(f'ExtWorst{k}', cols['extworst'][m])
            mac(f'MaxDisp{k}', disp[m], '{:.2f}')
            for sc in INT_FAIL:
                mac(f'Int{camel(sc)}{k}', fi[sc].unstack()[m].mean())
            for sc in EXT_FAIL + ['rna_noisy', 'mri_noisy', 'rna_high', 'rna_low']:
                if sc in xe.columns or sc in ('rna_noisy', 'mri_noisy', 'rna_high', 'rna_low'):
                    v = X[(X.design == 'carf2') & (X.model == m) & (X.scenario == sc)].auroc.mean()
                    mac(f'Ext{camel(sc)}{k}', v)
    mac('RppaInt', extra['rppa_int'])
    for c, (lo, hi) in ci.items():
        k = camel(c); mac(f'DExt{k}Lo', lo); mac(f'DExt{k}Hi', hi)
    # calibration (intact): Brier and log loss
    for m in models + ['clinical']:
        k = camel(m)
        xi = I[(I.design == 'carf2') & (I.scenario == 'intact') & (I.model == m)]
        xx = X[(X.design == 'carf2') & (X.scenario == 'intact') & (X.model == m)]
        mac(f'IntBrier{k}', xi.brier.mean()); mac(f'IntLogLoss{k}', xi.log_loss.mean())
        mac(f'ExtBrier{k}', xx.brier.mean()); mac(f'ExtLogLoss{k}', xx.log_loss.mean())
    # internal paired tests
    rows = []
    for col in ['intact', 'fail', 'worst', 'rna_swap', 'rppa_swap']:
        t = fi[col].unstack()
        for m in FUSION:
            diff = (t['carf2'] - t[m]).to_numpy()
            rows.append({'metric': col, 'model': m, 'delta': diff.mean(), 'p': corrected_p(diff),
                         'frac': (diff > 0).mean()})
            k = camel(m); c = camel(col)
            pv = corrected_p(diff)
            mac(f'DInt{c}{k}', diff.mean()); MACROS[f'PInt{c}{k}'] = '<0.001' if pv < 1e-3 else f'={pv:.3f}'
            mac(f'FracInt{c}{k}', 100 * (diff > 0).mean(), '{:.0f}')
    pd.DataFrame(rows).to_csv(RES / 'internal_paired.csv', index=False)
    # budgets
    bud = [v['carf2']['budget'] for k, v in G.items() if k.startswith('carf2/fold')]
    vals, cnt = np.unique(bud, return_counts=True)
    mac('BudgetExt', G['carf2/external']['carf2']['budget'], '{:g}')
    mac('BudgetModeCount', int(cnt.max()), '{:d}'); mac('BudgetMode', vals[cnt.argmax()], '{:g}')
    mac('BudgetList', ', '.join(f'{v:g}' for v in vals), None)
    MACROS['BudgetList'] = ', '.join(f'{v:g}~({c})' for v, c in zip(vals, cnt))
    mac('AllMissingDispCarf', X[(X.design == 'carf2') & (X.model == 'carf2') & (X.scenario == 'all_missing')].max_disp.max(), '{:.1e}')
    return fi, xe


# ------------------------------------------------------------------ stress table
def tab_stress(I, X, extra):
    fi = fold_table(I, 'carf2', INT_FAIL)
    xe = ext_table(X, 'carf2', EXT_FAIL)
    inp = extra.get('input')
    models = BASE + ['carf2']
    cols = [('int', 'rna_swap'), ('int', 'rppa_swap'), ('int', 'mri_swap'), ('int', 'rna_missing'),
            ('ext', 'rna_swap'), ('ext', 'mri_swap')]
    icols = ['rna_dropout50', 'rna_block_shift25', 'mri_noise_sd1', 'mri_dropout60']
    V = {}
    for src, sc in cols:
        V[(src, sc)] = {m: (fi[sc].unstack()[m].mean() if src == 'int' else xe.loc[m, sc]) for m in models}
    if inp is not None:
        s = inp[inp.design == 'carf2'].groupby(['scenario', 'model']).auroc.mean()
        for sc in icols:
            V[('inp', sc)] = {m: s.loc[(sc, m)] for m in models}
            for m in models + ['clinical']:
                mac(f'Inp{camel(sc)}{camel(m)}', s.loc[(sc, m)])
    keys = list(V.keys())
    Bd = {k: bold_best(V[k]) for k in keys}
    lines = [r'\begin{tabular}{@{}l' + 'c' * len(keys) + '@{}}', r'\toprule',
             r' & \multicolumn{4}{c}{Internal (output level)} & \multicolumn{2}{c}{External (output)} & \multicolumn{4}{c}{External (input level)} \\',
             r'\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(l){8-11}',
             r'Model & RNA sw. & RPPA sw. & MRI sw. & RNA miss. & RNA sw. & MRI sw. & RNA drop & RNA shift & MRI noise & MRI drop \\',
             r'\midrule']
    ci = fi['intact'].unstack()['clinical'].mean()
    lines.append(f'\\hspace{{0.5em}}Clinical & {ci:.3f} & {ci:.3f} & {ci:.3f} & {ci:.3f} & ' +
                 ' & '.join([f'{xe.loc["clinical", "intact"]:.3f}'] * (len(keys) - 4)) + r' \\')
    lines.append(r'\midrule')
    for m in models:
        if m == 'carf2':
            lines.append(r'\midrule')
        lines.append(f'\\hspace{{0.5em}}{NAMES[m]} & ' + ' & '.join(Bd[k][m] for k in keys) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_stress.tex').write_text('% AUTO-GENERATED by carf2/make_assets.py\n' + '\n'.join(lines) + '\n')
    mac('IntClinical', ci)
    unc = ['mean', 'stack', 'mask_stack', 'moddrop_mlp']
    loss = max(max(fi['intact'].unstack()[m].mean() - fi['worst'].unstack()[m].mean(), xe.loc[m, 'intact'] - xe.loc[m, 'worst'])
               for m in unc)
    mac('MaxLossUnconstrained', loss, '{:.2f}')


# ------------------------------------------------------------------ semi-realistic mix-ups
def tab_mixup(I, X):
    """AUROC under semi-realistic specimen mix-ups (mean over realizations and, internally, folds)."""
    models = FUSION + ['carf2_no_gate', 'carf2']
    ii = I[I.design == 'carf2']; xx = X[X.design == 'carf2']
    ti = ii.groupby(['split', 'model', 'scenario']).auroc.mean().groupby(['model', 'scenario']).mean().unstack()
    tx = xx.groupby(['model', 'scenario']).auroc.mean().unstack()
    keys = [('int', f'rna_{k}') for k in MIX] + [('int', f'rppa_{k}') for k in MIX] + [('ext', f'rna_{k}') for k in MIX]
    V = {k: {m: (ti if k[0] == 'int' else tx).loc[m, k[1]] for m in models} for k in keys}
    Bd = {k: bold_best(V[k]) for k in keys}
    lab = {'offset': 'Offset', 'batchswap': 'Batch', 'subtypeswap': 'Subtype'}
    lines = [r'\begin{tabular}{@{}l' + 'c' * len(keys) + '@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{Internal, RNA} & \multicolumn{3}{c}{Internal, RPPA} & \multicolumn{3}{c}{External, RNA} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}',
             'Model & ' + ' & '.join(lab[k[1].split('_', 1)[1]] for k in keys) + r' \\', r'\midrule']
    ci, cx = ti.loc['clinical', 'intact'], tx.loc['clinical', 'intact']
    lines.append('\\hspace{0.5em}Clinical & ' + ' & '.join(f'{ci if k[0] == "int" else cx:.3f}' for k in keys) + r' \\')
    lines.append(r'\midrule')
    for m in models:
        if m in ('gated_mean', 'carf2_no_gate'):
            lines.append(r'\midrule')
        nm = 'CARF without gate' if m == 'carf2_no_gate' else NAMES[m]
        lines.append(f'\\hspace{{0.5em}}{nm} & ' + ' & '.join(Bd[k][m] for k in keys) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_mixup.tex').write_text('% AUTO-GENERATED by carf2/make_assets.py\n' + '\n'.join(lines) + '\n')
    for (src, sc), vals in V.items():
        for m, v in vals.items():
            mac(f'Mix{"Int" if src == "int" else "Ext"}{camel(sc)}{camel(m)}', v)
    # internal AUROC lost by any fusion model under offsets and batch swaps, relative to its own intact AUROC
    loss = max(ti.loc[m, 'intact'] - ti.loc[m, f'{mod}_{k}'] for m in models for mod in ('rna', 'rppa')
               for k in ('offset', 'batchswap'))
    mac('MixMaxLossInt', loss, '{:.2f}')
    spread = max(max(V[k].values()) - min(V[k].values()) for k in keys if k[0] == 'int' and 'subtype' not in k[1])
    mac('MixSpreadInt', spread, '{:.2f}')
    # accuracy change of the baselines on intact inputs when they receive the gate
    mac('GateIntactDropInt', max(ti.loc[m, 'intact'] - ti.loc['gated_' + m, 'intact'] for m in BASE), '{:.3f}')
    mac('GateIntactDropExt', max(tx.loc[m, 'intact'] - tx.loc['gated_' + m, 'intact'] for m in BASE), '{:.3f}')
    # how often the best gated baseline beats CARF across the nine mix-up columns
    best_g = {k: max(V[k][m] for m in GATED) for k in keys}
    mac('MixNGatedBetter', int(sum(best_g[k] > V[k]['carf2'] for k in keys)), '{:d}')
    mac('MixNCols', len(keys), '{:d}')
    for k in keys:
        b = max(GATED, key=lambda m: V[k][m])
        MACROS[f'MixBestGated{"Int" if k[0] == "int" else "Ext"}{camel(k[1])}'] = NAMES[b]


# ------------------------------------------------------------------ design ablation
def tab_design(I, X):
    rows = [('v1', 'Age/HR/HER2 anchor; RNA, MRI'), ('treat', '+ treatment-aware anchor'),
            ('rppa', '+ RPPA modality'), ('carf2', '+ both (final design)')]
    models = ['mean', 'stack', 'mask_stack', 'carf2']
    lines = [r'\begin{tabular}{@{}l' + 'c' * (len(models) + 2) + '@{}}', r'\toprule',
             r' & \multicolumn{4}{c}{Internal AUROC, intact} & \multicolumn{2}{c}{CARF, internal} \\',
             r'\cmidrule(lr){2-5}\cmidrule(l){6-7}',
             'Design & ' + ' & '.join(NAMES[m].replace(' (ours)', '') for m in models) + r' & RNA sw. & Fail.\ avg \\', r'\midrule']
    for d, label in rows:
        scen = INT_FAIL if d in ('rppa', 'carf2') else [s for s in INT_FAIL if not s.startswith('rppa')]
        f = fold_table(I, d, scen)
        t = f['intact'].unstack()
        vals = {m: t[m].mean() for m in models if m in t}
        b = bold_best(vals)
        lines.append(f'{label} & ' + ' & '.join(b.get(m, '--') for m in models) +
                     f' & {f["rna_swap"].unstack()["carf2"].mean():.3f} & {f["fail"].unstack()["carf2"].mean():.3f} \\\\')
        for m in models:
            if m in t:
                mac(f'Des{camel(d)}{camel(m)}', t[m].mean())
        mac(f'Des{camel(d)}RnaSwapCarfTwo', f['rna_swap'].unstack()['carf2'].mean())
        mac(f'Des{camel(d)}FailCarfTwo', f['fail'].unstack()['carf2'].mean())
        x = X[(X.design == d) & (X.scenario == 'intact')].set_index('model').auroc
        for m in models:
            if m in x:
                mac(f'DesExt{camel(d)}{camel(m)}', x[m])
    T = {d: fold_table(I, d, INT_FAIL if d in ('rppa', 'carf2') else [s for s in INT_FAIL if not s.startswith('rppa')])['intact'].unstack()
         for d, _ in rows}
    g = lambda d, m: T[d][m].mean()
    mac('DesGapVOne', abs(g('v1', 'carf2') - g('v1', 'stack')))
    mac('DesGainRppaCarfTwo', g('rppa', 'carf2') - g('v1', 'carf2'))
    mac('DesGainRppaStack', g('rppa', 'stack') - g('v1', 'stack'))
    mac('DesGapFinal', abs(g('carf2', 'carf2') - g('carf2', 'stack')))
    mac('DesVOneClinical', g('v1', 'clinical'))
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_design.tex').write_text('% AUTO-GENERATED by carf2/make_assets.py\n' + '\n'.join(lines) + '\n')


# ------------------------------------------------------------------ component ablation
def tab_ablation(I, X, G):
    fi = fold_table(I, 'carf2', INT_FAIL)
    xe = ext_table(X, 'carf2', EXT_FAIL)
    disp = X[(X.design == 'carf2')].groupby('model').max_disp.max()
    models = ['carf2'] + ABL
    lines = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{Internal} & \multicolumn{3}{c}{External} \\', r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
             r'Variant & Intact & RNA sw. & Fail.\ avg & Intact & RNA sw. & $\max|z{-}a|$ \\', r'\midrule']
    for m in models:
        name = 'CARF (full)' if m == 'carf2' else '\\hspace{0.5em}' + NAMES[m]
        lines.append(f'{name} & {fi["intact"].unstack()[m].mean():.3f} & {fi["rna_swap"].unstack()[m].mean():.3f} & '
                     f'{fi["fail"].unstack()[m].mean():.3f} & {xe.loc[m, "intact"]:.3f} & {xe.loc[m, "rna_swap"]:.3f} & '
                     f'{disp[m]:.2f} \\\\')
        k = camel(m)
        mac(f'Int{k}', fi['intact'].unstack()[m].mean()); mac(f'IntRnaSwap{k}', fi['rna_swap'].unstack()[m].mean())
        mac(f'IntFail{k}', fi['fail'].unstack()[m].mean()); mac(f'Ext{k}', xe.loc[m, 'intact'])
        mac(f'ExtRnaSwap{k}', xe.loc[m, 'rna_swap']); mac(f'MaxDisp{k}', disp[m], '{:.2f}')
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_ablation.tex').write_text('% AUTO-GENERATED by carf2/make_assets.py\n' + '\n'.join(lines) + '\n')


# ------------------------------------------------------------------ extras
def extras():
    out = {}
    p = RES / 'extra/input_stress.csv'
    if p.exists():
        out['input'] = pd.read_csv(p)
    p = RES / 'extra/natural.csv'
    if p.exists():
        n = pd.read_csv(p); out['natural'] = n
        for sub in ['all', 'partial', 'rna_only', 'mri_only']:
            s = n[(n.design == 'carf2') & (n.subset == sub)].set_index('model')
            mac(f'Nat{camel(sub)}N', int(s.n.iloc[0]), '{:d}'); mac(f'Nat{camel(sub)}Pos', int(s.pos.iloc[0]), '{:d}')
            for m in ['clinical'] + FUSION + ['carf2']:
                if m in s.index:
                    mac(f'Nat{camel(sub)}{camel(m)}', s.loc[m, 'auroc'])
    p = RES / 'extra/gse25066.csv'
    if p.exists():
        g = pd.read_csv(p); out['gse'] = g
        s = g[g.design == 'carf2'].groupby(['scenario', 'model']).agg(auroc=('auroc', 'mean'), disp=('max_disp', 'max'))
        mac('GseN', int(g.n.iloc[0]), '{:d}'); mac('GsePos', int(g.pos.iloc[0]), '{:d}')
        for sc in ['intact', 'rna_swap', 'rna_dropout50']:
            for m in ['clinical'] + FUSION + ['carf2']:
                if (sc, m) in s.index:
                    mac(f'Gse{camel(sc)}{camel(m)}', s.loc[(sc, m), 'auroc'])
        for m in FUSION + ['carf2']:
            if m in s.index.get_level_values('model'):
                mac(f'GseDisp{camel(m)}', s.xs(m, level='model').disp.max(), '{:.2f}')
    p = RES / 'extra/permutation.csv'
    if p.exists():
        pr = pd.read_csv(p); out['perm'] = pr
        obs = MACROS.get('ExtCarfTwo')
        mac('NPerm', len(pr), '{:d}')
        mac('PermMaxCarf', pr.carf2.max()); mac('PermMeanCarf', pr.carf2.mean())
        o = float(obs) if obs else np.nan
        MACROS['PermPCarf'] = f'{(1 + (pr.carf2 >= o).sum()) / (1 + len(pr)):.3f}'
    return out


def rppa_internal():
    """RPPA expert AUROC among available patients, mean over the 25 outer folds (from the caches)."""
    from pipeline import Design
    import sys
    sys.path.insert(0, str(HERE))
    from run_main import DESIGNS
    from sklearn.model_selection import RepeatedStratifiedKFold
    from pipeline import load_data
    dev, y, ext, ye, M, Me = load_data()
    d = DESIGNS['carf2']
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(np.zeros(len(y)), y))
    aucs = []
    for i, (tr, te) in enumerate(folds):
        with np.load(RES / 'cache' / d.key() / f'fold_{i}.npz') as z:
            q = z['Q'][:, 2]
        ok = np.isfinite(q); aucs.append(roc_auc_score(y[te][ok], q[ok]))
    mac('NRppa', int(dev['rppa'].notna().any(axis=1).sum()), '{:d}')
    mac('NDev', len(y), '{:d}'); mac('NDevPos', int(y.sum()), '{:d}'); mac('NExt', len(ye), '{:d}'); mac('NExtPos', int(ye.sum()), '{:d}')
    mac('NGenes', f'{dev["rna"].shape[1]:,}'.replace(',', '{,}'), None)
    mac('NMriFeat', dev['mri'].shape[1], '{:d}'); mac('NRppaFeat', dev['rppa'].shape[1], '{:d}')
    return float(np.mean(aucs)), (d, folds, dev, y)


# ------------------------------------------------------------------ figures
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


OFFS = {('int', 'carf2'): (4, 3), ('int', 'carf_v1'): (-4, -7), ('int', 'offset_lr'): (4, -4), ('int', 'mask_stack'): (4, -3),
        ('int', 'mean'): (4, -5), ('int', 'stack'): (4, 2), ('ext', 'carf2'): (-4, -8), ('ext', 'carf_v1'): (-4, 3),
        ('ext', 'mask_stack'): (-4, 3), ('ext', 'moddrop_mlp'): (4, -6), ('ext', 'robust_offset_lr'): (-4, 3)}


def fig_tradeoff(I, X):
    fi = fold_table(I, 'carf2', INT_FAIL); xe = ext_table(X, 'carf2', EXT_FAIL)
    models = BASE + ['carf2']
    col = {'carf2': '#C0392B'}
    fig, axs = plt.subplots(1, 2, figsize=(12 * CM, 4.6 * CM))
    for ax, (src, title) in zip(axs, [('int', '(a) Internal, I-SPY2 (25 folds)'), ('ext', '(b) External, I-SPY1')]):
        for m in models:
            if src == 'int':
                x, yv = fi['intact'].unstack()[m].mean(), fi['worst'].unstack()[m].mean()
            else:
                x, yv = xe.loc[m, 'intact'], xe.loc[m, 'worst']
            c = col.get(m, '#5D6D7E')
            ax.scatter(x, yv, s=26 if m == 'carf2' else 16, color=c, zorder=3, marker='*' if m == 'carf2' else 'o')
            off = OFFS.get((src, m), (3, 2))
            ax.annotate(NAMES[m].replace(' (ours)', ''), (x, yv), xytext=off, textcoords='offset points', fontsize=5.8, color=c,
                        ha='right' if off[0] < 0 else 'left')
        ax.set_title(title, fontsize=7, loc='left')
        ax.set_xlabel('AUROC, intact inputs'); ax.set_ylabel('AUROC, worst failure scenario')
        ax.grid(alpha=.25, lw=.4)
    fig.tight_layout(); fig.savefig(FIG / 'fig_tradeoff.pdf'); fig.savefig(FIG / 'fig_tradeoff.png', dpi=220)


def fig_gate(ctx_data):
    """Anchor-match posterior for true and swapped samples, pooled over the 25 internal test folds."""
    from carf2 import AnchorContext
    from pipeline import cached
    d, folds, dev, y = ctx_data
    rng = np.random.default_rng(SEED)
    true, swap = {j: [] for j in range(3)}, {j: [] for j in range(3)}
    for i, (tr, te) in enumerate(folds):
        with np.load(RES / 'cache' / d.key() / f'fold_{i}.npz') as z:
            PA, QA = z['PA'], z['QA']
        lab = dev['clinical'].iloc[te][['hr', 'her2']].to_numpy(float)
        c = AnchorContext(QA, lab).scores('posterior', PA, 0.95)
        for j in range(3):
            ok = np.where(np.isfinite(QA[:, j]).all(1))[0]
            src = rng.permutation(ok)
            cs = AnchorContext(QA, lab).swapped(j, ok, src).scores('posterior', PA, 0.95)
            true[j] += list(c[ok, j]); swap[j] += list(cs[ok, j])
    fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 4.2 * CM), sharey=True, layout='constrained')
    for j, (ax, nm, tag) in enumerate(zip(axs, ['RNA', 'MRI', 'RPPA'], 'abc')):
        bins = np.linspace(0, 1, 21)
        w1 = np.full(len(true[j]), 1 / len(true[j])); w2 = np.full(len(swap[j]), 1 / len(swap[j]))
        ax.hist(true[j], bins, weights=w1, histtype='step', lw=1.4, color='#0072B2', label='Own sample')
        ax.hist(swap[j], bins, weights=w2, histtype='step', lw=1.4, color='#D55E00', label='Swapped sample')
        ax.set_yscale('log'); ax.set_ylim(1e-3, 1.6)
        ax.set_xlim(-0.03, 1.03); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '0.5', '1'])
        auc = roc_auc_score(np.r_[np.ones(len(true[j])), np.zeros(len(swap[j]))], np.r_[true[j], swap[j]])
        ax.set_title(f'({tag}) {nm}, AUROC {auc:.2f}', loc='left', fontsize=8); ax.set_xlabel('Gate $g_j$')
        mac(f'GateAuc{nm.capitalize()}', auc, '{:.2f}')
        mac(f'GateTrue{nm.capitalize()}', np.mean(true[j]), '{:.2f}'); mac(f'GateSwap{nm.capitalize()}', np.mean(swap[j]), '{:.2f}')
        mac(f'GateLow{nm.capitalize()}', 100 * np.mean(np.array(swap[j]) < .5), '{:.0f}')
        mac(f'GateLowTrue{nm.capitalize()}', 100 * np.mean(np.array(true[j]) < .5), '{:.0f}')
    axs[0].set_ylabel('Fraction of patients')
    axs[0].legend(loc='upper left', handlelength=1.8, borderaxespad=0.1, labelspacing=0.3)
    fig.savefig(FIG / 'fig_gate.pdf'); fig.savefig(FIG / 'fig_gate.png', dpi=200); plt.close(fig)


def certificate(G, P):
    """Effective and sharp budgets of the external CARF gate, and the certified share of patient pairs."""
    from scipy.special import logit
    g = G['carf2/external']['carf2']; B = g['budget']; th = np.array(g['coef']).reshape(-1, 3)
    kappa = max(t * np.exp(-t) for t in np.linspace(0, B, 10001))
    b = B * th.sum(); sharp = (B * th[:, 0] + kappa * th[:, 1] + B * th[:, 2])
    mac('CoefSum', th.sum(), '{:.2f}'); mac('EffBudget', b, '{:.2f}'); mac('SharpBudget', sharp.sum(), '{:.2f}')
    for nm, v in zip(['Rna', 'Mri', 'Rppa'], sharp):
        mac(f'SharpBudget{nm}', v, '{:.2f}')
    e = P[(P.design == 'carf2') & (P.split == 'external')]
    a = logit(np.clip(e.clinical.to_numpy(), 1e-7, 1 - 1e-7))
    diff = np.abs(a[:, None] - a[None, :])[np.triu_indices(len(a), 1)]
    mac('CertifiedPairs', 100 * np.mean(diff > 2 * sharp.sum()), '{:.0f}')
    mac('SharpBudgetExt', sharp[0] + sharp[1], '{:.2f}')  # RPPA absent in I-SPY1
    mac('CertifiedPairsExt', 100 * np.mean(diff > 2 * (sharp[0] + sharp[1])), '{:.0f}')


def _write(name, header, rows, spec):
    lines = [r'\begin{tabular}{@{}' + spec + '@{}}', r'\toprule', header + r' \\', r'\midrule'] + rows + [r'\bottomrule', r'\end{tabular}']
    (GEN / name).write_text('% AUTO-GENERATED by carf2/make_assets.py\n' + '\n'.join(lines) + '\n')


def supp_tables(I, X, ex):
    models = ['clinical'] + FUSION + ['carf2'] + ABL
    x = X[X.design == 'carf2']
    sc = ['intact', 'all_missing', 'rna_missing', 'rna_swap', 'rna_noisy', 'rna_low', 'rna_high', 'rna_neutral',
          'mri_missing', 'mri_swap', 'mri_noisy', 'mri_low', 'mri_high', 'mri_neutral']
    t = x.groupby(['model', 'scenario']).auroc.mean().unstack()
    d = x.groupby('model').max_disp.max()
    head = 'Model & ' + ' & '.join(s_.replace('rna_', 'R ').replace('mri_', 'M ').replace('_', ' ') for s_ in sc) + r' & $\max|z{-}a|$'
    rows = []
    for m in models:
        vals = ' & '.join(f'{t.loc[m, s_]:.3f}' if (m in t.index and s_ in t.columns and np.isfinite(t.loc[m, s_])) else '--' for s_ in sc)
        rows.append(f'{NAMES[m]} & {vals} & ' + (f'{d[m]:.2f}' if m != 'clinical' else '0') + r' \\')
    _write('tab_supp_external.tex', head, rows, 'l' + 'c' * (len(sc) + 1))
    # internal all scenarios
    ii = I[I.design == 'carf2']
    sci = ['intact', 'all_missing'] + INT_FAIL
    t = ii.groupby(['model', 'scenario']).auroc.mean().unstack()
    head = 'Model & ' + ' & '.join(s_.replace('_', ' ') for s_ in sci)
    rows = [f'{NAMES[m]} & ' + ' & '.join(f'{t.loc[m, s_]:.3f}' if np.isfinite(t.loc[m, s_]) else '--' for s_ in sci) + r' \\'
            for m in models if m in t.index]
    _write('tab_supp_internal.tex', head, rows, 'l' + 'c' * len(sci))
    # calibration
    rows = []
    for m in models:
        a = ii[(ii.scenario == 'intact') & (ii.model == m)]; b = x[(x.scenario == 'intact') & (x.model == m)]
        rows.append(f'{NAMES[m]} & {a.brier.mean():.3f} & {a.log_loss.mean():.3f} & {a.ece.mean():.3f} & '
                    f'{b.brier.mean():.3f} & {b.log_loss.mean():.3f} & {b.ece.mean():.3f} \\\\')
    _write('tab_supp_calib.tex', r' & \multicolumn{3}{c}{Internal} & \multicolumn{3}{c}{External} \\ Model & Brier & Log loss & ECE & Brier & Log loss & ECE',
           rows, 'lcccccc')
    # input-level
    if 'input' in ex:
        inp = ex['input'][ex['input'].design == 'carf2']
        order = ['rna_dropout30', 'rna_dropout50', 'rna_block_shift10', 'rna_block_shift25', 'mri_noise_sd0.5', 'mri_noise_sd1',
                 'mri_dropout30', 'mri_dropout60']
        t = inp.groupby(['model', 'scenario']).auroc.mean().unstack()
        order = [o for o in order if o in t.columns]
        head = 'Model & ' + ' & '.join(o.replace('_', ' ') for o in order)
        rows = [f'{NAMES[m]} & ' + ' & '.join(f'{t.loc[m, o]:.3f}' for o in order) + r' \\' for m in models if m in t.index]
        _write('tab_supp_input.tex', head, rows, 'l' + 'c' * len(order))
    if 'natural' in ex:
        n = ex['natural'][ex['natural'].design == 'carf2']
        subs = ['all', 'complete', 'partial', 'rna_only', 'mri_only']
        t = n.pivot(index='model', columns='subset', values='auroc'); c = n.drop_duplicates('subset').set_index('subset')
        head = 'Model & ' + ' & '.join(f'{s_.replace("_", " ")} ({int(c.loc[s_, "n"])}/{int(c.loc[s_, "pos"])})' for s_ in subs)
        rows = [f'{NAMES[m]} & ' + ' & '.join(f'{t.loc[m, s_]:.3f}' for s_ in subs) + r' \\' for m in models if m in t.index]
        _write('tab_supp_natural.tex', head, rows, 'l' + 'c' * len(subs))
    if 'gse' in ex:
        g = ex['gse'][ex['gse'].design == 'carf2']
        t = g.groupby(['model', 'scenario']).auroc.mean().unstack(); dd = g.groupby('model').max_disp.max()
        scg = ['intact', 'rna_swap', 'rna_dropout50']
        head = 'Model & intact & RNA swap & RNA drop 50\\% & $\\max|z{-}a|$'
        rows = [f'{NAMES[m]} & ' + ' & '.join(f'{t.loc[m, s_]:.3f}' for s_ in scg) + f' & {dd[m]:.2f}' + r' \\' for m in models if m in t.index]
        _write('tab_supp_gse.tex', head, rows, 'lcccc')


def main():
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); style()
    I, X, P, G = load()
    rppa, ctx_data = rppa_internal()
    ex = extras()
    ex['rppa_int'] = rppa
    tab_main(I, X, P, G, ex)
    tab_stress(I, X, ex)
    tab_design(I, X)
    tab_mixup(I, X)
    tab_ablation(I, X, G)
    certificate(G, P)
    supp_tables(I, X, ex)
    extras()  # permutation p-value uses ExtCarfTwo
    fig_tradeoff(I, X)
    fig_gate(ctx_data)
    body = ['% AUTO-GENERATED by carf2/make_assets.py -- do not edit by hand.']
    body += [f'\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}' for k, v in sorted(MACROS.items())]
    (GEN / 'macros.tex').write_text('\n'.join(body) + '\n')
    print(len(MACROS), 'macros')


if __name__ == '__main__':
    main()
