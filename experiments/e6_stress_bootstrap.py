#!/usr/bin/env python3
"""E6: paired patient bootstrap of delta = AUROC(carf) - AUROC(comparator) on external
I-SPY1 (n=112) under output- and input-level stress, using the e1 protocol exactly:
2000 resamples from default_rng(SEED) (same index sequence as e1_bootstrap; single-class
resamples skipped), realizations held fixed, AUROC averaged over the 30 realizations
inside each resample, then quantiles of the delta.

Per-realization predictions are regenerated with the original deterministic seeds:
  output-level: tier1/revision3 stress_scenarios(Q) (SEED + 1000*j + repeat);
  input-level:  a8.corrupt_Q with default_rng([SEED, 8, si, rep]) (as a8.job / e5).
Models: fusions fit on external P (fit_fusions), offset_lr (offset_models),
robust_offset_lr (e5 RobustOffsetLR, gamma 0.25), moddrop_mlp fixed at (32, 16),
alpha 1e-2 (a7's choice on the external split, as in e1/e5).
Mean AUROCs are checked against results/stress.csv, a7/stress.csv,
e5/robust_offset_stress.csv, a8/input_stress_summary.csv, e5/moddrop_input_stress_summary.csv.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a8_input_corruption as a8  # noqa: E402
import a7_moddrop as a7  # noqa: E402
from tier1 import load, labels, SEED, OUT as RES, stress_scenarios  # noqa: E402
from scripts.run_revision3_analysis import predict_fusions  # noqa: E402
from e5_robust_offset import RobustOffsetLR  # noqa: E402
from e1_bootstrap import auc_rows, NBOOT  # noqa: E402

OUT = RES / 'e6'
TEX = Path(__file__).resolve().parent.parent / 'latex/generated/tab_stress_ci.tex'
MODELS = ['carf', 'stack', 'mask_stack', 'offset_lr', 'robust_offset_lr', 'moddrop_mlp', 'carf_no_robust']
COMPS = MODELS[1:]
OUTPUT_SC = ['rna_permuted', 'mri_permuted', 'rna_noisy', 'mri_noisy']
INPUT_SC = ['rna_dropout50', 'rna_block25_shift+2', 'mri_noise_sd1', 'mri_dropout60']
TOL = 2e-3
_S = None


def predict_all(S, q):
    p = predict_fusions(S['fus'], q)
    p.update({k: m.predict_proba(q)[:, 1] for k, m in S['off'].items()})
    p['robust_offset_lr'] = S['rob'].predict_proba(q)[:, 1]
    p['moddrop_mlp'] = a7.predict(S['mlp'], q)
    return np.column_stack([p[m] for m in MODELS])


def input_job(t):
    si, kind, a, b, rep = t
    rng = np.random.default_rng([SEED, 8, si, rep])  # identical to a8.job / e5
    return a8.name_of(kind, a, b), rep, predict_all(_S, a8.corrupt_Q(_S, kind, a, b, rng))


def boot_job(A):
    ye = _S['ye']
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(ye), len(ye))
        if len(np.unique(ye[ix])) < 2:
            continue
        auc = np.mean([auc_rows(ye[ix], a[ix]) for a in A], axis=0)
        out.append(auc[0] - auc[1:])
    return np.array(out)


def check(reals, ye):
    ref = {}
    st = pd.read_csv(RES / 'stress.csv').groupby(['scenario', 'model']).auroc.mean()
    s7 = pd.read_csv(RES / 'a7/stress.csv').query("model == 'moddrop_mlp'").groupby('scenario').auroc.mean()
    sr = pd.read_csv(RES / 'e5/robust_offset_stress.csv').query("model == 'robust_offset_lr'").set_index('scenario').auroc_mean
    for sc in OUTPUT_SC:
        for m in MODELS:
            ref[(sc, m)] = s7[sc] if m == 'moddrop_mlp' else sr[sc] if m == 'robust_offset_lr' else st[(sc, m)]
    s8 = pd.read_csv(RES / 'a8/input_stress_summary.csv').set_index(['scenario', 'model']).auroc_mean
    s5 = pd.read_csv(RES / 'e5/moddrop_input_stress_summary.csv').set_index(['scenario', 'model']).auroc_mean
    for sc in INPUT_SC:
        for m in MODELS:
            ref[(sc, m)] = s5[(sc, m)] if m in ('moddrop_mlp', 'robust_offset_lr') else s8[(sc, m)]
    rows = []
    for sc, R in reals.items():
        v = np.mean([auc_rows(ye, a) for a in R], axis=0)
        for m, x in zip(MODELS, v):
            rows.append({'scenario': sc, 'model': m, 'regenerated': x, 'saved': ref[(sc, m)],
                         'abs_diff': abs(x - ref[(sc, m)])})
    chk = pd.DataFrame(rows)
    chk.to_csv(OUT / 'reproduction_check.csv', index=False)
    print(chk.round(5).to_string(index=False))
    print(f'max |regenerated - saved| mean AUROC: {chk.abs_diff.max():.2e}')
    assert chk.abs_diff.max() < TOL, chk.sort_values('abs_diff').tail()
    # e1's saved per-patient rna_permuted predictions
    e1 = pd.read_csv(RES / 'e1/external_stress_predictions.csv.gz').query("scenario == 'rna_permuted'")
    cols = [m for m in MODELS if m in e1.columns]
    E = np.stack([g.sort_values('patient')[cols].to_numpy() for _, g in e1.groupby('repeat')])
    R = np.stack(reals['rna_permuted'])[:, :, [MODELS.index(c) for c in cols]]
    print(f'max |pred - e1 saved pred| (rna_permuted, {cols}): {np.abs(E - R).max():.2e}')
    return chk


def fmt(r):
    return f"{r.mid:.3f} [{r.lo:.3f}, {r.hi:.3f}]"


LAB_M = {'stack': 'Stack', 'mask_stack': 'Mask-stack', 'offset_lr': 'Offset LR',
         'robust_offset_lr': 'Robust offset LR', 'moddrop_mlp': 'ModDrop MLP', 'carf_no_robust': 'CARF w/o robust'}
LAB_S = {'rna_permuted': 'RNA perm.', 'mri_permuted': 'MRI perm.', 'rna_noisy': 'RNA noisy',
         'mri_noisy': 'MRI noisy', 'rna_dropout50': 'RNA 50\\% drop', 'rna_block25_shift+2': 'RNA block 25\\%/+2',
         'mri_noise_sd1': 'MRI noise SD 1', 'mri_dropout60': 'MRI 60\\% drop'}


def write_tex(df):
    # Transposed: rows = scenarios, columns = comparators. 6 cells of "mid [lo, hi]"
    # do not fit 12.2 cm in one row, and 4 per row measured 13.1 cm, so three stacked
    # blocks: fusion baselines / offset family / ModDrop + ablation.
    blocks = [COMPS[:2], COMPS[2:4], COMPS[4:]]
    L = ['% generated by experiments/e6_stress_bootstrap.py -- do not edit',
         '{\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
         '\\begin{tabular}{@{}l' + 'c' * max(map(len, blocks)) + '@{}}', '\\toprule']
    for bi, cs in enumerate(blocks):
        if bi:
            L.append('\\midrule')
        L.append('Scenario & ' + ' & '.join(LAB_M[c] for c in cs) + ' \\\\')
        L.append('\\midrule')
        for gi, (g, scs) in enumerate([('Output-level', OUTPUT_SC), ('Input-level', INPUT_SC)]):
            L.append(f'\\multicolumn{{{len(cs) + 1}}}{{@{{}}l}}{{\\emph{{{g}}}}} \\\\')
            for sc in scs:
                cells = [fmt(df[(df.scenario == sc) & (df.comparator == c)].iloc[0]) for c in cs]
                L.append(LAB_S[sc] + ' & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '}']
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text('\n'.join(L) + '\n')


def main():
    global _S
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    S = _S = a8.setup()  # frozen experts, fus / off fit on external P, cached Q, ye
    P, _, ytr, ye = load('external', labels())
    X, yy = a7.masked_blocks(P, ytr)
    S['mlp'] = a7.mlp((32, 16), 1e-2).fit(X, yy)
    S['rob'] = RobustOffsetLR().fit(P, ytr)

    reals = {sc: [] for sc in OUTPUT_SC + INPUT_SC}
    for sc, _, q in stress_scenarios(S['Q']):
        if sc in OUTPUT_SC:
            reals[sc].append(predict_all(S, q))
    tasks = [(si, k, a, b, r) for si, (k, a, b) in enumerate(a8.SCENARIOS)
             if a8.name_of(k, a, b) in INPUT_SC for r in range(a8.REPEATS)]
    with mp.get_context('fork').Pool(36) as pool:
        for name, rep, arr in sorted(pool.map(input_job, tasks, chunksize=1), key=lambda t: (t[0], t[1])):
            reals[name].append(arr)
    assert all(len(v) == 30 for v in reals.values())
    rows = []
    for sc, R in reals.items():
        for r, a in enumerate(R):
            rows.append(pd.DataFrame(a, columns=MODELS).assign(scenario=sc, repeat=r, pcr=ye, patient=np.arange(len(ye))))
    pd.concat(rows).to_csv(OUT / 'stress_predictions.csv.gz', index=False)

    check(reals, ye)

    with mp.get_context('fork').Pool(8) as pool:
        boots = pool.map(boot_job, [np.stack(reals[sc]) for sc in reals])
    rows = []
    for sc, D in zip(reals, boots):
        for k, c in enumerate(COMPS):
            lo, mid, hi = np.quantile(D[:, k], [.025, .5, .975])
            rows.append({'scenario': sc, 'comparator': c, 'lo': lo, 'mid': mid, 'hi': hi,
                         'mean_delta': D[:, k].mean(), 'n_boot': len(D)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'stress_bootstrap.csv', index=False)
    write_tex(df)
    df['excl0'] = np.where(df.lo > 0, 'CARF>', np.where(df.hi < 0, 'CARF<', ''))
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    if '--tex-only' in sys.argv:  # rebuild the table from the saved CSV
        write_tex(pd.read_csv(OUT / 'stress_bootstrap.csv'))
    else:
        main()
