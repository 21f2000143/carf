#!/usr/bin/env python3
"""E1 task 3: paired patient bootstrap (2000 resamples, seed SEED).
(a) External I-SPY1 (n=112) under RNA permutation: per-patient predictions are
    regenerated for the 30 deterministic stress_scenarios realizations; per
    resample, AUROC is averaged over realizations, then
    delta = mean AUROC(carf) - mean AUROC(model). Intact deltas also reported.
(b) Natural missingness (a6 predictions, 199 patients with clinical + >=1 of
    RNA/MRI): delta = AUROC(carf) - AUROC(comparator); subgroups as extra rows."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1 import load, labels, offset_models, SEED, OUT as RES, stress_scenarios  # noqa: E402
from scripts.run_revision3_analysis import fit_fusions, predict_fusions  # noqa: E402
import a7_moddrop  # noqa: E402

OUT = RES / 'e1'
NBOOT = 2000
COMP_A = ['stack', 'mask_stack', 'offset_lr', 'carf_no_robust', 'moddrop_mlp']
COMP_B = ['clinical', 'stack', 'mask_stack', 'offset_lr_masked']


def auc_rows(y, S):
    """AUROC of y against each column of S (n x k), via rank formula (ties averaged)."""
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    R = np.apply_along_axis(rankdata, 0, S)
    return (R[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def summarize(boot, point, extra):
    rows = []
    for c, d in boot.items():
        d = np.asarray(d)
        q = np.quantile(d, [.025, .5, .975])
        rows.append({**extra, 'comparator': c, 'point': point[c], 'q025': q[0], 'q50': q[1], 'q975': q[2],
                     'p_delta_le_0': float((d <= 0).mean()), 'n_boot': len(d)})
    return rows


def part_a(y):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    P, Q, ytr, ye = load('external', y)
    fus = fit_fusions(P, ytr)
    off = offset_models(P, ytr)
    mlp, params = a7_moddrop.fit(P, ytr, 'moddrop_mlp')
    print('moddrop params', params)
    models = ['carf'] + COMP_A

    def predict_all(q):
        p = predict_fusions(fus, q)
        p.update({k: m.predict_proba(q)[:, 1] for k, m in off.items()})
        p['moddrop_mlp'] = a7_moddrop.predict(mlp, q)
        return np.column_stack([p[m] for m in models])

    sets = {'intact': [predict_all(Q)],
            'rna_permuted': [predict_all(q) for sc, _, q in stress_scenarios(Q) if sc == 'rna_permuted']}
    assert len(sets['rna_permuted']) == 30
    # consistency with saved stress results
    st = pd.read_csv(RES / 'stress.csv').query("scenario == 'rna_permuted'").groupby('model').auroc.mean()
    s7 = pd.read_csv(RES / 'a7/stress.csv').query("scenario == 'rna_permuted' and model == 'moddrop_mlp'").auroc.mean()
    ref = {**st.to_dict(), 'moddrop_mlp': s7}
    rows, per_patient = [], []
    for name, reals in sets.items():
        A = np.stack(reals)  # R x n x models
        point_auc = np.mean([auc_rows(ye, a) for a in A], axis=0)
        if name == 'rna_permuted':
            for m, v in zip(models, point_auc):
                print(f'check {m}: regenerated {v:.4f} vs saved {ref[m]:.4f}')
                assert abs(v - ref[m]) < 2e-3, m
        point = {m: point_auc[0] - point_auc[k] for k, m in enumerate(models) if k}
        rng = np.random.default_rng(SEED)
        boot = {m: [] for m in COMP_A}
        for b in range(NBOOT):
            ix = rng.integers(0, len(ye), len(ye))
            if len(np.unique(ye[ix])) < 2:
                continue
            auc = np.mean([auc_rows(ye[ix], a[ix]) for a in A], axis=0)
            for k, m in enumerate(models[1:], 1):
                boot[m].append(auc[0] - auc[k])
        rows += summarize(boot, point, {'cohort': 'external_112', 'scenario': name,
                                         'carf_auroc': point_auc[0]})
        for r, a in enumerate(A):
            per_patient.append(pd.DataFrame(a, columns=models).assign(scenario=name, repeat=r, pcr=ye,
                                                                       patient=np.arange(len(ye))))
    pd.concat(per_patient).to_csv(OUT / 'external_stress_predictions.csv.gz', index=False)
    return rows


def part_b():
    d = pd.read_csv(RES / 'a6/predictions.csv')
    d = d[d.group != 'clinical_only']
    rows = []
    for grp, sub in [('all_available', d)] + list(d.groupby('group')):
        yy = sub.pcr.to_numpy()
        S = sub[['carf'] + COMP_B].to_numpy()
        a0 = auc_rows(yy, S)
        point = {m: a0[0] - a0[k] for k, m in enumerate(COMP_B, 1)}
        rng = np.random.default_rng(SEED)
        boot = {m: [] for m in COMP_B}
        for b in range(NBOOT):
            ix = rng.integers(0, len(yy), len(yy))
            if len(np.unique(yy[ix])) < 2:
                continue
            a = auc_rows(yy[ix], S[ix])
            for k, m in enumerate(COMP_B, 1):
                boot[m].append(a[0] - a[k])
        rows += summarize(boot, point, {'cohort': f'natural_missing_{grp}', 'scenario': 'natural',
                                         'n': len(yy), 'carf_auroc': a0[0]})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y = labels()
    df = pd.DataFrame(part_a(y) + part_b())
    df.to_csv(OUT / 'bootstrap_deltas.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
