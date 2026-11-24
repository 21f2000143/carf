#!/usr/bin/env python3
"""E1 task 1: anchor failure. Corrupt I-SPY1 *clinical inputs* before the frozen
clinical expert (fit on all I-SPY2, as in a8), then recompute every fusion.
RNA/MRI expert outputs are the cached Q. Fusions are fit once on cached OOF P."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1 import offset_models, load, labels, SEED  # noqa: E402
from scripts.run_revision2_analysis import common_data, experts, predict_expert  # noqa: E402
from scripts.run_revision3_analysis import fit_fusions, predict_fusions  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results/e1'
REPEATS = 30
MODELS = ['clinical', 'rna', 'stack', 'mask_stack', 'offset_lr', 'carf', 'carf_no_robust']
SCENARIOS = [('intact', None), ('hr_flip', .10), ('hr_flip', .25), ('her2_flip', .10),
             ('her2_flip', .25), ('age_noise', 10.), ('clinical_output_permuted', None)]


def name_of(kind, a):
    if kind in ('hr_flip', 'her2_flip'):
        return f'{kind}{int(a * 100)}'
    if kind == 'age_noise':
        return f'age_noise_sd{a:g}'
    return kind


def setup():
    E2, M2, I2, E1, M1, I1 = common_data()
    y2 = M2.pcr.astype(int).to_numpy()
    fitted = experts(E2, M2, I2, y2, SEED + 100)
    P, Q, ytr, ye = load('external', labels())
    qc = predict_expert(fitted[1], M1, 'clinical')
    diff = np.abs(qc - Q[:, 1]).max()
    print('clinical expert reproduction max abs diff:', diff)
    if diff > 1e-6:
        raise SystemExit('Clinical expert reproduction failed')
    fus = fit_fusions(P, ytr)
    off = offset_models(P, ytr)
    return dict(clin=fitted[1], M1=M1, Q=Q, ye=ye, fus=fus, off=off)


_S = None


def job(t):
    si, kind, a, rep = t
    S = _S
    rng = np.random.default_rng([SEED, 11, si, rep])
    Q = S['Q'].copy()
    M = S['M1'].copy()
    if kind in ('hr_flip', 'her2_flip'):
        col = kind.split('_')[0]
        obs = np.flatnonzero(M[col].notna().to_numpy())
        pick = rng.choice(obs, int(round(a * len(M))), replace=False)
        v = M[col].to_numpy(float).copy()
        v[pick] = 1 - v[pick]
        M[col] = v
        Q[:, 1] = predict_expert(S['clin'], M, 'clinical')
    elif kind == 'age_noise':
        M['age'] = M['age'].to_numpy(float) + rng.normal(0, a, len(M))
        Q[:, 1] = predict_expert(S['clin'], M, 'clinical')
    elif kind == 'clinical_output_permuted':
        Q[:, 1] = rng.permutation(Q[:, 1])
    pred = predict_fusions(S['fus'], Q)
    for k, m in S['off'].items():
        pred[k] = m.predict_proba(Q)[:, 1]
    return [dict(scenario=name_of(kind, a), repeat=rep, model=k, auroc=roc_auc_score(S['ye'], pred[k]))
            for k in MODELS]


def main():
    global _S
    _S = setup()
    tasks = [(si, k, a, r) for si, (k, a) in enumerate(SCENARIOS)
             for r in (range(1) if k == 'intact' else range(REPEATS))]
    with mp.get_context('fork').Pool(40) as pool:
        out = pool.map(job, tasks, chunksize=2)
    df = pd.DataFrame([r for rows in out for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'anchor_failure.csv', index=False)
    order = [name_of(*s) for s in SCENARIOS]
    s = df.groupby(['scenario', 'model']).auroc.agg(['mean', 'std', 'size']).reset_index()
    s['scenario'] = pd.Categorical(s.scenario, order, ordered=True)
    s = s.sort_values(['scenario', 'model'])
    s.to_csv(OUT / 'anchor_failure_summary.csv', index=False)
    tab = s.assign(v=s['mean'].map('{:.3f}'.format) + '±' + s['std'].fillna(0).map('{:.3f}'.format))
    print(tab.pivot(index='scenario', columns='model', values='v')[MODELS].to_string())


if __name__ == '__main__':
    main()
