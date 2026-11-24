#!/usr/bin/env python3
"""E5 task 2: ModDrop MLP (and robust offset LR) under A8's input-level corruptions.

Reuses a8.setup / a8.corrupt_Q and a8's per-(scenario, repeat) RNG seeds, so the
corrupted expert outputs are identical to A8's (they were not saved, so regenerated).
ModDrop MLP is fixed at hidden (32, 16), alpha 1e-2: a7's most frequent choice for
moddrop_mlp (14/26 splits) and its choice on the external split. Trained on external P.
CARF / clinical are re-evaluated and checked against results/a8/input_stress.csv.
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
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a8_input_corruption as a8  # noqa: E402
import a7_moddrop as a7  # noqa: E402
from tier1 import load, labels  # noqa: E402
from scripts.run_revision3_analysis import predict_fusions  # noqa: E402
from e5_robust_offset import RobustOffsetLR  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results/e5'
HIDDEN, ALPHA = (32, 16), 1e-2
_S = None


def job(t):
    si, kind, a, b, rep = t
    S = _S
    rng = np.random.default_rng([a8.SEED, 8, si, rep])  # identical to a8.job
    Q = S['Q'].copy() if kind == 'intact' else a8.corrupt_Q(S, kind, a, b, rng)
    p = predict_fusions(S['fus'], Q)
    pred = {'clinical': p['clinical'], 'carf': p['carf'],
            'moddrop_mlp': a7.predict(S['mlp'], Q),
            'robust_offset_lr': S['rob'].predict_proba(Q)[:, 1]}
    zc = a8.lg(Q[:, 1])
    return [dict(scenario=a8.name_of(kind, a, b), repeat=rep, model=k, auroc=roc_auc_score(S['ye'], v),
                 brier=brier_score_loss(S['ye'], v), max_shift=np.max(np.abs(a8.lg(v) - zc)))
            for k, v in pred.items()]


def main():
    global _S
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    S = _S = a8.setup()
    P, _, ytr, _ = load('external', labels())
    X, yy = a7.masked_blocks(P, ytr)
    S['mlp'] = a7.mlp(HIDDEN, ALPHA).fit(X, yy)
    S['rob'] = RobustOffsetLR().fit(P, ytr)
    tasks = [(si, k, a, b, r) for si, (k, a, b) in enumerate(a8.SCENARIOS)
             for r in (range(1) if k == 'intact' else range(a8.REPEATS))]
    with mp.get_context('fork').Pool(32) as pool:
        out = pool.map(job, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'moddrop_input_stress.csv', index=False)
    # check against saved A8 values
    ref = pd.read_csv(a8.OUT / 'input_stress.csv').set_index(['scenario', 'repeat', 'model']).auroc
    for m in ['carf', 'clinical']:
        d = df[df.model == m]
        diff = np.abs(d.auroc.to_numpy() - ref.loc[list(zip(d.scenario, d.repeat, d.model))].to_numpy()).max()
        print(f'max |AUROC - a8| for {m}: {diff:.2e}')
        assert diff < 1e-6
    clin = df[df.model == 'clinical'].set_index(['scenario', 'repeat']).auroc
    df['auroc_loss_vs_clinical'] = clin.loc[list(zip(df.scenario, df.repeat))].to_numpy() - df.auroc
    order = [a8.name_of(*s) for s in a8.SCENARIOS]
    summ = (df[df.model.isin(['moddrop_mlp', 'robust_offset_lr'])].groupby(['scenario', 'model'])
            .agg(auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'), brier_mean=('brier', 'mean'),
                 max_shift_mean=('max_shift', 'mean'), max_shift_max=('max_shift', 'max'),
                 loss_vs_clinical_mean=('auroc_loss_vs_clinical', 'mean'),
                 loss_vs_clinical_sd=('auroc_loss_vs_clinical', 'std'), n=('auroc', 'size'))
            .reset_index())
    summ['scenario'] = pd.Categorical(summ.scenario, order, ordered=True)
    summ = summ.sort_values(['scenario', 'model'])
    summ.to_csv(OUT / 'moddrop_input_stress_summary.csv', index=False)
    a8s = pd.read_csv(a8.OUT / 'input_stress_summary.csv')
    a8s = a8s[a8s.model.isin(['clinical', 'carf', 'offset_lr', 'mask_stack'])]
    both = pd.concat([summ, a8s])
    both['cell'] = both.auroc_mean.map('{:.3f}'.format) + ' ± ' + both.auroc_sd.fillna(0).map('{:.3f}'.format)
    print(both.pivot(index='scenario', columns='model', values='cell').loc[order].to_string())


if __name__ == '__main__':
    main()
