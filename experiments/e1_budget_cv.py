#!/usr/bin/env python3
"""E1 task 2: CARF budget B chosen by inner 5-fold CV log loss on P only
(gamma=0.25). No test labels enter the choice."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1 import load, labels, SEED, AnchoredResidualFusion, stress_scenarios  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results/e1'
BUDGETS = [.25, .5, 1., 2., 4.]
GAMMA = .25


def run(key, seed):
    y = labels()
    P, Q, ytr, yte = load(key, y)
    losses = {B: [] for B in BUDGETS}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(P, ytr):
        for B in BUDGETS:
            m = AnchoredResidualFusion(budget=B, robust_weight=GAMMA).fit(P[tr], ytr[tr])
            losses[B].append(log_loss(ytr[va], m.predict_proba(P[va])[:, 1], labels=[0, 1]))
    mean = {B: float(np.mean(v)) for B, v in losses.items()}
    Bstar = min(BUDGETS, key=lambda B: mean[B])
    rec = {'split': key, 'chosen_B': Bstar, **{f'inner_ll_B{B:g}': mean[B] for B in BUDGETS}}
    for tag, B in [('cv', Bstar), ('fixed1', 1.)]:
        m = AnchoredResidualFusion(budget=B, robust_weight=GAMMA).fit(P, ytr)
        rec[f'auroc_{tag}'] = roc_auc_score(yte, m.predict_proba(Q)[:, 1])
        if key == 'external':
            perm = [roc_auc_score(yte, m.predict_proba(q)[:, 1])
                    for sc, _, q in stress_scenarios(Q) if sc == 'rna_permuted']
            assert len(perm) == 30
            rec[f'rna_perm_mean_{tag}'], rec[f'rna_perm_sd_{tag}'] = np.mean(perm), np.std(perm, ddof=1)
    return rec


def main():
    keys = [(f'fold_{k}', SEED + k) for k in range(25)] + [('external', SEED + 100)]
    recs = Parallel(n_jobs=26)(delayed(run)(k, s) for k, s in keys)
    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'budget_cv.csv', index=False)
    internal = df[df.split != 'external']
    print(df[['split', 'chosen_B', 'auroc_cv', 'auroc_fixed1']].to_string(index=False))
    print('chosen B counts (internal):', internal.chosen_B.value_counts().sort_index().to_dict())
    print('internal mean AUROC cv-B %.4f (SD %.4f) | fixed B=1 %.4f (SD %.4f)' % (
        internal.auroc_cv.mean(), internal.auroc_cv.std(), internal.auroc_fixed1.mean(), internal.auroc_fixed1.std()))
    print(df[df.split == 'external'].T.to_string())


if __name__ == '__main__':
    import e1_budget_cv
    e1_budget_cv.main()
