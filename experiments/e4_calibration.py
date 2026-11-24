#!/usr/bin/env python3
"""Like-for-like calibration: Platt-scale every fusion model on its own inner
out-of-fold predictions (as for CARF), then report ECE, Brier, and calibration
intercept/slope with patient-bootstrap intervals on I-SPY1."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1 import load, labels, ece, offset_models, SEED  # noqa: E402
from scripts.run_revision3_analysis import fit_fusions, predict_fusions  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results/e4'
MODELS = ['clinical', 'stack', 'mask_stack', 'offset_lr', 'carf', 'carf_no_robust']
EPS = 1e-6


def all_preds(P, y, Q):
    pred = predict_fusions(fit_fusions(P, y), Q)
    pred['offset_lr'] = offset_models(P, y)['offset_lr'].predict_proba(Q)[:, 1]
    return pred


def cal_line(y, p):
    z = logit(np.clip(p, EPS, 1 - EPS))
    slope = LogisticRegression(C=1e6).fit(z[:, None], y).coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1 (offset)
    from scipy.optimize import minimize_scalar
    nll = lambda b: np.mean(np.logaddexp(0, z + b) - y * (z + b))
    return minimize_scalar(nll).x, slope


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y = labels()
    P, Q, ytr, ye = load('external', y)
    oof = {m: np.zeros(len(ytr)) for m in MODELS}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + 100).split(P, ytr):
        pr = all_preds(P[tr], ytr[tr], P[te])
        for m in MODELS:
            oof[m][te] = pr[m]
    ext = all_preds(P, ytr, Q)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(ye), len(ye)) for _ in range(2000)]
    rows = []
    for m in MODELS:
        platt = LogisticRegression(C=1e6).fit(logit(np.clip(oof[m], EPS, 1 - EPS))[:, None], ytr)
        for variant, p in [('raw', ext[m]),
                           ('platt', platt.predict_proba(logit(np.clip(ext[m], EPS, 1 - EPS))[:, None])[:, 1])]:
            a, s = cal_line(ye, p)
            bs = np.array([cal_line(ye[i], p[i]) + (brier_score_loss(ye[i], p[i]),) for i in boots
                           if len(np.unique(ye[i])) == 2])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            rows.append(dict(model=m, variant=variant, auroc=roc_auc_score(ye, p), brier=brier_score_loss(ye, p),
                             ece10=ece(ye, p, 10), ece5=ece(ye, p, 5), cal_intercept=a, cal_slope=s,
                             int_lo=lo[0], int_hi=hi[0], slope_lo=lo[1], slope_hi=hi[1],
                             brier_lo=lo[2], brier_hi=hi[2], mean_pred=p.mean(), prevalence=ye.mean()))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'calibration.csv', index=False)
    print(d.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
