#!/usr/bin/env python3
"""Tier-1 experiments on cached expert predictions: offset LR baseline,
budget x robustness sweep, and calibration. No expert is refitted."""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MIKE = Path(__file__).resolve().parents[1] / 'mike2026'
sys.path.insert(0, str(MIKE))
from src.anchored_fusion import AnchoredResidualFusion  # noqa: E402
from scripts.run_revision3_analysis import (fit_fusions, predict_fusions, scores,  # noqa: E402
                                            stress_scenarios, SEED)

CACHE = MIKE / 'results/revision3/cache/d562c1894fd42bb6'
DATA = MIKE / 'data/revision2/processed'
OUT = Path(__file__).resolve().parent / 'results'
EPS = 1e-7


def labels():
    y = {}
    for cohort in ['ISPY2', 'ISPY1']:
        m = pd.read_csv(DATA / f'{cohort}_meta.csv', dtype={'patient_id': str}).set_index('patient_id')
        y.update(m.pcr.astype(int).to_dict())
    return y


def load(key, y):
    with np.load(CACHE / f'{key}.npz') as z:
        P, Q = z['P'], z['Q']
        ytr = np.array([y[i] for i in z['train_ids']])
        yte = np.array([y[i] for i in z['test_ids']])
    return P, Q, ytr, yte


class OffsetLR:
    """Logistic regression with the clinical logit as a fixed offset.

    z = a + b + sum_j m_j w_j logit(p_j), with L2 on w. Missing experts
    contribute zero. With ``masked`` the loss averages over all four
    availability masks, as CARF does. No sign, clip, or budget constraint.
    """

    def __init__(self, l2=0.01, masked=True, intercept=True):
        self.l2, self.masked, self.intercept = l2, masked, intercept

    @staticmethod
    def _design(P):
        a = logit(np.clip(P[:, 1], EPS, 1 - EPS))
        X = np.column_stack([np.where(np.isfinite(P[:, j]), logit(np.clip(np.nan_to_num(P[:, j], nan=.5), EPS, 1 - EPS)), 0.)
                             for j in (0, 2)])
        return a, X

    def fit(self, P, y):
        envs = []
        masks = list(itertools.product([False, True], repeat=2)) if self.masked else [(True, True)]
        for keep in masks:
            q = P.copy()
            for j, k in zip((0, 2), keep):
                if not k:
                    q[:, j] = np.nan
            envs.append(self._design(q))
        y = np.asarray(y, float)

        def objective(theta):
            b, w = (theta[0], theta[1:]) if self.intercept else (0., theta)
            value, gw, gb = 0., np.zeros_like(w), 0.
            for a, X in envs:
                z = a + b + X @ w
                value += np.mean(np.logaddexp(0, z) - y * z)
                r = (expit(z) - y) / len(y)
                gw += X.T @ r
                gb += r.sum()
            k = len(envs)
            value, gw, gb = value / k + self.l2 * w @ w / 2, gw / k + self.l2 * w, gb / k
            return value, (np.r_[gb, gw] if self.intercept else gw)

        res = minimize(objective, np.zeros(3 if self.intercept else 2), jac=True, method='L-BFGS-B')
        self.theta_ = res.x
        return self

    def decision_function(self, P):
        a, X = self._design(P)
        b, w = (self.theta_[0], self.theta_[1:]) if self.intercept else (0., self.theta_)
        return a + b + X @ w

    def predict_proba(self, P):
        p = expit(self.decision_function(P))
        return np.column_stack([1 - p, p])


def offset_models(P, y):
    return {'offset_lr': OffsetLR(masked=False).fit(P, y),
            'offset_lr_masked': OffsetLR(masked=True).fit(P, y)}


def platt_carf(P, y, seed):
    """CARF plus Platt scaling fit on inner cross-fitted CARF logits."""
    z = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(P, y):
        z[te] = AnchoredResidualFusion().fit(P[tr], y[tr]).decision_function(P[te])
    platt = LogisticRegression(C=1e6).fit(z[:, None], y)
    carf = AnchoredResidualFusion().fit(P, y)
    return lambda Q: platt.predict_proba(carf.decision_function(Q)[:, None])[:, 1]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def all_predictions(P, ytr, Q, seed):
    models = fit_fusions(P, ytr)
    pred = predict_fusions(models, Q)
    for name, m in offset_models(P, ytr).items():
        pred[name] = m.predict_proba(Q)[:, 1]
    pred['carf_platt'] = platt_carf(P, ytr, seed)(Q)
    return models, pred


def main_comparison(y):
    rows = []
    for fold in range(25):
        P, Q, ytr, yte = load(f'fold_{fold}', y)
        _, pred = all_predictions(P, ytr, Q, SEED + fold)
        rows += [{'fold': fold, 'model': k, **scores(yte, v), 'ece': ece(yte, v)} for k, v in pred.items()]
    internal = pd.DataFrame(rows)
    internal.to_csv(OUT / 'internal.csv', index=False)

    P, Q, ytr, ye = load('external', y)
    models, pred = all_predictions(P, ytr, Q, SEED + 100)
    ext = pd.DataFrame([{'model': k, **scores(ye, v), 'ece': ece(ye, v)} for k, v in pred.items()])
    ext.to_csv(OUT / 'external.csv', index=False)
    pd.DataFrame({'pcr': ye, **pred}).to_csv(OUT / 'external_predictions.csv', index=False)

    offs = offset_models(P, ytr)
    platt = platt_carf(P, ytr, SEED + 100)
    stress = []
    for scenario, rep, q in stress_scenarios(Q):
        p = predict_fusions(models, q)
        for name, m in offs.items():
            p[name] = m.predict_proba(q)[:, 1]
        p['carf_platt'] = platt(q)
        a = logit(np.clip(q[:, 1], EPS, 1 - EPS))
        for name, v in p.items():
            stress.append({'scenario': scenario, 'repeat': rep, 'model': name,
                           'auroc': roc_auc_score(ye, v),
                           'max_disp': float(np.abs(logit(np.clip(v, EPS, 1 - EPS)) - a).max())})
    pd.DataFrame(stress).to_csv(OUT / 'stress.csv', index=False)

    rng = np.random.default_rng(SEED)
    boot = []
    for b in range(2000):
        ix = rng.integers(0, len(ye), len(ye))
        if len(np.unique(ye[ix])) < 2:
            continue
        base = roc_auc_score(ye[ix], pred['carf'][ix])
        for comp in ['clinical', 'stack', 'offset_lr', 'offset_lr_masked', 'carf_no_robust']:
            boot.append({'comparator': comp, 'delta': base - roc_auc_score(ye[ix], pred[comp][ix])})
    boot = pd.DataFrame(boot).groupby('comparator').delta.quantile([.025, .5, .975]).unstack()
    boot.to_csv(OUT / 'paired_bootstrap.csv')

    gates = {name: {'coef': m.coef_.tolist(), 'budgets': m.modality_budgets().tolist()}
             for name, m in models.items() if name.startswith('carf') and m.bounded}
    gates['carf_no_robust_raw_coef_sum'] = float(models['carf_no_robust'].coef_.sum())
    gates['offset_lr'] = offs['offset_lr'].theta_.tolist()
    gates['offset_lr_masked'] = offs['offset_lr_masked'].theta_.tolist()
    (OUT / 'gates.json').write_text(json.dumps(gates, indent=2))


def sweep_one(B, g, y, folds):
    rec = {'budget': B, 'robust_weight': g}
    aucs = []
    for fold in folds:
        P, Q, ytr, yte = load(f'fold_{fold}', y)
        aucs.append(roc_auc_score(yte, AnchoredResidualFusion(budget=B, robust_weight=g).fit(P, ytr).predict_proba(Q)[:, 1]))
    rec['internal_auroc'], rec['internal_sd'] = np.mean(aucs), np.std(aucs, ddof=1)
    P, Q, ytr, ye = load('external', y)
    m = AnchoredResidualFusion(budget=B, robust_weight=g).fit(P, ytr)
    rec['external_auroc'] = roc_auc_score(ye, m.predict_proba(Q)[:, 1])
    bud = m.modality_budgets()
    rec.update(rna_budget=bud[0], mri_budget=bud[1], coef_sum=m.coef_.sum())
    per = {}
    for scenario, _, q in stress_scenarios(Q):
        per.setdefault(scenario, []).append(roc_auc_score(ye, m.predict_proba(q)[:, 1]))
    rec.update({f'ext_{k}': np.mean(v) for k, v in per.items()})
    return rec


def sweep(y):
    grid = list(itertools.product([.25, .5, 1, 2, 4], [0, .1, .25, .5, 1]))
    recs = Parallel(n_jobs=len(grid))(delayed(sweep_one)(B, g, y, range(25)) for B, g in grid)
    pd.DataFrame(recs).to_csv(OUT / 'sweep.csv', index=False)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    y = labels()
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'main'):
        main_comparison(y)
    if which in ('all', 'sweep'):
        sweep(y)
