#!/usr/bin/env python3
"""E5 task 1: robust offset LR -- OffsetLR's functional form trained on CARF's objective.

z = a + b + sum_j m_j w_j logit(p_j)  (free intercept b, unconstrained w, no clip/budget)
objective = mean_{4 masks} logloss + gamma * tau * (LSE(L_e / tau) - log 4) + l2/2 ||w||^2
over the 4 CARF corruption environments (one optional expert set to 0.01 / 0.99 where
observed); gamma=0.25, tau=0.05, l2=0.01 (b unpenalized). L-BFGS-B, analytic gradient.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize, check_grad
from scipy.special import expit, logit, logsumexp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1 import (OffsetLR, load, labels, ece, offset_models, SEED, EPS, OUT as RES,  # noqa: E402
                   stress_scenarios, scores)
from scripts.run_revision3_analysis import fit_fusions, predict_fusions  # noqa: E402
from e1_bootstrap import auc_rows, summarize, NBOOT  # noqa: E402

OUT = RES / 'e5'


class RobustOffsetLR(OffsetLR):
    def __init__(self, l2=0.01, robust_weight=0.25, temperature=0.05):
        super().__init__(l2=l2, masked=True, intercept=True)
        self.robust_weight, self.temperature = robust_weight, temperature

    def _envs(self, P):
        masks, bad = [], []
        for keep in itertools.product([False, True], repeat=2):
            q = P.copy()
            for j, k in zip((0, 2), keep):
                if not k:
                    q[:, j] = np.nan
            masks.append(self._design(q))
        for j in (0, 2):
            for value in (0.01, 0.99):
                q = P.copy()
                q[:, j] = np.where(np.isfinite(P[:, j]), value, np.nan)
                bad.append(self._design(q))
        return masks, bad

    def objective(self, theta, masks, bad, y):
        b, w = theta[0], theta[1:]

        def lg(env):
            a, X = env
            z = a + b + X @ w
            r = (expit(z) - y) / len(y)
            return np.mean(np.logaddexp(0, z) - y * z), np.r_[r.sum(), X.T @ r]

        Lm = [lg(e) for e in masks]
        Lb = [lg(e) for e in bad]
        mval = np.mean([v for v, _ in Lm]); mgrad = np.mean([g for _, g in Lm], axis=0)
        s = np.array([v for v, _ in Lb]) / self.temperature
        wts = np.exp(s - logsumexp(s))
        smooth = self.temperature * (logsumexp(s) - np.log(len(s)))
        bgrad = wts @ np.array([g for _, g in Lb])
        pen = np.r_[0., w]
        value = mval + self.robust_weight * smooth + self.l2 * w @ w / 2
        return value, mgrad + self.robust_weight * bgrad + self.l2 * pen

    def fit(self, P, y):
        y = np.asarray(y, float)
        masks, bad = self._envs(P)
        f = lambda t: self.objective(t, masks, bad, y)
        res = minimize(f, np.zeros(3), jac=True, method='L-BFGS-B',
                       options={'maxiter': 2000, 'gtol': 1e-9, 'ftol': 1e-14})
        self.theta_, self.opt_ = res.x, res
        self.grad_err_ = check_grad(lambda t: f(t)[0], lambda t: f(t)[1], res.x + 0.1)
        return self


def fold_job(fold, y):
    P, Q, ytr, yte = load(f'fold_{fold}', y)
    rows = []
    for name, g in [('robust_offset_lr', .25), ('robust_offset_lr_g1', 1.)]:
        m = RobustOffsetLR(robust_weight=g).fit(P, ytr)
        rows.append({'fold': fold, 'repeat': fold // 5, 'model': name, **scores(yte, m.predict_proba(Q)[:, 1])})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y = labels()
    # --- internal
    rows = sum([fold_job(f, y) for f in range(25)], [])
    internal = pd.DataFrame(rows)
    ref = pd.read_csv(RES / 'internal.csv')
    ref = ref[ref.model.isin(['carf', 'offset_lr', 'offset_lr_masked', 'carf_no_robust', 'clinical'])]
    ref = ref.assign(repeat=ref.fold // 5)
    allint = pd.concat([internal, ref[internal.columns.intersection(ref.columns)]], ignore_index=True)
    allint.to_csv(OUT / 'robust_offset_internal_folds.csv', index=False)
    summ = []
    for m, d in allint.groupby('model'):
        summ.append({'model': m, 'auroc_mean': d.auroc.mean(), 'sd_over_folds': d.auroc.std(ddof=1),
                     'sd_over_repeats': d.groupby('repeat').auroc.mean().std(ddof=1), 'n_folds': len(d)})
    pd.DataFrame(summ).to_csv(OUT / 'robust_offset_internal.csv', index=False)
    print(pd.DataFrame(summ).round(4).to_string(index=False))

    # --- external
    P, Q, ytr, ye = load('external', y)
    fus = fit_fusions(P, ytr)
    off = offset_models(P, ytr)
    rob = {'robust_offset_lr': RobustOffsetLR(robust_weight=.25).fit(P, ytr),
           'robust_offset_lr_g1': RobustOffsetLR(robust_weight=1.).fit(P, ytr)}

    def predict_all(q):
        p = predict_fusions(fus, q)
        p = {k: p[k] for k in ['clinical', 'carf', 'carf_no_robust', 'stack', 'mask_stack']}
        p.update({k: m.predict_proba(q)[:, 1] for k, m in {**off, **rob}.items()})
        return p

    pred = predict_all(Q)
    ext = pd.DataFrame([{'model': k, **scores(ye, v), 'ece': ece(ye, v)} for k, v in pred.items()])
    ext.to_csv(OUT / 'robust_offset_external.csv', index=False)
    print(ext.round(4).to_string(index=False))

    # --- output-level stress
    stress, reals = [], {'intact': [], 'rna_permuted': [], 'mri_permuted': []}
    models = ['carf', 'robust_offset_lr', 'robust_offset_lr_g1', 'offset_lr', 'offset_lr_masked']
    for sc, rep, q in stress_scenarios(Q):
        p = predict_all(q)
        a = logit(np.clip(q[:, 1], EPS, 1 - EPS))
        for k, v in p.items():
            stress.append({'scenario': sc, 'repeat': rep, 'model': k, 'auroc': roc_auc_score(ye, v),
                           'max_disp': float(np.abs(logit(np.clip(v, EPS, 1 - EPS)) - a).max())})
        if sc in reals:
            reals[sc].append(np.column_stack([p[m] for m in models]))
    stress = pd.DataFrame(stress)
    stress.to_csv(OUT / 'robust_offset_stress_raw.csv', index=False)
    order = list(dict.fromkeys(stress.scenario))
    ss = (stress.groupby(['scenario', 'model'])
          .agg(auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'), max_disp=('max_disp', 'max'),
               n=('auroc', 'size')).reset_index())
    ss['scenario'] = pd.Categorical(ss.scenario, order, ordered=True)
    ss = ss.sort_values(['scenario', 'model'])
    ss.to_csv(OUT / 'robust_offset_stress.csv', index=False)
    print(ss.pivot(index='scenario', columns='model', values='auroc_mean')[models].round(4).to_string())
    print(ss.pivot(index='scenario', columns='model', values='max_disp')[models].round(2).to_string())
    # consistency with tier1 saved stress for carf / offset models
    st = pd.read_csv(RES / 'stress.csv').groupby(['scenario', 'model']).auroc.mean()
    for m in ['carf', 'offset_lr', 'offset_lr_masked']:
        for sc in ['rna_permuted', 'mri_permuted', 'intact']:
            v = stress.query('scenario == @sc and model == @m').auroc.mean()
            assert abs(v - st[(sc, m)]) < 2e-3, (m, sc, v, st[(sc, m)])

    # --- paired bootstrap (e1 protocol), CARF minus comparator
    comps = models[1:]
    brows = []
    for name, R in reals.items():
        A = np.stack(R)
        point_auc = np.mean([auc_rows(ye, a) for a in A], axis=0)
        point = {m: point_auc[0] - point_auc[k] for k, m in enumerate(models) if k}
        rng = np.random.default_rng(SEED)
        boot = {m: [] for m in comps}
        for _ in range(NBOOT):
            ix = rng.integers(0, len(ye), len(ye))
            if len(np.unique(ye[ix])) < 2:
                continue
            auc = np.mean([auc_rows(ye[ix], a[ix]) for a in A], axis=0)
            for k, m in enumerate(comps, 1):
                boot[m].append(auc[0] - auc[k])
        brows += summarize(boot, point, {'cohort': 'external_112', 'scenario': name, 'n_real': len(A),
                                         'carf_auroc': point_auc[0]})
    bs = pd.DataFrame(brows)
    bs.to_csv(OUT / 'robust_offset_bootstrap.csv', index=False)
    print(bs.round(4).to_string(index=False))

    wj = {k: {'intercept_b': float(m.theta_[0]), 'w_rna': float(m.theta_[1]), 'w_mri': float(m.theta_[2])}
          for k, m in {**rob, **off}.items()}
    for k, m in rob.items():
        wj[k].update(converged=bool(m.opt_.success), objective=float(m.opt_.fun), grad_check_err=float(m.grad_err_))
    wj['carf_coef'] = fus['carf'].coef_.tolist()
    wj['carf_budgets'] = fus['carf'].modality_budgets().tolist()
    (OUT / 'weights.json').write_text(json.dumps(wj, indent=2))
    print(json.dumps(wj, indent=2))


if __name__ == '__main__':
    main()
