#!/usr/bin/env python3
"""A7: learned neural fusion baselines (MLP gate, ModDrop MLP) on cached expert
probabilities. Same features as mask_stack; grid chosen by inner 5-fold CV log
loss on the training OOF probabilities P only."""
import itertools
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from tier1 import EPS, OUT, ece, labels, load, scores, stress_scenarios, SEED
from scripts.run_revision3_analysis import mask_stack_features  # noqa: E402 (path set by tier1)

A7 = OUT / 'a7'
GRID = [(h, a) for h in [(16,), (32, 16)] for a in [1e-3, 1e-2]]
MODELS = ['mlp_gate', 'moddrop_mlp']


def masked_blocks(P, y):
    """Replicate P under all four RNA/MRI availability masks."""
    X = []
    for keep in itertools.product([False, True], repeat=2):
        q = P.copy()
        for j, k in zip((0, 2), keep):
            if not k:
                q[:, j] = np.nan
        X.append(mask_stack_features(q))
    return np.vstack(X), np.tile(y, 4)


def design(P, y, model):
    return masked_blocks(P, y) if model == 'moddrop_mlp' else (mask_stack_features(P), y)


def mlp(h, a):
    return MLPClassifier(hidden_layer_sizes=h, alpha=a, early_stopping=False,
                         max_iter=2000, random_state=SEED)


def fit(P, y, model):
    """Select (hidden, alpha) by inner CV log loss on P, refit on all of P.
    ModDrop validation loss is computed on the masked-augmented inner fold."""
    losses = []
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for h, a in GRID:
        ll = []
        for tr, va in skf.split(P, y):
            Xt, yt = design(P[tr], y[tr], model)
            Xv, yv = design(P[va], y[va], model)
            ll.append(log_loss(yv, mlp(h, a).fit(Xt, yt).predict_proba(Xv)[:, 1], labels=[0, 1]))
        losses.append(np.mean(ll))
    h, a = GRID[int(np.argmin(losses))]
    X, yy = design(P, y, model)
    return mlp(h, a).fit(X, yy), {'hidden': str(h), 'alpha': a, 'inner_logloss': min(losses)}


def predict(m, Q):
    return m.predict_proba(mask_stack_features(Q))[:, 1]


def run_split(key, model, y):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    P, Q, ytr, yte = load(key, y)
    m, params = fit(P, ytr, model)
    out = {'key': key, 'model': model, 'params': params, 'pred': predict(m, Q)}
    if key == 'external':
        stress = []
        for scenario, rep, q in stress_scenarios(Q):
            v = predict(m, q)
            a = logit(np.clip(q[:, 1], EPS, 1 - EPS))
            stress.append({'scenario': scenario, 'repeat': rep, 'model': model,
                           'auroc': roc_auc_score(yte, v),
                           'max_disp': float(np.abs(logit(np.clip(v, EPS, 1 - EPS)) - a).max())})
        out['stress'] = stress
    return out


def main():
    A7.mkdir(parents=True, exist_ok=True)
    y = labels()
    keys = [f'fold_{k}' for k in range(25)] + ['external']
    res = Parallel(n_jobs=30)(delayed(run_split)(k, m, y) for k in keys for m in MODELS)

    internal, chosen, ext_rows = [], [], []
    ext_pred = {}
    for r in res:
        _, _, _, yte = load(r['key'], y)
        chosen.append({'split': r['key'], 'model': r['model'], **r['params']})
        row = {'model': r['model'], **scores(yte, r['pred']), 'ece': ece(yte, r['pred'])}
        if r['key'] == 'external':
            ext_pred[r['model']] = r['pred']
            ext_rows.append(row)
        else:
            internal.append({'fold': int(r['key'].split('_')[1]), **row})
    pd.DataFrame(internal).sort_values(['model', 'fold']).to_csv(A7 / 'internal.csv', index=False)
    pd.DataFrame(ext_rows).to_csv(A7 / 'external.csv', index=False)
    pd.DataFrame(chosen).to_csv(A7 / 'chosen_params.csv', index=False)
    pd.DataFrame([s for r in res if 'stress' in r for s in r['stress']]).to_csv(A7 / 'stress.csv', index=False)

    ep = pd.read_csv(OUT / 'external_predictions.csv')
    ye, carf = ep.pcr.to_numpy(), ep.carf.to_numpy()
    _, _, _, ye2 = load('external', y)
    assert np.array_equal(ye, ye2)
    pd.DataFrame({'pcr': ye, 'carf': carf, **ext_pred}).to_csv(A7 / 'external_predictions.csv', index=False)
    rng = np.random.default_rng(SEED)
    boot = []
    for b in range(2000):
        ix = rng.integers(0, len(ye), len(ye))
        if len(np.unique(ye[ix])) < 2:
            continue
        base = roc_auc_score(ye[ix], carf[ix])
        for comp in MODELS:
            boot.append({'comparator': comp, 'delta': base - roc_auc_score(ye[ix], ext_pred[comp][ix])})
    boot = pd.DataFrame(boot)
    point = {c: roc_auc_score(ye, carf) - roc_auc_score(ye, ext_pred[c]) for c in MODELS}
    summ = boot.groupby('comparator').delta.quantile([.025, .5, .975]).unstack()
    summ['point'] = pd.Series(point)
    summ['p_le_0'] = boot.groupby('comparator').delta.apply(lambda d: (d <= 0).mean())
    summ.to_csv(A7 / 'bootstrap.csv')


if __name__ == '__main__':
    import a7_moddrop  # run via the module so joblib pickles functions by reference
    a7_moddrop.main()
