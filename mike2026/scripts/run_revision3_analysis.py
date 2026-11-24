#!/usr/bin/env python3
"""Matched CARF comparisons; resumable fold caches keyed to code and input hashes."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.anchored_fusion import AnchoredResidualFusion
from scripts.run_revision2_analysis import (common_data, experts, predict_expert,
                                           oof, metrics, DATA)

SEED = 20260923
OUT = ROOT / 'results/revision3'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def provenance():
    files = [ROOT / 'src/anchored_fusion.py', Path(__file__),
             ROOT / 'scripts/run_revision2_analysis.py', ROOT / 'revision3/ANALYSIS_PLAN.md']
    files += sorted(DATA.glob('*_meta.csv')) + sorted(DATA.glob('*_expression.csv.gz'))
    files += sorted(DATA.glob('*_mri.csv'))
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def scores(y, p):
    return {**metrics(y, p), 'log_loss': log_loss(y, p, labels=[0, 1])}


def mask_stack_features(P):
    return np.column_stack([np.nan_to_num(P, nan=.5), np.isfinite(P).astype(float)])


def fit_fusions(P, y):
    standard = lambda: LogisticRegression(C=1, class_weight='balanced', solver='liblinear')
    models = {'stack': standard().fit(P, y),
              'no_mri_stack': standard().fit(P[:, :2], y)}
    blocks = []
    for active in itertools.product([False, True], repeat=2):
        q = P.copy()
        for j, keep in zip([0, 2], active):
            if not keep:
                q[:, j] = np.nan
        blocks.append(mask_stack_features(q))
    models['mask_stack'] = standard().fit(np.vstack(blocks), np.tile(y, 4),
                                         sample_weight=np.full(len(y) * 4, .25))
    configs = {'carf': {}, 'carf_no_robust': {'robust_weight': 0},
               'carf_no_context': {'context': False}, 'carf_unbounded': {'bounded': False}}
    for name, kwargs in configs.items():
        models[name] = AnchoredResidualFusion(**kwargs).fit(P, y)
    q = P.copy()
    q[:, 2] = np.nan
    models['carf_no_mri'] = AnchoredResidualFusion().fit(q, y)
    return models


def predict_fusions(models, Q):
    filled = np.nan_to_num(Q, nan=.5)
    count = np.isfinite(Q).sum(axis=1)
    predictions = {'rna': filled[:, 0], 'clinical': Q[:, 1], 'mri': filled[:, 2],
                   'mean': np.nansum(Q, axis=1) / count,
                   'stack': models['stack'].predict_proba(filled)[:, 1],
                   'no_mri_stack': models['no_mri_stack'].predict_proba(filled[:, :2])[:, 1],
                   'mask_stack': models['mask_stack'].predict_proba(mask_stack_features(Q))[:, 1]}
    for name, model in models.items():
        if name.startswith('carf'):
            q = Q.copy()
            if name == 'carf_no_mri':
                q[:, 2] = np.nan
            predictions[name] = model.predict_proba(q)[:, 1]
    return predictions


def cached_predictions(cache, key, E, M, I, y, tr, te, external=None):
    path = cache / f'{key}.npz'
    if path.exists():
        with np.load(path) as z:
            return z['P'], z['Q']
    seed = SEED + (int(key.split('_')[-1]) if key.startswith('fold') else 100)
    P = oof(E.iloc[tr], M.iloc[tr], I.iloc[tr], y[tr], seed)
    fitted = experts(E.iloc[tr], M.iloc[tr], I.iloc[tr], y[tr], seed)
    target = [E.iloc[te], M.iloc[te], I.iloc[te]] if external is None else external
    Q = np.column_stack([predict_expert(model, x, kind) for model, x, kind in
                         zip(fitted, target, ['rna', 'clinical', 'mri'])])
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, P=P, Q=Q, train_ids=M.index[tr].to_numpy(dtype=str),
                        test_ids=target[1].index.to_numpy(dtype=str))
    tmp.replace(path)
    return P, Q


def stress_scenarios(Q):
    yield 'intact', 0, Q.copy()
    q = Q.copy()
    q[:, [0, 2]] = np.nan
    yield 'both_missing', 0, q
    for j, modality in [(0, 'rna'), (2, 'mri')]:
        for name, value in [('missing', np.nan), ('neutral', .5), ('low', .01), ('high', .99)]:
            q = Q.copy()
            q[:, j] = value
            yield f'{modality}_{name}', 0, q
        for repeat in range(30):
            rng = np.random.default_rng(SEED + 1000 * j + repeat)
            q = Q.copy()
            q[:, j] = rng.permutation(Q[:, j])
            yield f'{modality}_permuted', repeat, q
            q = Q.copy()
            q[:, j] = np.clip(Q[:, j] + rng.normal(0, .25, len(Q)), .001, .999)
            yield f'{modality}_noisy', repeat, q


def rppa_extension(E, M, I, y):
    rp = pd.read_csv(DATA / 'ISPY2_RPPA_expression.csv.gz', index_col=0)
    rp.index = rp.index.astype(str)
    ids = M.index.intersection(rp.index)
    E, M, I = E.loc[ids], M.loc[ids], I.loc[ids]
    rp = rp.loc[ids].replace([np.inf, -np.inf], np.nan)
    y = M.pcr.to_numpy(dtype=int)
    records = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(E, y)):
        P = oof(E.iloc[tr], M.iloc[tr], I.iloc[tr], y[tr], SEED + fold)
        fitted = experts(E.iloc[tr], M.iloc[tr], I.iloc[tr], y[tr])
        Q = np.column_stack([predict_expert(model, X.iloc[te], kind) for model, X, kind in
                             zip(fitted, [E, M, I], ['rna', 'clinical', 'mri'])])
        # Held-out patients and labels never enter a fit; RPPA is a feature extension.
        base = LogisticRegression(C=1, class_weight='balanced', solver='liblinear').fit(P, y[tr])
        four = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             LogisticRegression(C=.1, class_weight='balanced', solver='liblinear', max_iter=1000))
        four.fit(np.column_stack([P, rp.iloc[tr]]), y[tr])
        for name, pred in [('three_acquisition', base.predict_proba(Q)[:, 1]),
                           ('three_plus_rppa', four.predict_proba(np.column_stack([Q, rp.iloc[te]]))[:, 1])]:
            records.append({'fold': fold, 'model': name, 'n': len(te), **scores(y[te], pred)})
        print(f'RPPA fold {fold + 1}/5 complete', flush=True)
    pd.DataFrame(records).to_csv(OUT / 'rppa_internal.csv', index=False)


def run(repeats):
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = provenance()
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    cache = OUT / 'cache' / signature
    cache.mkdir(parents=True, exist_ok=True)
    (OUT / 'provenance.json').write_text(json.dumps({'hashes': hashes, 'cache_signature': signature,
        'seed': SEED, 'repeats': repeats, 'python': platform.python_version(),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
        'external_status': 'Exploratory reuse; previous external outcomes informed research context'}, indent=2))
    E, M, I, Ee, Me, Ie = common_data()
    y, ye = M.pcr.to_numpy(dtype=int), Me.pcr.to_numpy(dtype=int)
    assert M.index.is_unique and Me.index.is_unique
    rows, patients, budgets = [], [], []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(E, y)):
        P, Q = cached_predictions(cache, f'fold_{fold}', E, M, I, y, tr, te)
        models = fit_fusions(P, y[tr])
        predictions = predict_fusions(models, Q)
        for name, pred in predictions.items():
            rows.append({'fold': fold, 'repeat': fold // 5, 'model': name, **scores(y[te], pred)})
        patients.extend({'patient_id': str(M.index[index]), 'fold': fold, 'repeat': fold // 5,
                          'pcr': int(y[index]), **{name: float(p[k]) for name, p in predictions.items()}}
                         for k, index in enumerate(te))
        budgets.append({'fold': fold, **{f'coef_{i}': float(w) for i, w in enumerate(models['carf'].coef_)},
                        **models['carf'].optimization_})
        pd.DataFrame(rows).to_csv(OUT / 'internal_cv.csv', index=False)
        print(f'Internal fold {fold + 1}/{5 * repeats} complete', flush=True)
    pd.DataFrame(patients).to_csv(OUT / 'internal_predictions.csv', index=False)
    pd.DataFrame(budgets).to_csv(OUT / 'internal_budgets.csv', index=False)
    P, Q = cached_predictions(cache, 'external', E, M, I, y, np.arange(len(y)), None, [Ee, Me, Ie])
    models = fit_fusions(P, y)
    predictions = predict_fusions(models, Q)
    pd.DataFrame([{'model': name, 'n': len(ye), **scores(ye, pred)} for name, pred in predictions.items()]).to_csv(OUT / 'external_metrics.csv', index=False)
    pd.DataFrame({'patient_id': Me.index, 'pcr': ye, **predictions}).to_csv(OUT / 'external_predictions.csv', index=False)
    (OUT / 'fitted_gates.json').write_text(json.dumps({name: {'coef': model.coef_.tolist(),
        'settings': model.get_params(), 'optimization': model.optimization_}
        for name, model in models.items() if name.startswith('carf')}, indent=2))
    stress = []
    for scenario, repeat, q in stress_scenarios(Q):
        for name, pred in predict_fusions(models, q).items():
            z = logit(np.clip(pred, 1e-7, 1 - 1e-7))
            a = logit(np.clip(q[:, 1], 1e-7, 1 - 1e-7))
            stress.append({'scenario': scenario, 'repeat': repeat, 'model': name,
                           'max_anchor_logit_displacement': float(np.abs(z - a).max()), **scores(ye, pred)})
    pd.DataFrame(stress).to_csv(OUT / 'stress.csv', index=False)
    rng = np.random.default_rng(SEED)
    boot = []
    for b in range(2000):
        ix = rng.integers(0, len(ye), len(ye))
        if len(np.unique(ye[ix])) < 2:
            continue
        for comparator in ['clinical', 'stack', 'mask_stack', 'carf_no_mri']:
            boot.append({'bootstrap': b, 'comparator': comparator,
                         'delta_auroc': roc_auc_score(ye[ix], predictions['carf'][ix]) - roc_auc_score(ye[ix], predictions[comparator][ix])})
    pd.DataFrame(boot).to_csv(OUT / 'paired_bootstrap.csv', index=False)
    rppa_extension(E, M, I, y)
    pd.DataFrame({'cohort': ['ISPY2', 'ISPY1'], 'n': [len(y), len(ye)],
                  'positive': [int(y.sum()), int(ye.sum())], 'common_genes': [E.shape[1], Ee.shape[1]],
                  'mri_features': [I.shape[1], Ie.shape[1]]}).to_csv(OUT / 'cohorts.csv', index=False)
    print(pd.read_csv(OUT / 'external_metrics.csv').to_string(index=False), flush=True)
    (OUT / 'COMPLETE.json').write_text(json.dumps({'signature': signature, 'repeats': repeats,
        'internal_folds': 5 * repeats, 'bootstrap_samples': 2000}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args.repeats)
