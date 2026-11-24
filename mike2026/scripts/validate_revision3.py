#!/usr/bin/env python3
"""Audit completed experiment artifacts against source hashes and saved predictions."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import logit
from src.anchored_fusion import AnchoredResidualFusion
from scripts.run_revision3_analysis import OUT, provenance, scores, stress_scenarios


def validate():
    done = json.loads((OUT / 'COMPLETE.json').read_text())
    meta = json.loads((OUT / 'provenance.json').read_text())
    assert done['signature'] == meta['cache_signature']
    assert meta['hashes'] == provenance(), 'Source/input hashes changed after analysis'
    assert done['internal_folds'] == 25, 'Paper requires five repeats of five folds'
    cache = OUT / 'cache' / done['signature']
    # Preserve ULP-level differences/ties: default CSV parsing can change rank metrics.
    internal = pd.read_csv(OUT / 'internal_predictions.csv', float_precision='round_trip', dtype={'patient_id': str})
    cv = pd.read_csv(OUT / 'internal_cv.csv')
    external = pd.read_csv(OUT / 'external_predictions.csv', float_precision='round_trip', dtype={'patient_id': str})
    ext_metrics = pd.read_csv(OUT / 'external_metrics.csv').set_index('model')
    model_names = ext_metrics.index.tolist()
    for repeat, frame in internal.groupby('repeat'):
        assert frame.patient_id.is_unique and len(frame) == 976
    for fold, frame in internal.groupby('fold'):
        with np.load(cache / f'fold_{fold}.npz') as z:
            train_ids, test_ids = set(z['train_ids']), set(z['test_ids'])
            assert not train_ids.intersection(test_ids)
            assert set(frame.patient_id) == test_ids
            assert len(train_ids | test_ids) == 976
        for name in model_names:
            recorded = cv[(cv.fold == fold) & (cv.model == name)].iloc[0]
            for metric, value in scores(frame.pcr.to_numpy(), frame[name].to_numpy()).items():
                np.testing.assert_allclose(value, recorded[metric], atol=1e-12)
    assert external.patient_id.is_unique and len(external) == 112
    for name in model_names:
        assert external[name].between(0, 1).all()
        for metric, value in scores(external.pcr.to_numpy(), external[name].to_numpy()).items():
            np.testing.assert_allclose(value, ext_metrics.loc[name, metric], atol=1e-12)
    with np.load(cache / 'external.npz') as z:
        Q = z['Q']
        assert np.array_equal(z['test_ids'], external.patient_id.to_numpy())
    assert np.ptp(Q[:, 0]) > 1e-6, 'RNA expert collapsed to constant external predictions'
    gates = json.loads((OUT / 'fitted_gates.json').read_text())
    config = gates['carf']
    model = AnchoredResidualFusion(**config['settings'])
    model.coef_ = np.asarray(config['coef'])
    model.n_features_in_ = 3
    model.classes_ = np.array([0, 1])
    np.testing.assert_allclose(model.predict_proba(Q)[:, 1], external.carf, atol=1e-12)
    total = model.modality_budgets().sum()
    stress = pd.read_csv(OUT / 'stress.csv')
    for scenario, repeat, q in stress_scenarios(Q):
        p = model.predict_proba(q)[:, 1]
        displacement = np.abs(model.decision_function(q) - logit(np.clip(q[:, 1], 1e-7, 1 - 1e-7)))
        assert displacement.max() <= total + 1e-10
        row = stress[(stress.scenario == scenario) & (stress.repeat == repeat) & (stress.model == 'carf')].iloc[0]
        for metric, value in scores(external.pcr.to_numpy(), p).items():
            np.testing.assert_allclose(value, row[metric], atol=1e-12)
        if scenario == 'both_missing':
            np.testing.assert_allclose(p, q[:, 1], atol=1e-12)
    boot = pd.read_csv(OUT / 'paired_bootstrap.csv')
    assert boot.groupby('comparator').size().eq(2000).all()
    for name, frame in boot.groupby('comparator'):
        sample = frame[frame.bootstrap == 0].iloc[0]
        ix = np.random.default_rng(meta['seed']).integers(0, len(external), len(external))
        from sklearn.metrics import roc_auc_score
        delta = roc_auc_score(external.pcr.iloc[ix], external.carf.iloc[ix]) - roc_auc_score(external.pcr.iloc[ix], external[name].iloc[ix])
        np.testing.assert_allclose(sample.delta_auroc, delta, atol=1e-12)
    rppa = pd.read_csv(OUT / 'rppa_internal.csv')
    assert len(rppa) == 10 and rppa.groupby('model').n.sum().eq(728).all()
    budgets = pd.read_csv(OUT / 'internal_budgets.csv')
    assert budgets.success.all() and len(budgets) == 25
    print('PASS: hashes, all 25 folds, patient separation, 13 external models,')
    print('all CARF stress metrics/bounds, fallback, bootstrap samples, and RPPA counts.')


if __name__ == '__main__':
    validate()
