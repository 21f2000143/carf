import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from src.anchored_fusion import AnchoredResidualFusion
from scripts.run_revision2_analysis import TopVariance, oof, rank_normalize


class AnchoredFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        cls.P = expit(rng.normal(size=(160, 3)))
        cls.y = (rng.random(160) < expit(logit(cls.P[:, 1]) +
                 2 * (cls.P[:, 2] - .5))).astype(int)
        cls.model = AnchoredResidualFusion().fit(cls.P, cls.y)

    def test_learns_and_clones(self):
        self.assertGreater(self.model.coef_.sum(), 0)
        self.assertTrue(self.model.optimization_['success'])
        other = clone(self.model).fit(self.P, self.y)
        np.testing.assert_allclose(other.coef_, self.model.coef_)

    def test_missing_fallback_and_row_locality(self):
        P = self.P.copy()
        P[:, [0, 2]] = np.nan
        np.testing.assert_allclose(self.model.predict_proba(P)[:, 1], P[:, 1], atol=1e-15)
        np.testing.assert_allclose(self.model.predict_proba(self.P[:1]),
                                   self.model.predict_proba(self.P)[:1])

    def test_displacement_and_single_modality_replacement_bounds(self):
        rng = np.random.default_rng(71)
        P = rng.uniform(0, 1, (1000, 3))
        base = logit(P[:, 1])
        score = self.model.decision_function(P)
        self.assertLessEqual(np.max(np.abs(score - base)), 1 + 1e-12)
        for slot, j in enumerate([0, 2]):
            q = P.copy()
            q[:, j] = rng.choice([0., 1., np.nan], len(q))
            difference = np.abs(score - self.model.decision_function(q))
            self.assertLessEqual(difference.max(), 2 * self.model.modality_budgets()[slot] + 1e-12)

    def test_certified_anchor_order_survives_extreme_optional_inputs(self):
        # Reverse optional evidence while placing anchors more than 2B apart.
        P = np.array([[1., expit(-2.), 1.], [0., expit(2.), 0.]])
        scores = self.model.decision_function(P)
        self.assertLess(scores[0], scores[1])

    def test_input_validation(self):
        P = self.P.copy()
        P[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'anchor'):
            self.model.predict_proba(P)
        P[0, 1] = 1.5
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            self.model.predict_proba(P)
        with self.assertRaisesRegex(ValueError, 'binary'):
            AnchoredResidualFusion().fit(self.P, np.zeros(len(self.P)))

    def test_column_permutation_equivariance(self):
        perm = [2, 0, 1]
        model = AnchoredResidualFusion(anchor=2).fit(self.P[:, perm], self.y)
        np.testing.assert_allclose(model.predict_proba(self.P[:, perm]),
                                   self.model.predict_proba(self.P), atol=2e-6)

    def test_partial_expression_missingness_preserves_observed_ranks(self):
        X = np.array([[4., np.nan, 2., 2.], [np.nan, np.nan, np.nan, np.nan]])
        ranked = rank_normalize(X)
        np.testing.assert_allclose(ranked[0, [0, 2, 3]], [1., .5, .5])
        self.assertTrue(np.isnan(ranked[0, 1]))
        self.assertTrue(np.isnan(ranked[1]).all())
        np.testing.assert_allclose(rank_normalize(X[:1]), ranked[:1])

    def test_variance_selection_uses_training_only(self):
        train = np.array([[0, 0], [1, 10], [2, 20]], dtype=float)
        selection = TopVariance(k=1).fit(train)
        np.testing.assert_array_equal(selection.indices_, [1])
        selection.transform(np.array([[1e9, 0]]))
        np.testing.assert_array_equal(selection.indices_, [1])

    def test_oof_prediction_never_fits_held_out_patients(self):
        X = pd.DataFrame({'x': np.arange(30)}, index=np.arange(30))
        y = np.tile([0, 1], 15)
        def fit(frame, labels, kind, seed):
            return set(frame.index)
        def predict(model, frame, kind):
            self.assertFalse(model.intersection(frame.index))
            return np.full(len(frame), .5)
        with patch('scripts.run_revision2_analysis.fit_expert', side_effect=fit), \
             patch('scripts.run_revision2_analysis.predict_expert', side_effect=predict):
            np.testing.assert_array_equal(oof(X, X, X, y), np.full((30, 3), .5))


class RppaIsolationTests(unittest.TestCase):
    def test_rppa_heldout_probabilities_come_from_training_patients(self):
        from scripts import run_revision3_analysis as analysis
        rng = np.random.default_rng(21)
        ids = pd.Index([str(i) for i in range(30)])
        E = pd.DataFrame(rng.normal(size=(30, 4)), index=ids)
        M = pd.DataFrame({'pcr': np.tile([0, 1], 15)}, index=ids)
        I = E.copy()
        protein = E.copy()
        heldout_calls = []
        def fit_experts(Etrain, Mtrain, Itrain, labels, *args):
            return [set(Etrain.index)] * 3
        def predict_expert(model, frame, kind):
            self.assertFalse(model.intersection(frame.index))
            self.assertEqual(len(model), 24)
            heldout_calls.append(set(frame.index))
            return np.linspace(.2, .8, len(frame))
        def fake_oof(Etrain, Mtrain, Itrain, labels, *args):
            self.assertEqual(len(labels), 24)
            return rng.uniform(.1, .9, (len(labels), 3))
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(analysis, 'OUT', Path(folder)), \
             patch.object(analysis.pd, 'read_csv', return_value=protein), \
             patch.object(analysis, 'experts', side_effect=fit_experts), \
             patch.object(analysis, 'predict_expert', side_effect=predict_expert), \
             patch.object(analysis, 'oof', side_effect=fake_oof):
            analysis.rppa_extension(E, M, I, M.pcr.to_numpy())
        self.assertEqual(len(heldout_calls), 15)
        self.assertEqual(set.union(*heldout_calls), set(ids))


if __name__ == '__main__':
    unittest.main()
