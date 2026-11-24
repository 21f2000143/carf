"""Convex, bounded clinical-anchored residual fusion of OOF expert probabilities."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class AnchoredResidualFusion(ClassifierMixin, BaseEstimator):
    """An always-available anchor plus optional, mask-aware bounded residuals.

    Fit on out-of-fold expert probabilities, never in-sample expert predictions.
    Input columns are arbitrary modalities; ``anchor`` selects the mandatory one.
    NaN optional probabilities mean missing. The confidence basis is heuristic.
    """

    def __init__(self, anchor=1, budget=1.0, robust_weight=0.25,
                 temperature=0.05, l2=0.01, context=True, bounded=True):
        self.anchor = anchor
        self.budget = budget
        self.robust_weight = robust_weight
        self.temperature = temperature
        self.l2 = l2
        self.context = context
        self.bounded = bounded

    def _validate(self, P):
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[1] < 2 or not 0 <= self.anchor < P.shape[1]:
            raise ValueError('Expected at least two expert columns and a valid anchor')
        if hasattr(self, 'n_features_in_') and P.shape[1] != self.n_features_in_:
            raise ValueError('Expert column count changed')
        if not np.isfinite(P[:, self.anchor]).all():
            raise ValueError('Clinical anchor must be available and finite')
        observed = P[~np.isnan(P)]
        if not np.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
            raise ValueError('Expert probabilities must lie in [0, 1]')
        return P

    def _design(self, P):
        P = self._validate(P)
        a = logit(np.clip(P[:, self.anchor], 1e-7, 1 - 1e-7))
        columns = []
        for j in range(P.shape[1]):
            if j == self.anchor:
                continue
            present = np.isfinite(P[:, j])
            p = np.clip(np.where(present, P[:, j], 0.5), 1e-7, 1 - 1e-7)
            d = logit(p) - a
            residual = np.clip(d, -self.budget, self.budget) if self.bounded else d
            residual = np.where(present, residual, 0.)
            columns.append(residual)
            if self.context:
                confidence = 1 + (p * np.log(p) + (1 - p) * np.log1p(-p)) / np.log(2)
                columns.extend([residual * np.exp(-np.abs(d)), residual * confidence])
        return a, np.column_stack(columns)

    def fit(self, P, y):
        # Allow refitting the estimator on a different number of modalities.
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_
        P = self._validate(P)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(P),) or set(np.unique(y)) != {0., 1.}:
            raise ValueError('Both binary classes are required')
        if self.budget <= 0 or self.temperature <= 0 or self.l2 < 0 or self.robust_weight < 0:
            raise ValueError('Invalid objective hyperparameters')
        self.n_features_in_ = P.shape[1]
        self.classes_ = np.array([0, 1])
        optional = [j for j in range(P.shape[1]) if j != self.anchor]
        if len(optional) > 8:
            raise ValueError('Exact mask enumeration supports at most eight optional experts')
        masks, corruptions = [], []
        for bits in range(2 ** len(optional)):
            q = P.copy()
            for k, j in enumerate(optional):
                if not bits & (1 << k):
                    q[:, j] = np.nan
            masks.append(self._design(q)[1])
        for j in optional:
            for value in (0.01, 0.99):
                q = P.copy()
                q[:, j] = np.where(np.isfinite(P[:, j]), value, np.nan)
                corruptions.append(self._design(q)[1])
        a, design = self._design(P)
        masks, corruptions = np.asarray(masks), np.asarray(corruptions)

        def loss_grad(theta, X):
            z = a[None, :] + X @ theta
            loss = np.mean(np.logaddexp(0, z) - y[None, :] * z, axis=1)
            grad = np.einsum('en,end->ed', expit(z) - y[None, :], X) / len(y)
            return loss, grad

        def objective(theta):
            losses, gradients = loss_grad(theta, masks)
            bad_losses, bad_gradients = loss_grad(theta, corruptions)
            scale = bad_losses / self.temperature
            weights = np.exp(scale - logsumexp(scale))
            smooth_max = self.temperature * (logsumexp(scale) - np.log(len(scale)))
            value = losses.mean() + self.robust_weight * smooth_max + self.l2 * (theta @ theta) / 2
            gradient = gradients.mean(axis=0) + self.robust_weight * (weights @ bad_gradients) + self.l2 * theta
            return value, gradient

        constraints = []
        if self.bounded:
            constraints = [{'type': 'ineq', 'fun': lambda w: 1 - w.sum(),
                            'jac': lambda w: -np.ones_like(w)}]
        result = minimize(objective, np.zeros(design.shape[1]), jac=True,
                          method='SLSQP', bounds=[(0, None)] * design.shape[1],
                          constraints=constraints, options={'ftol': 1e-11, 'maxiter': 500})
        if not result.success:
            raise RuntimeError(f'Fusion optimization failed: {result.message}')
        self.coef_ = np.maximum(result.x, 0.)
        if self.bounded and self.coef_.sum() > 1:
            self.coef_ /= self.coef_.sum()
        self.optimization_ = {'success': bool(result.success), 'iterations': int(result.nit),
                              'objective': float(objective(self.coef_)[0])}
        return self

    def decision_function(self, P):
        check_is_fitted(self, 'coef_')
        a, X = self._design(P)
        return a + X @ self.coef_

    def predict_proba(self, P):
        p = expit(self.decision_function(P))
        return np.column_stack([1 - p, p])

    def predict(self, P):
        return (self.predict_proba(P)[:, 1] >= 0.5).astype(int)

    def modality_budgets(self):
        """Per-optional-modality logit displacement budgets, not probabilities."""
        check_is_fitted(self, 'coef_')
        if not self.bounded:
            raise ValueError('No finite residual guarantee in the unbounded ablation')
        width = 3 if self.context else 1
        return self.budget * self.coef_.reshape(-1, width).sum(axis=1)
