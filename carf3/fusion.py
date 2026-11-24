"""Fusion baselines and CARF variants on expert probabilities with the anchor in the last column.

All models take P (n x (J+1)) with NaN for missing optional experts and, where used,
ctx (n x J) anchor-consistency scores.
"""
import itertools
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from carf2 import CARF2, select_budget, AnchorContext

EPS = 1e-7
SEED = 20260923


def L(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def mask_variants(P):
    """All 2^J availability patterns of the optional experts (anchor last, always kept)."""
    J = P.shape[1] - 1
    out = []
    for keep in itertools.product([False, True], repeat=J):
        q = P.copy()
        for j, k in enumerate(keep):
            if not k:
                q[:, j] = np.nan
        out.append(q)
    return out


def mask_features(P):
    return np.column_stack([np.nan_to_num(P, nan=.5), np.isfinite(P[:, :-1]).astype(float)])


class Mean:
    def fit(self, P, y, ctx=None):
        return self

    def predict(self, P, ctx=None):
        return np.nanmean(P, axis=1)


class Stack:
    """Logistic regression on the expert probabilities, missing experts filled with 0.5."""
    def fit(self, P, y, ctx=None):
        self.m = LogisticRegression(C=1, class_weight='balanced', solver='liblinear').fit(np.nan_to_num(P, nan=.5), y)
        return self

    def predict(self, P, ctx=None):
        return self.m.predict_proba(np.nan_to_num(P, nan=.5))[:, 1]


class MaskStack:
    """Stacking with availability indicators, trained on all availability patterns."""
    def fit(self, P, y, ctx=None):
        V = mask_variants(P)
        X = np.vstack([mask_features(q) for q in V])
        self.m = LogisticRegression(C=1, class_weight='balanced', solver='liblinear').fit(
            X, np.tile(y, len(V)), sample_weight=np.full(len(X), 1 / len(V)))
        return self

    def predict(self, P, ctx=None):
        return self.m.predict_proba(mask_features(P))[:, 1]


class OffsetLR:
    """z = a + b + sum_j m_j w_j logit(p_j); masked loss over all availability patterns; optional robust term."""
    def __init__(self, l2=0.01, robust_weight=0.0, temperature=0.05):
        self.l2, self.robust_weight, self.temperature = l2, robust_weight, temperature

    @staticmethod
    def _design(P):
        a = L(P[:, -1])
        X = np.column_stack([np.where(np.isfinite(P[:, j]), L(np.nan_to_num(P[:, j], nan=.5)), 0.)
                             for j in range(P.shape[1] - 1)])
        return a, X

    def fit(self, P, y, ctx=None):
        y = np.asarray(y, float)
        envs = [self._design(q) for q in mask_variants(P)]
        bad = []
        for j in range(P.shape[1] - 1):
            for v in (.01, .99):
                q = P.copy(); q[:, j] = np.where(np.isfinite(P[:, j]), v, np.nan)
                bad.append(self._design(q))

        def lg(env, t):
            a, X = env
            z = a + t[0] + X @ t[1:]
            r = (expit(z) - y) / len(y)
            return np.mean(np.logaddexp(0, z) - y * z), np.r_[r.sum(), X.T @ r]

        def f(t):
            vals = [lg(e, t) for e in envs]
            v = np.mean([a for a, _ in vals]); g = np.mean([b for _, b in vals], axis=0)
            if self.robust_weight > 0:
                bv = [lg(e, t) for e in bad]
                s = np.array([a for a, _ in bv]) / self.temperature
                w = np.exp(s - logsumexp(s))
                v += self.robust_weight * self.temperature * (logsumexp(s) - np.log(len(s)))
                g = g + self.robust_weight * (w @ np.array([b for _, b in bv]))
            w_ = t[1:]
            return v + self.l2 * w_ @ w_ / 2, g + self.l2 * np.r_[0., w_]
        self.theta_ = minimize(f, np.zeros(P.shape[1]), jac=True, method='L-BFGS-B',
                               options={'maxiter': 2000}).x
        return self

    def decision(self, P):
        a, X = self._design(P)
        return a + self.theta_[0] + X @ self.theta_[1:]

    def predict(self, P, ctx=None):
        return expit(self.decision(P))


class ModDropMLP:
    """MLP on mask features trained on all availability patterns; grid chosen by inner CV log loss."""
    GRID = [(h, a) for h in [(16,), (32, 16)] for a in [1e-3, 1e-2]]

    @staticmethod
    def _blocks(P, y):
        V = mask_variants(P)
        return np.vstack([mask_features(q) for q in V]), np.tile(y, len(V))

    def fit(self, P, y, ctx=None):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        mk = lambda h, a: MLPClassifier(hidden_layer_sizes=h, alpha=a, max_iter=2000, random_state=SEED)
        losses = []
        for h, a in self.GRID:
            ll = []
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(P, y):
                Xt, yt = self._blocks(P[tr], y[tr]); Xv, yv = self._blocks(P[va], y[va])
                ll.append(log_loss(yv, mk(h, a).fit(Xt, yt).predict_proba(Xv)[:, 1], labels=[0, 1]))
            losses.append(np.mean(ll))
        self.params_ = self.GRID[int(np.argmin(losses))]
        X, yy = self._blocks(P, y)
        self.m = mk(*self.params_).fit(X, yy)
        return self

    def predict(self, P, ctx=None):
        return self.m.predict_proba(mask_features(P))[:, 1]


class CARFWrap:
    """CARF-v1 (original settings) or CARF-2 variants via CARF2 keyword arguments.

    With ``cv_budgets`` the budget B is chosen by inner CV of the held-out robust objective.
    """
    def __init__(self, cv_budgets=None, **kw):
        self.kw, self.cv_budgets = kw, cv_budgets

    def fit(self, P, y, ctx=None):
        kw = dict(self.kw)
        if self.cv_budgets:
            kw.pop('budget', None)
            self.budget_, self.cv_scores_ = select_budget(P, y, self.cv_budgets, ctx=ctx, seed=SEED, **kw)
            kw['budget'] = self.budget_
        self.m = CARF2(**kw).fit(P, y, ctx=ctx)
        return self

    def decision(self, P, ctx=None):
        return self.m.decision_function(P, ctx=ctx)

    def predict(self, P, ctx=None):
        return expit(self.decision(P, ctx))


class Gated:
    """Any fusion model applied to gated experts: each optional probability is shrunk towards the anchor,
    logit p~_j = a + g_j (logit p_j - a), with the anchor-verification gate g_j of CARF."""
    def __init__(self, base, prior=0.95):
        self.base, self.prior = base, prior

    def _tx(self, P, ctx):
        g = ctx.scores('posterior', self.ref_, self.prior)
        a = L(P[:, -1]); out = P.copy()
        for j in range(P.shape[1] - 1):
            m = np.isfinite(P[:, j])
            d = L(np.nan_to_num(P[:, j], nan=.5)) - a
            out[:, j] = np.where(m, expit(a + np.nan_to_num(g[:, j], nan=0.) * d), np.nan)
        return out

    def fit(self, P, y, ctx=None):
        self.ref_ = ctx.aux.copy()  # reference samples for the swap null, as in CARF
        self.base.fit(self._tx(P, ctx), y)
        return self

    def predict(self, P, ctx=None):
        return self.base.predict(self._tx(P, ctx))
