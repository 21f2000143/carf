"""Experts for CARF-2: treatment-aware clinical anchor and anchor-residual optional experts."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA

EPS = 1e-7
BASE_CLIN = ['age', 'hr', 'her2']
TREAT_CLIN = BASE_CLIN + ['anti_her2', 'immunotherapy', 'platinum', 'her2_tki']


def clinical_frame(M):
    """Clinical table with therapy-class indicators shared by I-SPY1 and I-SPY2."""
    X = M[BASE_CLIN].astype(float).copy()
    arm = M.arm.fillna('').astype(str)
    X['anti_her2'] = (arm.str.contains('Trastuzumab') | arm.str.contains('Pertuzumab') |
                      arm.str.contains('T-DM1')).astype(float)
    X['immunotherapy'] = arm.str.contains('Pembrolizumab').astype(float)
    X['platinum'] = arm.str.contains('Carboplatin').astype(float)
    X['her2_tki'] = arm.str.contains('Neratinib').astype(float)
    ispy1 = arm == 'AC_based'
    if ispy1.any():
        s1 = pd.read_csv(DATA / 'ISPY1_source_metadata.csv', dtype={'patient_id': str})
        col = [c for c in s1.columns if c.startswith('neoadjuvant chemotherapy')][0]
        code = s1.drop_duplicates('patient_id').set_index('patient_id')[col].str[-1].astype(float)
        X.loc[ispy1, 'anti_her2'] = (code.reindex(X.index[ispy1]) == 3).astype(float).to_numpy()
    return X


class TopVariance(BaseEstimator, TransformerMixin):
    def __init__(self, k=3000):
        self.k = k

    def fit(self, X, y=None):
        self.indices_ = np.argsort(-np.nanvar(np.asarray(X, float), axis=0), kind='stable')[:self.k]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices_]


def _offset_lr(X, y, offset, C, balanced=True):
    """L2 logistic regression with a fixed offset; returns (intercept, w)."""
    n, p = X.shape
    sw = np.ones(n)
    if balanced:
        pos = y.mean(); sw = np.where(y == 1, .5 / pos, .5 / (1 - pos))
    sw = sw / sw.sum()

    def f(t):
        b, w = t[0], t[1:]
        z = offset + b + X @ w
        v = np.sum(sw * (np.logaddexp(0, z) - y * z)) + (w @ w) / (2 * C * n)
        r = sw * (expit(z) - y)
        return v, np.r_[r.sum(), X.T @ r + w / (C * n)]
    t = minimize(f, np.zeros(p + 1), jac=True, method='L-BFGS-B', options={'maxiter': 2000}).x
    return t[0], t[1:]


class Expert:
    """Logistic expert on one modality.

    kind: 'clinical' | 'rna' | 'mri'. With ``residual=True`` the optional expert is fitted with the
    anchor logit as a fixed offset, so that its logit equals anchor + learned residual.
    """

    def __init__(self, kind, residual=False, clin_cols=BASE_CLIN, k=250, C=None, anchor=None):
        self.kind, self.residual, self.clin_cols, self.k, self.C, self.anchor = kind, residual, clin_cols, k, C, anchor

    def _prep(self):
        if self.kind == 'rna':
            return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), TopVariance(3000),
                                 SelectKBest(f_classif, k=self.k), StandardScaler())
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

    def _C(self):
        return self.C if self.C is not None else {'rna': 0.1, 'rppa': 0.01}.get(self.kind, 1.0)

    def fit(self, X, y, a=None):
        y = np.asarray(y, float)
        if self.kind == 'clinical':
            X = X[self.clin_cols]
        else:
            ok = available(X)
            X, y = X[ok], y[ok]
            a = None if a is None else np.asarray(a)[ok]
        self.prep_ = self._prep().fit(X, y)
        Z = self.prep_.transform(X)
        if self.residual and self.kind != 'clinical':
            self.b_, self.w_ = _offset_lr(Z, y, a, self._C())
        else:
            m = LogisticRegression(C=self._C(), class_weight='balanced', max_iter=1000, solver='liblinear').fit(Z, y)
            self.b_, self.w_ = m.intercept_[0], m.coef_[0]
        return self

    def residual_logit(self, X):
        if self.kind == 'clinical':
            return self.b_ + self.prep_.transform(X[self.clin_cols]) @ self.w_
        out = np.full(len(X), np.nan)
        ok = available(X)
        if ok.any():
            out[ok] = self.b_ + self.prep_.transform(X[ok]) @ self.w_
        return out

    def predict(self, X, a=None):
        f = self.residual_logit(X)
        return expit(a + f) if (self.residual and self.kind != 'clinical') else expit(f)


def available(X):
    """A modality is available for a patient when at least one of its features is observed."""
    return np.isfinite(np.asarray(X, float)).any(axis=1)


class AuxHead:
    """Predicts always-recorded anchor variables (HR, HER2 status) from one optional modality.

    Fitted with unweighted logistic regression so that its probabilities are usable as likelihoods.
    """
    TARGETS = ('hr', 'her2')

    def __init__(self, kind, k=50, C=None):
        self.kind, self.k, self.C = kind, k, C

    def fit(self, X, clin):
        self.models_ = {}
        for t in self.TARGETS:
            lab = clin[t].to_numpy(float); ok = np.isfinite(lab) & available(X)
            if self.kind == 'rna':
                m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), TopVariance(3000),
                                  SelectKBest(f_classif, k=self.k), StandardScaler(),
                                  LogisticRegression(C=self.C or 0.1, max_iter=2000, solver='liblinear'))
            else:
                m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                  LogisticRegression(C=self.C or 1.0, max_iter=2000, solver='liblinear'))
            self.models_[t] = m.fit(X[ok], lab[ok].astype(int))
        return self

    def predict(self, X):
        out = np.full((len(X), len(self.TARGETS)), np.nan)
        ok = available(X)
        if ok.any():
            out[ok] = np.column_stack([self.models_[t].predict_proba(X[ok])[:, 1] for t in self.TARGETS])
        return out


def consistency(aux_prob, clin):
    """Geometric-mean likelihood of the recorded HR/HER2 status under a modality's auxiliary head.

    Returns values in (0, 1]; unrecorded anchor variables are skipped (score 1 if none recorded).
    """
    lab = clin[list(AuxHead.TARGETS)].to_numpy(float)
    q = np.clip(aux_prob, 1e-4, 1 - 1e-4)
    ll = np.where(np.isfinite(lab), np.where(lab == 1, np.log(q), np.log1p(-q)), 0.)
    cnt = np.isfinite(lab).sum(1)
    return np.exp(ll.sum(1) / np.maximum(cnt, 1))


def _label_loglik(aux_prob, lab):
    q = np.clip(aux_prob, 1e-4, 1 - 1e-4)
    return np.where(np.isfinite(lab), np.where(lab == 1, np.log(q), np.log1p(-q)), 0.)


def match_posterior(aux_prob, clin, ref_aux, prior=0.5):
    """Posterior probability that a modality sample belongs to the patient whose anchor record it sits with.

    Likelihood under 'match': product over recorded anchor variables of the auxiliary head's probability
    of the recorded value. Likelihood under 'swap' (the sample comes from another patient): the same
    quantity averaged over the reference (training) patients' auxiliary predictions. Missing modality: NaN.
    """
    lab = clin[list(AuxHead.TARGETS)].to_numpy(float)
    ll_match = _label_loglik(aux_prob, lab).sum(1)
    ref = ref_aux[np.isfinite(ref_aux).all(1)]
    # log mean_i P_i(lab) computed per patient over reference auxiliary predictions
    lr = np.stack([_label_loglik(np.broadcast_to(ref[:, k][None, :], (len(lab), len(ref))), lab[:, [k]])
                   for k in range(lab.shape[1])], axis=0).sum(0)
    ll_swap = np.log(np.mean(np.exp(lr), axis=1))
    logit_post = np.log(prior) - np.log1p(-prior) + ll_match - ll_swap
    post = 1 / (1 + np.exp(-logit_post))
    return np.where(np.isfinite(aux_prob).all(1), post, np.nan)
