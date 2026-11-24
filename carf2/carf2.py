"""CARF-2: context-conditioned, bounded clinical-anchored residual fusion.

Extends CARF (mike2026/src/anchored_fusion.py, unchanged) with
  * context-conditioned gates: one coefficient block per anchor context
    (e.g. HR/HER2 subtype), with the budget constraint enforced per context;
  * an optional cross-modal agreement basis;
  * optional affine recalibration of the anchor (fallback to alpha*a + beta);
  * robust environments made of permutations and/or extremes of each expert.
The fused logit is linear in all free parameters, so every objective is convex.
"""
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

EPS = 1e-7


class AnchorContext:
    """Auxiliary predictions of the anchor variables from each optional modality, plus the recorded values.

    aux: n x J x T probabilities (NaN when the modality is missing); lab: n x T recorded anchor variables
    (NaN when unrecorded). Scores are recomputed after any manipulation of a modality's sample, so a swapped
    sample is always checked against the receiving patient's record.
    """

    def __init__(self, aux, lab):
        self.aux, self.lab = np.asarray(aux, float), np.asarray(lab, float)

    def take(self, idx):
        return AnchorContext(self.aux[idx], self.lab[idx])

    def swapped(self, jj, rows, src):
        aux = self.aux.copy(); aux[rows, jj] = self.aux[src, jj]
        return AnchorContext(aux, self.lab)

    def loglik(self, jj, aux=None):
        q = np.clip(self.aux[:, jj] if aux is None else aux, 1e-4, 1 - 1e-4)
        lab = self.lab
        return np.where(np.isfinite(lab), np.where(lab == 1, np.log(q), np.log1p(-q)), 0.)

    def scores(self, kind, ref=None, prior=0.9):
        """n x J matrix in [0, 1]: 'consistency' = geometric-mean likelihood of the record;
        'posterior' = posterior probability that the sample matches the record (null: a random
        reference patient's sample)."""
        n, J, T = self.aux.shape
        out = np.full((n, J), np.nan)
        cnt = np.maximum(np.isfinite(self.lab).sum(1), 1)
        for jj in range(J):
            ll = self.loglik(jj).sum(1)
            if kind == 'consistency':
                s = np.exp(ll / cnt)
            else:
                r = ref[:, jj]; r = r[np.isfinite(r).all(1)]
                lr = np.zeros((n, len(r)))
                for t in range(T):
                    q = np.clip(r[:, t], 1e-4, 1 - 1e-4)[None, :]
                    lab = self.lab[:, [t]]
                    lr += np.where(np.isfinite(lab), np.where(lab == 1, np.log(q), np.log1p(-q)), 0.)
                ll0 = np.log(np.mean(np.exp(lr), axis=1))
                s = expit(np.log(prior / (1 - prior)) + ll - ll0)
            out[:, jj] = np.where(np.isfinite(self.aux[:, jj]).all(1), s, np.nan)
        return out


def _logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


class CARF2:
    def __init__(self, anchor=-1, budget=1.0, robust_weight=0.25, temperature=0.05, l2=0.01,
                 bases=('const', 'agree_anchor', 'confidence'), agreement=False, n_groups=1,
                 anchor_affine=False, envs=('extreme',), n_perm=4, seed=0, bounded=True,
                 group_budget=None, gate_power=0.0, scales=None, gated_copies=False,
                 gate='consistency', prior=0.9):
        self.anchor, self.budget, self.robust_weight = anchor, budget, robust_weight
        self.temperature, self.l2, self.bases, self.agreement = temperature, l2, tuple(bases), agreement
        self.n_groups, self.anchor_affine, self.envs, self.n_perm = n_groups, anchor_affine, tuple(envs), n_perm
        self.seed, self.bounded, self.group_budget = seed, bounded, group_budget
        self.scales = scales  # None: hard clip at B; tuple: saturating residuals B*tanh(d/s) per scale s
        self.gate, self.prior = gate, prior  # how an AnchorContext is turned into per-modality scores
        self.gated_copies = gated_copies  # add consistency-gated copies of every basis
        self.gate_power = gate_power  # > 0: every basis of modality j is multiplied by ctx_j ** gate_power

    # ----------------------------------------------------------------- design
    def _anchor(self, P):
        return self.anchor % P.shape[1]

    def optional(self, P):
        return [j for j in range(P.shape[1]) if j != self._anchor(P)]

    def _design(self, P, g, ctx=None):
        """Return anchor logit a and residual design X (n x p).

        ctx: optional AnchorContext (or a precomputed n x n_optional score matrix in [0, 1]).
        """
        P = np.asarray(P, float)
        if isinstance(ctx, AnchorContext):
            ctx = ctx.scores(self.gate, getattr(self, 'ref_', None), self.prior)
        a = _logit(P[:, self._anchor(P)])
        cols, rs, present = [], [], []
        for jj, j in enumerate(self.optional(P)):
            m = np.isfinite(P[:, j])
            p = np.clip(np.where(m, P[:, j], .5), EPS, 1 - EPS)
            d = logit(p) - a
            if self.scales is None:
                R = [np.where(m, np.clip(d, -self.budget, self.budget) if self.bounded else d, 0.)]
            else:
                R = [np.where(m, self.budget * np.tanh(d / (s_ * self.budget)), 0.) for s_ in self.scales]
            rs.append(R[0]); present.append(m)
            gp = self.gate_power[jj] if isinstance(self.gate_power, (tuple, list)) else self.gate_power
            if gp > 0:
                R = [r * np.nan_to_num(np.clip(ctx[:, jj], 0, 1), nan=0.) ** gp for r in R]
            if self.gated_copies and ctx is not None:
                c = np.nan_to_num(np.clip(ctx[:, jj], 0, 1), nan=0.)
                R = R + [r * c for r in R]
            for r, b in itertools.product(R, self.bases):
                if b == 'const':
                    h = 1.
                elif b == 'agree_anchor':
                    h = np.exp(-np.abs(d))
                elif b == 'confidence':
                    h = 1 + (p * np.log(p) + (1 - p) * np.log1p(-p)) / np.log(2)
                elif b == 'consistency':
                    h = np.nan_to_num(np.clip(ctx[:, jj], 0, 1), nan=0.)
                elif b == 'consistency2':
                    h = np.nan_to_num(np.clip(ctx[:, jj], 0, 1), nan=0.) ** 2
                else:
                    raise ValueError(b)
                cols.append(r * h)
        if self.agreement:
            # h = 1 when every other present optional expert moves the anchor the same way.
            opt = self.optional(P)
            for k, j in enumerate(opt):
                agree = np.ones(len(P), bool); anyother = np.zeros(len(P), bool)
                for l in range(len(opt)):
                    if l == k:
                        continue
                    agree &= ~present[l] | (np.sign(rs[l]) == np.sign(rs[k]))
                    anyother |= present[l]
                cols.append(rs[k] * (agree & anyother))
        X = np.column_stack(cols)
        G = self.n_groups
        if G > 1:
            g = np.asarray(g, int)
            X = np.column_stack([X * (g == c)[:, None] for c in range(G)])
        return a, X

    def _environments(self, P, g, ctx):
        opt = self.optional(P)
        masks = []
        for bits in itertools.product([False, True], repeat=len(opt)):
            q = P.copy()
            for keep, j in zip(bits, opt):
                if not keep:
                    q[:, j] = np.nan
            masks.append(self._design(q, g, ctx))
        bad = []
        rng = np.random.default_rng(self.seed)
        for jj, j in enumerate(opt):
            if 'extreme' in self.envs:
                for v in (.01, .99):
                    q = P.copy(); q[:, j] = np.where(np.isfinite(P[:, j]), v, np.nan)
                    bad.append(self._design(q, g, ctx))
            avail = np.where(np.isfinite(P[:, j]))[0]
            if 'perm' in self.envs:
                for _ in range(self.n_perm):
                    q = P.copy(); q[avail, j] = P[rng.permutation(avail), j]
                    bad.append(self._design(q, g, ctx))
            if 'swap' in self.envs:
                # Sample swap among patients with the modality: its pCR output and its
                # anchor-consistency score move together, as when specimens are exchanged.
                for _ in range(self.n_perm):
                    src = rng.permutation(avail)
                    q = P.copy(); q[avail, j] = P[src, j]
                    if isinstance(ctx, AnchorContext):
                        c = ctx.swapped(jj, avail, src)
                    elif ctx is not None:
                        raise ValueError('swap environments need an AnchorContext')
                    else:
                        c = None
                    bad.append(self._design(q, g, c))
        return masks, bad

    # -------------------------------------------------------------------- fit
    def fit(self, P, y, g=None, ctx=None):
        P = np.asarray(P, float); y = np.asarray(y, float)
        g = np.zeros(len(P), int) if g is None else np.asarray(g, int)
        if isinstance(ctx, AnchorContext):
            self.ref_ = ctx.aux.copy()  # reference samples for the swap null
        masks, bad = self._environments(P, g, ctx)
        A = np.array([e[0] for e in masks]); X = np.array([e[1] for e in masks])
        Ab = np.array([e[0] for e in bad]) if bad else None
        Xb = np.array([e[1] for e in bad]) if bad else None
        p = X.shape[2]; na = 2 if self.anchor_affine else 0
        n = len(y)

        def losses(theta, A, X):
            if na:
                al, be, w = theta[0], theta[1], theta[2:]
                z = al * A + be + X @ w
            else:
                w = theta; z = A + X @ w
            L = np.mean(np.logaddexp(0, z) - y * z, axis=1)
            r = (expit(z) - y) / n
            gw = np.einsum('en,enp->ep', r, X)
            if na:
                gw = np.column_stack([(r * A).sum(1), r.sum(1), gw])
            return L, gw

        def objective(theta):
            L, G = losses(theta, A, X)
            v = L.mean(); gr = G.mean(0)
            if Xb is not None and self.robust_weight > 0:
                Lb, Gb = losses(theta, Ab, Xb)
                s = Lb / self.temperature
                wts = np.exp(s - logsumexp(s))
                v += self.robust_weight * self.temperature * (logsumexp(s) - np.log(len(s)))
                gr = gr + self.robust_weight * (wts @ Gb)
            w = theta[na:]
            v += self.l2 * w @ w / 2
            gr = gr.copy(); gr[na:] += self.l2 * w
            return v, gr

        per = p // self.n_groups
        cons = []
        if self.bounded:
            for c in range(self.n_groups):
                idx = np.arange(na + c * per, na + (c + 1) * per)
                cons.append({'type': 'ineq', 'fun': lambda t, idx=idx: 1 - t[idx].sum(),
                             'jac': lambda t, idx=idx: -np.eye(len(t))[idx].sum(0)})
        x0 = np.r_[[1., 0.] if na else [], np.zeros(p)]
        bounds = ([(1e-3, None), (None, None)] if na else []) + [(0, None)] * p
        res = minimize(objective, x0, jac=True, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-11, 'maxiter': 1000})
        th = res.x.copy(); th[na:] = np.maximum(th[na:], 0)
        if self.bounded:
            for c in range(self.n_groups):
                s = th[na + c * per: na + (c + 1) * per].sum()
                if s > 1:
                    th[na + c * per: na + (c + 1) * per] /= s
        self.theta_ = th; self.success_ = bool(res.success); self.n_affine_ = na; self.per_group_ = per
        return self

    @property
    def coef_(self):
        return self.theta_[self.n_affine_:]

    def anchor_map(self, a):
        return self.theta_[0] * a + self.theta_[1] if self.n_affine_ else a

    def decision_function(self, P, g=None, ctx=None):
        P = np.asarray(P, float)
        g = np.zeros(len(P), int) if g is None else np.asarray(g, int)
        a, X = self._design(P, g, ctx)
        return self.anchor_map(a) + X @ self.coef_

    def predict_proba(self, P, g=None, ctx=None):
        return expit(self.decision_function(P, g, ctx))


def robust_objective(model, P, y, ctx=None):
    """Held-out value of the CARF-2 objective (without the penalty) for fixed coefficients."""
    P = np.asarray(P, float); y = np.asarray(y, float)
    g = np.zeros(len(P), int)
    masks, bad = model._environments(P, g, ctx)

    def loss(env):
        a, X = env
        z = model.anchor_map(a) + X @ model.coef_
        return np.mean(np.logaddexp(0, z) - y * z)
    v = np.mean([loss(e) for e in masks])
    if bad and model.robust_weight > 0:
        s = np.array([loss(e) for e in bad]) / model.temperature
        v += model.robust_weight * model.temperature * (logsumexp(s) - np.log(len(s)))
    return v


def select_budget(P, y, budgets=(1.0, 1.5, 2.0, 3.0), ctx=None, folds=5, seed=0, **kw):
    """Choose B by inner cross-validation of the held-out robust objective."""
    from sklearn.model_selection import StratifiedKFold
    scores = np.zeros(len(budgets))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(P, y):
        for i, B in enumerate(budgets):
            m = CARF2(budget=B, **kw).fit(P[tr], y[tr], ctx=None if ctx is None else ctx.take(tr))
            scores[i] += robust_objective(m, P[va], y[va], None if ctx is None else ctx.take(va))
    return budgets[int(np.argmin(scores))], scores / folds
