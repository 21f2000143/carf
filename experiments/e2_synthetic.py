#!/usr/bin/env python3
"""E2: controlled synthetic experiment for CARF under known ground truth.

Latent model
    x_R, x_C, x_M ~ iid N(0, 1)                      (column order R, C, M as in P)
    eta = beta_R x_R + beta_C x_C + beta_M x_M
    y ~ Bernoulli(sigmoid(eta + c)),  c set for 30% prevalence
    observed feature  x~_j = x_j + TAU * e_j,  e_j ~ N(0, 1)   (measurement noise)
Betas were tuned (large Monte Carlo) so that the clinical-anchor expert alone has
AUROC ~0.70 and a logistic model on all three noisy features ~0.85; RNA (R) is the
strong optional modality and MRI (M) the weak one (beta_M = beta_R / 2).

Experts: one univariate logistic regression per modality on its own noisy feature.
Fusion training uses 5-fold cross-fitted (out-of-fold) expert probabilities on the
n=1000 training set; experts are then refit on all training data and frozen.
Fusions: fit_fusions/predict_fusions (revision3), masked OffsetLR (tier1), ModDrop MLP
(a7, fixed hidden=(32, 16), alpha=1e-3), and CARF at B in {0.5, 1, 2}.

Deployment scenarios act only on the n=2000 test set (experts and fusions frozen),
separately for j = R and j = M:
    missing   fraction q of patients have p_j = NaN
    silent    for a fraction q, x_j is replaced by independent N(0,1) noise before
              measurement (expert input present but uninformative)
    flip      for a fraction q, the observed association reverses (x~_j -> -x~_j)
    shift     additive covariate offset delta (SD units) on x_j for all patients
Cost scenarios (R only):
    cost      test data drawn with beta_R,test = k * beta_R,train, k in {1,1.5,2,3};
              intercept c held at its training value (so prevalence rises with k)
    strong    a separate setting where training AND test use beta_R = 2 beta_R
              (the gate is trained on a modality that deserves more than the budget)

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 mike2026/env/bin/python experiments/e2_synthetic.py
     (add `plot` as argument to only redraw the figure from results.csv)
     Extension: `e2_synthetic.py sweep` runs the anchor-strength sweep (anchor_sweep*.csv,
     fig_anchor_sweep.pdf); `plot_sweep` only redraws that figure.
"""
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tier1 import OffsetLR  # noqa: E402  (also puts mike2026 on sys.path)
from a7_moddrop import masked_blocks, mlp, predict as moddrop_predict  # noqa: E402
from src.anchored_fusion import AnchoredResidualFusion  # noqa: E402
from scripts.run_revision3_analysis import fit_fusions, predict_fusions, scores  # noqa: E402

OUT = HERE / 'results' / 'e2'
SEED = 20260924
EPS = 1e-7
R, C, M = 0, 1, 2
NAME = {R: 'R', M: 'M'}
P_ = dict(beta_R=1.8, beta_C=1.45, beta_M=0.9, tau=0.5, prevalence=0.30,
          n_train=1000, n_test=2000, n_rep=50, n_folds=5,
          moddrop_hidden=(32, 16), moddrop_alpha=1e-3, carf_budgets=[0.5, 1.0, 2.0])
QS = [0.0, 0.25, 0.5, 1.0]
DELTAS = [0.0, 0.5, 1.0, 2.0]
COST_K = [1.0, 1.5, 2.0, 3.0]
MODELS = ['clinical', 'stack', 'mask_stack', 'mean', 'offset_lr', 'moddrop_mlp',
          'carf_no_robust', 'carf', 'carf_B0.5', 'carf_B2', 'carf_unbounded', 'carf_no_context']


def betas(strong=False):
    return np.array([P_['beta_R'] * (2 if strong else 1), P_['beta_C'], P_['beta_M']])


def intercept(b, n=400_000):
    """Intercept giving the target prevalence (large Monte Carlo, fixed seed)."""
    eta = np.random.default_rng(1).normal(size=(n, 3)) @ b
    return brentq(lambda c: expit(eta + c).mean() - P_['prevalence'], -10, 5)


def simulate(n, b, c, rng):
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) < expit(X @ b + c)).astype(int)
    E = P_['tau'] * rng.normal(size=(n, 3))
    return X, E, y


def fit_experts(Xo, y, seed):
    """Per-modality LR; returns OOF probabilities P and the refit experts."""
    P = np.zeros((len(y), 3))
    for tr, te in StratifiedKFold(P_['n_folds'], shuffle=True, random_state=seed).split(Xo, y):
        for j in range(3):
            P[te, j] = LogisticRegression().fit(Xo[tr, j:j + 1], y[tr]).predict_proba(Xo[te, j:j + 1])[:, 1]
    experts = [LogisticRegression().fit(Xo[:, j:j + 1], y) for j in range(3)]
    return P, experts


def expert_probs(experts, Xo):
    return np.column_stack([e.predict_proba(Xo[:, j:j + 1])[:, 1] for j, e in enumerate(experts)])


def fit_all(P, y):
    models = fit_fusions(P, y)
    models['offset_lr'] = OffsetLR(masked=True).fit(P, y)
    models['moddrop_mlp'] = mlp(P_['moddrop_hidden'], P_['moddrop_alpha']).fit(*masked_blocks(P, y))
    for B in (0.5, 2.0):
        models[f'carf_B{B:g}'] = AnchoredResidualFusion(budget=B).fit(P, y)
    return models


def predict_all(models, Q):
    pred = predict_fusions({k: v for k, v in models.items()
                            if k not in ('offset_lr', 'moddrop_mlp') and not k.startswith('carf_B')}, Q)
    pred['offset_lr'] = models['offset_lr'].predict_proba(Q)[:, 1]
    pred['moddrop_mlp'] = moddrop_predict(models['moddrop_mlp'], Q)
    for k in models:
        if k.startswith('carf_B'):
            pred[k] = models[k].predict_proba(Q)[:, 1]
    return {k: pred[k] for k in MODELS}


def deployments(X, E, rng):
    """Yield (scenario, modality, level, observed features, missing mask) on a test set."""
    n = len(X)
    for j in (R, M):
        for q in QS:
            idx = rng.permutation(n)[:int(round(q * n))]
            miss = np.zeros((n, 3), bool)
            miss[idx, j] = True
            yield 'missing', NAME[j], q, X + E, miss
            Xs = X.copy()
            Xs[idx, j] = rng.normal(size=len(idx))
            yield 'silent', NAME[j], q, Xs + E, None
            Xo = X + E
            Xo[idx, j] *= -1
            yield 'flip', NAME[j], q, Xo, None
        for d in DELTAS:
            Xs = X.copy()
            Xs[:, j] += d
            yield 'shift', NAME[j], d, Xs + E, None


def evaluate(models, experts, Xo, miss, y, meta):
    Q = expert_probs(experts, Xo)
    if miss is not None:
        Q[miss] = np.nan
    a = logit(np.clip(Q[:, C], EPS, 1 - EPS))
    pred = predict_all(models, Q)
    base = roc_auc_score(y, pred['clinical'])
    rows = []
    for k, p in pred.items():
        s = scores(y, np.clip(p, EPS, 1 - EPS))
        rows.append({**meta, 'model': k, **s, 'delta_auroc_vs_anchor': s['auroc'] - base,
                     'max_disp': float(np.abs(logit(np.clip(p, EPS, 1 - EPS)) - a).max()),
                     'prevalence_test': float(y.mean())})
    return rows


def replicate(rep, setting):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    strong = setting == 'strong'
    b = betas(strong)
    c = intercept(b)
    rng = np.random.default_rng([SEED, rep, int(strong)])
    X, E, y = simulate(P_['n_train'], b, c, rng)
    P, experts = fit_experts(X + E, y, SEED + rep)
    models = fit_all(P, y)
    gates = {'rep': rep, 'setting': setting,
             **{f'carf_budget_{NAME[j]}': v for j, v in zip((R, M), models['carf'].modality_budgets())},
             **{f'carf_no_robust_budget_{NAME[j]}': v for j, v in zip((R, M), models['carf_no_robust'].modality_budgets())},
             'offset_w_R': models['offset_lr'].theta_[1], 'offset_w_M': models['offset_lr'].theta_[2],
             'stack_coef': json.dumps(models['stack'].coef_.ravel().round(4).tolist())}
    Xt, Et, yt = simulate(P_['n_test'], b, c, rng)
    rows = []
    if strong:
        rows += evaluate(models, experts, Xt + Et, None, yt,
                         {'rep': rep, 'setting': setting, 'scenario': 'strong', 'modality': 'R', 'level': 2.0})
        return rows, gates
    rows += evaluate(models, experts, Xt + Et, None, yt,
                     {'rep': rep, 'setting': setting, 'scenario': 'intact', 'modality': '-', 'level': 0.0})
    for scen, mod, lev, Xo, miss in deployments(Xt, Et, rng):
        rows += evaluate(models, experts, Xo, miss, yt,
                         {'rep': rep, 'setting': setting, 'scenario': scen, 'modality': mod, 'level': lev})
    for k in COST_K:  # stronger R at test; same train model, same intercept
        bt = b.copy()
        bt[R] *= k
        Xk, Ek, yk = simulate(P_['n_test'], bt, c, np.random.default_rng([SEED, rep, 99, int(10 * k)]))
        rows += evaluate(models, experts, Xk + Ek, None, yk,
                         {'rep': rep, 'setting': setting, 'scenario': 'cost', 'modality': 'R', 'level': k})
    return rows, gates


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, s) for r in range(P_['n_rep']) for s in ('base', 'strong')]
    res = Parallel(n_jobs=40, verbose=5)(delayed(replicate)(r, s) for r, s in jobs)
    df = pd.DataFrame([row for rows, _ in res for row in rows])
    df.to_csv(OUT / 'results.csv', index=False)
    pd.DataFrame([g for _, g in res]).to_csv(OUT / 'gates.csv', index=False)
    keys = ['setting', 'scenario', 'modality', 'level', 'model']
    metrics = ['auroc', 'log_loss', 'delta_auroc_vs_anchor', 'max_disp', 'brier', 'auprc']
    summ = df.groupby(keys)[metrics].agg(['mean', 'std'])
    summ.columns = [f'{m}_{s.replace("std", "sd")}' for m, s in summ.columns]
    summ.reset_index().to_csv(OUT / 'summary.csv', index=False, float_format='%.4f')
    (OUT / 'params.json').write_text(json.dumps({**P_, 'moddrop_hidden': list(P_['moddrop_hidden']),
        'intercept_base': intercept(betas()), 'intercept_strong': intercept(betas(True)),
        'seed': SEED, 'qs': QS, 'deltas': DELTAS, 'cost_k': COST_K}, indent=2))


# ----------------------------------------------------------------------------- figure
LABEL = {'stack': 'Stacking', 'mask_stack': 'Mask-stacking', 'offset_lr': 'Offset LR (masks)',
         'moddrop_mlp': 'ModDrop MLP', 'carf_no_robust': 'CARF, no robust loss', 'carf': 'CARF'}
COLOR = {'clinical': '#666666', 'stack': '#E69F00', 'mask_stack': '#009E73', 'offset_lr': '#CC79A7',
         'moddrop_mlp': '#8C6D1F', 'carf_no_robust': '#0072B2', 'carf': '#D55E00'}  # make_assets.py
MARK = {'stack': 's', 'mask_stack': '^', 'offset_lr': 'v', 'moddrop_mlp': 'P',
        'carf_no_robust': 'D', 'carf': 'o'}


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    CM = 1 / 2.54
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'dejavusans',
        'legend.frameon': False, 'savefig.bbox': 'tight'})
    df = pd.read_csv(OUT / 'results.csv')
    g = df.groupby(['scenario', 'modality', 'level', 'model']).auroc.agg(['mean', 'std']).reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2 * CM, 5.4 * CM))
    panels = [(a1, ('silent', 'R'), QS, 'Fraction $q$ with silent failure of $R$',
               '(a) Silent failure of $R$'),
              (a2, ('cost', 'R'), COST_K, r'$\beta_{R,\mathrm{test}}/\beta_{R,\mathrm{train}}$',
               '(b) Cost: $R$ stronger at test')]
    for ax, (scen, mod), xs, xl, title in panels:
        s = g[(g.scenario == scen) & (g.modality == mod)]
        anc = s[s.model == 'clinical'].sort_values('level')
        ax.plot(anc.level, anc['mean'], color='#888888', ls='--', lw=1.0, zorder=1, label='Anchor only')
        for m in LABEL:
            t = s[s.model == m].sort_values('level')
            ax.errorbar(t.level, t['mean'], yerr=t['std'], color=COLOR[m], marker=MARK[m], ms=3.2,
                        lw=1.6 if m == 'carf' else 1.0, elinewidth=0.6, capsize=1.2,
                        zorder=3 if m == 'carf' else 2, label=LABEL[m])
        if scen == 'silent':  # CARF with a tighter clip, B = 0.5
            t = s[s.model == 'carf_B0.5'].sort_values('level')
            ax.plot(t.level, t['mean'], color=COLOR['carf'], ls=':', marker='o', mfc='white', ms=3.2,
                    lw=1.1, zorder=3, label='CARF, $B=0.5$')
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{x:g}' for x in xs])
        ax.set_xlabel(xl)
        ax.set_title(title, loc='left', fontsize=8)
    a1.set_ylabel('Test AUROC')
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=7,
               handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.2)
    fig.savefig(OUT / 'fig_synthetic.pdf')
    fig.savefig(OUT / 'fig_synthetic.png', dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- extension: anchor-strength sweep
# beta_C values giving clinical-expert AUROC ~0.65/0.70/0.75/0.80/0.85 (beta_R, beta_M fixed),
# found by bisection on a 400k Monte Carlo sample; 0.70 is the main-experiment setting.
ANCHOR_BETA_C = {0.65: 1.028, 0.70: 1.45, 0.75: 1.954, 0.80: 2.652, 0.85: 3.731}
CV_BUDGETS = [0.25, 0.5, 1.0, 2.0, 4.0]
SWEEP_MODELS = ['clinical', 'stack', 'mask_stack', 'offset_lr', 'moddrop_mlp', 'carf_no_robust', 'carf',
                'carf_cv', 'carf_B4', 'carf_unbounded']


def carf_cv(P, y, seed):
    """CARF with B chosen by inner 5-fold CV log loss on the training OOF probabilities only."""
    from sklearn.metrics import log_loss
    losses = []
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(P, y))
    for B in CV_BUDGETS:
        losses.append(np.mean([log_loss(y[va], AnchoredResidualFusion(budget=B).fit(P[tr], y[tr])
                                        .predict_proba(P[va])[:, 1], labels=[0, 1]) for tr, va in folds]))
    B = CV_BUDGETS[int(np.argmin(losses))]
    return AnchoredResidualFusion(budget=B).fit(P, y), B


def sweep_replicate(target, rep):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    b = np.array([P_['beta_R'], ANCHOR_BETA_C[target], P_['beta_M']])
    c = intercept(b)
    rng = np.random.default_rng([SEED, rep, 7, int(round(100 * target))])
    X, E, y = simulate(P_['n_train'], b, c, rng)
    P, experts = fit_experts(X + E, y, SEED + rep)
    models = fit_fusions(P, y)
    models['offset_lr'] = OffsetLR(masked=True).fit(P, y)
    models['moddrop_mlp'] = mlp(P_['moddrop_hidden'], P_['moddrop_alpha']).fit(*masked_blocks(P, y))
    models['carf_cv'], B_cv = carf_cv(P, y, SEED + rep)
    models['carf_B4'] = AnchoredResidualFusion(budget=4.0).fit(P, y)
    Xt, Et, yt = simulate(P_['n_test'], b, c, rng)
    n = len(yt)
    idx_half = rng.permutation(n)[:n // 2]
    cases = [('intact', 0.0, Xt.copy(), None)]
    for q, idx in [(0.5, idx_half), (1.0, np.arange(n))]:
        Xs = Xt.copy()
        Xs[idx, R] = rng.normal(size=len(idx))
        cases.append(('silent', q, Xs, None))
    miss = np.zeros((n, 3), bool)
    miss[:, R] = True
    cases.append(('missing', 1.0, Xt.copy(), miss))
    Xs = Xt.copy()
    Xs[:, R] += 2.0
    cases.append(('shift', 2.0, Xs, None))
    rows = []
    for scen, lev, Xs, m in cases:
        Q = expert_probs(experts, Xs + Et)
        if m is not None:
            Q[m] = np.nan
        a = logit(np.clip(Q[:, C], EPS, 1 - EPS))
        pred = predict_fusions({k: v for k, v in models.items() if k in
                                ('stack', 'no_mri_stack', 'mask_stack', 'carf', 'carf_no_robust',
                                 'carf_unbounded')}, Q)
        pred['offset_lr'] = models['offset_lr'].predict_proba(Q)[:, 1]
        pred['moddrop_mlp'] = moddrop_predict(models['moddrop_mlp'], Q)
        for k in ('carf_cv', 'carf_B4'):
            pred[k] = models[k].predict_proba(Q)[:, 1]
        base = roc_auc_score(yt, pred['clinical'])
        for k in SWEEP_MODELS:
            p = np.clip(pred[k], EPS, 1 - EPS)
            s = scores(yt, p)
            rows.append({'anchor_target': target, 'beta_C': b[C], 'rep': rep, 'scenario': scen, 'modality': 'R',
                         'level': lev, 'model': k, **s, 'delta_auroc_vs_anchor': s['auroc'] - base,
                         'max_disp': float(np.abs(logit(p) - a).max()), 'carf_cv_B': B_cv})
    return rows


def run_sweep():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(t, r) for t in ANCHOR_BETA_C for r in range(P_['n_rep'])]
    res = Parallel(n_jobs=40, verbose=5)(delayed(sweep_replicate)(t, r) for t, r in jobs)
    df = pd.DataFrame([row for rows in res for row in rows])
    df.to_csv(OUT / 'anchor_sweep.csv', index=False)
    keys = ['anchor_target', 'beta_C', 'scenario', 'level', 'model']
    summ = df.groupby(keys)[['auroc', 'log_loss', 'delta_auroc_vs_anchor', 'max_disp']].agg(['mean', 'std'])
    summ.columns = [f'{m}_{s.replace("std", "sd")}' for m, s in summ.columns]
    summ.reset_index().to_csv(OUT / 'anchor_sweep_summary.csv', index=False, float_format='%.4f')
    # paired gaps to stacking, per replicate
    w = df.pivot_table(index=['anchor_target', 'rep', 'scenario', 'level'], columns='model', values='auroc')
    gaps = pd.DataFrame({f'{m}_minus_stack': w[m] - w['stack'] for m in SWEEP_MODELS if m != 'stack'})
    g = gaps.groupby(['anchor_target', 'scenario', 'level']).agg(['mean', 'std'])
    g.columns = [f'{m}_{s.replace("std", "sd")}' for m, s in g.columns]
    g = g.join(df[df.model == 'clinical'].groupby(['anchor_target', 'scenario', 'level']).auroc.mean()
               .rename('anchor_auroc'))
    g.reset_index().to_csv(OUT / 'anchor_sweep_gaps.csv', index=False, float_format='%.4f')
    df.groupby('anchor_target').carf_cv_B.value_counts().unstack(fill_value=0).to_csv(OUT / 'anchor_sweep_cvB.csv')


def plot_sweep():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    CM = 1 / 2.54
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'dejavusans',
        'legend.frameon': False, 'savefig.bbox': 'tight'})
    g = pd.read_csv(OUT / 'anchor_sweep_gaps.csv')
    col = {**COLOR, 'carf_cv': '#000000'}
    lab = {'carf': 'CARF ($B=1$)', 'carf_cv': 'CARF (CV-selected $B$)', 'mask_stack': 'Mask-stacking',
           'moddrop_mlp': 'ModDrop MLP'}
    mk = {'carf': 'o', 'carf_cv': 'D', 'mask_stack': '^', 'moddrop_mlp': 'P'}
    fig, ax = plt.subplots(figsize=(12.2 * CM, 5.6 * CM))
    ax.axhline(0, color='#888888', lw=0.8, ls=':', zorder=0)
    for scen, lev, ls, fill in [('intact', 0.0, '-', True), ('silent', 1.0, '--', False)]:
        s = g[(g.scenario == scen) & (g.level == lev)].sort_values('anchor_auroc')
        for m in lab:
            ax.errorbar(s.anchor_auroc, s[f'{m}_minus_stack_mean'], yerr=s[f'{m}_minus_stack_sd'],
                        color=col[m], ls=ls, marker=mk[m], ms=3.2, mfc=col[m] if fill else 'white',
                        lw=1.5 if m.startswith('carf') else 1.0, elinewidth=0.6, capsize=1.2)
    ax.set_xlabel('Anchor-only AUROC')
    ax.set_ylabel('AUROC $-$ stacking AUROC')
    h = [Line2D([], [], color=col[m], marker=mk[m], ms=3.2, lw=1.2, label=lab[m]) for m in lab]
    h += [Line2D([], [], color='#333333', ls='-', lw=1.0, label='Intact'),
          Line2D([], [], color='#333333', ls='--', lw=1.0, label='$R$ silent, $q=1$')]
    ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.01, 0.5), handlelength=2.0)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_anchor_sweep.pdf')
    fig.savefig(OUT / 'fig_anchor_sweep.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    import e2_synthetic  # run via the module so joblib pickles functions by reference
    arg = sys.argv[1:]
    if arg == ['sweep']:
        e2_synthetic.run_sweep()
        e2_synthetic.plot_sweep()
    elif arg == ['plot_sweep']:
        e2_synthetic.plot_sweep()
    else:
        if arg != ['plot']:
            e2_synthetic.run()
        e2_synthetic.plot()
