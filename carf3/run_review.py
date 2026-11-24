#!/usr/bin/env python3
"""CARF-3 review-response experiments, run on top of the frozen carf2 pipeline/models.

New analyses added in response to the MIKE 2026 review (202-review-640.docx), none of which
change carf2/latex2: run_main.py and run_extra.py are read-only.

  anchor   : external stress test that corrupts the clinical anchor itself (HR/HER2 charting
             errors), not just the optional modalities. Addresses "the approach assumes the
             clinical anchor remains reliable; failures affecting the anchor itself are not
             adequately addressed."
  combo    : simultaneous (not one-at-a-time) modality failures externally: RNA swap + MRI
             input noise, and RNA gene dropout + MRI feature dropout, together. Addresses
             "evaluate more realistic corruption scenarios, including simultaneous modality
             failures."
  calib    : post-hoc Platt recalibration of every model (fit on internal out-of-fold
             predictions, applied to the external cohort with no external labels used) with
             before/after Brier, log loss, and ECE, plus a decision-curve (net benefit)
             analysis on the recalibrated external probabilities. Addresses "the model tends
             to overestimate pCR prevalence, and calibration remains insufficiently
             addressed" and "include calibration metrics and decision-curve analysis."
  mrigate  : exploratory, I-SPY1-only cross-validated check of whether an MRI-tumor-size/
             T-stage consistency score separates true from swapped MRI samples better than
             the HR/HER2-based gate (AUROC 0.56). T-stage is not recorded in I-SPY2, so this
             cannot be deployed in the main pipeline; it is reported as an exploratory,
             small-sample analysis only. Addresses "improve the modality-verification gate,
             particularly its ability to detect unreliable MRI inputs."

All four reuse the frozen carf2 design fit on the whole of I-SPY2 and the cached inner
out-of-fold predictions used to fit the fusion models, exactly as run_extra.py does.
"""
import json
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, OUT, SEED
from pipeline import load_data
from run_extra import corrupt, evaluate, frozen
from run_main import ctx_of, logits, scores

RES = OUT / 'review'
RES.mkdir(parents=True, exist_ok=True)
EPS = 1e-7
REPEATS = 30


def stable_hash(s):
    """Deterministic across processes, unlike built-in hash() on str/tuple, which is seeded per
    process by PYTHONHASHSEED and would otherwise make these RNG streams non-reproducible."""
    return zlib.crc32(repr(s).encode())

HEADLINE = ['clinical', 'mean', 'stack', 'mask_stack', 'offset_lr', 'robust_offset_lr', 'moddrop_mlp',
            'carf2', 'carf2_no_gate', 'gated_stack', 'gated_robust_offset_lr']


# ------------------------------------------------------------------ A. anchor failure
def corrupt_anchor(blocks, frac, which, rng):
    """Flip the recorded HR and/or HER2 status for a random subset of patients, as after a
    charting/lab error. This corrupts (i) the anchor logit itself, since the optional experts'
    residuals and the final anchor column are both computed from blocks['clinical'], and
    (ii) the verification gate, since it checks optional samples against the same record."""
    c = blocks['clinical'].copy()
    idx = rng.choice(len(c), size=max(1, int(round(frac * len(c)))), replace=False)
    for v in which:
        col = c.columns.get_loc(v)
        c.iloc[idx, col] = 1.0 - c.iloc[idx, col].to_numpy()
    b = dict(blocks); b['clinical'] = c
    return b, idx


def anchor_job(frac, which, rep, F, ext, ye):
    d, experts, fus, Q, QA = F
    rng = np.random.default_rng([SEED, 42, int(round(frac * 1000)), stable_hash(which) % 1000, rep])
    b2, idx = corrupt_anchor(ext, frac, which, rng)
    Q2, QA2 = d.predict(experts, b2)
    rows = evaluate(d, fus, b2, experts, ye, Q2, QA2)
    name = f'anchor_{"_".join(which)}_{int(round(frac * 100))}'
    return [{'scenario': name, 'frac': frac, 'which': '+'.join(which), 'rep': rep, **r} for r in rows]


def run_anchor(dev, y, ext, ye):
    F = frozen('carf2', dev, y, ext)
    combos = [(f, w) for f in (0.05, 0.1, 0.2) for w in (('hr',), ('her2',), ('hr', 'her2'))]
    jobs = [(f, w, r) for f, w in combos for r in range(REPEATS)]
    out = Parallel(n_jobs=60, verbose=2)(delayed(anchor_job)(f, w, r, F, ext, ye) for f, w, r in jobs)
    df = pd.DataFrame([r for o in out for r in o])
    df.to_csv(RES / 'anchor_failure.csv', index=False)
    print(df[df.model.isin(HEADLINE)].groupby(['scenario', 'model']).auroc.mean().unstack().round(3))
    return df


# ------------------------------------------------------------------ B. simultaneous failures
def swap_output(Q, QA, j, rng):
    avail = np.where(np.isfinite(Q[:, j]))[0]
    src = rng.permutation(avail)
    q, qa = Q.copy(), QA.copy()
    q[avail, j] = Q[src, j]; qa[avail, j] = QA[src, j]
    return q, qa


def combo_job(name, rep, F, ext, ye, raw_rna, mri_sd, optional):
    d, experts, fus, Q, QA = F
    rng = np.random.default_rng([SEED, 77, stable_hash(name) % 100000, rep])
    j_rna, j_mri = optional.index('rna'), optional.index('mri')
    if name == 'rna_swap+mri_noise':
        b2 = corrupt(ext, raw_rna, mri_sd, 'mri_noise', 1.0, rng)
        Q2, QA2 = d.predict(experts, b2)
        Q2, QA2 = swap_output(Q2, QA2, j_rna, rng)
        rows = evaluate(d, fus, b2, experts, ye, Q2, QA2)
    elif name == 'rna_dropout+mri_dropout':
        b2 = corrupt(ext, raw_rna, mri_sd, 'rna_dropout', .5, rng)
        b2 = corrupt(b2, raw_rna, mri_sd, 'mri_dropout', .6, rng)
        rows = evaluate(d, fus, b2, experts, ye)
    elif name == 'rna_swap+mri_swap':
        Q2, QA2 = swap_output(Q, QA, j_rna, rng)
        Q2, QA2 = swap_output(Q2, QA2, j_mri, np.random.default_rng([SEED, 78, rep]))
        rows = evaluate(d, fus, ext, experts, ye, Q2, QA2)
    else:
        raise ValueError(name)
    return [{'scenario': name, 'rep': rep, **r} for r in rows]


def run_combo(dev, y, ext, ye):
    F = frozen('carf2', dev, y, ext)
    d = F[0]
    e1 = pd.read_csv(DATA / 'ISPY1_expression.csv.gz', index_col=0); e1.index = e1.index.astype(str)
    raw = e1.loc[ext['rna'].index, ext['rna'].columns].to_numpy(float)
    mri_sd = dev['mri'].std(axis=0, ddof=0).to_numpy()
    names = ['rna_swap+mri_noise', 'rna_dropout+mri_dropout', 'rna_swap+mri_swap']
    jobs = [(n, r) for n in names for r in range(REPEATS)]
    out = Parallel(n_jobs=60, verbose=2)(
        delayed(combo_job)(n, r, F, ext, ye, raw, mri_sd, d.optional) for n, r in jobs)
    df = pd.DataFrame([r for o in out for r in o])
    # Single-failure comparators computed from the same frozen models, for a like-for-like gap.
    singles = []
    for sname, kind, a in [('rna_swap', None, None), ('mri_noisy', 'mri_noise', 1.0),
                           ('rna_dropout50', 'rna_dropout', .5), ('mri_dropout60', 'mri_dropout', .6)]:
        for rep in range(REPEATS):
            rng = np.random.default_rng([SEED, 79, stable_hash(sname) % 10000, rep])
            if kind is None:
                Q2, QA2 = swap_output(F[3], F[4], d.optional.index('rna'), rng)
                rows = evaluate(d, F[2], ext, F[1], ye, Q2, QA2)
            else:
                b2 = corrupt(ext, raw, mri_sd, kind, a, rng)
                rows = evaluate(d, F[2], b2, F[1], ye)
            singles += [{'scenario': sname, 'rep': rep, **r} for r in rows]
    df = pd.concat([df, pd.DataFrame(singles)], ignore_index=True)
    df.to_csv(RES / 'combo_failure.csv', index=False)
    print(df[df.model.isin(HEADLINE)].groupby(['scenario', 'model']).auroc.mean().unstack().round(3))
    return df


# ------------------------------------------------------------------ C. calibration + decision curve
def platt_fit(p, y):
    z = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def platt_apply(p, ab):
    a, b = ab
    z = logit(np.clip(p, EPS, 1 - EPS))
    return expit(a * z + b)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def calib_metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return {'brier': brier_score_loss(y, p), 'log_loss': log_loss(y, p, labels=[0, 1]), 'ece': ece(y, p)}


def net_benefit(y, p, thresholds):
    n = len(y)
    out = []
    for pt in thresholds:
        pred = p >= pt
        tp = float(np.sum(pred & (y == 1))); fp = float(np.sum(pred & (y == 0)))
        out.append(tp / n - fp / n * (pt / (1 - pt)))
    return np.array(out)


def nested_internal_calibration(dfi, model, seed=SEED):
    """5-fold nested Platt recalibration of the pooled internal out-of-fold predictions."""
    y = dfi.pcr.to_numpy(); p = dfi[model].to_numpy()
    out = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(p, y):
        ab = platt_fit(p[tr], y[tr])
        out[te] = platt_apply(p[te], ab)
    return y, p, out


def run_calib(dev, y, ext, ye):
    preds = pd.read_csv(OUT / 'main' / 'predictions.csv.gz')
    dfi = preds[(preds.design == 'carf2') & (preds.split != 'external')]
    dfe = preds[(preds.design == 'carf2') & (preds.split == 'external')].set_index('patient_id')
    models = [m for m in HEADLINE if m in dfi.columns]
    rows = []
    recal_ext = {}
    for m in models:
        yi, pi_raw, pi_cal = nested_internal_calibration(dfi, m)
        before_i, after_i = calib_metrics(yi, pi_raw), calib_metrics(yi, pi_cal)
        ab_full = platt_fit(dfi[m].to_numpy(), dfi.pcr.to_numpy())  # fit on all internal OOF, no external labels
        pe_raw = dfe[m].to_numpy(); pe_cal = platt_apply(pe_raw, ab_full)
        before_e, after_e = calib_metrics(ye, pe_raw), calib_metrics(ye, pe_cal)
        recal_ext[m] = pe_cal
        rows.append({'model': m, 'cohort': 'internal', **{f'{k}_before': v for k, v in before_i.items()},
                     **{f'{k}_after': v for k, v in after_i.items()}, 'prev_pred_before': float(pi_raw.mean()),
                     'prev_pred_after': float(pi_cal.mean()), 'prev_true': float(yi.mean())})
        rows.append({'model': m, 'cohort': 'external', **{f'{k}_before': v for k, v in before_e.items()},
                     **{f'{k}_after': v for k, v in after_e.items()}, 'prev_pred_before': float(pe_raw.mean()),
                     'prev_pred_after': float(pe_cal.mean()), 'prev_true': float(ye.mean())})
    pd.DataFrame(rows).to_csv(RES / 'calibration.csv', index=False)

    thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
    dc_models = ['clinical', 'carf2', 'stack']
    dc = {'threshold': thresholds, 'treat_all': net_benefit(ye, np.ones_like(ye, float), thresholds),
          'treat_none': np.zeros_like(thresholds)}
    for m in dc_models:
        dc[m] = net_benefit(ye, recal_ext[m], thresholds)
    pd.DataFrame(dc).to_csv(RES / 'decision_curve.csv', index=False)
    print(pd.DataFrame(rows)[['model', 'cohort', 'brier_before', 'brier_after', 'ece_before', 'ece_after',
                              'prev_pred_before', 'prev_pred_after', 'prev_true']].round(3).to_string())
    return pd.DataFrame(rows), pd.DataFrame(dc)


# ------------------------------------------------------------------ D. exploratory MRI T-stage gate
def ispy1_tstage(ext_index):
    s1 = pd.read_csv(DATA / 'ISPY1_source_metadata.csv', dtype={'patient_id': str})
    col_t = [c for c in s1.columns if c.startswith('clinical t stage')][0]
    d = s1.drop_duplicates('patient_id').set_index('patient_id')[col_t]
    t = d.reindex(ext_index)
    return (t >= 3).astype(float).where(t.notna())


def posterior_single(ll_own, ll_null, prior=0.95):
    return expit(np.log(prior / (1 - prior)) + ll_own - ll_null)


def gate_auroc_single(q, label, seed, prior=0.95, n_rep=200):
    n = len(q)
    qc = np.clip(q, 1e-4, 1 - 1e-4)
    ll_mat = np.where(label[:, None] == 1, np.log(qc)[None, :], np.log1p(-qc)[None, :])
    np.fill_diagonal(ll_mat, -np.inf)
    ll_null = np.log(np.mean(np.exp(ll_mat), axis=1))
    ll_own = np.where(label == 1, np.log(qc), np.log1p(-qc))
    g_true = posterior_single(ll_own, ll_null, prior)
    rng = np.random.default_rng(seed)
    swaps = []
    for _ in range(n_rep):
        perm = rng.permutation(n)
        fixed = perm == np.arange(n)
        while fixed.any():
            perm[fixed] = rng.permutation(n)[fixed]
            fixed = perm == np.arange(n)
        ll_swap_own = np.where(label == 1, np.log(qc[perm]), np.log1p(-qc[perm]))
        swaps.append(posterior_single(ll_swap_own, ll_null, prior))
    g_swap = np.concatenate(swaps)
    y_bin = np.r_[np.ones(n), np.zeros(len(g_swap))]
    score = np.r_[g_true, g_swap]
    return float(roc_auc_score(y_bin, score)), float(g_true.mean()), float(np.mean(g_swap < 0.5))


def run_mrigate(dev, y, ext, ye):
    t3 = ispy1_tstage(ext['mri'].index)
    ok = t3.notna() & np.isfinite(ext['mri'].to_numpy(float)).any(axis=1)
    ids = ext['mri'].index[ok]
    X = ext['mri'].loc[ids]
    lab = t3.loc[ids].to_numpy(float)
    n = len(ids)
    q = np.full(n, np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(n), lab):
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000, solver='liblinear'))
        m.fit(X.iloc[tr], lab[tr])
        q[te] = m.predict_proba(X.iloc[te])[:, 1]
    direct_auc = roc_auc_score(lab, q)
    gate_auc, g_true_mean, g_swap_low = gate_auroc_single(q, lab, SEED)
    out = {'n': int(n), 'n_t3': int(lab.sum()), 'mri_predicts_tstage_auc': direct_auc,
           'gate_auc_tstage': gate_auc, 'gate_true_mean_tstage': g_true_mean, 'gate_swap_low_frac_tstage': g_swap_low}
    (RES / 'mri_tstage_gate.json').write_text(json.dumps(out, indent=1))
    print(out)
    return out


def main(which):
    dev, y, ext, ye, M, Me = load_data()
    warnings.filterwarnings('ignore')
    if 'anchor' in which: run_anchor(dev, y, ext, ye)
    if 'combo' in which: run_combo(dev, y, ext, ye)
    if 'calib' in which: run_calib(dev, y, ext, ye)
    if 'mrigate' in which: run_mrigate(dev, y, ext, ye)


if __name__ == '__main__':
    import run_review
    run_review.main(sys.argv[1:] or ['anchor', 'combo', 'calib', 'mrigate'])
