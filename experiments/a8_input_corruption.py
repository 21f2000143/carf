#!/usr/bin/env python3
"""A8: input-level corruption of external I-SPY1 inputs through frozen experts.

Experts are fit once on all I-SPY2 (as in cached_predictions, seed SEED+100);
fusion models are fit once on the cached OOF P. Only the targeted modality's
inputs are corrupted; the other expert outputs are the cached Q.
RNA corruptions act on the raw (log-ratio) I-SPY1 expression, followed by the
pipeline's patient-wise rank_normalize over observed genes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.special import logit
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1 import offset_models, platt_carf, load, labels, DATA, SEED  # noqa: E402
from scripts.run_revision2_analysis import common_data, experts, predict_expert, rank_normalize  # noqa: E402
from scripts.run_revision3_analysis import fit_fusions, predict_fusions  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results/a8'
EPS = 1e-7
REPEATS = 30


def lg(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def setup():
    E2, M2, I2, E1, M1, I1 = common_data()
    y2 = M2.pcr.astype(int).to_numpy()
    fitted = experts(E2, M2, I2, y2, SEED + 100)
    Qnew = np.column_stack([predict_expert(m, x, k) for m, x, k in
                            zip(fitted, [E1, M1, I1], ['rna', 'clinical', 'mri'])])
    P, Q, ytr, ye = load('external', labels())
    print('reproduction max abs diff per column:', np.abs(Qnew - Q).max(axis=0))
    # RNA/clinical match to <=2e-8; MRI LR (liblinear) differs by <=1e-4 in probability
    # (median 4e-6, identical AUROC, r=0.999999999) -- solver-level numerical noise.
    if not np.allclose(Qnew, Q, rtol=0, atol=1e-3):
        raise SystemExit('Expert reproduction failed; stopping.')
    e1 = pd.read_csv(DATA / 'ISPY1_expression.csv.gz', index_col=0); e1.index = e1.index.astype(str)
    raw = e1.loc[E1.index, E1.columns].to_numpy(float)
    assert np.allclose(rank_normalize(raw), E1.to_numpy(), equal_nan=True)
    print(f'raw I-SPY1 expression: shape {raw.shape}, median gene SD {np.nanmedian(np.nanstd(raw, 0)):.3f}, '
          f'overall SD {np.nanstd(raw):.3f}, NaN frac {np.isnan(raw).mean():.4f}')
    fus = fit_fusions(P, ytr)
    off = offset_models(P, ytr)
    platt = platt_carf(P, ytr, SEED + 100)
    mri_sd = I2.std(axis=0, ddof=0).to_numpy()
    return dict(fitted=fitted, E1=E1, I1=I1, raw=raw, Q=Q, ye=ye, fus=fus, off=off,
                platt=platt, mri_sd=mri_sd)


SCENARIOS = ([('intact', None, None)]
             + [('rna_dropout', f, None) for f in (.1, .3, .5)]
             + [('rna_block_shift', f, s) for f in (.1, .25) for s in (1., 2.)]
             + [('mri_noise', s, None) for s in (.5, 1.)]
             + [('mri_dropout', f, None) for f in (.3, .6)])


def name_of(kind, a, b):
    if kind == 'intact':
        return 'intact'
    if kind == 'rna_block_shift':
        return f'rna_block{int(a * 100)}_shift+{b:g}'
    if kind == 'mri_noise':
        return f'mri_noise_sd{a:g}'
    return f'{kind}{int(a * 100)}'


def corrupt_Q(S, kind, a, b, rng):
    Q = S['Q'].copy()
    rna, _, mri = S['fitted']
    if kind.startswith('rna'):
        raw = S['raw'].copy()
        n, g = raw.shape
        if kind == 'rna_dropout':
            for i in range(n):
                raw[i, rng.choice(g, int(round(a * g)), replace=False)] = np.nan
        else:  # one platform-wide block of genes shifted in all patients
            raw[:, rng.choice(g, int(round(a * g)), replace=False)] += b
        X = pd.DataFrame(rank_normalize(raw), index=S['E1'].index, columns=S['E1'].columns)
        Q[:, 0] = predict_expert(rna, X, 'rna')
    elif kind.startswith('mri'):
        X = S['I1'].to_numpy(float).copy()
        n, f = X.shape
        if kind == 'mri_noise':
            X = X + rng.normal(0, 1, X.shape) * (a * S['mri_sd'])
        else:
            for i in range(n):
                X[i, rng.choice(f, int(round(a * f)), replace=False)] = np.nan
        Q[:, 2] = predict_expert(mri, pd.DataFrame(X, index=S['I1'].index, columns=S['I1'].columns), 'mri')
    return Q


def evaluate(S, Q):
    pred = predict_fusions(S['fus'], Q)
    for k, m in S['off'].items():
        pred[k] = m.predict_proba(Q)[:, 1]
    pred['carf_platt'] = S['platt'](Q)
    zc = lg(Q[:, 1])
    return [dict(model=k, auroc=roc_auc_score(S['ye'], p), brier=brier_score_loss(S['ye'], p),
                 max_shift=np.max(np.abs(lg(p) - zc))) for k, p in pred.items()]


_S = None  # set before fork; workers inherit it (platt closure is not picklable)


def job(t):
    si, kind, a, b, rep = t
    S = _S
    rng = np.random.default_rng([SEED, 8, si, rep])
    Q = S['Q'].copy() if kind == 'intact' else corrupt_Q(S, kind, a, b, rng)
    rows = evaluate(S, Q)
    for r in rows:
        r.update(scenario=name_of(kind, a, b), repeat=rep)
    return rows


def main():
    global _S
    S = _S = setup()
    tasks = [(si, k, a, b, r) for si, (k, a, b) in enumerate(SCENARIOS)
             for r in (range(1) if k == 'intact' else range(REPEATS))]
    with mp.get_context('fork').Pool(32) as pool:
        out = pool.map(job, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows])[['scenario', 'repeat', 'model', 'auroc', 'brier', 'max_shift']]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'input_stress.csv', index=False)
    clin = df[df.model == 'clinical'].set_index(['scenario', 'repeat']).auroc
    df['auroc_loss_vs_clinical'] = clin.loc[list(zip(df.scenario, df.repeat))].to_numpy() - df.auroc
    order = [name_of(*s) for s in SCENARIOS]
    summ = (df.groupby(['scenario', 'model'])
            .agg(auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'), brier_mean=('brier', 'mean'),
                 max_shift_mean=('max_shift', 'mean'), max_shift_max=('max_shift', 'max'),
                 loss_vs_clinical_mean=('auroc_loss_vs_clinical', 'mean'),
                 loss_vs_clinical_sd=('auroc_loss_vs_clinical', 'std'), n=('auroc', 'size'))
            .reset_index())
    summ['scenario'] = pd.Categorical(summ.scenario, order, ordered=True)
    summ = summ.sort_values(['scenario', 'model'])
    summ.to_csv(OUT / 'input_stress_summary.csv', index=False)
    keep = ['clinical', 'rna', 'mri', 'stack', 'mask_stack', 'offset_lr', 'carf', 'carf_no_robust']
    print(summ.pivot(index='scenario', columns='model', values='auroc_mean')[keep].round(3).to_string())


if __name__ == '__main__':
    main()
