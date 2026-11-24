#!/usr/bin/env python3
"""CARF-2 secondary experiments on frozen models fitted on all of I-SPY2.

  input   : input-level corruption of I-SPY1 expression / MRI features (30 realisations)
  natural : I-SPY1 patients with a naturally missing modality (true masks)
  gse     : independent cohort GSE25066 without I-SPY1 samples (RNA only)
  perm    : whole-pipeline label permutation null for the external AUROC
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from common import OUT, SEED, DATA, ROOT
from pipeline import load_data, cached, subset, oof
from run_main import DESIGNS, ctx_of, scores, logits
from experts import clinical_frame
from models import make, BASELINES, GATED

RES = OUT / 'extra'
MODELS = BASELINES + ['carf_v1', 'carf2', 'carf2_no_robust', 'carf2_no_gate'] + GATED
REPEATS = 30


def frozen(dname, dev, y, ext):
    """Experts refit on all of I-SPY2 and fusions fit on the cached inner OOF predictions."""
    d = DESIGNS[dname]
    P, PA, Q, QA = cached(d, 'external', dev, y, np.arange(len(y)), ext, SEED + 100)
    experts = d.fit(dev, y)
    ctr = ctx_of(PA, dev['clinical'])
    warnings.filterwarnings('ignore')
    fus = {n: make(n).fit(P, y, ctr) for n in MODELS}
    return d, experts, fus, Q, QA


def evaluate(d, fus, blocks, experts, yt, Q=None, QA=None):
    if Q is None:
        Q, QA = d.predict(experts, blocks)
    cte = ctx_of(QA, blocks['clinical'])
    a = logits(Q[:, -1])
    out = {n: m.predict(Q, cte) for n, m in fus.items()}
    out['clinical'] = Q[:, -1]
    return [{'model': n, **scores(yt, p), 'max_disp': float(np.abs(logits(p) - a).max())} for n, p in out.items()]


# ------------------------------------------------------------------ input-level corruption
INPUT = [('rna_dropout', .5), ('rna_dropout', .3), ('rna_block_shift', .25), ('rna_block_shift', .10),
         ('mri_noise', 1.), ('mri_noise', .5), ('mri_dropout', .6), ('mri_dropout', .3)]


def corrupt(blocks, raw, mri_sd, kind, a, rng):
    from scripts.run_revision2_analysis import rank_normalize
    b = dict(blocks)
    if kind.startswith('rna'):
        x = raw.copy(); n, g = x.shape
        if kind == 'rna_dropout':
            for i in range(n):
                x[i, rng.choice(g, int(round(a * g)), replace=False)] = np.nan
        else:
            x[:, rng.choice(g, int(round(a * g)), replace=False)] += 2.
        b['rna'] = pd.DataFrame(rank_normalize(x), index=blocks['rna'].index, columns=blocks['rna'].columns)
    else:
        X = blocks['mri'].to_numpy(float).copy(); n, f = X.shape
        if kind == 'mri_noise':
            X = X + rng.normal(0, 1, X.shape) * (a * mri_sd)
        else:
            for i in range(n):
                X[i, rng.choice(f, int(round(a * f)), replace=False)] = np.nan
        b['mri'] = pd.DataFrame(X, index=blocks['mri'].index, columns=blocks['mri'].columns)
    return b


def input_job(dname, si, kind, a, rep, F, ext, ye, raw, mri_sd):
    d, experts, fus, Q, QA = F
    rng = np.random.default_rng([SEED, 8, si, rep])
    rows = evaluate(d, fus, corrupt(ext, raw, mri_sd, kind, a, rng), experts, ye)
    name = f'{kind}{int(round(a * 100))}' if kind != 'mri_noise' else f'mri_noise_sd{a:g}'
    return [{'design': dname, 'scenario': name, 'rep': rep, **r} for r in rows]


def run_input(dev, y, ext, ye):
    e1 = pd.read_csv(DATA / 'ISPY1_expression.csv.gz', index_col=0); e1.index = e1.index.astype(str)
    raw = e1.loc[ext['rna'].index, ext['rna'].columns].to_numpy(float)
    mri_sd = dev['mri'].std(axis=0, ddof=0).to_numpy()
    jobs = [(dn, si, k, a, r) for dn in ('v1', 'carf2') for si, (k, a) in enumerate(INPUT) for r in range(REPEATS)]
    F = {dn: frozen(dn, dev, y, ext) for dn in ('v1', 'carf2')}
    out = Parallel(n_jobs=60, verbose=2)(delayed(input_job)(*j, F[j[0]], ext, ye, raw, mri_sd) for j in jobs)
    pd.DataFrame([r for o in out for r in o]).to_csv(RES / 'input_stress.csv', index=False)


# ------------------------------------------------------------------ natural missingness
def natural_blocks(dev):
    from scripts.run_revision2_analysis import rank_normalize
    A6 = ROOT / 'experiments/results/a6'
    audit = pd.read_csv(A6 / 'ispy1_availability_audit.csv', dtype={'patient_id': str}).set_index('patient_id')
    audit = audit[~audit.excluded & audit.pcr.notna()]
    x = pd.read_csv(A6 / 'ISPY1_expression_all.csv.gz', index_col=0); x.index = x.index.astype(str)
    mri = pd.read_csv(A6 / 'ISPY1_mri_all.csv', index_col=0)
    mri.index = mri.index.str.replace('ISPY1_', '', regex=False)
    genes = dev['rna'].columns
    ids = audit.index
    E = pd.DataFrame(np.nan, index=ids, columns=genes)
    has_x = [i for i in ids if audit.loc[i, 'group'] in ('complete', 'rna_only') and i in x.index]
    E.loc[has_x] = rank_normalize(x.loc[has_x, genes].to_numpy(float))
    I = pd.DataFrame(np.nan, index=ids, columns=dev['mri'].columns)
    has_m = [i for i in ids if audit.loc[i, 'group'] in ('complete', 'mri_only') and i in mri.index]
    I.loc[has_m] = mri.loc[has_m, dev['mri'].columns].to_numpy(float)
    meta = audit[['age', 'hr', 'her2']].copy(); meta['arm'] = 'AC_based'
    clin = clinical_frame(meta)
    clin['anti_her2'] = clin['anti_her2'].fillna(0.)  # therapy code unavailable without expression record
    blocks = {'rna': E, 'mri': I, 'clinical': clin}
    blocks['rppa'] = pd.DataFrame(np.nan, index=ids, columns=dev['rppa'].columns)
    return blocks, audit.pcr.astype(int).to_numpy(), audit.group.to_numpy()


def run_natural(dev, y, ext):
    blocks, yn, group = natural_blocks(dev)
    rows, preds = [], {}
    for dn in ('v1', 'carf2'):
        d, experts, fus, _, _ = frozen(dn, dev, y, ext)
        Q, QA = d.predict(experts, blocks)
        cte = ctx_of(QA, blocks['clinical'])
        out = {n: m.predict(Q, cte) for n, m in fus.items()}
        out['clinical'] = Q[:, -1]
        for sub in ['all', 'partial', 'rna_only', 'mri_only', 'complete']:
            sel = {'all': np.ones(len(yn), bool), 'partial': np.isin(group, ['rna_only', 'mri_only'])}.get(sub, group == sub)
            if sub == 'all':
                sel = group != 'clinical_only'
            for n, p in out.items():
                rows.append({'design': dn, 'subset': sub, 'n': int(sel.sum()), 'pos': int(yn[sel].sum()),
                             'model': n, **scores(yn[sel], p[sel])})
        preds[dn] = out
    pd.DataFrame(rows).to_csv(RES / 'natural.csv', index=False)
    pd.DataFrame({'group': group, 'pcr': yn, **{f'{dn}:{n}': p for dn, o in preds.items() for n, p in o.items()}}
                 ).to_csv(RES / 'natural_predictions.csv', index=False)


# ------------------------------------------------------------------ GSE25066
def run_gse(dev, y):
    sys.path.insert(0, str(ROOT / 'experiments'))
    import e7_gse25066 as e7  # read-only reuse of the parsers
    from scripts.run_revision2_analysis import rank_normalize
    e7.RAW = OUT.parent / 'data_gse25066'
    meta, expr = e7.parse_series()
    clin = e7.clinical_map(meta)
    genes, _ = e7.probe_to_symbol(expr)
    lab = clin[clin.pcr.notna() & (clin.source != 'ISPY')].copy()
    lab['pcr'] = lab.pcr.astype(int)
    e2 = pd.read_csv(DATA / 'ISPY2_expression.csv.gz', index_col=0); e2.index = e2.index.astype(str)
    G = e2.columns.intersection(genes.columns)
    ids2 = dev['rna'].index
    dev_g = dict(dev); dev_g['rna'] = pd.DataFrame(rank_normalize(e2.loc[ids2, G].to_numpy()), index=ids2, columns=G)
    raw = genes.loc[lab.index, G].to_numpy(float)
    m = lab[['age', 'hr', 'her2']].copy(); m['arm'] = 'AC_based'
    blocks = {'rna': pd.DataFrame(rank_normalize(raw), index=lab.index, columns=G),
              'mri': pd.DataFrame(np.nan, index=lab.index, columns=dev['mri'].columns),
              'rppa': pd.DataFrame(np.nan, index=lab.index, columns=dev['rppa'].columns)}
    c = clinical_frame(m.assign(arm='')); c['anti_her2'] = 0.  # GSE25066: HER2-directed therapy not given
    blocks['clinical'] = c
    ye = lab.pcr.to_numpy()
    rows = []
    for dn in ('v1', 'carf2'):
        d = DESIGNS[dn]
        bl = {k: blocks[k] for k in ['rna', 'mri', 'clinical'] + (['rppa'] if 'rppa' in d.optional else [])}
        P, PA = oof(d, dev_g, y, SEED + 100)
        experts = d.fit(dev_g, y)
        ctr = ctx_of(PA, dev_g['clinical'])
        fus = {n: make(n).fit(P, y, ctr) for n in MODELS}
        Q, QA = d.predict(experts, bl)
        for r in evaluate(d, fus, bl, experts, ye, Q, QA):
            rows.append({'design': dn, 'scenario': 'intact', 'rep': 0, **r})
        avail = np.arange(len(ye))
        for rep in range(REPEATS):
            rng = np.random.default_rng([SEED, 25066, rep])
            src = rng.permutation(avail)
            q, qa = Q.copy(), QA.copy(); q[:, 0] = Q[src, 0]; qa[:, 0] = QA[src, 0]
            for r in evaluate(d, fus, bl, experts, ye, q, qa):
                rows.append({'design': dn, 'scenario': 'rna_swap', 'rep': rep, **r})
            x = raw.copy()
            for i in range(len(x)):
                x[i, rng.choice(x.shape[1], x.shape[1] // 2, replace=False)] = np.nan
            b2 = dict(bl); b2['rna'] = pd.DataFrame(rank_normalize(x), index=lab.index, columns=G)
            for r in evaluate(d, fus, b2, experts, ye):
                rows.append({'design': dn, 'scenario': 'rna_dropout50', 'rep': rep, **r})
        print('GSE25066 done', dn, flush=True)
    df = pd.DataFrame(rows); df['n'] = len(ye); df['pos'] = int(ye.sum())
    df.to_csv(RES / 'gse25066.csv', index=False)


# ------------------------------------------------------------------ permutation
def perm_job(b, dev, y, ext, ye):
    from pipeline import Design
    warnings.filterwarnings('ignore')
    d = DESIGNS['carf2']
    yp = np.random.default_rng([SEED, 555, b]).permutation(y)
    P, PA = oof(d, dev, yp, SEED + 100)
    Q, QA = d.predict(d.fit(dev, yp), ext)
    ctr, cte = ctx_of(PA, dev['clinical']), ctx_of(QA, ext['clinical'])
    out = {'perm': b}
    for n in ['carf2', 'stack', 'mask_stack']:
        out[n] = roc_auc_score(ye, make(n).fit(P, yp, ctr).predict(Q, cte))
    return out


def run_perm(dev, y, ext, ye, n=200):
    out = Parallel(n_jobs=60, verbose=2)(delayed(perm_job)(b, dev, y, ext, ye) for b in range(n))
    pd.DataFrame(out).to_csv(RES / 'permutation.csv', index=False)


def main(which):
    RES.mkdir(parents=True, exist_ok=True)
    dev, y, ext, ye, M, Me = load_data()
    if 'input' in which: run_input(dev, y, ext, ye)
    if 'natural' in which: run_natural(dev, y, ext)
    if 'gse' in which: run_gse(dev, y)
    if 'perm' in which: run_perm(dev, y, ext, ye)


if __name__ == '__main__':
    import run_extra
    run_extra.main(sys.argv[1:])
