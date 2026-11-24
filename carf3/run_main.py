#!/usr/bin/env python3
"""CARF-2 main study: internal 5x5 CV on I-SPY2 (with internal stress), external I-SPY1 (frozen).

Designs
  v1    : original study (clinical age/HR/HER2 anchor; RNA + MRI).
  carf2 : treatment-aware anchor; RNA + MRI + RPPA (RPPA available for 75% of I-SPY2, none of I-SPY1).
  treat, rppa : design ablations (one change each).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import sys
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold

from common import OUT, SEED, DATA, subtype_group
from pipeline import load_data, Design, cached
from experts import TREAT_CLIN, consistency
from models import make, ALL, GATED
from carf2 import AnchorContext

EPS = 1e-7
RES = OUT / 'main'
DESIGNS = {
    'v1': Design('v1'),
    'treat': Design('treat', clin_cols=TREAT_CLIN),
    'rppa': Design('rppa', optional=('rna', 'mri', 'rppa')),
    'carf2': Design('carf2', optional=('rna', 'mri', 'rppa'), clin_cols=TREAT_CLIN),
}
MODELS_BY_DESIGN = {'v1': ALL, 'carf2': ALL + GATED,
                    'treat': ['mean', 'stack', 'mask_stack', 'carf_v1', 'carf2', 'carf2_no_gate'],
                    'rppa': ['mean', 'stack', 'mask_stack', 'carf_v1', 'carf2', 'carf2_no_gate']}
N_INT_REP = 5
N_EXT_REP = 30
MIX_FRAC = 0.2   # fraction of a modality's samples affected by a semi-realistic mix-up
MIX_MODS = ('rna', 'rppa')  # specimen-based modalities


def mixup_keys():
    """Processing order (GEO accession) and processing batch of each specimen-based sample.

    RNA, I-SPY2: accession order, batch = array platform x normalization; RNA, I-SPY1: accession order,
    one batch; RPPA: accession order, batch = RPPA array (normalization key)."""
    def acc(df):
        df = df.drop_duplicates('patient_id').copy()
        df['order'] = df.gsm.str[3:].astype(int)
        return df.set_index('patient_id')
    r2 = acc(pd.read_csv(DATA / 'ISPY2_source_metadata.csv', dtype={'patient_id': str}))
    r2['batch'] = r2.platform_id + '/' + r2['adjustment method']
    r1 = acc(pd.read_csv(DATA / 'ISPY1_source_metadata.csv', dtype={'patient_id': str}))
    r1['batch'] = 'ISPY1'
    rp = acc(pd.read_csv(DATA / 'ISPY2_RPPA_source_metadata.csv', dtype={'patient_id': str}))
    rp['batch'] = rp['rppa array (normalization key)']
    return {'rna': pd.concat([r2[['order', 'batch']], r1[['order', 'batch']]]), 'rppa': rp[['order', 'batch']]}


def mixups(Q, QA, j, avail, keys, sub, rng):
    """Semi-realistic specimen mix-ups of modality j among the rows in avail; yields (name, q, qa)."""
    k = max(2, int(round(MIX_FRAC * len(avail))))
    def apply(rows, src):
        q = Q.copy(); qa = QA.copy(); q[rows, j] = Q[src, j]; qa[rows, j] = QA[src, j]
        return q, qa
    # Offset: a contiguous run of k samples in processing order is assigned to the next patient.
    order = avail[np.argsort(keys.order.to_numpy()[avail], kind='stable')]
    start = rng.integers(len(order))
    run = order[(start + np.arange(k)) % len(order)]
    yield ('offset',) + apply(run, np.roll(run, 1))
    # Within-batch swap: k random samples are exchanged among themselves within their processing batch.
    pick = rng.choice(avail, k, replace=False)
    src = pick.copy(); b = keys.batch.to_numpy()[pick]
    for v in np.unique(b):
        idx = np.where(b == v)[0]
        src[idx] = pick[rng.permutation(idx)]
    yield ('batchswap',) + apply(pick, src)
    # Same-subtype swap: all samples permuted within HR/HER2 subtype (invisible to the verification gate).
    src = avail.copy()
    for v in np.unique(sub[avail]):
        idx = np.where(sub[avail] == v)[0]
        src[idx] = avail[rng.permutation(idx)]
    yield ('subtypeswap',) + apply(avail, src)


def ctx_of(PA, clin):
    """Anchor context: auxiliary HR/HER2 predictions of each optional modality and the recorded values."""
    return AnchorContext(PA, clin[['hr', 'her2']].to_numpy(float))


def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {'auroc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p), 'log_loss': log_loss(y, p, labels=[0, 1])}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def scenarios(Q, QA, optional, n_rep, seed, full=False, mix=None):
    """Output-level scenarios. Swap = the modality's outputs (pCR and auxiliary) are exchanged
    between patients who have that modality (sample mix-up / uninformative modality)."""
    yield 'intact', 0, Q, QA
    q = Q.copy(); q[:, :-1] = np.nan
    yield 'all_missing', 0, q, QA
    for j, mod in enumerate(optional):
        avail = np.where(np.isfinite(Q[:, j]))[0]
        if len(avail) == 0:
            continue
        q = Q.copy(); q[:, j] = np.nan
        yield f'{mod}_missing', 0, q, QA
        if full:
            for name, v in [('low', .01), ('high', .99), ('neutral', .5)]:
                q = Q.copy(); q[avail, j] = v
                yield f'{mod}_{name}', 0, q, QA
        for r in range(n_rep):
            rng = np.random.default_rng([seed, j, r])
            src = rng.permutation(avail)
            q = Q.copy(); qa = QA.copy(); q[avail, j] = Q[src, j]; qa[avail, j] = QA[src, j]
            yield f'{mod}_swap', r, q, qa
            if mix is not None and mod in MIX_MODS:
                keys, sub = mix
                for name, q2, qa2 in mixups(Q, QA, j, avail, keys[mod], sub, np.random.default_rng([seed, j, r, 99])):
                    yield f'{mod}_{name}', r, q2, qa2
            if full:
                q = Q.copy(); q[avail, j] = np.clip(Q[avail, j] + rng.normal(0, .25, len(avail)), .001, .999)
                yield f'{mod}_noisy', r, q, QA


def fit_all(names, P, ytr, ctr):
    warnings.filterwarnings('ignore')
    fitted = {}
    for n in names:
        fitted[n] = make(n).fit(P, ytr, ctr)
    return fitted


def logits(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def run_split(dname, key, tr, te, dev, y, ext, ye, keys=None):
    d = DESIGNS[dname]
    tb = ext if te is None else {k: v.iloc[te] for k, v in dev.items()}
    seed = SEED + (100 if te is None else int(key.split('_')[1]))
    P, PA, Q, QA = cached(d, key, dev, y, tr, tb, seed)
    clin_tr = dev['clinical'].iloc[tr]; clin_te = tb['clinical']
    ctr, yt = ctx_of(PA, clin_tr), (ye if te is None else y[te])
    fitted = fit_all(MODELS_BY_DESIGN[dname], P, y[tr], ctr)
    rows, preds, disp = [], {}, []
    experts = {mod: j for j, mod in enumerate(d.optional)}
    ids = tb['clinical'].index
    mix = None if keys is None or dname != 'carf2' else (
        {m: v.reindex(ids) for m, v in keys.items()}, subtype_group(tb['clinical']))
    scen_preds = {}
    for sc, rep, q, qa in scenarios(Q, QA, d.optional, N_EXT_REP if te is None else N_INT_REP,
                                    SEED + 7, full=te is None, mix=mix):
        cte = ctx_of(qa, clin_te)
        a = logits(q[:, -1])
        out = {n: m.predict(q, cte) for n, m in fitted.items()}
        if sc == 'intact':
            out['clinical'] = q[:, -1]
            for mod, j in experts.items():
                if np.isfinite(q[:, j]).all():
                    out[mod] = q[:, j]
            preds = out
        if te is None and dname == 'carf2':
            scen_preds[(sc, rep)] = {n: p.astype(np.float32) for n, p in out.items()}
        for n, p in out.items():
            r = {'design': dname, 'split': key, 'scenario': sc, 'rep': rep, 'model': n, **scores(yt, p),
                 'max_disp': float(np.abs(logits(p) - a).max())}
            if sc == 'intact':
                r['ece'] = ece(yt, p)
            rows.append(r)
    info = {n: {'budget': float(getattr(m, 'budget_', m.m.budget)), 'coef': m.m.coef_.tolist()}
            for n, m in fitted.items() if n.startswith('carf')}
    return {'rows': rows, 'preds': preds, 'yt': yt, 'test_ids': list(tb['clinical'].index), 'gates': info,
            'key': key, 'design': dname, 'scen_preds': scen_preds}


def main(repeats=5):
    RES.mkdir(parents=True, exist_ok=True)
    dev, y, ext, ye, M, Me = load_data()
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED).split(np.zeros(len(y)), y))
    jobs = [(dn, f'fold_{i}', tr, te) for dn in DESIGNS for i, (tr, te) in enumerate(folds)]
    jobs += [(dn, 'external', np.arange(len(y)), None) for dn in DESIGNS]
    keys = mixup_keys()
    res = Parallel(n_jobs=int(os.environ.get('NJOBS', 60)), verbose=5)(
        delayed(run_split)(dn, k, tr, te, dev, y, ext, ye, keys) for dn, k, tr, te in jobs)
    rows = [r for o in res for r in o['rows']]
    df = pd.DataFrame(rows)
    df[df.split != 'external'].to_csv(RES / 'internal.csv', index=False)
    df[df.split == 'external'].to_csv(RES / 'external.csv', index=False)
    gates, pred_rows = {}, []
    for o in res:
        gates[f"{o['design']}/{o['key']}"] = o['gates']
        for k, pid in enumerate(o['test_ids']):
            pred_rows.append({'design': o['design'], 'split': o['key'], 'patient_id': pid, 'pcr': int(o['yt'][k]),
                              **{n: float(p[k]) for n, p in o['preds'].items()}})
    (RES / 'gates.json').write_text(json.dumps(gates, indent=1))
    # External per-scenario predictions of the main design, for patient-bootstrap intervals.
    sp = next(o for o in res if o['design'] == 'carf2' and o['key'] == 'external')
    names = sorted({n for v in sp['scen_preds'].values() for n in v})
    keys_ = sorted(sp['scen_preds'])
    arr = np.full((len(keys_), len(names), len(sp['yt'])), np.nan, np.float32)
    for i, k in enumerate(keys_):
        for jn, n in enumerate(names):
            if n in sp['scen_preds'][k]:
                arr[i, jn] = sp['scen_preds'][k][n]
    np.savez_compressed(RES / 'external_scenarios.npz', pred=arr, y=sp['yt'], models=np.array(names),
                        scenario=np.array([k[0] for k in keys_]), rep=np.array([k[1] for k in keys_]))
    pd.DataFrame(pred_rows).to_csv(RES / 'predictions.csv.gz', index=False)
    s = df[df.scenario.isin(['intact', 'rna_swap'])].groupby(['design', 'model', df.split == 'external', 'scenario']).auroc.mean()
    print(s.unstack([2, 3]).round(3).to_string())


if __name__ == '__main__':
    import run_main  # run via the module so workers unpickle functions by reference
    run_main.main()
