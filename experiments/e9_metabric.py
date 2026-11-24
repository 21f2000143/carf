#!/usr/bin/env python3
"""E9: CARF on METABRIC (cBioPortal brca_metabric), a new task with its own split.

Task (pre-specified in results/e9_metabric/summary.md before any model was fit):
  outcome  = breast-cancer death within 60 months (VITAL_STATUS 'Died of Disease' & OS_MONTHS <= 60)
             vs OS_MONTHS > 60; DSS-censored before 60 months excluded.
  anchor   = clinical expert (age, size, grade, nodes+, ER, HER2), column 1.
  optional = expression expert (rank-normalised Illumina array), column 0. Column 2 (mri) is NaN for all.
  split    = COHORT 1-2 train, 3-5 test. Experts + fusions fit on train; gates on inner 5-fold OOF.
All CARF/baseline settings are the fixed I-SPY settings (B=1, gamma=0.25). Nothing tuned on test labels.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import hashlib
import itertools
import json
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tier1 import offset_models, ece, MIKE  # noqa: E402  (puts mike2026 on sys.path)
from e5_robust_offset import RobustOffsetLR  # noqa: E402
from e1_bootstrap import auc_rows, summarize  # noqa: E402
import a7_moddrop  # noqa: E402
from src.anchored_fusion import AnchoredResidualFusion  # noqa: E402
from scripts.run_revision2_analysis import fit_expert, rank_normalize  # noqa: E402
from scripts.run_revision3_analysis import (fit_fusions, predict_fusions, mask_stack_features,  # noqa: E402
                                            stress_scenarios)

SEED = 20260923
DATA = HERE / 'data_e9'
OUT = HERE / 'results/e9_metabric'
EPS = 1e-7
NBOOT = 2000
REPEATS = 30
URLS = {
    'tarball_attempted': 'https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz (HTTP 403)',
    'data_clinical_patient.txt': 'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_patient.txt',
    'data_clinical_sample.txt': 'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_sample.txt',
    'data_mrna_illumina_microarray.txt': 'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_mrna_illumina_microarray.txt',
}
CLIN = ['age', 'size', 'grade', 'nodes', 'er', 'her2']
# Models reported. 'mri' and 'carf_no_mri' are dropped (mri is absent for everyone, so
# carf_no_mri == carf); 'no_mri_stack' is kept (it equals the two-expert stack).
DROP = {'mri', 'carf_no_mri'}


def lg(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- data
def load_data():
    p = pd.read_csv(DATA / 'data_clinical_patient.txt', sep='\t', comment='#').set_index('PATIENT_ID')
    s = pd.read_csv(DATA / 'data_clinical_sample.txt', sep='\t', comment='#').set_index('PATIENT_ID')
    assert p.index.is_unique and s.index.is_unique
    d = p.join(s[['ER_STATUS', 'HER2_STATUS', 'GRADE', 'TUMOR_SIZE']])
    M = pd.DataFrame({'age': d.AGE_AT_DIAGNOSIS, 'size': d.TUMOR_SIZE, 'grade': d.GRADE,
                      'nodes': d.LYMPH_NODES_EXAMINED_POSITIVE,
                      'er': d.ER_STATUS.map({'Positive': 1., 'Negative': 0.}),
                      'her2': d.HER2_STATUS.map({'Positive': 1., 'Negative': 0.})}, index=d.index).astype(float)
    t, vs = d.OS_MONTHS, d.VITAL_STATUS
    y = pd.Series(np.nan, index=d.index)
    known = t.notna() & vs.notna()
    y[known & (t > 60)] = 0.
    y[known & (t <= 60) & (vs == 'Died of Disease')] = 1.
    # Living or Died of Other Causes with t <= 60: DSS-censored before 5 y -> excluded (NaN).
    M['y'] = y
    M['cohort'] = d.COHORT
    M['vital'] = vs
    M['os_months'] = t

    cache = DATA / 'expression_raw.npz'
    if not cache.exists():
        e = pd.read_csv(DATA / 'data_mrna_illumina_microarray.txt', sep='\t')
        genes = (e.Hugo_Symbol.astype(str) + '|' + e.Entrez_Gene_Id.astype(str)).to_numpy()
        X = e.drop(columns=['Hugo_Symbol', 'Entrez_Gene_Id'])
        np.savez_compressed(cache, raw=X.to_numpy(float).T, samples=X.columns.to_numpy(str), genes=genes.astype(str))
    with np.load(cache, allow_pickle=False) as z:
        raw, samples, genes = z['raw'], z['samples'], z['genes']
    genes = np.array([f'{g}|r{i}' for i, g in enumerate(genes)])  # unique names (duplicate symbol/Entrez rows)
    return M, raw, samples, genes


# ----------------------------------------------------------------------------- experts
def fit_clinical(X, y):
    # Mirrors fit_expert(kind='clinical'): median impute -> scaler -> LR C=1 balanced (liblinear),
    # on the six METABRIC clinical variables (fit_expert hard-codes the I-SPY columns).
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=1, class_weight='balanced', max_iter=1000,
                                            solver='liblinear')).fit(X[CLIN], y)


def pred_clinical(m, X):
    return m.predict_proba(X[CLIN])[:, 1]


def pred_rna(m, E):
    return m.predict_proba(E)[:, 1]


def oof_experts(E, M, y, seed):
    P = np.full((len(y), 3), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(E, y):
        P[te, 0] = pred_rna(fit_expert(E.iloc[tr], y[tr], 'rna', seed), E.iloc[te])
        P[te, 1] = pred_clinical(fit_clinical(M.iloc[tr], y[tr]), M.iloc[te])
    return P


# ----------------------------------------------------------------------------- fusions
def fit_fusions_2mod(P, y):
    """fit_fusions with an all-NaN mri column.

    fit_fusions(P) itself fails on the NaN column (sklearn LR in 'stack'). Here stack
    and no_mri_stack are fit on P with NaN filled by 0.5 -- the same fill
    predict_fusions applies at prediction time -- while mask_stack and all CARF variants
    are fit by fit_fusions on the NaN matrix (they handle missingness natively).
    """
    filled = np.nan_to_num(P, nan=.5)
    std = lambda: LogisticRegression(C=1, class_weight='balanced', solver='liblinear')
    models = {'stack': std().fit(filled, y),
              'no_mri_stack': std().fit(filled[:, :2], y)}
    # mask_stack + CARF variants: same code as fit_fusions, on the NaN matrix.
    blocks = []
    for active in itertools.product([False, True], repeat=2):
        q = P.copy()
        for j, keep in zip([0, 2], active):
            if not keep:
                q[:, j] = np.nan
        blocks.append(mask_stack_features(q))
    models['mask_stack'] = std().fit(np.vstack(blocks), np.tile(y, 4), sample_weight=np.full(len(y) * 4, .25))
    configs = {'carf': {}, 'carf_no_robust': {'robust_weight': 0},
               'carf_no_context': {'context': False}, 'carf_unbounded': {'bounded': False}}
    for name, kw in configs.items():
        models[name] = AnchoredResidualFusion(**kw).fit(P, y)
    q = P.copy(); q[:, 2] = np.nan
    models['carf_no_mri'] = AnchoredResidualFusion().fit(q, y)
    return models


def fit_all(P, y):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    fus = fit_fusions_2mod(P, y)
    off = offset_models(P, y)
    off['robust_offset_lr'] = RobustOffsetLR(robust_weight=.25).fit(P, y)
    mlp, params = a7_moddrop.fit(P, y, 'moddrop_mlp')
    return {'fus': fus, 'off': off, 'mlp': mlp, 'mlp_params': params}


def predict_all(F, Q):
    p = {k: v for k, v in predict_fusions(F['fus'], Q).items() if k not in DROP}
    p.update({k: m.predict_proba(Q)[:, 1] for k, m in F['off'].items()})
    p['moddrop_mlp'] = a7_moddrop.predict(F['mlp'], Q)
    return p


def score(y, p, a):
    return {'auroc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p), 'ece': ece(y, p),
            'log_loss': log_loss(y, p, labels=[0, 1]), 'max_disp': float(np.abs(lg(p) - a).max())}


# ----------------------------------------------------------------------------- bootstrap
def paired_boot(y, reals, models, extra, ref='carf', seed=SEED):
    """reals: list of dicts model->pred (R realisations). AUROC averaged over realisations."""
    A = np.stack([np.column_stack([r[m] for m in models]) for r in reals])
    k0 = models.index(ref)
    pt = np.mean([auc_rows(y, a) for a in A], axis=0)
    point = {m: pt[k0] - pt[k] for k, m in enumerate(models) if k != k0}
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in point}
    for _ in range(NBOOT):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        auc = np.mean([auc_rows(y[ix], a[ix]) for a in A], axis=0)
        for k, m in enumerate(models):
            if k != k0:
                boot[m].append(auc[k0] - auc[k])
    return summarize(boot, point, {**extra, 'metric': 'auroc', 'carf_value': pt[k0]})


def paired_boot_brier(y, pred, models, extra, seed=SEED):
    rng = np.random.default_rng(seed)
    point = {m: brier_score_loss(y, pred['carf']) - brier_score_loss(y, pred[m]) for m in models if m != 'carf'}
    boot = {m: [] for m in point}
    for _ in range(NBOOT):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        b0 = np.mean((pred['carf'][ix] - y[ix]) ** 2)
        for m in point:
            boot[m].append(b0 - np.mean((pred[m][ix] - y[ix]) ** 2))
    return summarize(boot, point, {**extra, 'metric': 'brier', 'carf_value': brier_score_loss(y, pred['carf'])})


# ----------------------------------------------------------------------------- main
def dropout_Q(Q, rna_model, raw_te, ids, genes, rep):
    rng = np.random.default_rng([SEED, 9, rep])
    raw = raw_te.copy()
    n, g = raw.shape
    for i in range(n):
        raw[i, rng.choice(g, int(round(.5 * g)), replace=False)] = np.nan
    q = Q.copy()
    q[:, 0] = pred_rna(rna_model, pd.DataFrame(rank_normalize(raw), index=ids, columns=genes))
    return q


def main():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    M, raw, samples, genes = load_data()
    sidx = {s: i for i, s in enumerate(samples)}
    M['has_expr'] = M.index.isin(samples)
    elig = M[M.y.notna() & M.cohort.isin([1, 2, 3, 4, 5])]
    tr_ids = elig.index[elig.cohort.isin([1, 2]) & elig.has_expr]
    te_ids = elig.index[elig.cohort.isin([3, 4, 5]) & elig.has_expr]
    nat_ids = elig.index[elig.cohort.isin([3, 4, 5]) & ~elig.has_expr]
    counts = {
        'cohort1_5_total': int(M.cohort.isin([1, 2, 3, 4, 5]).sum()),
        'excluded_dss_censored_or_unknown': int((M.cohort.isin([1, 2, 3, 4, 5]) & M.y.isna()).sum()),
        'excluded_no_expression_train': int((elig.cohort.isin([1, 2]) & ~elig.has_expr).sum()),
        'train_n': len(tr_ids), 'train_events': int(M.loc[tr_ids, 'y'].sum()),
        'test_n': len(te_ids), 'test_events': int(M.loc[te_ids, 'y'].sum()),
        'natural_missing_test_n': len(nat_ids),
        'died_other_causes_le60_excluded': int((M.cohort.isin([1, 2, 3, 4, 5]) & (M.vital == 'Died of Other Causes') & (M.os_months <= 60)).sum()),
        'n_genes': len(genes),
    }
    print(counts, flush=True)
    raw_tr = raw[[sidx[i] for i in tr_ids]]
    raw_te = raw[[sidx[i] for i in te_ids]]
    Etr = pd.DataFrame(rank_normalize(raw_tr), index=tr_ids, columns=genes)
    Ete = pd.DataFrame(rank_normalize(raw_te), index=te_ids, columns=genes)
    Mtr, Mte = M.loc[tr_ids], M.loc[te_ids]
    ytr, yte = Mtr.y.to_numpy(int), Mte.y.to_numpy(int)

    # Experts: inner-OOF P on train for the gates; final experts on all train -> Q on test.
    P = oof_experts(Etr, Mtr, ytr, SEED + 100)
    rna_m = fit_expert(Etr, ytr, 'rna', SEED + 100)
    clin_m = fit_clinical(Mtr, ytr)
    Q = np.column_stack([pred_rna(rna_m, Ete), pred_clinical(clin_m, Mte), np.full(len(yte), np.nan)])
    np.savez_compressed(OUT / 'expert_probs.npz', P=P, Q=Q, train_ids=tr_ids.to_numpy(str),
                        test_ids=te_ids.to_numpy(str), ytr=ytr, yte=yte)

    # Document NaN handling of the unmodified fit_fusions.
    nan_check = {}
    try:
        fit_fusions(P, ytr)
        nan_check['fit_fusions_on_nan_mri'] = 'ok'
    except Exception as ex:  # noqa: BLE001
        nan_check['fit_fusions_on_nan_mri'] = f'fails: {type(ex).__name__}: {str(ex)[:120]}'
    F = fit_all(P, ytr)
    pred = predict_all(F, Q)
    a = lg(Q[:, 1])
    models = list(pred)
    for k, v in pred.items():
        assert np.isfinite(v).all(), k
    nan_check['all_predictions_finite'] = True
    nan_check['carf_equals_carf_no_mri_max_abs_diff'] = float(np.abs(
        F['fus']['carf'].predict_proba(Q)[:, 1] - F['fus']['carf_no_mri'].predict_proba(Q)[:, 1]).max())

    oof_auc = {'rna_oof_train': roc_auc_score(ytr, P[:, 0]), 'clinical_oof_train': roc_auc_score(ytr, P[:, 1])}
    metrics = pd.DataFrame([{'model': k, 'n': len(yte), 'events': int(yte.sum()), **score(yte, v, a)}
                            for k, v in pred.items()])
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame({'patient_id': te_ids, 'y': yte, **pred}).to_csv(OUT / 'test_predictions.csv', index=False)
    print(metrics.round(4).to_string(index=False), flush=True)

    # Stress (frozen models).
    keep = {'intact', 'rna_missing', 'rna_neutral', 'rna_low', 'rna_high', 'rna_permuted', 'rna_noisy'}
    scen = [(s, r, q) for s, r, q in stress_scenarios(Q) if s in keep]
    drops = Parallel(n_jobs=30)(delayed(dropout_Q)(Q, rna_m, raw_te, te_ids, genes, r) for r in range(REPEATS))
    scen += [('rna_dropout50', r, q) for r, q in enumerate(drops)]
    stress, reals = [], {}
    for s, r, q in scen:
        p = predict_all(F, q)
        reals.setdefault(s, []).append(p)
        stress += [{'scenario': s, 'repeat': r, 'model': k, **score(yte, v, a)} for k, v in p.items()]
    stress = pd.DataFrame(stress)
    stress.to_csv(OUT / 'stress_raw.csv', index=False)
    order = list(dict.fromkeys(stress.scenario))
    ss = (stress.groupby(['scenario', 'model'])
          .agg(auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'), brier_mean=('brier', 'mean'),
               ece_mean=('ece', 'mean'), max_disp=('max_disp', 'max'), n_real=('auroc', 'size')).reset_index())
    ss['scenario'] = pd.Categorical(ss.scenario, order, ordered=True)
    ss = ss.sort_values(['scenario', 'model'])
    ss.to_csv(OUT / 'stress.csv', index=False)

    # Paired bootstrap: intact (AUROC, Brier) + stress scenarios (AUROC over realisations).
    jobs = [(s, reals[s]) for s in ['intact', 'rna_permuted', 'rna_noisy', 'rna_missing', 'rna_dropout50']]
    brows = Parallel(n_jobs=len(jobs))(delayed(paired_boot)(yte, R, models, {'scenario': s, 'n_real': len(R)})
                                        for s, R in jobs)
    brows = [r for b in brows for r in b] + paired_boot_brier(yte, pred, models, {'scenario': 'intact', 'n_real': 1})
    boot = pd.DataFrame(brows)
    boot.to_csv(OUT / 'bootstrap.csv', index=False)

    fus = F['fus']
    gates = {n: {'coef': m.coef_.tolist(), 'optimization': m.optimization_}
             for n, m in fus.items() if n.startswith('carf')}
    gates['carf']['modality_budgets_[rna,mri]'] = fus['carf'].modality_budgets().tolist()
    gates['offset'] = {k: m.theta_.tolist() for k, m in F['off'].items()}
    gates['moddrop_params'] = F['mlp_params']
    files = {p.name: sha256(p) for p in sorted(DATA.glob('data_*.txt'))}
    code = {str(p.relative_to(HERE.parent)): sha256(p) for p in
            [Path(__file__), HERE / 'tier1.py', HERE / 'e5_robust_offset.py', HERE / 'a7_moddrop.py',
             HERE / 'e1_bootstrap.py', MIKE / 'src/anchored_fusion.py',
             MIKE / 'scripts/run_revision2_analysis.py', MIKE / 'scripts/run_revision3_analysis.py']}
    prov = {'urls': URLS, 'data_sha256': files, 'code_sha256': code, 'counts': counts,
            'expert_oof_auroc_train': oof_auc, 'nan_handling': nan_check, 'gates': gates, 'seed': SEED,
            'split': 'train COHORT 1-2, test COHORT 3-5; complete cases with expression',
            'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'pandas': pd.__version__}
    (OUT / 'provenance.json').write_text(json.dumps(prov, indent=2, default=str))

    show = ['clinical', 'rna', 'mean', 'stack', 'mask_stack', 'offset_lr', 'offset_lr_masked',
            'robust_offset_lr', 'moddrop_mlp', 'carf_no_robust', 'carf']
    print(json.dumps({**oof_auc, **nan_check}, indent=1))
    print(ss.pivot(index='scenario', columns='model', values='auroc_mean')[show].round(3).to_string())
    print(ss.pivot(index='scenario', columns='model', values='max_disp')[show].round(2).to_string())
    print(boot[boot.comparator.isin(show)].round(4).to_string(index=False))


if __name__ == '__main__':
    import e9_metabric  # run via module so joblib pickles functions by reference
    e9_metabric.main()
