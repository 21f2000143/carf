#!/usr/bin/env python3
"""E8: frozen external evaluation of CARF and all baselines on GEO GSE20194
(MAQC-II BR / MDACC, pCR, Affymetrix U133A = GPL96).

Protocol (frozen before any GSE20194 label was looked at by a model):
  * GSE20194 cohort = one array per patient, the canonical MAQC-II BR set
    (MAQC_T training + MAQC_V validation, n=230). Technical replicate arrays
    (title suffix R1, status MDA_R) and MAQC_Q "Not used" arrays are excluded
    (label-free decision, from GEO sample descriptions only).
  * Clinical: age; hr = ER positive OR PR positive (IHC status); her2 = HER2 status P.
  * GPL96 probe -> gene symbol (single-symbol probes only), mean of log2 MAS5
    values per symbol, then patient-wise rank_normalize over G.
  * G = I-SPY2 expression genes intersected with GSE20194 symbols.
  * Experts, OOF P and every fusion are refit on all of I-SPY2 exactly like the
    external model of run_revision3_analysis (oof/experts with SEED+100), with
    the RNA expert restricted to G. CARF fixed at B=1, gamma=0.25. MRI absent (NaN).
  * No step uses GSE20194 labels except the final metric computation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import gzip
import hashlib
import json
import platform
import sys
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tier1 import offset_models, platt_carf, ece, SEED, EPS, MIKE  # noqa: E402
from e5_robust_offset import RobustOffsetLR  # noqa: E402
from e1_bootstrap import auc_rows  # noqa: E402
import a7_moddrop  # noqa: E402
from scripts.run_revision2_analysis import (common_data, experts, predict_expert, oof,  # noqa: E402
                                            rank_normalize, DATA)
from scripts.run_revision3_analysis import (fit_fusions, predict_fusions, scores,  # noqa: E402
                                            stress_scenarios)

RAW = HERE / 'data_e8'
OUT = HERE / 'results/e8_gse20194'
NBOOT = 2000
REPEATS = 30
DROPOUT = 0.5
URLS = {
    'GSE20194_series_matrix.txt.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20194/matrix/GSE20194_series_matrix.txt.gz',
    'GPL96_full.txt': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL96&targ=self&form=text&view=data',
    'GSE20194_MDACC_Sample_Info.xls.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20194/suppl/GSE20194_MDACC_Sample_Info.xls.gz',
    'GSE25066_series_matrix.txt.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25066/matrix/GSE25066_series_matrix.txt.gz',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def lg(p):
    return logit(np.clip(p, EPS, 1 - EPS))


# ----------------------------------------------------------------- parsing
def series_meta(path):
    d = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            if not line.startswith('!Sample_'):
                continue
            r = line.rstrip('\n').split('\t')
            key, vals = r[0], [v.strip('"') for v in r[1:]]
            if key == '!Sample_characteristics_ch1':
                for i, v in enumerate(vals):
                    if ':' in v:
                        a, b = v.split(':', 1)
                        d.setdefault(a.strip(), [''] * len(vals))[i] = b.strip()
            elif key not in d:  # first line wins (e.g. first !Sample_description)
                d[key] = vals
    return pd.DataFrame(d)


def series_table(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
        df = pd.read_csv(f, sep='\t', index_col=0, comment='!', na_values=['null', 'NA', ''])
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors='coerce')


def gpl96_symbols():
    lines = open(RAW / 'GPL96_full.txt').read().split('\n')
    start = lines.index('!platform_table_begin') + 1
    end = lines.index('!platform_table_end')
    from io import StringIO
    t = pd.read_csv(StringIO('\n'.join(lines[start:end])), sep='\t', dtype=str)
    sym = t.set_index('ID')['Gene Symbol'].fillna('').str.strip()
    return sym


def parse():
    m = series_meta(RAW / 'GSE20194_series_matrix.txt.gz')
    m['status'] = m['!Sample_description'].str.extract(r'Status: (\S+)')[0]
    m['patient'] = m['!Sample_title'].str.replace('BR_FNA_', '', regex=False)
    audit = {'arrays_total': len(m), 'status_counts': m.status.value_counts().to_dict()}
    keep = m[m.status.isin(['MAQC_T', 'MAQC_V'])].copy()
    assert keep.patient.is_unique and not keep.patient.str.contains('R1').any()
    unlabeled = ~keep.pcr_vs_rd.isin(['pCR', 'RD'])
    audit['excluded_unlabeled'] = int(unlabeled.sum())
    keep = keep[~unlabeled]
    meta = pd.DataFrame({
        'patient_id': keep.patient.values, 'gsm': keep['!Sample_geo_accession'].values,
        'maqc_status': keep.status.values,
        'pcr': (keep.pcr_vs_rd == 'pCR').astype(int).values,
        'age': pd.to_numeric(keep.age, errors='coerce').values,
        'er_status': keep.er_status.values, 'pr_status': keep.pr_status.values,
        'her2_status': keep['her2 status'].values,
        'treatment': keep['treatment code'].values})
    for col in ['er_status', 'pr_status', 'her2_status']:
        assert meta[col].isin(['P', 'N']).all(), col
    meta['hr'] = ((meta.er_status == 'P') | (meta.pr_status == 'P')).astype(int)
    meta['her2'] = (meta.her2_status == 'P').astype(int)
    meta = meta.set_index('patient_id')
    x = series_table(RAW / 'GSE20194_series_matrix.txt.gz')
    x = x[meta.gsm]
    audit['probes'] = x.shape[0]
    audit['value_range'] = [float(np.nanmin(x.values)), float(np.nanmax(x.values))]
    assert np.nanmax(x.values) < 30, 'expected log2 values'
    sym = gpl96_symbols().reindex(x.index).fillna('')
    single = (sym != '') & ~sym.str.contains('///')
    audit['probes_single_symbol'] = int(single.sum())
    audit['probes_multi_symbol_dropped'] = int(sym.str.contains('///').sum())
    audit['probes_no_symbol_dropped'] = int((sym == '').sum())
    g = x[single].groupby(sym[single]).mean()
    g.columns = meta.index
    expr = g.T  # patients x symbols (log2 MAS5)
    audit['symbols'] = expr.shape[1]
    meta.to_csv(RAW / 'GSE20194_meta_processed.csv')
    expr.to_csv(RAW / 'GSE20194_expression_symbol_log2.csv.gz')
    return meta, expr, audit


# ----------------------------------------------------------------- overlap audit
def overlap_audit(meta):
    """GSE25066 MDACC titles (e.g. M157) share the MDACC patient id with GSE20194
    (BR_FNA_M157). Confirm matches with clinical fields and array correlation."""
    m25 = series_meta(RAW / 'GSE25066_series_matrix.txt.gz')
    md = m25[m25.source == 'MDACC'].copy()
    md['pid'] = md['!Sample_title'].str.strip()
    ids = sorted(set(md.pid) & set(meta.index))
    res = {'gse25066_sources': m25.source.value_counts().to_dict(),
           'gse25066_mdacc_n': int(len(md)), 'id_matches': len(ids),
           'gse20194_n': int(len(meta))}
    mm = md.set_index('pid').loc[ids]
    res['age_abs_diff_max'] = float(np.nanmax(np.abs(pd.to_numeric(mm.age_years, errors='coerce').to_numpy()
                                                      - meta.loc[ids, 'age'].to_numpy())))
    y25 = mm.pathologic_response_pcr_rd.map({'pCR': 1, 'RD': 0})
    res['pcr_label_agreement'] = float((y25.to_numpy() == meta.loc[ids, 'pcr'].to_numpy()).mean())
    res['er_agreement'] = float((mm.er_status_ihc.to_numpy() == meta.loc[ids, 'er_status'].to_numpy()).mean())
    # array correlation on common probes: matched pairs vs non-matched pairs
    x20 = series_table(RAW / 'GSE20194_series_matrix.txt.gz')[meta.gsm]
    x20.columns = meta.index
    x25 = series_table(RAW / 'GSE25066_series_matrix.txt.gz')[mm['!Sample_geo_accession']]
    x25.columns = ids
    probes = x20.index.intersection(x25.index)
    a = x20.loc[probes].rank().to_numpy(); b = x25.loc[probes].rank().to_numpy()
    a = (a - a.mean(0)) / a.std(0); b = (b - b.mean(0)) / b.std(0)
    C = a.T @ b / len(probes)  # spearman, GSE20194 patients x matched GSE25066 samples
    pos = {p: i for i, p in enumerate(meta.index)}
    matched = np.array([C[pos[p], j] for j, p in enumerate(ids)])
    rest = C.copy(); rest[[pos[p] for p in ids], range(len(ids))] = np.nan
    best = np.array([meta.index[k] for k in np.nanargmax(C, axis=0)])
    res.update(common_probes=int(len(probes)), spearman_matched_median=float(np.median(matched)),
               spearman_matched_min=float(matched.min()),
               spearman_unmatched_max=float(np.nanmax(rest)),
               spearman_unmatched_median=float(np.nanmedian(rest)),
               best_match_is_same_id=float((best == np.array(ids)).mean()))
    res['id_matched_low_corr'] = [(p, float(r)) for p, r in zip(ids, matched) if r < 0.99]
    # exact-array duplicates against ALL GSE25066 samples (any source), rho > 0.99
    all25 = series_table(RAW / 'GSE25066_series_matrix.txt.gz')
    names = (m25.source + ':' + m25['!Sample_title'].str.strip()).to_numpy()
    all25 = all25[m25['!Sample_geo_accession']]
    c = all25.loc[probes].rank().to_numpy(); c = (c - c.mean(0)) / c.std(0)
    C2 = a.T @ c / len(probes)
    dup = C2 > 0.99
    res['gse20194_arrays_with_rho_gt_0.99_in_gse25066'] = int(dup.any(1).sum())
    res['dup_pairs_non_id'] = [(meta.index[i], names[j], float(C2[i, j])) for i, j in zip(*np.where(dup))
                               if names[j] != 'MDACC:' + meta.index[i]]
    res['max_rho_per_gse20194_array_summary'] = {q: float(np.quantile(C2.max(1), q)) for q in (0, .1, .5)}
    res['gse20194_not_in_gse25066'] = [meta.index[i] for i in np.where(~dup.any(1))[0]]
    res['overlap_ids'] = ids
    res['ispy_note'] = ('GSE20194 is a single-institution MDACC FNA cohort (2000-2008, Affymetrix); '
                        'I-SPY2/I-SPY1 training/external cohorts used here are Agilent I-SPY data; '
                        'no shared patient identifiers exist. GSE25066 additionally contains 83 ISPY '
                        '(I-SPY1) Affymetrix samples, relevant to the GSE25066 analysis, not to GSE20194.')
    return res


# ----------------------------------------------------------------- models
def fit_all(P, y):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    fus = fit_fusions(P, y)
    off = offset_models(P, y)
    rob = RobustOffsetLR(robust_weight=.25).fit(P, y)
    Xm, ym = a7_moddrop.masked_blocks(P, y)
    mlp = a7_moddrop.mlp((32, 16), 1e-2).fit(Xm, ym)
    platt = platt_carf(P, y, SEED + 100)
    assert fus['carf'].budget == 1.0 and fus['carf'].robust_weight == .25
    return dict(fus=fus, off=off, rob=rob, mlp=mlp, platt=platt)


def predict_all(F, Q):
    p = predict_fusions(F['fus'], Q)
    p.update({k: m.predict_proba(Q)[:, 1] for k, m in F['off'].items()})
    p['robust_offset_lr'] = F['rob'].predict_proba(Q)[:, 1]
    p['moddrop_mlp'] = a7_moddrop.predict(F['mlp'], Q)
    p['carf_platt'] = F['platt'](Q)
    return p


_S = None


def dropout_job(rep):
    S = _S
    rng = np.random.default_rng([SEED, 8, 3, rep])  # a8 seeding; rna_dropout50 is scenario index 3
    raw = S['raw'].copy()
    n, g = raw.shape
    for i in range(n):
        raw[i, rng.choice(g, int(round(DROPOUT * g)), replace=False)] = np.nan
    X = pd.DataFrame(rank_normalize(raw), index=S['index'], columns=S['genes'])
    Q = S['Q'].copy()
    Q[:, 0] = predict_expert(S['rna'], X, 'rna')
    return Q


def boot_deltas(y, sets, models, extra):
    """e1 protocol: per resample, AUROC averaged over realizations; CARF minus model."""
    rows = []
    for name, reals in sets.items():
        A = np.stack(reals)  # R x n x models
        point_auc = np.mean([auc_rows(y, a) for a in A], axis=0)
        rng = np.random.default_rng(SEED)
        bd, ba = [], []
        for _ in range(NBOOT):
            ix = rng.integers(0, len(y), len(y))
            if len(np.unique(y[ix])) < 2:
                continue
            auc = np.mean([auc_rows(y[ix], a[ix]) for a in A], axis=0)
            ba.append(auc); bd.append(auc[models.index('carf')] - auc)
        bd, ba = np.array(bd), np.array(ba)
        ic = models.index('carf')
        for k, m in enumerate(models):
            q = np.quantile(bd[:, k], [.025, .5, .975]); qa = np.quantile(ba[:, k], [.025, .975])
            rows.append({**extra, 'scenario': name, 'model': m, 'n_real': len(A),
                         'auroc': point_auc[k], 'auroc_q025': qa[0], 'auroc_q975': qa[1],
                         'carf_auroc': point_auc[ic], 'delta_carf_minus_model': point_auc[ic] - point_auc[k],
                         'delta_q025': q[0], 'delta_q50': q[1], 'delta_q975': q[2],
                         'p_delta_le_0': float((bd[:, k] <= 0).mean()), 'n_boot': len(bd)})
    return rows


def main():
    global _S
    OUT.mkdir(parents=True, exist_ok=True)
    meta, expr, audit = parse()
    y = meta.pcr.to_numpy(int)
    print('cohort', len(y), 'pCR', y.sum(), audit, flush=True)
    ov = overlap_audit(meta)
    print('overlap', {k: v for k, v in ov.items() if k != 'overlap_ids'}, flush=True)

    # I-SPY2 discovery data with the RNA expert restricted to G
    E2, M2, I2, _, _, _ = common_data()
    e2 = pd.read_csv(DATA / 'ISPY2_expression.csv.gz', index_col=0); e2.index = e2.index.astype(str)
    G = e2.columns.intersection(expr.columns)
    G = pd.Index(sorted(G))
    E2g = pd.DataFrame(rank_normalize(e2.loc[M2.index, G].to_numpy(float)), index=M2.index, columns=G)
    y2 = M2.pcr.astype(int).to_numpy()
    raw_ext = expr.loc[meta.index, G].to_numpy(float)
    Ee = pd.DataFrame(rank_normalize(raw_ext), index=meta.index, columns=G)
    genes = {'ispy2_genes': int(e2.shape[1]), 'gse20194_symbols': int(expr.shape[1]), 'G': int(len(G)),
             'ispy2_ispy1_common_genes': int(E2.shape[1]),
             'G_within_ispy2_ispy1_common': int(len(G.intersection(E2.columns))),
             'ext_nan_frac': float(np.isnan(raw_ext).mean())}
    print('genes', genes, flush=True)

    seed = SEED + 100
    P = oof(E2g, M2, I2, y2, seed)
    fitted = experts(E2g, M2, I2, y2, seed)
    Q = np.column_stack([predict_expert(fitted[0], Ee, 'rna'),
                         predict_expert(fitted[1], meta[['age', 'hr', 'her2']], 'clinical'),
                         np.full(len(y), np.nan)])
    F = fit_all(P, y2)
    pred = predict_all(F, Q)
    models = list(pred)
    zc = lg(Q[:, 1])
    rows = [{'model': k, 'n': len(y), 'events': int(y.sum()), **scores(y, v), 'ece': ece(y, v),
             'max_abs_z_minus_a': float(np.max(np.abs(lg(v) - zc)))} for k, v in pred.items()]
    met = pd.DataFrame(rows)
    pd.DataFrame({'patient_id': meta.index, 'pcr': y, **pred}).to_csv(OUT / 'predictions.csv', index=False)
    np.savez_compressed(OUT / 'expert_probs.npz', P=P, Q=Q, ispy2_ids=M2.index.to_numpy(str),
                        ext_ids=meta.index.to_numpy(str))

    # output-level stress (stress_scenarios; RNA permuted uses its seeds)
    stress_raw, perm_sets = [], []
    for sc, rep, q in stress_scenarios(Q):
        if sc.startswith('mri'):  # MRI is absent externally; MRI stress is undefined
            continue
        p = predict_all(F, q)
        a = lg(q[:, 1])
        for k, v in p.items():
            stress_raw.append({'scenario': sc, 'repeat': rep, 'model': k, 'auroc': roc_auc_score(y, v),
                               'brier': float(np.mean((v - y) ** 2)),
                               'max_abs_z_minus_a': float(np.abs(lg(v) - a).max())})
        if sc == 'rna_permuted':
            perm_sets.append(np.column_stack([p[m] for m in models]))
    assert len(perm_sets) == REPEATS

    # input-level stress: 50% per-patient gene dropout in GSE20194 RNA
    _S = dict(raw=raw_ext, index=meta.index, genes=G, Q=Q, rna=fitted[0])
    with mp.get_context('fork').Pool(REPEATS) as pool:
        drop_Q = pool.map(dropout_job, range(REPEATS))
    drop_sets = []
    for rep, q in enumerate(drop_Q):
        p = predict_all(F, q)
        a = lg(q[:, 1])
        for k, v in p.items():
            stress_raw.append({'scenario': 'rna_input_dropout50', 'repeat': rep, 'model': k,
                               'auroc': roc_auc_score(y, v), 'brier': float(np.mean((v - y) ** 2)),
                               'max_abs_z_minus_a': float(np.abs(lg(v) - a).max())})
        drop_sets.append(np.column_stack([p[m] for m in models]))
    sr = pd.DataFrame(stress_raw)
    sr.to_csv(OUT / 'stress_raw.csv', index=False)

    intact = np.column_stack([pred[m] for m in models])
    sets = {'intact': [intact], 'rna_permuted': perm_sets, 'rna_input_dropout50': drop_sets}
    b = pd.DataFrame(boot_deltas(y, sets, models, {'cohort': 'GSE20194'}))
    bi = b[b.scenario == 'intact']
    bi.to_csv(OUT / 'bootstrap.csv', index=False)
    met = met.merge(bi[['model', 'auroc_q025', 'auroc_q975']], on='model')
    met.to_csv(OUT / 'metrics.csv', index=False)
    agg = (sr.groupby(['scenario', 'model'])
           .agg(auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'),
                auroc_real_q025=('auroc', lambda s: s.quantile(.025)),
                auroc_real_q975=('auroc', lambda s: s.quantile(.975)),
                brier_mean=('brier', 'mean'), max_abs_z_minus_a=('max_abs_z_minus_a', 'max'),
                n_real=('auroc', 'size')).reset_index())
    bs = b[b.scenario != 'intact'][['scenario', 'model', 'auroc_q025', 'auroc_q975', 'delta_carf_minus_model',
                                    'delta_q025', 'delta_q975', 'p_delta_le_0']]
    stress = agg.merge(bs.rename(columns={'auroc_q025': 'auroc_boot_q025', 'auroc_q975': 'auroc_boot_q975'}),
                       on=['scenario', 'model'], how='left')
    stress.to_csv(OUT / 'stress.csv', index=False)

    carf = F['fus']['carf']
    prov = {
        'protocol': 'frozen; see module docstring of experiments/e8_gse20194.py',
        'sources': {k: {'url': URLS[k], 'sha256': sha256(RAW / k), 'bytes': (RAW / k).stat().st_size}
                    for k in URLS if (RAW / k).exists()},
        'processed': {p.name: sha256(p) for p in [RAW / 'GSE20194_meta_processed.csv',
                                                  RAW / 'GSE20194_expression_symbol_log2.csv.gz']},
        'code': {p.name: sha256(p) for p in [Path(__file__), HERE / 'tier1.py', HERE / 'e5_robust_offset.py',
                                             HERE / 'a7_moddrop.py', HERE / 'e1_bootstrap.py',
                                             MIKE / 'scripts/run_revision2_analysis.py',
                                             MIKE / 'scripts/run_revision3_analysis.py',
                                             MIKE / 'src/anchored_fusion.py']},
        'ispy2_inputs': {p.name: sha256(p) for p in [DATA / 'ISPY2_meta.csv', DATA / 'ISPY2_expression.csv.gz',
                                                    DATA / 'ISPY2_mri.csv']},
        'cohort': {'n': int(len(y)), 'events': int(y.sum()), **audit,
                   'treatment_counts': meta.treatment.value_counts().to_dict(),
                   'hr_pos': int(meta.hr.sum()), 'her2_pos': int(meta.her2.sum()),
                   'age_missing': int(meta.age.isna().sum())},
        'clinical_mapping': {'age': 'characteristics "age"', 'hr': 'er_status == P or pr_status == P',
                             'her2': '"her2 status" == P', 'pcr': 'pcr_vs_rd == pCR (RD = 0)'},
        'genes': genes, 'ispy2_n': int(len(y2)), 'ispy2_events': int(y2.sum()),
        'seed': SEED, 'expert_seed': seed, 'n_boot': NBOOT, 'repeats': REPEATS,
        'carf': {'budget': carf.budget, 'robust_weight': carf.robust_weight, 'coef': carf.coef_.tolist(),
                 'modality_budgets': carf.modality_budgets().tolist()},
        'offset_theta': {k: m.theta_.tolist() for k, m in F['off'].items()},
        'robust_offset_theta': F['rob'].theta_.tolist(),
        'moddrop': {'hidden': [32, 16], 'alpha': 0.01, 'selection': 'fixed (a7 external choice), no tuning'},
        'overlap': ov,
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'sklearn': sklearn.__version__, 'pandas': pd.__version__,
    }
    (OUT / 'provenance.json').write_text(json.dumps(prov, indent=2, default=str))
    pd.set_option('display.width', 250)
    print(met.round(4).to_string(index=False))
    print(stress.round(4).to_string(index=False))
    print(b.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
