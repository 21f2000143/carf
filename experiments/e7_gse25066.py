#!/usr/bin/env python3
"""E7: frozen external evaluation of CARF and all baselines on GEO GSE25066
(Hatzis et al., JAMA 2011; neoadjuvant taxane-anthracycline, pCR vs RD, Affymetrix
U133A / GPL96). MRI is absent in this cohort (column 2 = NaN for every patient).

Protocol (frozen before any GSE25066 outcome was examined):
  * GSE25066: series matrix (log2 MAS5), probe -> symbol by GPL96 annotation
    (single-symbol probes only; probes averaged per symbol on the log2 scale).
    hr = ER or PR positive (ER 'I' resolved by the er_status_ihc_esr1_for
    indeterminate field; PR 'I'/NA = unknown); her2 P->1, N->0, I/NA -> NaN
    (median-imputed by the frozen clinical expert). Unlabelled patients excluded.
  * G = I-SPY2 genes intersected with GSE25066 symbols (label-free); both cohorts
    restricted to G and then rank-normalized per patient (rank_normalize).
  * All I-SPY2: inner OOF P and experts at seed SEED+100 (as cached_predictions
    'external'); fusions = fit_fusions + offset_lr(+masked) + robust_offset_lr
    (gamma .25) + moddrop_mlp (hidden (32,16), alpha .01, fixed) + carf_platt.
    CARF B=1, gamma=.25 (package defaults). No GSE25066 label is used in fitting.
  * PRIMARY cohort: GSE25066 without its 83 I-SPY1 samples (source == 'ISPY'), since
    I-SPY1 was seen during method design (coordinator instruction). Full labelled
    cohort is secondary (*_all.csv). Experts/fusions do not depend on GSE25066.
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

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tier1 import offset_models, platt_carf, ece, SEED, EPS  # noqa: E402  (adds mike2026 to path)
from e5_robust_offset import RobustOffsetLR  # noqa: E402
from e1_bootstrap import auc_rows, summarize, NBOOT  # noqa: E402
from a7_moddrop import masked_blocks  # noqa: E402
from scripts.run_revision2_analysis import (DATA, rank_normalize, common_data, experts,  # noqa: E402
                                           predict_expert, oof)
from scripts.run_revision3_analysis import (fit_fusions, predict_fusions, scores,  # noqa: E402
                                           stress_scenarios, mask_stack_features)

RAW = HERE / 'data_e7'
OUT = HERE / 'results/e7_gse25066'
URLS = {'GSE25066_series_matrix.txt.gz':
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25066/matrix/GSE25066_series_matrix.txt.gz',
        'GPL96.annot.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz'}
REPEATS = 30
DROPOUT_SI = 3  # a8 scenario index of rna_dropout50 -> rng = default_rng([SEED, 8, 3, rep])


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ parsing
def parse_series():
    chars, head, table = [], {}, []
    with gzip.open(RAW / 'GSE25066_series_matrix.txt.gz', 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            p = line.rstrip('\n').split('\t')
            v = [x.strip('"') for x in p[1:]]
            if p[0] == '!Sample_characteristics_ch1':
                chars.append(v)
            elif p[0] in ('!Sample_geo_accession', '!Sample_source_name_ch1', '!Sample_title'):
                head[p[0]] = v
        expr = pd.read_csv(f, sep='\t', index_col=0, comment='!')
    recs = []
    for i, g in enumerate(head['!Sample_geo_accession']):
        r = {'gsm': g, 'source_name': head['!Sample_source_name_ch1'][i]}
        for row in chars:  # characteristics order differs between samples: parse by key
            if ': ' in row[i]:
                k, val = row[i].split(': ', 1)
                r[k] = val
        recs.append(r)
    meta = pd.DataFrame(recs).set_index('gsm')
    expr.columns = [c.strip('"') for c in expr.columns]
    assert list(expr.columns) == list(meta.index)
    return meta, expr


def clinical_map(meta):
    pos = lambda s: s.map({'P': 1., 'N': 0.})
    er = pos(meta['er_status_ihc'])
    er = er.fillna(pos(meta['er_status_ihc_esr1_for indeterminate']))  # 'I' resolved by ESR1 field
    pr = pos(meta['pr_status_ihc'])
    hr = pd.Series(np.where((er == 1) | (pr == 1), 1., np.where((er == 0) & (pr == 0), 0., np.nan)), index=meta.index)
    out = pd.DataFrame({'sample_id': meta['sample id'], 'source': meta['source'],
                        'pcr': meta['pathologic_response_pcr_rd'].map({'pCR': 1, 'RD': 0}),
                        'age': pd.to_numeric(meta['age_years'], errors='coerce'),
                        'er': er, 'pr': pr, 'hr': hr, 'her2': pos(meta['her2_status'])}, index=meta.index)
    return out


def probe_to_symbol(expr):
    with gzip.open(RAW / 'GPL96.annot.gz', 'rt') as f:
        lines = f.read().split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('ID\t'))
    end = next(i for i, l in enumerate(lines) if l.startswith('!platform_table_end'))
    from io import StringIO
    ann = pd.read_csv(StringIO('\n'.join(lines[start:end])), sep='\t', dtype=str)
    sym = ann.set_index('ID')['Gene symbol'].dropna()
    sym = sym[~sym.str.contains('///')]  # ambiguous multi-gene probes dropped
    x = expr.loc[expr.index.intersection(sym.index)]
    stats = {'probes_total': int(len(expr)), 'probes_single_symbol': int(len(x))}
    genes = x.groupby(sym.loc[x.index].to_numpy()).mean().T  # mean of log2 values per symbol
    stats['symbols'] = int(genes.shape[1])
    return genes, stats


# ------------------------------------------------------------------ models
def moddrop_fit(P, y):
    X, yy = masked_blocks(P, y)
    return MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=False,
                         max_iter=2000, random_state=SEED).fit(X, yy)


def build():
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    meta, expr = parse_series()
    clin = clinical_map(meta)
    genes, gstats = probe_to_symbol(expr)
    lab = clin[clin.pcr.notna()].copy()
    lab['pcr'] = lab.pcr.astype(int)

    m2 = pd.read_csv(DATA / 'ISPY2_meta.csv', dtype={'patient_id': str}).set_index('patient_id')
    e2 = pd.read_csv(DATA / 'ISPY2_expression.csv.gz', index_col=0); e2.index = e2.index.astype(str)
    i2 = pd.read_csv(DATA / 'ISPY2_mri.csv', index_col=0); i2.index = i2.index.astype(str)
    ids2 = m2.index.intersection(e2.index).intersection(i2.index)
    G = e2.columns.intersection(genes.columns)  # label-free gene intersection
    E2 = pd.DataFrame(rank_normalize(e2.loc[ids2, G].to_numpy()), index=ids2, columns=G)
    M2, I2 = m2.loc[ids2], i2.loc[ids2]
    y2 = M2.pcr.astype(int).to_numpy()
    ref = common_data()[0]
    assert list(ref.index) == list(ids2)  # same I-SPY2 patients as the frozen pipeline

    raw = genes.loc[lab.index, G].to_numpy(float)
    Ee = pd.DataFrame(rank_normalize(raw), index=lab.index, columns=G)
    Me = lab[['age', 'hr', 'her2']]
    ye = lab.pcr.to_numpy()

    seed = SEED + 100
    P = oof(E2, M2, I2, y2, seed)
    fitted = experts(E2, M2, I2, y2, seed)
    Q = np.column_stack([predict_expert(fitted[0], Ee, 'rna'), predict_expert(fitted[1], Me, 'clinical'),
                         np.full(len(ye), np.nan)])  # MRI absent: the MRI expert never predicts
    fus = fit_fusions(P, y2)
    off = offset_models(P, y2)
    rob = RobustOffsetLR(robust_weight=.25).fit(P, y2)
    mlp = moddrop_fit(P, y2)
    platt = platt_carf(P, y2, seed)
    assert fus['carf'].budget == 1.0 and fus['carf'].robust_weight == .25

    def predict_all(q):
        p = predict_fusions(fus, q)
        p.update({k: m.predict_proba(q)[:, 1] for k, m in off.items()})
        p['robust_offset_lr'] = rob.predict_proba(q)[:, 1]
        p['moddrop_mlp'] = mlp.predict_proba(mask_stack_features(q))[:, 1]
        p['carf_platt'] = platt(q)
        return p

    S = dict(lab=lab, ye=ye, Q=Q, raw=raw, Ee=Ee, fitted=fitted, predict_all=predict_all, G=G,
             gstats=gstats, clin=clin, n_ispy2=len(y2), ispy2_events=int(y2.sum()),
             carf_coef=fus['carf'].coef_.tolist(), carf_budgets=fus['carf'].modality_budgets().tolist(),
             robust_theta=rob.theta_.tolist(), offset_theta={k: m.theta_.tolist() for k, m in off.items()})
    return S


def realizations(S):
    """Prediction matrices (n x models) per scenario."""
    Q, pa = S['Q'], S['predict_all']
    names = list(pa(Q).keys())
    def mat(q):
        p = pa(q)
        return np.column_stack([p[k] for k in names])
    sets = {'intact': [(Q, mat(Q))]}
    sets['rna_permuted'] = [(q, mat(q)) for sc, _, q in stress_scenarios(Q) if sc == 'rna_permuted']
    drop = []
    n, g = S['raw'].shape
    for rep in range(REPEATS):
        rng = np.random.default_rng([SEED, 8, DROPOUT_SI, rep])
        raw = S['raw'].copy()
        for i in range(n):
            raw[i, rng.choice(g, int(round(.5 * g)), replace=False)] = np.nan
        q = Q.copy()
        q[:, 0] = predict_expert(S['fitted'][0], pd.DataFrame(rank_normalize(raw), index=S['Ee'].index,
                                                              columns=S['G']), 'rna')
        drop.append((q, mat(q)))
    sets['rna_dropout50_input'] = drop
    assert len(sets['rna_permuted']) == REPEATS
    return names, sets


def evaluate(names, sets, ye, mask, cohort):
    ye = ye[mask]
    metric_rows, stress_rows, boot_rows = [], [], []
    ci = names.index('carf')
    comps = [m for m in names if m not in ('carf', 'mri')]  # mri is a constant 0.5 here
    for sc, reals in sets.items():
        for rep, (q, A) in enumerate(reals):
            a = logit(np.clip(q[mask, 1], EPS, 1 - EPS))
            for k, m in enumerate(names):
                p = A[mask, k]
                row = {'cohort': cohort, 'scenario': sc, 'repeat': rep, 'model': m, 'n': int(len(ye)),
                       'events': int(ye.sum()), **scores(ye, p), 'ece': ece(ye, p),
                       'max_abs_z_minus_a': float(np.abs(logit(np.clip(p, EPS, 1 - EPS)) - a).max())}
                (metric_rows if sc == 'intact' else stress_rows).append(row)
        Ast = np.stack([A[mask] for _, A in reals])
        point_auc = np.mean([auc_rows(ye, x) for x in Ast], axis=0)
        point = {m: point_auc[ci] - point_auc[names.index(m)] for m in comps}
        rng = np.random.default_rng(SEED)
        boot = {m: [] for m in comps}
        for _ in range(NBOOT):
            ix = rng.integers(0, len(ye), len(ye))
            if len(np.unique(ye[ix])) < 2:
                continue
            auc = np.mean([auc_rows(ye[ix], x[ix]) for x in Ast], axis=0)
            for m in comps:
                boot[m].append(auc[ci] - auc[names.index(m)])
        boot_rows += summarize(boot, point, {'cohort': cohort, 'scenario': sc, 'n_real': len(reals),
                                             'carf_auroc': point_auc[ci]})
    return metric_rows, stress_rows, boot_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = build()
    names, sets = realizations(S)
    lab, ye = S['lab'], S['ye']
    cohorts = {'gse25066_all': np.ones(len(ye), bool),
               'gse25066_non_ispy': (lab.source != 'ISPY').to_numpy()}
    M, St, B = [], [], []
    for c, mask in cohorts.items():
        m, s, b = evaluate(names, sets, ye, mask, c)
        M += m; St += s; B += b
    metrics = pd.DataFrame(M).drop(columns=['scenario', 'repeat'])
    metrics[metrics.cohort == 'gse25066_non_ispy'].to_csv(OUT / 'metrics.csv', index=False)
    metrics[metrics.cohort == 'gse25066_all'].to_csv(OUT / 'metrics_all.csv', index=False)
    boot = pd.DataFrame(B)
    boot[boot.cohort == 'gse25066_non_ispy'].to_csv(OUT / 'bootstrap.csv', index=False)
    boot[boot.cohort == 'gse25066_all'].to_csv(OUT / 'bootstrap_all.csv', index=False)
    stress = pd.DataFrame(St)
    stress.to_csv(OUT / 'stress_raw.csv', index=False)
    ss = (stress.groupby(['cohort', 'scenario', 'model'])
          .agg(auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'), brier_mean=('brier', 'mean'),
               max_abs_z_minus_a=('max_abs_z_minus_a', 'max'), n_real=('auroc', 'size')).reset_index())
    ss = ss.merge(boot.rename(columns={'comparator': 'model'})[['cohort', 'scenario', 'model', 'point', 'q025',
                                                                 'q975', 'p_delta_le_0']]
                  .rename(columns={'point': 'carf_minus_model', 'q025': 'delta_q025', 'q975': 'delta_q975'}),
                  on=['cohort', 'scenario', 'model'], how='left')
    ss[ss.cohort == 'gse25066_non_ispy'].to_csv(OUT / 'stress.csv', index=False)
    ss[ss.cohort == 'gse25066_all'].to_csv(OUT / 'stress_all.csv', index=False)
    pd.DataFrame({'patient': lab.index, 'source': lab.source, 'pcr': ye,
                  **{m: sets['intact'][0][1][:, k] for k, m in enumerate(names)}}).to_csv(
        OUT / 'predictions.csv', index=False)

    m1 = pd.read_csv(DATA / 'ISPY1_meta.csv', dtype={'patient_id': str})
    m2 = pd.read_csv(DATA / 'ISPY2_meta.csv', dtype={'patient_id': str})
    clin = S['clin']
    isp = clin[clin.source == 'ISPY']
    prov = {
        'sources': {k: {'url': u, 'sha256': sha256(RAW / k), 'bytes': (RAW / k).stat().st_size} for k, u in URLS.items()},
        'script_sha256': sha256(Path(__file__)), 'seed': SEED, 'expert_seed': SEED + 100,
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'sklearn': sklearn.__version__,
        'cohort': {'samples_total': int(len(clin)), 'labelled': int(len(lab)), 'events_pcr': int(ye.sum()),
                   'unlabelled_excluded': int(clin.pcr.isna().sum()),
                   'by_source': lab.groupby('source').pcr.agg(['size', 'sum']).astype(int).to_dict('index'),
                   'hr_counts': lab.hr.value_counts(dropna=False).rename(str).to_dict(),
                   'her2_counts': lab.her2.value_counts(dropna=False).rename(str).to_dict(),
                   'age_missing': int(lab.age.isna().sum())},
        'clinical_mapping': 'er=er_status_ihc (P/N; I resolved by er_status_ihc_esr1_for indeterminate); '
                            'pr=pr_status_ihc (P/N; I/NA unknown); hr=1 if ER or PR positive, 0 if both '
                            'negative, else NaN; her2=her2_status P->1 N->0 I/NA->NaN; NaN median-imputed '
                            'by frozen I-SPY2 clinical expert; I-SPY2 codes hr/her2 as 0/1.',
        'genes': {**S['gstats'], 'ispy2_genes': 19134, 'intersection_G': int(len(S['G']))},
        'overlap': {'ispy2_id_overlap': int(clin['sample_id'].isin(m2.patient_id).sum()),
                    'ispy_source_samples': int(len(isp)),
                    'ispy_source_labelled': int(isp.pcr.notna().sum()),
                    'non_ispy_sample_id_matching_ispy1_ids': int(clin[clin.source != 'ISPY'].sample_id.isin(m1.patient_id).sum()),
                    'ispy_source_in_our_ispy1_cohort': int(isp['sample_id'].isin(m1.patient_id).sum()),
                    'note': 'GSE25066 includes 83 I-SPY1 (source=ISPY) samples, 62 of which are in the '
                            'I-SPY1 external cohort (n=112) used elsewhere; no I-SPY2 patients. Training '
                            'uses I-SPY2 only. PRIMARY = non-ISPY cohort (metrics.csv etc.); full cohort secondary (*_all.csv).'},
        'ispy2_train': {'n': S['n_ispy2'], 'events': S['ispy2_events']},
        'fitted': {'carf_coef': S['carf_coef'], 'carf_budgets': S['carf_budgets'],
                   'robust_offset_theta': S['robust_theta'], 'offset_theta': S['offset_theta']},
        'settings': {'carf_budget': 1.0, 'carf_robust_weight': .25, 'robust_offset_gamma': .25,
                     'moddrop': {'hidden': [32, 16], 'alpha': .01}, 'n_boot': NBOOT, 'stress_repeats': REPEATS}}
    (OUT / 'provenance.json').write_text(json.dumps(prov, indent=2, default=str))
    pd.set_option('display.width', 250)
    print(json.dumps(prov['cohort'], indent=1, default=str), json.dumps(prov['genes']))
    print(metrics.round(4).to_string(index=False))
    print(boot.round(4).to_string(index=False))
    print(ss.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
