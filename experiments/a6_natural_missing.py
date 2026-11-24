#!/usr/bin/env python3
"""A6: I-SPY1 patients with naturally missing modalities (clinical + exactly one
of {expression, usable MRI}). Experts are refit on all I-SPY2 exactly as in the
revision3 external run; fusions are fit on the cached I-SPY2 OOF matrix P.
Nothing under mike2026/ is written; outputs go to experiments/results/a6/."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

HERE = Path(__file__).resolve().parent
MIKE = HERE.parent / 'mike2026'
sys.path.insert(0, str(MIKE))
sys.path.insert(0, str(HERE))
from src.revision2_data import read_ispy1_expression  # noqa: E402
from src.revision2_imaging import image_features  # noqa: E402
import nibabel as nib  # noqa: E402
from scripts.run_revision2_analysis import (common_data, experts, predict_expert,  # noqa: E402
                                           rank_normalize)
from scripts.run_revision3_analysis import fit_fusions, predict_fusions, SEED  # noqa: E402
from tier1 import load, labels, offset_models, platt_carf  # noqa: E402

RAW = MIKE / 'data/revision2/raw'
PROC = MIKE / 'data/revision2/processed'
IMG = MIKE / 'data/revision2/images/BreastDCEDL_ISPY1_min_crop'
OUT = HERE / 'results/a6'
MODELS = ['clinical', 'carf', 'carf_no_robust', 'mask_stack', 'stack', 'offset_lr_masked', 'mean']


def excel(path, **kw):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        return pd.read_excel(path, **kw)


def expression_all():
    """Full GSE22226 matrix (all participants), cached under results/a6."""
    xp, mp = OUT / 'ISPY1_expression_all.csv.gz', OUT / 'ISPY1_source_metadata_all.csv'
    if xp.exists():
        x = pd.read_csv(xp, index_col=0); x.index = x.index.astype(str)
        return x, pd.read_csv(mp, dtype={'patient_id': str})
    x, m = read_ispy1_expression(RAW / 'GSE22226_family.soft.gz')
    m['pcr_source'] = m['pathological complete response (pcr)'].map({'Yes': 1, 'No': 0})
    x.to_csv(xp); m.to_csv(mp, index=False)
    return x, m


def clinical():
    # Same construction as prepare_revision2_molecular.prepare() for ISPY1.
    c1 = excel(RAW / 'ispy1_clinical.xlsx', sheet_name='TCIA Patient Clinical Subset')
    o1 = excel(RAW / 'ispy1_clinical.xlsx', sheet_name='TCIA Outcomes Subset')
    c1 = c1.merge(o1[['SUBJECTID', 'PCR']], on='SUBJECTID', validate='one_to_one')
    c1 = c1.rename(columns={'SUBJECTID': 'patient_id', 'PCR': 'pcr', 'HR Pos': 'hr', 'Her2MostPos': 'her2'})
    c1.patient_id = c1.patient_id.astype(str)
    return c1.set_index('patient_id')


def measure(args):
    """Copy of prepare_revision2_images.measure (same QC rules). Differences: phase
    files may be .nii or .nii.gz (one on-disk ISPY1_1239 phase is stored
    decompressed), and paths are not made relative to mike2026/."""
    pid, row, base = args
    audit = {'image_id': pid, 'included': False}
    try:
        phases = []
        for column in ['pre', 'post_early', 'post_late']:
            paths = list((base / 'dce').glob(f'{pid}_*_vis1_dce_aqc_{int(row[column])}.nii*'))
            if len(paths) != 1: raise ValueError(f'{column}: expected one phase, found {len(paths)}')
            phases.append(nib.load(paths[0]).get_fdata(dtype=np.float32))
        masks = list((base / 'mask').glob(f'{pid}_*_vis1_mask.nii*'))
        if len(masks) != 1: raise ValueError(f'Expected one mask, found {len(masks)}')
        mask = nib.load(masks[0]).get_fdata(dtype=np.float32)
        if not np.isfinite(mask).all(): raise ValueError('Nonfinite segmentation')
        if not np.isin(np.unique(mask), [0, 1]).all(): raise ValueError('Segmentation is not binary')
        spacing = [row.slice_thick, row.xy_spacing, row.xy_spacing]
        features = image_features(*phases, mask, spacing)
        audit.update(included=True, shape=str(mask.shape), mask_voxels=int(mask.sum()))
        return {'image_id': pid, **features}, audit
    except (ValueError, OSError) as error:
        audit['exclusion_reason'] = str(error)
        return None, audit


def mri_features(image_ids):
    path = OUT / 'ISPY1_mri_all.csv'
    if path.exists():
        return pd.read_csv(path, index_col=0), pd.read_csv(OUT / 'ISPY1_image_qc_all.csv')
    meta = pd.read_csv(RAW / 'BreastDCEDL_metadata_min_crop.csv').set_index('pid')
    tasks = [(pid, meta.loc[pid], IMG) for pid in image_ids]
    rows, audits = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for feats, audit in ex.map(measure, tasks):
            audits.append(audit)
            if feats:
                rows.append(feats)
    frame = pd.DataFrame(rows).set_index('image_id')
    frame.to_csv(path); pd.DataFrame(audits).to_csv(OUT / 'ISPY1_image_qc_all.csv', index=False)
    return frame, pd.DataFrame(audits)


def metric_row(group, y, pred):
    n, pos = len(y), int(y.sum())
    rows = []
    for m in MODELS:
        p = pred[m]
        auc = roc_auc_score(y, p) if min(pos, n - pos) >= 5 else np.nan
        rows.append({'subgroup': group, 'n': n, 'pcr_pos': pos, 'model': m, 'auroc': auc,
                     'brier': brier_score_loss(y, p)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x1, m1 = expression_all()
    c1 = clinical()
    geo = m1.dropna(subset=['patient_id']).groupby('patient_id').pcr_source.first()
    image = pd.read_csv(RAW / 'BreastDCEDL_metadata_min_crop.csv')
    image['patient_id'] = image.pid.str.extract(r'(\d+)$')[0]
    im = image.loc[image.dataset == 'spy1'].set_index('patient_id')
    feats, qc = mri_features(im.pid.tolist())
    usable_img = set(feats.index)

    complete = pd.read_csv(PROC / 'ISPY1_meta.csv', dtype={'patient_id': str}).set_index('patient_id')
    audit = []
    for pid, row in c1.iterrows():
        has_rna = pid in x1.index
        has_img = pid in im.index
        img_ok = has_img and im.loc[pid, 'pid'] in usable_img
        reasons = []
        if pd.isna(row.pcr):
            reasons.append('missing_pcr')
        else:
            if has_rna and pd.notna(geo.get(pid)) and row.pcr != geo.loc[pid]:
                reasons.append('geo_tcia_pcr_disagreement')
            if has_img and pd.notna(im.loc[pid, 'pCR']) and row.pcr != im.loc[pid, 'pCR']:
                reasons.append('image_tcia_pcr_disagreement')
        if has_img and not img_ok:
            reasons_img = qc.set_index('image_id').loc[im.loc[pid, 'pid']].get('exclusion_reason')
        else:
            reasons_img = ''
        group = ('complete' if has_rna and img_ok else 'rna_only' if has_rna else
                 'mri_only' if img_ok else 'clinical_only')
        audit.append({'patient_id': pid, 'has_expression': has_rna, 'in_image_manifest': has_img,
                      'usable_mri': img_ok, 'image_qc_reason': reasons_img, 'group': group,
                      'pcr': row.pcr, 'age': row.age, 'hr': row.hr, 'her2': row.her2,
                      'in_cached_112': pid in complete.index,
                      'excluded': bool(reasons), 'exclusion_reason': ';'.join(reasons)})
    audit = pd.DataFrame(audit).set_index('patient_id')
    audit.to_csv(OUT / 'ispy1_availability_audit.csv')
    kept = audit.loc[~audit.excluded]
    assert set(kept.index[kept.group == 'complete']) == set(complete.index), 'complete-case mismatch'
    counts = {g: {'n': int((kept.group == g).sum()), 'pcr_pos': int(kept.pcr[kept.group == g].sum()),
                  'missing_hr_or_her2': int(kept.loc[kept.group == g, ['hr', 'her2']].isna().any(axis=1).sum())}
              for g in ['complete', 'rna_only', 'mri_only', 'clinical_only']}
    counts['excluded'] = audit.loc[audit.excluded].groupby(['group', 'exclusion_reason']).size().rename('n').reset_index().to_dict('records')
    print(json.dumps(counts, indent=1), flush=True)

    # Experts on all I-SPY2, exactly as the revision3 external run.
    E2, M2, I2, E1c, M1c, I1c = common_data()
    y2 = M2.pcr.to_numpy(dtype=int)
    fitted = experts(E2, M2, I2, y2, SEED + 100)
    genes = E2.columns
    ids = kept.index.tolist()
    rna_ids = [i for i in ids if kept.loc[i, 'has_expression']]
    E = pd.DataFrame(rank_normalize(x1.loc[rna_ids].reindex(columns=genes).to_numpy()), index=rna_ids, columns=genes)
    img_ids = [i for i in ids if kept.loc[i, 'usable_mri']]
    I = feats.loc[im.loc[img_ids, 'pid']].copy(); I.index = img_ids; I = I[I2.columns]
    M = kept.loc[ids, ['age', 'hr', 'her2']].astype(float)
    Q = np.full((len(ids), 3), np.nan)
    pos = {i: k for k, i in enumerate(ids)}
    Q[[pos[i] for i in rna_ids], 0] = predict_expert(fitted[0], E, 'rna')
    Q[:, 1] = predict_expert(fitted[1], M, 'clinical')
    Q[[pos[i] for i in img_ids], 2] = predict_expert(fitted[2], I, 'mri')

    # Reproduction check against cached external Q (112 complete cases).
    y = labels()
    P, Qc, ytr, ye = load('external', y)
    with np.load(MIKE / 'results/revision3/cache/d562c1894fd42bb6/external.npz') as z:
        test_ids = z['test_ids'].astype(str)
    Qmine = Q[[pos[i] for i in test_ids]]
    # Also check the processed-matrix path (the pipeline's own E1/I1) for isolation.
    Qproc = np.column_stack([predict_expert(fitted[j], X, k) for j, (X, k) in
                             enumerate([(E1c, 'rna'), (M1c, 'clinical'), (I1c, 'mri')])])
    feat_diff = np.abs(I.loc[test_ids].to_numpy() - I1c.loc[test_ids].to_numpy())
    rel = feat_diff / np.maximum(np.abs(I1c.loc[test_ids].to_numpy()), 1e-9)
    repro = {'max_abs_diff_rebuilt_vs_cached': np.abs(Qmine - Qc).max(axis=0).tolist(),
             'max_abs_diff_processed_vs_cached': np.abs(Qproc - Qc).max(axis=0).tolist(),
             'mri_feature_max_rel_diff_rebuilt_vs_processed': float(rel.max()),
             'n_checked': len(test_ids)}
    print(json.dumps(repro, indent=1), flush=True)
    (OUT / 'reproduction_check.json').write_text(json.dumps(repro, indent=2))

    # Fusions fit once on cached OOF P; predict with true NaN masks.
    models = fit_fusions(P, ytr)
    pred = predict_fusions(models, Q)
    for name, m in offset_models(P, ytr).items():
        pred[name] = m.predict_proba(Q)[:, 1]
    pred['carf_platt'] = platt_carf(P, ytr, SEED + 100)(Q)
    yk = kept.loc[ids, 'pcr'].to_numpy(dtype=int)
    grp = kept.loc[ids, 'group'].to_numpy()
    pd.DataFrame({'patient_id': ids, 'group': grp, 'pcr': yk, 'q_rna': Q[:, 0], 'q_clinical': Q[:, 1],
                  'q_mri': Q[:, 2], **pred}).to_csv(OUT / 'predictions.csv', index=False)
    rows = []
    for name, sel in [('rna_only', grp == 'rna_only'), ('mri_only', grp == 'mri_only'),
                      ('complete_112', grp == 'complete'),
                      ('all_available', np.isin(grp, ['complete', 'rna_only', 'mri_only'])),
                      ('clinical_only', grp == 'clinical_only'),
                      ('all_incl_clinical_only', np.ones(len(grp), bool))]:
        if sel.sum():
            rows += metric_row(name, yk[sel], {k: v[sel] for k, v in pred.items()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'metrics.csv', index=False)
    (OUT / 'counts.json').write_text(json.dumps(counts, indent=2))
    print(res.pivot(index='model', columns='subgroup', values='auroc').round(3).to_string())
    print(res.pivot(index='model', columns='subgroup', values='brier').round(3).to_string())


if __name__ == '__main__':
    main()
