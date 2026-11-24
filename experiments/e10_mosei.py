#!/usr/bin/env python3
"""E10: CARF on CMU-MOSEI, same protocol as E3 (CMU-MOSI), MMSA unaligned_50 features.

Reuses the E3 functions unchanged (pooling, experts, fusions, stress, diagnostics) by import.
Modality mapping onto the I-SPY code's fixed column order [rna, clinical, mri]:
    column 0 = audio (COVAREP), column 1 = text (BERT, anchor), column 2 = visual (FACET).
Task: binary sentiment, label >= 0 vs < 0. Experts: standardised LR per modality, C tuned
on validation (fit on train). Gates use 5-fold OOF probabilities on train+valid; test
predictions use experts refit on train+valid. Test labels never drive a choice.

Additions over E3:
  * tuning and fusion fitting are parallelised (same grids and estimators);
  * natural missingness (segment with zero valid length, or all-zero / non-finite frames
    within its valid length) is evaluated as a subgroup, under three regimes:
      zero_features : main fit, test features as delivered (what E3 did);
      mask_frozen   : main fusions, test expert probability set to NaN (true mask);
      mask_refit    : fusions refit on OOF probabilities with the true train+valid masks,
                      tested with true test masks. Experts are unchanged in all regimes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import json
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e3_second_domain as e3  # noqa: E402
from e3_second_domain import (lg, sha256, pool, lr, text_mlp, make, oof, predict_all, score,  # noqa: E402
                              input_scenarios, stress_rows, rename_scenario, diagnostics, MAIN, MODS,
                              C_GRID, SEED)
import a7_moddrop  # noqa: E402
from tier1 import offset_models  # noqa: E402
from src.anchored_fusion import AnchoredResidualFusion  # noqa: E402
from scripts.run_revision3_analysis import fit_fusions, stress_scenarios  # noqa: E402

DATA = HERE / 'data_e10'
OUT = HERE / 'results/e10_mosei'
NJOBS = int(os.environ.get('E10_NJOBS', 64))
NBOOT = 2000
SOURCE = 'https://huggingface.co/datasets/tamb2203579/CMU-MOSEI/resolve/main/Processed/unaligned_50.pkl'
SPLITS = {'train': 16326, 'valid': 1871, 'test': 4659}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- data
def natural_missing(x, lengths):
    x = np.asarray(x)
    out = np.zeros(len(x), bool)
    for i, L in enumerate(lengths):
        L = int(L)
        if L <= 0:
            out[i] = True
            continue
        s = x[i, :min(L, x.shape[1])]
        out[i] = not np.any(np.isfinite(s) & (s != 0))
    return out


def load_split(d):
    text_mask = d['text_bert'][:, 1, :].astype(float)
    t = d['text'].astype(float)
    n = np.maximum(text_mask.sum(1, keepdims=True), 1)
    mean = (t * text_mask[:, :, None]).sum(1) / n
    std = np.sqrt(np.maximum((((t - mean[:, None]) ** 2) * text_mask[:, :, None]).sum(1) / n, 0))
    feats = {'text': np.c_[mean, std],
             'audio': pool(d['audio'], d['audio_lengths']),
             'visual': pool(d['vision'], d['vision_lengths'])}
    nat = {'audio': natural_missing(d['audio'], d['audio_lengths']),
           'visual': natural_missing(d['vision'], d['vision_lengths'])}
    y = (np.asarray(d['regression_labels']) >= 0).astype(int)
    shapes = {k: list(np.shape(d[k])) for k in ['text', 'text_bert', 'audio', 'vision']}
    extra = {'audio_len0': int((np.asarray(d['audio_lengths']) <= 0).sum()),
             'vision_len0': int((np.asarray(d['vision_lengths']) <= 0).sum()),
             'audio_nonfinite_values': int((~np.isfinite(np.asarray(d['audio'], float))).sum()),
             'vision_nonfinite_values': int((~np.isfinite(np.asarray(d['vision'], float))).sum()),
             'text_empty': int((text_mask.sum(1) == 0).sum()), 'shapes': shapes}
    return feats, y, nat, list(d['id']), extra


def data():
    cache = DATA / 'pooled.pkl'
    if cache.exists():
        return pickle.load(open(cache, 'rb'))
    log('loading pickle')
    raw = pickle.load(open(DATA / 'unaligned_50.pkl', 'rb'))
    log('loaded; keys', {s: sorted(raw[s].keys()) for s in raw})
    D = {s: load_split(raw[s]) for s in ['train', 'valid', 'test']}
    del raw
    pickle.dump(D, open(cache, 'wb'))
    return D


# ---------------------------------------------------------------- experts (parallel version of e3.tune)
def _val(kind, param, Xtr, ytr, Xva, yva, modality):
    est = lr(param) if kind == 'lr' else text_mlp(param)
    p = est.fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    return {'modality': modality, 'model': kind, 'param': param, 'val_logloss': log_loss(yva, p),
            'val_auroc': roc_auc_score(yva, p)}


def tune(Xtr, ytr, Xva, yva, modality):
    rows = Parallel(n_jobs=len(C_GRID))(delayed(_val)('lr', C, Xtr, ytr, Xva, yva, modality) for C in C_GRID)
    best = min(rows, key=lambda r: r['val_logloss'])
    if modality == 'text' and best['val_auroc'] < .75:
        rows += Parallel(n_jobs=5)(delayed(_val)('mlp', a, Xtr, ytr, Xva, yva, modality)
                                   for a in [1e-3, 1e-2, 1e-1, 1., 10.])
        best = min(rows, key=lambda r: r['val_logloss'])
    return best, rows


def _oof_fold(X, y, best, tr, te):
    return te, make(best).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def oof_par(X, y, best, seed):
    from sklearn.model_selection import StratifiedKFold
    p = np.zeros(len(y))
    res = Parallel(n_jobs=5)(delayed(_oof_fold)(X, y, best, tr, te)
                             for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    for te, v in res:
        p[te] = v
    return p


def expert_job(m, Ftr, ytr, Fva, yva, Fte):
    best, rows = tune(Ftr, ytr, Fva, yva, m)
    Xtv, ytv = np.vstack([Ftr, Fva]), np.r_[ytr, yva]
    P = oof_par(Xtv, ytv, best, SEED)
    model = make(best).fit(Xtv, ytv)
    return m, best, rows, P, model, model.predict_proba(Fte)[:, 1]


# ---------------------------------------------------------------- fusions (parallel version of e3.fit_all)
def fit_fusions_nan(P, y):
    """fit_fusions when P has true NaN masks: stack/no_mri_stack are fit on the 0.5-filled
    matrix (the fill predict_fusions applies at test time); mask_stack and the CARF variants are
    refit on P with its NaNs, replicating fit_fusions lines for those models."""
    if np.isfinite(P).all():
        return fit_fusions(P, y)
    from sklearn.linear_model import LogisticRegression
    from scripts.run_revision3_analysis import mask_stack_features
    import itertools
    models = fit_fusions(np.nan_to_num(P, nan=.5), y)
    standard = lambda: LogisticRegression(C=1, class_weight='balanced', solver='liblinear')
    blocks = []
    for active in itertools.product([False, True], repeat=2):
        q = P.copy()
        for j, keep in zip([0, 2], active):
            if not keep:
                q[:, j] = np.nan
        blocks.append(mask_stack_features(q))
    models['mask_stack'] = standard().fit(np.vstack(blocks), np.tile(y, 4), sample_weight=np.full(len(y) * 4, .25))
    configs = {'carf': {}, 'carf_no_robust': {'robust_weight': 0},
               'carf_no_context': {'context': False}, 'carf_unbounded': {'bounded': False}}
    for name, kwargs in configs.items():
        models[name] = AnchoredResidualFusion(**kwargs).fit(P, y)
    q = P.copy(); q[:, 2] = np.nan
    models['carf_no_mri'] = AnchoredResidualFusion().fit(q, y)
    return models


def _fit_part(part, P, y):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    if part == 'fus':
        return part, fit_fusions_nan(P, y)
    if part == 'off':
        return part, offset_models(P, y)
    if part.startswith('carf_B'):
        return part, AnchoredResidualFusion(budget=float(part[6:])).fit(P, y)
    return part, a7_moddrop.fit(P, y, part)


def fit_all_many(Ps, y):
    """e3.fit_all for several probability matrices, all parts in one flat parallel map."""
    parts = ['fus', 'off', 'carf_B0.5', 'carf_B2'] + list(a7_moddrop.MODELS)
    res = Parallel(n_jobs=len(parts) * len(Ps))(delayed(_fit_part)(p, P, y) for P in Ps for p in parts)
    out = []
    for i in range(len(Ps)):
        r = dict(res[i * len(parts):(i + 1) * len(parts)])
        out.append(dict(fus=r['fus'], off=r['off'], budgets={k: r[k] for k in ['carf_B0.5', 'carf_B2']},
                        neural={m: r[m] for m in a7_moddrop.MODELS}))
    return out


def gates_of(M):
    g = {k: {'coef': m.coef_.tolist(), 'budgets': m.modality_budgets().tolist() if m.bounded else None,
             'optimization': m.optimization_}
         for k, m in {**M['fus'], **M['budgets']}.items() if isinstance(m, AnchoredResidualFusion)}
    g['offset_lr'] = M['off']['offset_lr'].theta_.tolist()
    g['offset_lr_masked'] = M['off']['offset_lr_masked'].theta_.tolist()
    g['neural_params'] = {k: v[1] for k, v in M['neural'].items()}
    return g


# ---------------------------------------------------------------- bootstrap
def _boot_chunk(seed_seq, n_b, y, pred):
    rng = np.random.default_rng(seed_seq)
    out = []
    for b in range(n_b):
        ix = rng.integers(0, len(y), len(y))
        base = roc_auc_score(y[ix], pred['carf'][ix])
        out.append({k: base - roc_auc_score(y[ix], v[ix]) for k, v in pred.items() if k != 'carf'})
    return out


def bootstrap(y, pred):
    seeds = np.random.SeedSequence(SEED).spawn(40)
    res = Parallel(n_jobs=40)(delayed(_boot_chunk)(s, NBOOT // 40, y, pred) for s in seeds)
    boot = pd.DataFrame([r for chunk in res for r in chunk])
    boot = boot.reset_index().rename(columns={'index': 'bootstrap'}).melt(
        id_vars='bootstrap', var_name='comparator', value_name='delta')
    bs = boot.groupby('comparator').delta.quantile([.025, .5, .975]).unstack()
    bs.columns = ['q025', 'median', 'q975']
    bs['point'] = pd.Series({k: roc_auc_score(y, pred['carf']) - roc_auc_score(y, v)
                             for k, v in pred.items() if k != 'carf'})
    bs['p_le_0'] = boot.groupby('comparator').delta.apply(lambda d: (d <= 0).mean())
    return bs


# ---------------------------------------------------------------- natural missingness
def mask(P, nat):
    q = P.copy()
    q[nat['audio'], 0] = np.nan
    q[nat['visual'], 2] = np.nan
    return q


def natural_rows(y, nat, regimes):
    groups = {'all': np.ones(len(y), bool), 'any_missing': nat['audio'] | nat['visual'],
              'audio_missing': nat['audio'], 'visual_missing': nat['visual'],
              'both_missing': nat['audio'] & nat['visual'], 'complete': ~(nat['audio'] | nat['visual'])}
    rows = []
    for regime, pred in regimes.items():
        for g, ix in groups.items():
            if ix.sum() < 10 or len(np.unique(y[ix])) < 2:
                rows.append({'regime': regime, 'group': g, 'n': int(ix.sum()), 'model': None})
                continue
            for k, v in pred.items():
                rows.append({'regime': regime, 'group': g, 'n': int(ix.sum()), 'positive_rate': float(y[ix].mean()),
                             'model': k, **score(y[ix], v[ix])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = data()
    (Ftr, ytr, ntr, _, xtr), (Fva, yva, nva, _, xva), (Fte, yte, nte, ids_te, xte) = D['train'], D['valid'], D['test']
    counts = {s: {'n': len(D[s][1]), 'positive': int(D[s][1].sum()),
                  'nat_missing_audio': int(D[s][2]['audio'].sum()),
                  'nat_missing_visual': int(D[s][2]['visual'].sum()),
                  'nat_missing_both': int((D[s][2]['audio'] & D[s][2]['visual']).sum()),
                  **D[s][4]} for s in D}
    log(json.dumps(counts))
    assert {s: counts[s]['n'] for s in SPLITS} == SPLITS

    ytv = np.r_[ytr, yva]
    res = {m: expert_job(m, Ftr[m], ytr, Fva[m], yva, Fte[m]) for m in MODS}
    tuning, best, experts, Pcols, Qcols, Xte = [], {}, {}, [], [], {}
    for m in MODS:
        _, best[m], rows, p, experts[m], q = res[m]
        tuning += rows; Pcols.append(p); Qcols.append(q); Xte[m] = Fte[m]
        log(m, best[m], 'OOF AUROC', roc_auc_score(ytv, p))
    pd.DataFrame(tuning).to_csv(OUT / 'expert_tuning.csv', index=False)
    P, Q = np.column_stack(Pcols), np.column_stack(Qcols)
    np.savez_compressed(OUT / 'expert_probs.npz', P=P, Q=Q, ytv=ytv, yte=yte)
    expert_rows = [{'modality': m, 'model': best[m]['model'], 'param': best[m]['param'],
                    'val_auroc': best[m]['val_auroc'], 'oof_auroc_trainvalid': roc_auc_score(ytv, P[:, j]),
                    'test_auroc': roc_auc_score(yte, Q[:, j])} for j, m in enumerate(MODS)]
    pd.DataFrame(expert_rows).to_csv(OUT / 'experts.csv', index=False)

    ntv = {m: np.r_[ntr[m], nva[m]] for m in ['audio', 'visual']}
    Pm, Qm = mask(P, ntv), mask(Q, nte)
    log('fitting fusions (main and true-mask refit)')
    M, Mm = fit_all_many([P, Pm], ytv)
    pred = predict_all(M, Q)
    clean = pd.DataFrame([{'model': k, 'n': len(yte), **score(yte, v)} for k, v in pred.items()])
    clean.to_csv(OUT / 'clean.csv', index=False)
    pd.DataFrame({'id': ids_te, 'y': yte, 'nat_missing_audio': nte['audio'], 'nat_missing_visual': nte['visual'],
                  **pred}).to_csv(OUT / 'test_predictions.csv', index=False)
    gates = {'main': gates_of(M), 'mask_refit': gates_of(Mm)}
    (OUT / 'gates.json').write_text(json.dumps(gates, indent=2, default=float))
    log('\n' + clean.to_string(index=False))

    log('bootstrap')
    bs = bootstrap(yte, pred)
    bs.to_csv(OUT / 'bootstrap.csv')

    log('stress')
    S = {'experts': experts, 'Xte': Xte, 'Q': Q}
    jobs = [(rename_scenario(s), r, q) for s, r, q in stress_scenarios(Q)]
    jobs += list(input_scenarios(S))
    res = Parallel(n_jobs=NJOBS, batch_size=4)(delayed(stress_rows)(M, s, r, q, yte) for s, r, q in jobs)
    stress = pd.DataFrame([row for rows in res for row in rows])
    stress.to_csv(OUT / 'stress.csv', index=False)

    log('natural missingness')
    regimes = {'zero_features': pred, 'mask_frozen': predict_all(M, Qm), 'mask_refit': predict_all(Mm, Qm)}
    nat_df = natural_rows(yte, nte, regimes)
    nat_df.to_csv(OUT / 'natural_missingness.csv', index=False)
    ixs = nte['audio'] | nte['visual']
    nat_boot = {}
    if ixs.sum() >= 10:
        for regime in ['zero_features', 'mask_refit']:
            b = bootstrap(yte[ixs], {k: v[ixs] for k, v in regimes[regime].items()})
            nat_boot[regime] = b.reset_index().to_dict(orient='records')

    prov = {'source_url': SOURCE,
            'dataset_repo': 'https://huggingface.co/datasets/tamb2203579/CMU-MOSEI (MMSA-format processed pickles)',
            'files': {f.name: {'sha256': sha256(f), 'bytes': f.stat().st_size}
                      for f in sorted(DATA.iterdir()) if f.suffix in ('.pkl', '.csv') and f.name != 'pooled.pkl'},
            'hf_lfs_sha256_unaligned_50': 'ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f',
            'file_used': 'unaligned_50.pkl',
            'features': {'text': 'BERT token embeddings (768-d), mean+std over attention-masked tokens',
                         'audio': f"COVAREP ({xte['shapes']['audio'][2]}-d MMSA), mean+std over first audio_length frames",
                         'visual': f"FACET ({xte['shapes']['vision'][2]}-d MMSA), mean+std over first vision_length frames"},
            'label': 'regression_labels >= 0 -> 1 (non-negative), < 0 -> 0 (Acc-2 incl. zeros as positive)',
            'natural_missing_definition': 'valid length <= 0, or no finite non-zero value within the valid length',
            'splits': counts, 'seed': SEED, 'n_jobs': NJOBS,
            'code_sha256': {p.name: sha256(p) for p in [Path(__file__).resolve(), HERE / 'e3_second_domain.py',
                            HERE / 'tier1.py', HERE / 'a7_moddrop.py',
                            HERE.parent / 'mike2026/src/anchored_fusion.py',
                            HERE.parent / 'mike2026/scripts/run_revision3_analysis.py']}}
    (OUT / 'provenance.json').write_text(json.dumps(prov, indent=2))
    diag = {'main': diagnostics(P, ytv), 'mask_refit': diagnostics(Pm, ytv),
            'natural_missingness_subgroup_bootstrap_carf_minus_model': nat_boot}
    (OUT / 'diagnostics.json').write_text(json.dumps(diag, indent=2, default=float))
    summarize(expert_rows, clean, bs, stress, counts, gates, diag, nat_df)
    log('done')


def summarize(expert_rows, clean, bs, stress, counts, gates, diag, nat_df):
    f = lambda x: f'{x:.3f}'
    lines = ['# E10: CARF on CMU-MOSEI (E3 protocol)', '',
             'Data: MMSA `unaligned_50.pkl` (HF mirror tamb2203579/CMU-MOSEI); splits '
             f"{counts['train']['n']}/{counts['valid']['n']}/{counts['test']['n']}. "
             'Label: sentiment >= 0 vs < 0 (Acc-2, zeros counted positive). '
             f"Test positive rate {counts['test']['positive'] / counts['test']['n']:.3f}.",
             'Columns: audio -> col 0 (rna slot), text -> col 1 (anchor), visual -> col 2 (mri slot).', '',
             '## Experts', '', '| modality | model | C/alpha | val AUROC | OOF AUROC (train+valid) | test AUROC |',
             '|---|---|---|---|---|---|']
    lines += [f"| {r['modality']} | {r['model']} | {r['param']:.3g} | {f(r['val_auroc'])} | "
              f"{f(r['oof_auroc_trainvalid'])} | {f(r['test_auroc'])} |" for r in expert_rows]
    c = clean.set_index('model')
    lines += ['', '## Clean test', '', '| model | AUROC | Acc@0.5 | Brier | ECE | CARF-model dAUROC [95% CI] |',
              '|---|---|---|---|---|---|']
    for k in MAIN + [k for k in c.index if k not in MAIN]:
        if k not in c.index:
            continue
        ci = '' if k == 'carf' else f"{bs.loc[k, 'point']:+.3f} [{bs.loc[k, 'q025']:+.3f}, {bs.loc[k, 'q975']:+.3f}]"
        lines.append(f"| {k} | {f(c.loc[k, 'auroc'])} | {f(c.loc[k, 'accuracy'])} | {f(c.loc[k, 'brier'])} | "
                     f"{f(c.loc[k, 'ece'])} | {ci} |")
    agg = stress.groupby(['scenario', 'model']).agg(auroc=('auroc', 'mean'),
                                                    disp=('max_anchor_logit_displacement', 'max')).reset_index()
    piv = agg.pivot(index='scenario', columns='model', values='auroc')
    cols = [k for k in ['text', 'mean', 'stack', 'mask_stack', 'offset_lr_masked', 'moddrop_mlp', 'carf',
                        'carf_B0.5', 'carf_B2', 'carf_no_robust'] if k in piv.columns]
    lines += ['', '## Stress (test AUROC, mean over repeats for random scenarios)', '',
              '| scenario | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for s in piv.index:
        lines.append(f'| {s} | ' + ' | '.join(f(piv.loc[s, k]) for k in cols) + ' |')
    worst = piv.min()
    disp = agg.groupby('model').disp.max()
    lines += ['', '## Worst-case AUROC over scenarios and max |fused - anchor| logit', '',
              '| model | clean AUROC | worst AUROC | worst scenario | max displacement |', '|---|---|---|---|---|']
    for k in MAIN:
        if k in piv.columns:
            lines.append(f"| {k} | {f(c.loc[k, 'auroc'])} | {f(worst[k])} | {piv[k].idxmin()} | {disp[k]:.2f} |")
    t = counts['test']
    lines += ['', '## Natural missingness (true masks)', '',
              f"Test: audio missing {t['nat_missing_audio']}, visual missing {t['nat_missing_visual']}, "
              f"both {t['nat_missing_both']} of {t['n']}. Train: audio {counts['train']['nat_missing_audio']}, "
              f"visual {counts['train']['nat_missing_visual']}; valid: audio {counts['valid']['nat_missing_audio']}, "
              f"visual {counts['valid']['nat_missing_visual']}.",
              'Regimes: zero_features = main fit, test features as delivered; mask_frozen = main fusions, '
              'test probability of a naturally missing modality set to NaN; mask_refit = fusions refit '
              'on OOF probabilities with true train+valid masks, true test masks.', '']
    nd = nat_df.dropna(subset=['model'])
    ncols = [k for k in ['text', 'mean', 'stack', 'mask_stack', 'offset_lr_masked', 'moddrop_mlp', 'carf',
                         'carf_B2'] if k in set(nd.model)]
    lines += ['| regime | group | n | ' + ' | '.join(ncols) + ' |', '|' + '---|' * (len(ncols) + 3)]
    for (regime, g), d in nd.groupby(['regime', 'group'], sort=False):
        d = d.set_index('model')
        lines.append(f"| {regime} | {g} | {int(d.n.iloc[0])} | " + ' | '.join(f(d.loc[k, 'auroc']) for k in ncols) + ' |')
    for regime, recs in diag['natural_missingness_subgroup_bootstrap_carf_minus_model'].items():
        s = ', '.join(f"{r['comparator']} {r['point']:+.3f} [{r['q025']:+.3f}, {r['q975']:+.3f}]"
                      for r in recs if r['comparator'] in ncols)
        lines.append(f'\nany_missing subgroup, {regime}, CARF minus model dAUROC [95% CI]: {s}')
    lines += ['', '## Notes', '',
              '- Time-shuffling before pooling skipped (mean/std pooling is permutation-invariant).',
              f"- CARF gate (B=1), main: coef {np.round(gates['main']['carf']['coef'], 4).tolist()}, "
              f"budgets [audio, visual] {np.round(gates['main']['carf']['budgets'], 4).tolist()}; "
              f"B=0.5 {np.round(gates['main']['carf_B0.5']['coef'], 4).tolist()}; "
              f"B=2 {np.round(gates['main']['carf_B2']['coef'], 4).tolist()}.",
              f"- CARF gate (B=1), mask_refit: coef {np.round(gates['mask_refit']['carf']['coef'], 4).tolist()}.",
              f"- Gate diagnostics (OOF, main): {json.dumps(diag['main'])}. A positive gradient at theta=0 "
              'with theta>=0 means theta=0 is the constrained optimum (KKT).',
              f"- offset_lr weights [intercept, audio, visual]: {np.round(gates['main']['offset_lr'], 3).tolist()}."]
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    import e10_mosei
    e10_mosei.main()
