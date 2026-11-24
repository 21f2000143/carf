#!/usr/bin/env python3
"""E3: second-domain evaluation of CARF on CMU-MOSI (MMSA unaligned_50 features).

Modality mapping onto the I-SPY code's fixed column order [rna, clinical, mri]:
    column 0 = audio (COVAREP, 5-d), column 1 = text (BERT, anchor), column 2 = visual (FACET, 20-d).
Task: binary sentiment, label >= 0 (positive, incl. neutral 0) vs < 0 (negative).
Experts: standardised LR per modality, C tuned on the validation split by log loss (fit on train).
Gate training uses 5-fold cross-fitted OOF expert probabilities on train+valid; test
predictions come from experts refit on train+valid. Test labels are never used for choices.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import hashlib
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tier1 import offset_models, ece  # noqa: E402  (also puts mike2026 on sys.path)
import a7_moddrop  # noqa: E402
from src.anchored_fusion import AnchoredResidualFusion  # noqa: E402
from scripts.run_revision3_analysis import fit_fusions, predict_fusions, stress_scenarios  # noqa: E402

SEED = 20260923
DATA = HERE / 'data_e3'
OUT = HERE / 'results/e3'
EPS = 1e-7
REPEATS = 30
C_GRID = np.logspace(-4, 2, 13)
SOURCE = 'https://huggingface.co/datasets/tamb2203579/CMU-MOSI/resolve/main/Processed/unaligned_50.pkl'
RENAME = {'rna': 'audio', 'clinical': 'text', 'mri': 'visual',
          'no_mri_stack': 'no_visual_stack', 'carf_no_mri': 'carf_no_visual'}
MODS = ['audio', 'text', 'visual']


def lg(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- data
def pool(x, lengths, rng=None):
    """Mean+std pool over the first `length` frames; inf/nan -> 0 before pooling."""
    x = np.nan_to_num(np.asarray(x, float), nan=0., posinf=0., neginf=0.)
    out = np.zeros((len(x), 2 * x.shape[2]))
    for i, L in enumerate(lengths):
        L = int(min(max(L, 1), x.shape[1]))
        s = x[i, :L]
        out[i] = np.r_[s.mean(0), s.std(0)]
    return out


def load_split(d):
    text_mask = d['text_bert'][:, 1, :].astype(float)
    t = d['text'].astype(float)
    n = text_mask.sum(1, keepdims=True)
    mean = (t * text_mask[:, :, None]).sum(1) / n
    std = np.sqrt(np.maximum((((t - mean[:, None]) ** 2) * text_mask[:, :, None]).sum(1) / n, 0))
    feats = {'text': np.c_[mean, std],
             'audio': pool(d['audio'], d['audio_lengths']),
             'visual': pool(d['vision'], d['vision_lengths'])}
    # natural missingness: all-zero visual/acoustic frames within the valid length
    nat = {}
    for m, k, L in [('audio', 'audio', 'audio_lengths'), ('visual', 'vision', 'vision_lengths')]:
        x = d[k]
        nat[m] = np.array([np.abs(x[i, :max(int(l), 1)]).sum() == 0 for i, l in enumerate(d[L])])
    y = (np.asarray(d['regression_labels']) >= 0).astype(int)
    return feats, y, nat, list(d['id'])


def data():
    raw = pickle.load(open(DATA / 'unaligned_50.pkl', 'rb'))
    return {s: load_split(raw[s]) for s in ['train', 'valid', 'test']}


# ---------------------------------------------------------------- experts
def lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def text_mlp(alpha):
    return make_pipeline(StandardScaler(), MLPClassifier((64,), alpha=alpha, early_stopping=True,
                                                         max_iter=500, random_state=SEED))


def tune(Xtr, ytr, Xva, yva, modality):
    rows = []
    for C in C_GRID:
        p = lr(C).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
        rows.append({'modality': modality, 'model': 'lr', 'param': C, 'val_logloss': log_loss(yva, p),
                     'val_auroc': roc_auc_score(yva, p)})
    best = min(rows, key=lambda r: r['val_logloss'])
    if modality == 'text' and best['val_auroc'] < .75:
        for a in [1e-3, 1e-2, 1e-1, 1., 10.]:
            p = text_mlp(a).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
            rows.append({'modality': modality, 'model': 'mlp', 'param': a,
                         'val_logloss': log_loss(yva, p), 'val_auroc': roc_auc_score(yva, p)})
        best = min(rows, key=lambda r: r['val_logloss'])
    return best, rows


def make(best):
    return lr(best['param']) if best['model'] == 'lr' else text_mlp(best['param'])


def oof(X, y, best, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[te] = make(best).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


# ---------------------------------------------------------------- fusions
def fit_all(P, y):
    fus = fit_fusions(P, y)
    off = offset_models(P, y)
    budgets = {f'carf_B{B:g}': AnchoredResidualFusion(budget=B).fit(P, y) for B in (.5, 2.)}
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    neural = {m: a7_moddrop.fit(P, y, m) for m in a7_moddrop.MODELS}
    return dict(fus=fus, off=off, budgets=budgets, neural=neural)


def predict_all(M, Q):
    pred = {RENAME.get(k, k): v for k, v in predict_fusions(M['fus'], Q).items()}
    for k, m in {**M['off'], **M['budgets']}.items():
        pred[k] = m.predict_proba(Q)[:, 1]
    for k, (m, _) in M['neural'].items():
        pred[k] = a7_moddrop.predict(m, Q)
    return pred


def score(y, p):
    return {'auroc': roc_auc_score(y, p), 'accuracy': float(((p >= .5).astype(int) == y).mean()),
            'brier': brier_score_loss(y, p), 'ece': ece(y, p)}


# ---------------------------------------------------------------- stress
def input_scenarios(S):
    """Input-level corruption of standardised pooled test features for one modality."""
    for j, m in [(0, 'audio'), (2, 'visual')]:
        expert = S['experts'][m]
        scaler = expert.steps[0][1]
        X = S['Xte'][m]
        for r in range(REPEATS):
            rng = np.random.default_rng(SEED + 5000 + 1000 * j + r)
            for sd in (.5, 1.):
                Xc = X + rng.normal(0, sd, X.shape) * scaler.scale_
                q = S['Q'].copy(); q[:, j] = expert.predict_proba(Xc)[:, 1]
                yield f'{m}_input_noise_sd{sd:g}', r, q
            drop = rng.random(X.shape) < .5
            Xc = np.where(drop, scaler.mean_[None, :], X)
            q = S['Q'].copy(); q[:, j] = expert.predict_proba(Xc)[:, 1]
            yield f'{m}_input_dropout50', r, q


def stress_rows(M, scenario, repeat, q, y):
    a = lg(q[:, 1])
    rows = []
    for name, v in predict_all(M, q).items():
        rows.append({'scenario': scenario, 'repeat': repeat, 'model': name, **score(y, v),
                     'max_anchor_logit_displacement': float(np.abs(lg(v) - a).max())})
    return rows


def rename_scenario(s):
    return s.replace('rna', 'audio').replace('mri', 'visual')


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = data()
    (Ftr, ytr, ntr, _), (Fva, yva, nva, _), (Fte, yte, nte, ids_te) = D['train'], D['valid'], D['test']
    counts = {s: {'n': len(D[s][1]), 'positive': int(D[s][1].sum()),
                  'nat_missing_audio': int(D[s][2]['audio'].sum()),
                  'nat_missing_visual': int(D[s][2]['visual'].sum())} for s in D}
    print(counts, flush=True)
    assert [counts[s]['n'] for s in ['train', 'valid', 'test']] == [1284, 229, 686]

    # experts: tune on valid, OOF on train+valid, refit on train+valid for test
    ytv = np.r_[ytr, yva]
    tuning, best, experts, Pcols, Qcols, Xte = [], {}, {}, [], [], {}
    for m in MODS:
        best[m], rows = tune(Ftr[m], ytr, Fva[m], yva, m)
        tuning += rows
        Xtv = np.vstack([Ftr[m], Fva[m]])
        Pcols.append(oof(Xtv, ytv, best[m], SEED))
        experts[m] = make(best[m]).fit(Xtv, ytv)
        Xte[m] = Fte[m]
        Qcols.append(experts[m].predict_proba(Fte[m])[:, 1])
        print(m, best[m], 'OOF AUROC', roc_auc_score(ytv, Pcols[-1]), flush=True)
    pd.DataFrame(tuning).to_csv(OUT / 'expert_tuning.csv', index=False)
    P, Q = np.column_stack(Pcols), np.column_stack(Qcols)
    np.savez_compressed(OUT / 'expert_probs.npz', P=P, Q=Q, ytv=ytv, yte=yte)
    expert_rows = [{'modality': m, 'model': best[m]['model'], 'param': best[m]['param'],
                    'val_auroc': best[m]['val_auroc'], 'oof_auroc_trainvalid': roc_auc_score(ytv, P[:, j]),
                    'test_auroc': roc_auc_score(yte, Q[:, j])} for j, m in enumerate(MODS)]
    pd.DataFrame(expert_rows).to_csv(OUT / 'experts.csv', index=False)

    # fusions on OOF probabilities
    M = fit_all(P, ytv)
    pred = predict_all(M, Q)
    clean = pd.DataFrame([{'model': k, 'n': len(yte), **score(yte, v)} for k, v in pred.items()])
    clean.to_csv(OUT / 'clean.csv', index=False)
    pd.DataFrame({'id': ids_te, 'y': yte, **pred}).to_csv(OUT / 'test_predictions.csv', index=False)
    gates = {k: {'coef': m.coef_.tolist(), 'budgets': m.modality_budgets().tolist() if m.bounded else None,
                 'optimization': m.optimization_}
             for k, m in {**M['fus'], **M['budgets']}.items() if isinstance(m, AnchoredResidualFusion)}
    gates['offset_lr'] = M['off']['offset_lr'].theta_.tolist()
    gates['offset_lr_masked'] = M['off']['offset_lr_masked'].theta_.tolist()
    gates['neural_params'] = {k: v[1] for k, v in M['neural'].items()}
    (OUT / 'gates.json').write_text(json.dumps(gates, indent=2, default=float))
    print(clean.to_string(index=False), flush=True)

    # paired bootstrap
    rng = np.random.default_rng(SEED)
    boot = []
    for b in range(2000):
        ix = rng.integers(0, len(yte), len(yte))
        base = roc_auc_score(yte[ix], pred['carf'][ix])
        for k, v in pred.items():
            if k != 'carf':
                boot.append({'bootstrap': b, 'comparator': k, 'delta': base - roc_auc_score(yte[ix], v[ix])})
    boot = pd.DataFrame(boot)
    bs = boot.groupby('comparator').delta.quantile([.025, .5, .975]).unstack()
    bs.columns = ['q025', 'median', 'q975']
    bs['point'] = pd.Series({k: roc_auc_score(yte, pred['carf']) - roc_auc_score(yte, v)
                             for k, v in pred.items() if k != 'carf'})
    bs['p_le_0'] = boot.groupby('comparator').delta.apply(lambda d: (d <= 0).mean())
    bs.to_csv(OUT / 'bootstrap.csv')

    # stress tests (frozen experts and fusions)
    S = {'experts': experts, 'Xte': Xte, 'Q': Q}
    jobs = [(rename_scenario(s), r, q) for s, r, q in stress_scenarios(Q)]
    jobs += list(input_scenarios(S))
    res = Parallel(n_jobs=36, batch_size=8)(delayed(stress_rows)(M, s, r, q, yte) for s, r, q in jobs)
    stress = pd.DataFrame([row for rows in res for row in rows])
    stress.to_csv(OUT / 'stress.csv', index=False)

    # natural missingness subgroup
    nat = {m: nte[m] for m in ['audio', 'visual']}
    natural = {'test_counts': {m: int(v.sum()) for m, v in nat.items()},
               'trainvalid_counts': {m: int(np.r_[ntr[m], nva[m]].sum()) for m in nat}}
    (OUT / 'natural_missingness.json').write_text(json.dumps(natural, indent=2))

    # provenance
    prov = {'source_url': SOURCE,
            'dataset_repo': 'https://huggingface.co/datasets/tamb2203579/CMU-MOSI (MMSA-format processed pickles)',
            'files': {f.name: {'sha256': sha256(f), 'bytes': f.stat().st_size} for f in sorted(DATA.iterdir())},
            'file_used': 'unaligned_50.pkl',
            'other_files_urls': {
                'aligned_50.pkl': SOURCE.replace('unaligned_50', 'aligned_50'),
                'label.csv': 'https://huggingface.co/datasets/tamb2203579/CMU-MOSI/resolve/main/label.csv'},
            'features': {'text': 'BERT token embeddings (768-d), mean+std over attention-masked tokens',
                         'audio': 'COVAREP (5-d MMSA subset), mean+std over first audio_length frames',
                         'visual': 'FACET (20-d), mean+std over first vision_length frames'},
            'label': 'regression_labels >= 0 -> 1 (non-negative), < 0 -> 0 (Acc-2 incl. zeros as positive)',
            'splits': counts, 'seed': SEED,
            'code_sha256': {p.name: sha256(p) for p in [Path(__file__).resolve(), HERE / 'tier1.py',
                            HERE / 'a7_moddrop.py', HERE.parent / 'mike2026/src/anchored_fusion.py',
                            HERE.parent / 'mike2026/scripts/run_revision3_analysis.py']}}
    (OUT / 'provenance.json').write_text(json.dumps(prov, indent=2))
    diag = diagnostics(P, ytv)
    (OUT / 'diagnostics.json').write_text(json.dumps(diag, indent=2))
    summarize(expert_rows, clean, bs, stress, counts, natural, gates, diag)


def diagnostics(P, y):
    """Why the CARF gate is (or is not) zero: KKT gradient at theta=0 and residual structure."""
    a = lg(P[:, 1])
    m = AnchoredResidualFusion(robust_weight=0, l2=0, context=False)
    a0, X = m._design(P)
    from scipy.special import expit
    grad0 = X.T @ (expit(a0) - y) / len(y)
    return {'anchor_platt_slope_oof': float(LogisticRegression(C=1e6).fit(a[:, None], y).coef_[0, 0]),
            'corr_residual_vs_minus_anchor': {mm: float(np.corrcoef(lg(P[:, j]) - a, -a)[0, 1])
                                              for j, mm in [(0, 'audio'), (2, 'visual')]},
            'sd_logit': {mm: float(lg(P[:, j]).std()) for j, mm in enumerate(MODS)},
            'grad_at_zero_no_robust_no_l2_no_context_full_mask': grad0.tolist(),
            'coef_no_robust_no_l2': AnchoredResidualFusion(robust_weight=0, l2=0).fit(P, y).coef_.tolist(),
            'coef_unbounded_no_context_no_robust_no_l2': AnchoredResidualFusion(
                robust_weight=0, l2=0, bounded=False, context=False).fit(P, y).coef_.tolist()}


MAIN = ['text', 'audio', 'visual', 'mean', 'stack', 'mask_stack', 'offset_lr', 'offset_lr_masked',
        'mlp_gate', 'moddrop_mlp', 'carf', 'carf_B0.5', 'carf_B2', 'carf_no_robust', 'carf_no_context',
        'carf_unbounded']


def summarize(expert_rows, clean, bs, stress, counts, natural, gates, diag):
    f = lambda x: f'{x:.3f}'
    lines = ['# E3: CARF on CMU-MOSI (second domain)', '',
             'Data: MMSA `unaligned_50.pkl` (HF mirror tamb2203579/CMU-MOSI); splits '
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
    agg = stress.groupby(['scenario', 'model']).agg(auroc=('auroc', 'mean'), acc=('accuracy', 'mean'),
                                                    disp=('max_anchor_logit_displacement', 'max')).reset_index()
    piv = agg.pivot(index='scenario', columns='model', values='auroc')
    cols = [k for k in ['text', 'mean', 'stack', 'mask_stack', 'offset_lr_masked', 'moddrop_mlp', 'carf',
                        'carf_B0.5', 'carf_B2', 'carf_no_robust'] if k in piv.columns]
    lines += ['', '## Stress (test AUROC, mean over repeats for random scenarios)', '',
              '| scenario | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for s in piv.index:
        lines.append(f'| {s} | ' + ' | '.join(f(piv.loc[s, k]) for k in cols) + ' |')
    worst = piv.min()
    lines += ['', '## Worst-case AUROC over scenarios and max |fused - anchor| logit', '',
              '| model | clean AUROC | worst AUROC | worst scenario | max displacement |', '|---|---|---|---|---|']
    disp = agg.groupby('model').disp.max()
    for k in MAIN:
        if k in piv.columns:
            lines.append(f"| {k} | {f(c.loc[k, 'auroc'])} | {f(worst[k])} | {piv[k].idxmin()} | {disp[k]:.2f} |")
    lines += ['', '## Notes', '',
              f"- Natural missingness (all-zero frames within valid length): {natural}. "
              'Too few for a subgroup analysis; these segments were kept with their (zero) pooled features.',
              '- Time-shuffling before pooling was skipped: mean/std pooling is permutation-invariant, '
              'so it is an exact no-op on these features.',
              f"- CARF gate (B=1): coef {np.round(gates['carf']['coef'], 4).tolist()}, "
              f"per-modality budgets [audio, visual] {np.round(gates['carf']['budgets'], 4).tolist()}.",
              f"- Gate diagnostics (OOF): {json.dumps(diag)}. A positive gradient at theta=0 with theta>=0 "
              'means theta=0 is the constrained optimum (KKT). The residual logit(p_j)-a is almost '
              'collinear with -a, so any positive weight mainly shrinks the (slightly underconfident, '
              'Platt slope>1) text anchor.',
              f"- offset_lr weights [intercept, audio, visual]: {np.round(gates['offset_lr'], 3).tolist()}."]
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    import e3_second_domain
    e3_second_domain.main()
