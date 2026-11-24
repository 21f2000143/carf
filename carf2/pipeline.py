"""Cross-fitted expert predictions for CARF-2 study designs.

Modality order in every prediction matrix: optional experts first in ``design.optional``
order, then the clinical anchor last. Missing optional inputs give NaN.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import OUT, DATA
from experts import Expert, AuxHead, clinical_frame, BASE_CLIN, TREAT_CLIN


def load_data(rppa=True):
    """Returns development and external blocks: dicts of DataFrames aligned to the meta index."""
    from scripts.run_revision2_analysis import common_data  # mike2026, read-only
    E, M, I, Ee, Me, Ie = common_data()
    dev = {'rna': E, 'mri': I, 'clinical': clinical_frame(M)}
    ext = {'rna': Ee, 'mri': Ie, 'clinical': clinical_frame(Me)}
    if rppa:
        rp = pd.read_csv(DATA / 'ISPY2_RPPA_expression.csv.gz', index_col=0)
        rp.index = rp.index.astype(str)
        rp = rp.replace([np.inf, -np.inf], np.nan)
        dev['rppa'] = rp.reindex(M.index)
        ext['rppa'] = pd.DataFrame(np.nan, index=Me.index, columns=rp.columns)
    return dev, M.pcr.to_numpy(int), ext, Me.pcr.to_numpy(int), M, Me


class Design:
    def __init__(self, name, optional=('rna', 'mri'), clin_cols=BASE_CLIN, residual=False, aux=True,
                 rna_k=250, C=None):
        self.name, self.optional, self.clin_cols = name, list(optional), list(clin_cols)
        self.residual, self.aux, self.rna_k, self.C = residual, aux, rna_k, dict(C or {})

    def key(self):
        d = {k: v for k, v in self.__dict__.items() if k != 'name'}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]

    def fit(self, blocks, y):
        clin = Expert('clinical', clin_cols=self.clin_cols).fit(blocks['clinical'], y)
        a = clin.residual_logit(blocks['clinical'])
        experts = [Expert(k, residual=self.residual, k=self.rna_k, C=self.C.get(k)).fit(blocks[k], y, a)
                   for k in self.optional]
        aux = [AuxHead(k).fit(blocks[k], blocks['clinical']) for k in self.optional] if self.aux else None
        return clin, experts, aux

    def predict(self, fitted, blocks):
        """Returns (p, aux): p is n x (J+1) (optional..., clinical); aux is n x J x 2 or None."""
        clin, experts, aux = fitted
        a = clin.residual_logit(blocks['clinical'])
        p = np.column_stack([e.predict(blocks[k], a) for e, k in zip(experts, self.optional)] +
                            [clin.predict(blocks['clinical'])])
        A = None if aux is None else np.stack([h.predict(blocks[k]) for h, k in zip(aux, self.optional)], axis=1)
        return p, A


def subset(blocks, idx):
    return {k: v.iloc[idx] for k, v in blocks.items()}


def oof(design, blocks, y, seed):
    J = len(design.optional)
    P = np.zeros((len(y), J + 1)); PA = np.full((len(y), J, 2), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        p, A = design.predict(design.fit(subset(blocks, tr), y[tr]), subset(blocks, te))
        P[te] = p
        if A is not None:
            PA[te] = A
    return P, PA


def cached(design, key, blocks, y, tr, te_blocks, seed):
    """P: inner OOF predictions on training rows; Q: experts refit on all training rows, applied to te_blocks."""
    path = OUT / 'cache' / design.key() / f'{key}.npz'
    if path.exists():
        with np.load(path) as z:
            return z['P'], z['PA'], z['Q'], z['QA']
    path.parent.mkdir(parents=True, exist_ok=True)
    btr = subset(blocks, tr)
    P, PA = oof(design, btr, y[tr], seed)
    Q, QA = design.predict(design.fit(btr, y[tr]), te_blocks)
    QA = np.full((len(Q), len(design.optional), 2), np.nan) if QA is None else QA
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, P=P, PA=PA, Q=Q, QA=QA)
    tmp.replace(path)
    (path.parent / 'design.json').write_text(json.dumps(design.__dict__))
    return P, PA, Q, QA
