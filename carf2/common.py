"""Shared loaders for CARF-2 experiments (read-only use of mike2026 and experiments)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIKE = ROOT / 'mike2026'
sys.path.insert(0, str(MIKE))
DATA = MIKE / 'data/revision2/processed'
CACHE_V1 = MIKE / 'results/revision3/cache/d562c1894fd42bb6'
OUT = Path(__file__).resolve().parent / 'results'
SEED = 20260923


def meta():
    frames = []
    for cohort in ['ISPY2', 'ISPY1']:
        m = pd.read_csv(DATA / f'{cohort}_meta.csv', dtype={'patient_id': str}).set_index('patient_id')
        m['cohort'] = cohort
        frames.append(m)
    return pd.concat(frames)


def subtype_group(m):
    """0 = HER2+, 1 = HR+/HER2-, 2 = TNBC; missing receptor values use training medians (HR+, HER2-)."""
    hr = m.hr.fillna(1).to_numpy(); her2 = m.her2.fillna(0).to_numpy()
    return np.where(her2 == 1, 0, np.where(hr == 1, 1, 2)).astype(int)


def load_v1(key, M):
    with np.load(CACHE_V1 / f'{key}.npz') as z:
        P, Q = z['P'], z['Q']
        tr, te = z['train_ids'].astype(str), z['test_ids'].astype(str)
    return P, Q, tr, te
