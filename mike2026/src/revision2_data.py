"""Patient-ID based molecular preparation for the imaging revision."""
from pathlib import Path
import gzip
import re
import numpy as np
import pandas as pd


def soft_records(path):
    """Stream SOFT, retaining metadata and only annotation/value columns."""
    current = None
    header = None
    mode = None
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            line = line.rstrip('\r\n')
            if line.startswith('^'):
                if current is not None:
                    yield current
                kind, _, name = line.partition(' = ')
                current = {'kind': kind[1:], 'id': name, 'metadata': {}, 'values': {}}
                mode = None
            elif line in ('!platform_table_begin', '!sample_table_begin'):
                mode = 'header'
            elif line in ('!platform_table_end', '!sample_table_end'):
                mode = None
            elif mode == 'header':
                header = line.split('\t'); mode = 'table'
                if current['kind'] == 'PLATFORM':
                    columns = {key: header.index(key) for key in ('ID', 'GENE_SYMBOL') if key in header}
                else:
                    columns = {key: header.index(key) for key in ('ID_REF', 'VALUE') if key in header}
            elif mode == 'table':
                fields = line.split('\t')
                if current['kind'] == 'PLATFORM' and 'GENE_SYMBOL' in columns:
                    if len(fields) > columns['GENE_SYMBOL']:
                        symbol = fields[columns['GENE_SYMBOL']].strip()
                        if re.fullmatch(r'[A-Za-z0-9_.-]+', symbol):
                            current['values'][fields[columns['ID']]] = symbol
                elif current['kind'] == 'SAMPLE' and 'VALUE' in columns:
                    if len(fields) > columns['VALUE']:
                        try: value = float(fields[columns['VALUE']])
                        except ValueError: value = np.nan
                        current['values'][fields[columns['ID_REF']]] = value
            elif current is not None and line.startswith('!') and ' = ' in line:
                key, value = line.split(' = ', 1)
                if key.startswith('!Sample_characteristics_ch'):
                    name, _, content = value.partition(': ')
                    current['metadata'][name.lower()] = content
                elif key in ('!Sample_title', '!Sample_platform_id', '!Sample_source_name_ch1', '!Sample_source_name_ch2'):
                    current['metadata'][key.removeprefix('!Sample_')] = value
        if current is not None:
            yield current


def read_ispy1_expression(path):
    maps = {}; metadata = []; expression = []
    for record in soft_records(path):
        if record['kind'] == 'PLATFORM': maps[record['id']] = record['values']
        if record['kind'] != 'SAMPLE': continue
        m = record['metadata']
        if 'i-spy id' not in m:
            if m.get('tissue') == 'normal breast':
                metadata.append({'patient_id': None, 'gsm': record['id'],
                                 'exclusion_reason': 'normal_breast_reference_control', **m})
                continue
            raise ValueError(f"Missing trial ID: {record['id']}")
        pid = str(int(m['i-spy id']))
        mapping = maps[m['platform_id']]
        values = pd.Series(record['values'], dtype=float)
        values.index = values.index.map(mapping)
        values = values.loc[values.index.notna()].groupby(level=0).mean()
        values.name = pid; expression.append(values)
        metadata.append({'patient_id': pid, 'gsm': record['id'], **m})
    meta = pd.DataFrame(metadata)
    expr = pd.DataFrame(expression)
    expr.index.name = 'patient_id'
    # Technical replicates retain their audit rows but become one participant.
    return expr.groupby(level=0).mean(), meta


def read_ispy2_expression(path):
    expression = pd.read_csv(path, sep='\t', index_col=0)
    expression.index = expression.index.astype(str)
    expression = expression.groupby(level=0).mean().T
    # pandas mangles repeated participant column names as ID.1; source records
    # must verify these are genuine technical repeats before averaging.
    expression.index = expression.index.str.replace(r'\.\d+$', '', regex=True)
    expression.index.name = 'patient_id'
    return expression.groupby(level=0).mean()


def read_proteins(path):
    metadata = []; values = []
    for record in soft_records(path):
        if record['kind'] != 'SAMPLE': continue
        m = record['metadata']; pid = str(int(m['title']))
        metadata.append({'patient_id': pid, 'gsm': record['id'], **m})
        values.append(pd.Series(record['values'], name=pid, dtype=float))
    frame = pd.DataFrame(values); frame.index.name = 'patient_id'
    if frame.index.has_duplicates: raise ValueError('Unexpected duplicate RPPA participant')
    return frame, pd.DataFrame(metadata)
