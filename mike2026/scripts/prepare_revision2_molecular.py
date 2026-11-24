#!/usr/bin/env python3
"""Prepare auditable molecular matrices and verify explicit trial-ID linkage."""
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.revision2_data import read_ispy1_expression, read_ispy2_expression, read_proteins, soft_records
RAW=ROOT/'data/revision2/raw'
OUT=ROOT/'data/revision2/processed'


def excel(path, **kwargs):
    # openpyxl warns about unsupported Excel UI extensions; cell values are read.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unknown extension is not supported and will be removed', category=UserWarning)
        return pd.read_excel(path,**kwargs)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    x1,m1=read_ispy1_expression(RAW/'GSE22226_family.soft.gz')
    x2=read_ispy2_expression(RAW/'GSE194040_expression.txt.gz')
    protein,mp=read_proteins(RAW/'GSE196093_family.soft.gz')
    m2=pd.DataFrame([{'patient_id':r['metadata']['patient id'],'gsm':r['id'],**r['metadata']}
                     for r in soft_records(RAW/'GSE194040_family.soft.gz') if r['kind']=='SAMPLE'])
    m1['pcr_source']=m1['pathological complete response (pcr)'].map({'Yes':1,'No':0})
    m2['pcr_source']=pd.to_numeric(m2.pcr)
    mp['pcr_source']=pd.to_numeric(mp.pcr)
    for label,meta in [('ISPY1',m1),('ISPY2',m2),('ISPY2_RPPA',mp)]:
        disagreements=meta.dropna(subset=['patient_id']).groupby('patient_id').pcr_source.nunique()
        if disagreements.gt(1).any(): raise ValueError(f'{label}: inconsistent technical-replicate outcomes')
        meta.to_csv(OUT/f'{label}_source_metadata.csv',index=False)
    c1=excel(RAW/'ispy1_clinical.xlsx',sheet_name='TCIA Patient Clinical Subset')
    o1=excel(RAW/'ispy1_clinical.xlsx',sheet_name='TCIA Outcomes Subset')
    c1=c1.merge(o1[['SUBJECTID','PCR']],on='SUBJECTID',validate='one_to_one')
    c1=c1.rename(columns={'SUBJECTID':'patient_id','PCR':'pcr','HR Pos':'hr','Her2MostPos':'her2'})
    c2=excel(RAW/'ispy2_clinical.xlsx').rename(columns={'Patient_ID':'patient_id','pCR':'pcr','HR':'hr','HER2':'her2','Age_at_Screening':'age','Arm':'arm'})
    image=pd.read_csv(RAW/'BreastDCEDL_metadata_min_crop.csv')
    image['patient_id']=image.pid.str.extract(r'(\d+)$')[0]
    summaries={}; links=[]
    for label,X,clinical,metadata,dataset in [('ISPY1',x1,c1,m1,'spy1'),('ISPY2',x2,c2,m2,'spy2')]:
        clinical.patient_id=clinical.patient_id.astype(str)
        clinical=clinical.set_index('patient_id')
        im=image.loc[image.dataset==dataset].set_index('patient_id')
        labels=metadata.dropna(subset=['patient_id']).groupby('patient_id').pcr_source.first()
        candidates=clinical.index.intersection(X.index).intersection(im.index)
        rows=[]
        for pid in candidates:
            row=clinical.loc[pid]
            reasons=[]
            if pd.isna(row.pcr): reasons.append('missing_pcr')
            elif pd.notna(labels.get(pid)) and row.pcr!=labels.loc[pid]: reasons.append('geo_tcia_pcr_disagreement')
            if pd.notna(im.loc[pid,'pCR']) and row.pcr!=im.loc[pid,'pCR']: reasons.append('image_tcia_pcr_disagreement')
            links.append({'cohort':label,'patient_id':pid,'image_id':im.loc[pid,'pid'],'included':not reasons,'exclusion_reason':';'.join(reasons)})
            if reasons: continue
            subtype='HER2+' if row.her2==1 else ('HR+/HER2-' if row.hr==1 and row.her2==0 else ('TNBC' if row.hr==0 and row.her2==0 else 'Unknown'))
            rows.append({'patient_id':pid,'image_id':im.loc[pid,'pid'],'pcr':int(row.pcr),'age':row.age,'hr':row.hr,'her2':row.her2,'subtype':subtype,'arm':row.get('arm','AC_based'), 'protein_available':pid in protein.index})
        final=pd.DataFrame(rows).set_index('patient_id')
        if final.index.has_duplicates or final.pcr.nunique()!=2: raise ValueError('Invalid matched cohort')
        X.loc[final.index].to_csv(OUT/f'{label}_expression.csv.gz')
        final.to_csv(OUT/f'{label}_meta.csv')
        summaries[label]={'clinical':len(clinical),'expression':len(X),'images_in_manifest':len(im), 'matched_candidates':len(candidates),'retained_before_image_QC':len(final),'pcr_positive':int(final.pcr.sum()),'gene_features':X.shape[1]}
        print(label,summaries[label],flush=True)
    protein.to_csv(OUT/'ISPY2_RPPA_expression.csv.gz')
    pd.DataFrame(links).to_csv(OUT/'patient_linkage_audit.csv',index=False)
    (OUT/'molecular_cohort_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print('Molecular preparation complete; image extraction/QC still required.',flush=True)

if __name__=='__main__': prepare()
