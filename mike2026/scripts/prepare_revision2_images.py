#!/usr/bin/env python3
"""Verify, safely extract, and measure the immutable pretreatment MRI archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.revision2_imaging import image_features
RAW=ROOT/'data/revision2/raw'
OUT=ROOT/'data/revision2/processed'

def md5(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'md5').hexdigest()

def extract(cohort):
    name=f'BreastDCEDL_{cohort}_min_crop.tar.gz'
    entry=next(x for x in json.loads((RAW/'BreastDCEDL_zenodo_record.json').read_text())['files'] if x['key']==name)
    archive=RAW/name
    if archive.stat().st_size!=entry['size'] or md5(archive)!=entry['checksum'].split(':')[1]:
        raise ValueError(f'{name}: publisher checksum failed')
    target=ROOT/'data/revision2/images'
    target.mkdir(parents=True,exist_ok=True)
    files=[]
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            if member.isdir(): continue
            relative=PurePosixPath(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe archive member: {member.name}')
            if relative.parts[0]!=f'BreastDCEDL_{cohort}_min_crop':
                raise ValueError(f'Unexpected archive member: {member.name}')
            # Ignore harmless desktop metadata included by the publisher.
            if not member.name.endswith('.nii.gz'):
                continue
            destination=target/str(relative)
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix('.part')
            with stream.extractfile(member) as source, temporary.open('wb') as sink:
                shutil.copyfileobj(source,sink)
            if temporary.stat().st_size!=member.size: raise ValueError('Incomplete extraction')
            temporary.replace(destination)
            files.append({'path':member.name,'bytes':member.size})
    (OUT/f'{cohort}_image_extraction.json').write_text(json.dumps({'archive':name,'publisher_md5':entry['checksum'],'files':files},indent=2)+'\n')
    return target/f'BreastDCEDL_{cohort}_min_crop'

def measure(args):
    pid,row,base=args
    audit={'image_id':pid,'included':False}
    try:
        phases=[]; phase_paths=[]
        for column in ['pre','post_early','post_late']:
            paths=list((base/'dce').glob(f'{pid}_*_vis1_dce_aqc_{int(row[column])}.nii.gz'))
            if len(paths)!=1: raise ValueError(f'{column}: expected one phase, found {len(paths)}')
            phases.append(nib.load(paths[0]).get_fdata(dtype=np.float32)); phase_paths.append(str(paths[0].relative_to(ROOT)))
        masks=list((base/'mask').glob(f'{pid}_*_vis1_mask.nii.gz'))
        if len(masks)!=1: raise ValueError(f'Expected one mask, found {len(masks)}')
        mask=nib.load(masks[0]).get_fdata(dtype=np.float32)
        if not np.isfinite(mask).all(): raise ValueError('Nonfinite segmentation')
        if not np.isin(np.unique(mask),[0,1]).all(): raise ValueError('Segmentation is not binary')
        spacing=[row.slice_thick,row.xy_spacing,row.xy_spacing]
        features=image_features(*phases,mask,spacing)
        audit.update(included=True,shape=str(mask.shape),mask_voxels=int(mask.sum()),
                     slice_spacing_mm=float(spacing[0]),xy_spacing_mm=float(spacing[1]),
                     metadata_n_z=float(row.n_z),actual_n_z=int(mask.shape[0]),
                     metadata_z_matches=bool(mask.shape[0]==row.n_z),
                     phase_paths=';'.join(phase_paths),mask_path=str(masks[0].relative_to(ROOT)))
        return {'image_id':pid,**features},audit
    except (ValueError,OSError) as error:
        audit['exclusion_reason']=str(error)
        return None,audit

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--cohort',choices=['ISPY1','ISPY2'],required=True)
    parser.add_argument('--workers',type=int,default=2); options=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    base=extract(options.cohort)
    metadata=pd.read_csv(RAW/'BreastDCEDL_metadata_min_crop.csv').set_index('pid')
    selected=pd.read_csv(OUT/f'{options.cohort}_meta.csv',dtype={'patient_id':str}).set_index('image_id')
    tasks=[(pid,metadata.loc[pid],base) for pid in selected.index]
    rows=[]; audits=[]
    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        for number,(features,audit) in enumerate(executor.map(measure,tasks),1):
            audits.append(audit)
            if features: rows.append(features)
            if number%20==0: print(f'{options.cohort}: {number}/{len(tasks)} images measured',flush=True)
    frame=pd.DataFrame(rows).set_index('image_id')
    frame.index=selected.loc[frame.index,'patient_id']; frame.index.name='patient_id'
    frame.to_csv(OUT/f'{options.cohort}_mri.csv')
    pd.DataFrame(audits).to_csv(OUT/f'{options.cohort}_image_qc.csv',index=False)
    print(f'{options.cohort}: {len(frame)}/{len(tasks)} retained, {frame.shape[1]} MRI features',flush=True)
    if not len(frame): raise ValueError('No usable MRI features')

if __name__=='__main__': main()
