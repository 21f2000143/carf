#!/usr/bin/env python3
"""Download public matched imaging/molecular sources for revision 2."""
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/revision2/raw'
URLS = {
 'ispy1_clinical.xlsx': 'https://www.cancerimagingarchive.net/wp-content/uploads/I-SPY-1-All-Patient-Clinical-and-Outcome-Data.xlsx',
 'ispy1_mri.xlsx': 'https://www.cancerimagingarchive.net/wp-content/uploads/I-SPY-1-level2a-Patient-Clinical-and-Outcome-Data.xlsx',
 'ispy1_radiomics.xlsx': 'https://www.cancerimagingarchive.net/wp-content/uploads/ISPY_DataPaper_features.xlsx',
 'ispy1_radiomics_README.txt': 'https://www.cancerimagingarchive.net/wp-content/uploads/ispy1_README.txt',
 'ispy2_clinical.xlsx': 'https://www.cancerimagingarchive.net/wp-content/uploads/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx',
 'ispy2_mri.xlsx': 'https://www.cancerimagingarchive.net/wp-content/uploads/Multi-feature-MRI-NACT-Data.xlsx',
 'GSE194040_expression.txt.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194040/suppl/GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt.gz',
 'GSE194040_family.soft.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194040/soft/GSE194040_family.soft.gz',
 'GSE22226_family.soft.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE22nnn/GSE22226/soft/GSE22226_family.soft.gz',
 'GSE196093_family.soft.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE196nnn/GSE196093/soft/GSE196093_family.soft.gz',
}

def main():
 RAW.mkdir(parents=True,exist_ok=True)
 records=[]
 for name,url in URLS.items():
  dest=RAW/name
  if not dest.exists():
   print('Downloading',name,flush=True)
   with requests.get(url,stream=True,timeout=(30,180)) as response:
    response.raise_for_status()
    temp=dest.with_suffix(dest.suffix+'.part')
    with temp.open('wb') as out:
     for chunk in response.iter_content(1024*1024):
      out.write(chunk)
    temp.replace(dest)
  with dest.open('rb') as source:
   digest=hashlib.file_digest(source,'sha256').hexdigest()
  records.append({'file':name,'url':url,'bytes':dest.stat().st_size,'sha256':digest})
  print(name,dest.stat().st_size,flush=True)
  (RAW/'download_manifest.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':
 main()
