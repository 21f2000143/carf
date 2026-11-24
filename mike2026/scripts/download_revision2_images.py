#!/usr/bin/env python3
"""Fetch immutable BreastDCEDL image archives with resume and MD5 validation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/revision2/raw'
RECORD=18114231
NAMES=['BreastDCEDL_ISPY1_min_crop.tar.gz','BreastDCEDL_ISPY2_min_crop.tar.gz',
       'BreastDCEDL_metadata_min_crop.csv','LICENSE']

def digest(path):
 with path.open('rb') as stream:
  return hashlib.file_digest(stream,'md5').hexdigest()

def fetch(item):
 name=item['key']; destination=RAW/name
 expected=item['checksum'].removeprefix('md5:')
 if destination.exists() and destination.stat().st_size==item['size'] and digest(destination)==expected:
  print('Verified existing',name,flush=True); return
 temporary=destination.with_name(destination.name+'.part')
 for attempt in range(5):
  start=temporary.stat().st_size if temporary.exists() else 0
  if start==item['size']: break
  try:
   with requests.get(item['links']['self'],headers={'Range':f'bytes={start}-'} if start else {},
                     stream=True,timeout=(30,120)) as response:
    response.raise_for_status()
    if start and response.status_code==206:
     if not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
      raise ValueError('Unexpected range response')
     mode='ab'
    else: mode='wb'; start=0
    print(f'Downloading {name} from byte {start}',flush=True)
    last=time.monotonic()
    with temporary.open(mode) as stream:
     for chunk in response.iter_content(4*1024*1024):
      stream.write(chunk); start+=len(chunk)
      if time.monotonic()-last>30:
       print(f'{name}: {start}/{item["size"]} bytes',flush=True); last=time.monotonic()
   break
  except (requests.RequestException, OSError) as error:
   print(f'{name} attempt {attempt+1}: {error}',flush=True)
   if attempt==4: raise
   time.sleep(2)
 if temporary.stat().st_size!=item['size'] or digest(temporary)!=expected:
  raise ValueError(f'{name}: downloaded size/checksum mismatch')
 temporary.replace(destination)
 print('Verified download',name,flush=True)

def main():
 RAW.mkdir(parents=True,exist_ok=True)
 record=requests.get(f'https://zenodo.org/api/records/{RECORD}',timeout=60)
 record.raise_for_status(); data=record.json()
 (RAW/'BreastDCEDL_zenodo_record.json').write_text(json.dumps(data,indent=2)+'\n')
 items=[item for item in data['files'] if item['key'] in NAMES]
 assert len(items)==len(NAMES)
 with ThreadPoolExecutor(max_workers=2) as executor:
  list(executor.map(fetch,items))
 (RAW/'image_download_manifest.json').write_text(json.dumps(items,indent=2)+'\n')
 print('All image sources downloaded and verified.',flush=True)

if __name__=='__main__': main()
