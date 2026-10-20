from pathlib import Path
import hashlib,json,os
import pandas as pd
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE/'reproduced'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return pd.read_csv(PACKAGE/'inputs'/path,float_precision='round_trip')
def jd(path):return json.loads((PACKAGE/'inputs'/path).read_text(encoding='utf-8'))
def dump(path,obj):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def text(path,value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value.strip()+'\n',encoding='utf-8')
def table(path,frame):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(p,index=False,float_format='%.17g')
def manifests():return [read(f'final_evidence/inputs/stage{stage}_candidate_manifest.csv') for stage in [1,2]]
def verify_inputs():
    for r in json.loads((PACKAGE/'MANIFEST.json').read_text(encoding='utf-8')):assert sha(PACKAGE/r['path'])==r['sha256'],r['path']
    return True
