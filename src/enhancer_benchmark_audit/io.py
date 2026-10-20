"""Repository-relative immutable inputs and separate generated outputs."""
from pathlib import Path
import os,json,hashlib,tempfile
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[2]
ROOT=PACKAGE/'results/reproduced'
INPUT=PACKAGE/'results/reference_inputs'
for key in ['TEMP','TMP','TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','JUPYTER_RUNTIME_DIR','IPYTHONDIR']:
    p=PACKAGE/'.runtime'/key.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
tempfile.tempdir=os.environ['TMP']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return pd.read_csv(INPUT/path,float_precision='round_trip',sep='\t' if str(path).endswith('.tsv') else ',')
def jd(path):return json.loads((INPUT/path).read_text(encoding='utf-8'))
def dump(path,obj):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def text(path,value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value.strip()+'\n',encoding='utf-8')
def table(path,frame):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(p,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0} if p.suffix=='.gz' else None)
def manifests():return [read(f'final_evidence/inputs/stage{stage}_candidate_manifest.csv') for stage in [1,2]]
def verify_inputs():
    manifest=json.loads((PACKAGE/'MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest:
        p=PACKAGE/item['path']
        assert p.is_file() and sha(p)==item['sha256'],item['path']
    return True
