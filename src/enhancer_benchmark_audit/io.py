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
    """Check the compact manuscript-facing release inputs without depending on the historical archive."""
    required=[
        INPUT/'phase21_outputs/historical_test_overlap_roster.csv',
        INPUT/'final_evidence/benchmark_audit.csv',
        INPUT/'final_evidence/inputs/stage1_candidate_manifest.csv',
        INPUT/'final_evidence/inputs/stage2_candidate_manifest.csv',
        INPUT/'final_statistics/classification_predictions.csv.gz',
        INPUT/'final_statistics/classification_summary.csv',
        INPUT/'phase32_outputs/seed_metrics_pooled_and_fold.csv',
        INPUT/'phase35_outputs/training_job_summary.csv',
        INPUT/'phase36_outputs/validation_candidate_scores.csv',
        INPUT/'phase37_outputs/verified_ensemble_predictions.csv.gz',
        INPUT/'final_statistics/mpra_internal_summary.csv',
        INPUT/'final_statistics/external_transfer_summary.csv',
        PACKAGE/'results/predictions/classification_current_all_seeds.csv.gz',
        PACKAGE/'configs/activity_selection_lock.json',
        PACKAGE/'configs/activity_protocol.json',
        PACKAGE/'configs/activity_input_manifest.csv',
        PACKAGE/'configs/classification_run_lock.json',
        PACKAGE/'configs/activity_run_lock.json',
    ]
    missing=[str(p.relative_to(PACKAGE)) for p in required if not p.is_file()]
    assert not missing,f'Missing release inputs: {missing}'
    return True
