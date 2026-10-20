"""Explicit opt-in entry point for the exact frozen GPU training implementations."""
from pathlib import Path
import argparse,subprocess,sys,os
REPO=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task',choices=['classification','activity'],required=True)
parser.add_argument('--workspace',type=Path,default=REPO/'data/training_workspace')
parser.add_argument('--execute',action='store_true',help='Explicitly start the complete frozen training sequence; requires GPU and authentic inputs.')
args=parser.parse_args();workspace=args.workspace.resolve()
assert workspace.is_relative_to(REPO.resolve()),'Use the new repository-local workspace.'
if not args.execute:
    print('No training started. Prepare authentic inputs, activate the original enhancer environment, and add --execute to opt in.');raise SystemExit(0)
assert workspace.is_dir() and (workspace/'analysis_tools').is_dir(),'Prepare and authenticate the workspace first.'
assert Path(sys.prefix).name.lower()=='enhancer','The frozen source requires the original environment name enhancer.'
if args.task=='classification':
    program="import sys,itertools;sys.path.insert(0,'analysis_tools');import phase11c_completion as s;s.phase31_setup();[(s.train_job(m,t,z,f)) for m,t,z,f in itertools.product(s.MODELS,s.STAGES,s.SEEDS,s.FOLDS)];s.phase31_finish()"
else:
    program="import sys;sys.path.insert(0,'analysis_tools');import mpra_fairness_training as s;s.setup();[s.train_job(*job) for job in s.jobs()];import mpra_fairness_analysis as a;a.selection_and_inference()"
result=subprocess.run([sys.executable,'-B','-X','utf8','-c',program],cwd=workspace,env=os.environ.copy())
raise SystemExit(result.returncode)
