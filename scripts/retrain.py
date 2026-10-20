"""Explicit opt-in entry point for the exact frozen GPU training implementations."""
from pathlib import Path
import argparse,subprocess,sys,os
REPO=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task',choices=['classification','activity'],required=True)
parser.add_argument('--workspace',type=Path,default=REPO/'data/training_workspace')
parser.add_argument('--execute',action='store_true',help='Explicitly start the complete frozen training sequence; requires GPU and authentic inputs.')
parser.add_argument('--training-only',action='store_true',help='Authenticate completed activity training without starting test/external inference.')
args=parser.parse_args();workspace=args.workspace.resolve()
assert workspace.is_relative_to(REPO.resolve()),'Use the new repository-local workspace.'
if not args.execute:
    print('No training started. Prepare authentic inputs, activate the original enhancer environment, and add --execute to opt in.');raise SystemExit(0)
assert workspace.is_dir() and (workspace/'analysis_tools').is_dir(),'Prepare and authenticate the workspace first.'
assert Path(sys.prefix).name.lower()=='enhancer','The frozen source requires the original environment name enhancer.'
program="import sys,json;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,'analysis_tools');from enhancer_benchmark_audit.training_workflow import run_training;print(json.dumps(run_training(sys.argv[2],evaluate=sys.argv[3]=='yes')))"
result=subprocess.run([sys.executable,'-B','-X','utf8','-c',program,str(REPO/'src'),args.task,'no' if args.training_only else 'yes'],cwd=workspace,env=os.environ.copy())
raise SystemExit(result.returncode)
