"""Copy separately obtained authentic inputs into a fresh, isolated study workspace."""
from pathlib import Path
import argparse,json,hashlib,shutil
REPO=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,required=True,help='Authorized artifact tree with the source-relative paths in training_artifacts.json.')
parser.add_argument('--output',type=Path,default=REPO/'data/training_workspace')
parser.add_argument('--check-only',action='store_true',help='Authenticate inputs without copying or starting training.')
args=parser.parse_args()
source=args.source_root.resolve();target=args.output.resolve()
assert target.is_relative_to(REPO.resolve()),'The reproduction workspace must be inside this repository.'
assert not target.exists() or args.check_only,'Use a new workspace; do not overwrite an existing run.'
manifest=json.loads((REPO/'configs/training_artifacts.json').read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
errors=[]
for item in manifest:
    p=source/item['path']
    if not p.is_file():errors.append('Missing: '+item['path']);continue
    if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:errors.append('Mismatched: '+item['path'])
if errors:raise SystemExit('\n'.join(errors)+'\nNo training has started. See data/README.md for acquisition and archival limits.')
if args.check_only:
    print('All required artifacts authenticated; no copies or training performed.');raise SystemExit(0)
target.mkdir(parents=True)
for item in manifest:
    p=target/item['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/item['path'],p)
for source_file in (REPO/'src/frozen/analysis_tools').glob('*.py'):
    p=target/'analysis_tools'/source_file.name;p.parent.mkdir(exist_ok=True);shutil.copy2(source_file,p)
(target/'reference_inputs').mkdir(exist_ok=True)
print('Prepared isolated workspace:',target)
print('No model training or inference has been started.')
