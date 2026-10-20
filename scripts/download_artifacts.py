"""Download a published Zenodo record's exact artifacts, or validate a local upload plan."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from enhancer_benchmark_audit.artifacts import materialize, safe_target, sha

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--record-id', type=int, help='Actual published artifact record ID; not a placeholder DOI')
parser.add_argument('--local-upload-plan', type=Path, help='Local prepared upload plan; no remote access')
parser.add_argument('--components', default='inputs', help='Comma-separated inputs,activity,classification,metadata or all')
parser.add_argument('--output', type=Path, default=REPO / 'data/artifacts')
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()
manifest = json.loads((REPO / 'configs/zenodo_artifacts.json').read_text(encoding='utf-8'))
components = set(args.components.split(','))
if 'all' in components:
    components = {'inputs', 'activity', 'classification', 'metadata'}
if not components.issubset({'inputs', 'activity', 'classification', 'metadata'}):
    raise ValueError('Unknown artifact component')
if components & {'activity', 'classification'}:
    components.add('inputs')
files = [f for f in manifest['files'] if f['category'] in components]
destination = args.output.resolve()
if not destination.is_relative_to(REPO):
    raise ValueError('Artifact destination must be inside the repository')
if not args.execute:
    print(json.dumps({'files': len(files), 'download_bytes': sum(f['bytes'] for f in files),
                      'record_id': args.record_id or manifest['record_id'],
                      'status': 'Plan only; add --execute. Artifact record publication is pending if record_id is null.'}, indent=2))
    raise SystemExit(0)
local = None
if args.local_upload_plan:
    local = {f['key']: f for f in json.loads(args.local_upload_plan.read_text(encoding='utf-8'))['files']}
else:
    record = args.record_id or manifest['record_id']
    if record is None:
        raise ValueError('No published artifact record exists yet. Supply its actual ID after Zenodo publication.')
    with urllib.request.urlopen(f'https://zenodo.org/api/records/{record}', timeout=60) as r:
        remote = {f['key']: f for f in json.load(r)['files']}
    if any(f['key'] not in remote for f in files):
        raise ValueError('Record does not contain the expected versioned artifact file set')
destination.mkdir(parents=True, exist_ok=True)
cache = destination / '.downloads'
cache.mkdir(exist_ok=True)
checks = []
for specification in files:
    if local is not None:
        item = local[specification['key']]
        if item['sha256'] != specification['sha256'] or item['bytes'] != specification['bytes']:
            raise ValueError('Local plan differs from the public artifact manifest')
        source = Path(item['local_path'])
    else:
        source = safe_target(cache, specification['key'])
        if not source.exists():
            needed = specification['bytes'] + sum(x['bytes'] for x in specification['members'])
            if shutil.disk_usage(destination).free < needed + 1_000_000_000:
                raise RuntimeError('Insufficient free space; preserve existing files and use a drive with more capacity')
            links = remote[specification['key']]['links']
            url = links.get('content') or links.get('self')
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme != 'https' or parsed.hostname != 'zenodo.org':
                raise ValueError('Unexpected artifact download host')
            temporary = source.with_name(source.name + '.writing')
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open('xb') as output:
                shutil.copyfileobj(response, output)
            if sha(temporary) != specification['sha256']:
                raise ValueError('Downloaded artifact hash mismatch')
            os.replace(temporary, source)
    count = materialize(source, specification, destination)
    checks.append({'key': specification['key'], 'passed': True, 'materialized_files': count})
    (destination / 'ACQUISITION_VALIDATION.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('Authenticated', specification['key'], flush=True)
