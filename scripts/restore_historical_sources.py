"""Restore the inventoried original research source tree; do not execute it."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(destination):
    repo = Path(__file__).resolve().parents[1]
    inventory = json.loads((repo / 'docs/ORIGINAL_SOURCE_INVENTORY.json').read_text(encoding='utf-8'))
    destination = destination.expanduser().resolve()
    if destination == repo or repo in destination.parents:
        raise ValueError('Choose a new empty directory outside the repository.')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('The destination must be new or empty; existing files are never overwritten.')
    plan = []
    for record in inventory['records']:
        relative = PurePosixPath(record['source'])
        if relative.is_absolute() or '..' in relative.parts or any(':' in p for p in relative.parts):
            raise ValueError('Unsafe source path in inventory.')
        archived = (repo / record['archive_path']).resolve()
        target = destination.joinpath(*relative.parts).resolve()
        if repo not in archived.parents or destination not in target.parents:
            raise ValueError('Path escapes its declared directory.')
        if digest(archived) != record['sha256'] or archived.stat().st_size != record['bytes']:
            raise ValueError('Archive hash or size mismatch: ' + record['archive_path'])
        plan.append((archived, target, record))
    destination.mkdir(parents=True, exist_ok=True)
    for archived, target, record in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(archived.read_bytes())
        if digest(target) != record['sha256']:
            raise ValueError('Restored-file hash mismatch: ' + record['source'])
    print(json.dumps({'restored_source_paths': len(plan),
                      'notebook_paths': inventory['notebook_source_copies'],
                      'code_paths': inventory['code_source_copies'],
                      'all_hashes_verified': True,
                      'historical_sources_executed': False,
                      'destination': str(destination)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New empty directory outside this repository')
    restore(parser.parse_args().destination)
