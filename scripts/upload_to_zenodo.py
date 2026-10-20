"""Stream a prepared local file plan into an EXISTING Zenodo draft; never publish it."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request


def checksums(path):
    s, m = hashlib.sha256(), hashlib.md5()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            s.update(block)
            m.update(block)
    return s.hexdigest(), 'md5:' + m.hexdigest()


def same_md5(remote, expected):
    return str(remote).lower().removeprefix('md5:') == expected.lower().removeprefix('md5:')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--draft-id', type=int, required=True)
    p.add_argument('--local-plan', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    plan = json.loads(args.local_plan.read_text(encoding='utf-8'))
    files = plan['files']
    if len(files) > 100 or sum(f['bytes'] for f in files) > 50_000_000_000:
        raise ValueError('Plan exceeds the default single-record Zenodo quota')
    if not args.execute:
        print(json.dumps({'files': len(files), 'bytes': sum(f['bytes'] for f in files),
                          'draft_id': args.draft_id, 'will_publish': False,
                          'instruction': 'Review draft metadata/custom rights in Zenodo, set ZENODO_ACCESS_TOKEN privately, then add --execute.'}, indent=2))
        return
    token = os.environ.get('ZENODO_ACCESS_TOKEN')
    if not token:
        raise ValueError('ZENODO_ACCESS_TOKEN is missing; do not store tokens in notebooks or source files.')
    headers = {'Authorization': 'Bearer ' + token, 'User-Agent': 'enhancer-benchmark-audit'}
    draft_url = f'https://zenodo.org/api/deposit/depositions/{args.draft_id}'

    def draft():
        with urllib.request.urlopen(urllib.request.Request(draft_url, headers=headers), timeout=60) as r:
            data = json.load(r)
        if data.get('submitted') or data.get('state') == 'done':
            raise ValueError('The destination is published; use an editable draft.')
        return data

    current = draft()
    bucket = current['links']['bucket']
    parsed = urllib.parse.urlparse(bucket)
    if parsed.scheme != 'https' or parsed.hostname != 'zenodo.org':
        raise ValueError('Unexpected upload destination')
    completed = []
    for item in files:
        source = Path(item['local_path'])
        if not source.is_file() or source.stat().st_size != item['bytes']:
            raise ValueError('Original upload file is missing or changed: ' + item['key'])
        sha, md5 = checksums(source)
        if sha != item['sha256']:
            raise ValueError('Original upload file hash mismatch: ' + item['key'])
        existing = {f['filename']: f for f in draft().get('files', [])}
        if item['key'] in existing:
            if not same_md5(existing[item['key']]['checksum'], md5):
                raise ValueError('A different file already exists in the draft; it was not overwritten.')
            action = 'reused_verified_upload'
        else:
            url = bucket.rstrip('/') + '/' + urllib.parse.quote(item['key'], safe='')
            with source.open('rb') as stream:
                request = urllib.request.Request(url, data=stream, method='PUT',
                    headers={**headers, 'Content-Type': 'application/octet-stream', 'Content-Length': str(item['bytes'])})
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
            if not same_md5(result['checksum'], md5):
                raise ValueError('Remote upload checksum mismatch')
            action = 'uploaded_and_verified'
        completed.append({'key': item['key'], 'sha256': sha, 'zenodo_checksum': md5, 'status': action})
        args.local_plan.with_name('ZENODO_UPLOAD_PROGRESS.json').write_text(json.dumps({
            'draft_id': args.draft_id, 'completed': completed, 'published': False}, indent=2) + '\n', encoding='utf-8')
        print(action, item['key'], flush=True)
    print('All planned files are verified in the draft. This command does not change metadata or publish the record.')


if __name__ == '__main__':
    main()
