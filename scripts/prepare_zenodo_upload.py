"""Prepare authenticated upload files without duplicating large fitted checkpoints."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', type=Path, required=True)
    p.add_argument('--legacy-bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, output = a.study_root.resolve(), a.output.resolve()
    if output.exists():
        raise ValueError('Use a new output directory.')
    output.mkdir(parents=True)
    tree = output / 'source_tree'
    tree.mkdir()
    required = json.loads((REPO / 'configs/training_artifacts.json').read_text(encoding='utf-8'))
    checkpoint_records = []
    for kind, phase in [('classification', 31), ('activity', 35)]:
        for meta in sorted((REPO / f'results/job_metadata/{kind}').glob('*.json')):
            m = json.loads(meta.read_text(encoding='utf-8'))
            artifacts = ([{'file': m['checkpoint_file'], 'sha256': m['checkpoint_sha256']}]
                         if kind == 'classification' else [r for r in m['artifacts'] if r['file'].endswith('.pt')])
            for r in artifacts:
                f = root / r['file']
                checkpoint_records.append({'path': r['file'], 'sha256': r['sha256'], 'bytes': f.stat().st_size,
                                           'task': kind, 'model': m['model'], 'job': m['job']})
    assert len(checkpoint_records) == 132
    cnn = [r for r in checkpoint_records if r['task'] == 'activity' and r['model'] != 'dnabert2']
    large = [r for r in checkpoint_records if r not in cnn]
    expected_space = sum(r['bytes'] for r in required) * 2 + 2 * 835_393_456 + sum(r['bytes'] for r in cnn) * 2 + 2_000_000_000
    if shutil.disk_usage(output).free < expected_space:
        raise RuntimeError('Insufficient free space for the bounded input/CNN staging; preserve existing files and free space elsewhere.')
    members = {}

    def add(source, relative, expected=None):
        source = Path(source)
        h = sha(source)
        if expected is not None and h != expected:
            raise ValueError('Artifact hash mismatch: ' + relative)
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != h:
            raise ValueError('Copy hash mismatch')
        members[relative] = {'path': relative, 'bytes': target.stat().st_size, 'sha256': h}

    for r in required:
        add(root / r['path'], r['path'], r['sha256'])
    prep = []

    def addprep(source, relative, expected=None):
        add(source, relative, expected)
        prep.append(members[relative])

    with zipfile.ZipFile(a.legacy_bundle) as outer:
        original = outer.read('DNABERT (Preserve).zip')
        audit = outer.read('Enhancer_Sequence_Audit_Checkpoint_02.zip')
    if hashlib.sha256(original).hexdigest() != '20f59d3b830f93ddae01813ab3a2a4a6a2fe41455062e62b512d3da1dd77ce0c':
        raise ValueError('Legacy source archive differs from the original notebook lock')
    if hashlib.sha256(audit).hexdigest() != 'f84392731ecad8b9a6549b927a4207cd731ebbde8525cbd3cd9e1d4c99969399':
        raise ValueError('Sequence-audit source archive differs from original notebook lock')
    legacy = output / 'legacy_fasta_sources.zip'
    names = {'enhancer.cv.txt', 'enhancer.ind.txt', 'non.cv.txt', 'non.ind.txt',
             'strong_742.txt', 'strong_100.txt', 'weak_742.txt', 'weak_100.txt'}
    with zipfile.ZipFile(io.BytesIO(original)) as z, zipfile.ZipFile(legacy, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        found = {Path(n).name: n for n in z.namelist() if Path(n).name in names}
        if set(found) != names:
            raise ValueError('Legacy raw input file set mismatch')
        for name in sorted(names):
            out.writestr(name, z.read(found[name]))
    addprep(legacy, 'preparation/legacy_fasta_sources.zip')
    auditfile = output / 'sequence_audit_checkpoint.zip'
    auditfile.write_bytes(audit)
    addprep(auditfile, 'preparation/sequence_audit_checkpoint.zip')
    source_index = []
    with zipfile.ZipFile(auditfile) as z:
        for name in z.namelist():
            if name.startswith('scripts/') and name.endswith(('.cpp', '.py')):
                target = REPO / 'src/audit_sources' / Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
                source_index.append({'archive_sha256': sha(auditfile), 'member': name,
                                     'repository_path': target.relative_to(REPO).as_posix(), 'sha256': sha(target)})
    provenance = json.loads((root / 'phase34_inputs/phase9a_audit_and_provenance.json').read_text(encoding='utf-8'))
    for name, r in provenance['sources'].items():
        addprep(root / 'reference_inputs/raw_sources' / name, 'preparation/raw_sources/' + name, r['sha256'])
    for phase in [4, 7]:
        name = f'{phase}_enhancer_phase{phase}_' + ('external_dataset.zip' if phase == 4 else 'chromosome_results.zip')
        addprep(root / 'reference_inputs/readiness_sources' / name, 'preparation/' + name)
    coda_name = '41586_2024_8070_MOESM14_ESM.txt'
    coda = output / coda_name
    urls = [
        'https://pmc.ncbi.nlm.nih.gov/articles/instance/11525185/bin/' + coda_name,
        'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08070-z/MediaObjects/' + coda_name]
    expected = json.loads((root / 'phase34_inputs/phase10_provenance.json').read_text(encoding='utf-8'))['config']['source_sha256']
    errors = []
    for url in urls:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'enhancer-benchmark-audit'}), timeout=180) as response, coda.open('wb') as f:
                shutil.copyfileobj(response, f)
            if sha(coda) != expected:
                raise ValueError('Publisher Table 12 hash mismatch')
            break
        except Exception as e:
            errors.append(type(e).__name__ + ': ' + str(e)[:200])
    else:
        raise RuntimeError('Cannot retrieve exact publisher source: ' + '; '.join(errors))
    addprep(coda, 'preparation/raw_sources/' + coda_name, expected)
    # Small authenticated records needed to audit/reuse completed fits accompany the inputs.
    for phase in [31, 35]:
        for folder in ['jobs', 'training_logs', 'predictions' if phase == 31 else 'validation']:
            for f in sorted((root / f'phase{phase}_outputs' / folder).glob('*')):
                if f.is_file():
                    add(f, f.relative_to(root).as_posix())
        for name in ['summary.json', 'run_lock.json']:
            f = root / f'phase{phase}_outputs' / name
            add(f, f.relative_to(root).as_posix())
    add(root / 'phase36_outputs/selection_lock.json', 'phase36_outputs/selection_lock.json')
    (REPO / 'configs/preparation_artifacts.json').write_text(json.dumps(prep, indent=2) + '\n', encoding='utf-8')
    (REPO / 'configs/checkpoint_artifacts.json').write_text(json.dumps(checkpoint_records, indent=2) + '\n', encoding='utf-8')
    (REPO / 'docs/PREPARATION_SOURCE_INDEX.json').write_text(json.dumps(source_index, indent=2) + '\n', encoding='utf-8')
    uploads = []

    def register(file, key, category, targets):
        uploads.append({'key': key, 'local_path': str(file), 'bytes': Path(file).stat().st_size,
                        'sha256': sha(file), 'category': category, 'members': targets})

    inputzip = output / 'study_inputs_and_records.zip'
    with zipfile.ZipFile(inputzip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for rel in sorted(members):
            z.write(tree / rel, rel)
    with zipfile.ZipFile(inputzip) as z:
        if z.testzip() is not None:
            raise ValueError('Input archive CRC failure')
    register(inputzip, inputzip.name, 'inputs', list(members.values()))
    cnnzip = output / 'activity_cnn_checkpoints.zip'
    with zipfile.ZipFile(cnnzip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for r in cnn:
            f = root / r['path']
            if sha(f) != r['sha256']:
                raise ValueError('CNN checkpoint authentication failed')
            z.write(f, r['path'])
    with zipfile.ZipFile(cnnzip) as z:
        if z.testzip() is not None:
            raise ValueError('CNN archive CRC failure')
    register(cnnzip, cnnzip.name, 'activity', cnn)
    for i, r in enumerate(large):
        f = root / r['path']
        h = sha(f)
        if h != r['sha256']:
            raise ValueError('Fitted checkpoint hash mismatch: ' + r['path'])
        key = r['task'] + '__' + Path(r['path']).name
        uploads.append({'key': key, 'local_path': str(f), 'bytes': r['bytes'], 'sha256': h,
                        'category': r['task'], 'members': [r]})
        if (i + 1) % 10 == 0:
            print('Authenticated large checkpoints', i + 1, '/', len(large), flush=True)
    metadata = output / 'artifact_metadata.zip'
    public_files = [{k: v for k, v in r.items() if k != 'local_path'} for r in uploads]
    with zipfile.ZipFile(metadata, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('ARTIFACT_MANIFEST.json', json.dumps({'files': public_files, 'checkpoints': checkpoint_records}, indent=2))
        z.write(REPO / 'LICENSE', 'LICENSE')
        z.write(REPO / 'THIRD_PARTY_NOTICES.md', 'THIRD_PARTY_NOTICES.md')
        z.writestr('README.txt', 'Research inputs and fitted checkpoints for enhancer-benchmark-audit. Original code retains the stated rights notice; third-party data, code and pretrained models retain their source terms. Dataset source identities and frozen transformations are recorded in the manuscript and repository. No new license is granted for third-party content.\n')
    register(metadata, metadata.name, 'metadata', [])
    if len(uploads) > 100 or sum(r['bytes'] for r in uploads) > 50_000_000_000:
        raise ValueError('Prepared deposit exceeds the default Zenodo record quota')
    public = {'schema_version': 1, 'status': 'prepared_local_pending_zenodo', 'record_id': None, 'doi': None,
              'files': [{k: v for k, v in r.items() if k != 'local_path'} for r in uploads]}
    (REPO / 'configs/zenodo_artifacts.json').write_text(json.dumps(public, indent=2) + '\n', encoding='utf-8')
    plan = {**public, 'files': uploads, 'upload_files': len(uploads), 'total_bytes': sum(r['bytes'] for r in uploads),
            'large_checkpoint_copies_created': 0, 'original_sources_modified': False,
            'original_checkpoint_files_to_select_directly': len(large), 'no_remote_deposit_created': True}
    (output / 'UPLOAD_PLAN.local.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: plan[k] for k in ['upload_files', 'total_bytes', 'large_checkpoint_copies_created', 'original_checkpoint_files_to_select_directly']}, indent=2))


if __name__ == '__main__':
    main()
