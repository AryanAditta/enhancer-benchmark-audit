"""Portable path adapters around the preserved original data-preparation cells."""
from pathlib import Path
import ast
import bisect
import gc
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import types
import urllib.request
import zipfile
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def notebook(phase):
    inventory = json.loads((REPO / 'docs/ORIGINAL_SOURCE_INVENTORY.json').read_text(encoding='utf-8'))
    suffix = 'phase' + str(phase)
    for r in inventory['records']:
        p = Path(r['source'])
        if r['kind'] == 'notebook' and len(p.parts) == 1 and p.name.startswith('[' + str(phase) + ']'):
            f = REPO / r['archive_path']
            if digest(f) != r['sha256']:
                raise ValueError('Original preparation notebook hash mismatch')
            return json.loads(f.read_text(encoding='utf-8-sig'))
    raise ValueError('Missing original notebook ' + suffix)


def shafin_notebook(phase):
    inventory = json.loads((REPO / 'docs/ORIGINAL_SOURCE_INVENTORY.json').read_text(encoding='utf-8'))
    for r in inventory['records']:
        p = Path(r['source'])
        if r['kind'] == 'notebook' and len(p.parts) == 2 and p.name.startswith('[' + str(phase) + ']'):
            f = REPO / r['archive_path']
            if digest(f) != r['sha256']:
                raise ValueError('Original notebook hash mismatch')
            return json.loads(f.read_text(encoding='utf-8-sig'))
    raise ValueError('Missing original notebook')


def execute(nb, cell, namespace, replacements=None):
    tree = ast.parse(''.join(nb['cells'][cell]['source']))

    class PathsOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for old, new in (replacements or {}).items():
                    if node.value == old or node.value.startswith(old + '/'):
                        return ast.copy_location(ast.Constant(new + node.value[len(old):]), node)
            return node

    tree = ast.fix_missing_locations(PathsOnly().visit(tree))
    exec(compile(tree, f'preserved_notebook_cell_{cell}', 'exec'), namespace)


def portable_genome_module():
    """Reuse the study's existing read-only 2bit reader on Windows."""
    nb = shafin_notebook(14)
    found = None
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.ClassDef) and node.name == 'TwoBit':
                    found = node
    if found is None:
        raise ValueError('Missing authenticated TwoBit reader')
    namespace = {'Path': Path, 'struct': struct, 'np': np, 'bisect': bisect}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[found], type_ignores=[])), 'preserved_TwoBit', 'exec'), namespace)
    cls = namespace['TwoBit']

    def chroms(self):
        result = {}
        for name, offset in self.offsets.items():
            self.f.seek(offset)
            result[name] = self.uints(1)[0]
        return result

    cls.chroms = chroms
    module = types.ModuleType('py2bit')
    module.open = cls
    return module


def compare_frame(got, expected):
    left = pd.read_csv(got, float_precision='round_trip')
    right = pd.read_csv(expected, float_precision='round_trip')
    pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=0, atol=1e-12)
    return {'rows': len(left), 'columns': len(left.columns), 'content_matches': True,
            'byte_identical': digest(got) == digest(expected), 'sha256': digest(got),
            'reference_sha256': digest(expected)}


def rebuild(artifact_root, output, tasks):
    artifact_root = Path(artifact_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; preserved sources are never overwritten.')
    output.mkdir(parents=True)
    checks = {}
    started = time.monotonic()
    if 'legacy' in tasks:
        import sklearn
        if sklearn.__version__ != '1.6.1':
            raise ValueError('Legacy fold reconstruction requires scikit-learn==1.6.1, recorded in the original audit. Use the separate preparation environment; later versions change shuffled group folds.')
    # Inputs are authenticated before any original notebook code is executed.
    extra = json.loads((REPO / 'configs/preparation_artifacts.json').read_text(encoding='utf-8'))
    locked = json.loads((REPO / 'configs/training_artifacts.json').read_text(encoding='utf-8'))
    for r in extra + locked:
        f = artifact_root / r['path']
        if not f.is_file() or digest(f) != r['sha256']:
            raise ValueError('Unauthenticated preparation input: ' + r['path'])
    if 'legacy' in tasks:
        nb = notebook(1)
        header = ast.parse(''.join(nb['cells'][1]['source']))
        body = [n for n in header.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'preserved_audit_helpers', 'exec'), ns)
        work = output / 'legacy'
        work.mkdir()
        ns.update(OUT=work, inputs={
            'DNABERT (Preserve).zip': artifact_root / 'preparation/legacy_fasta_sources.zip',
            'Enhancer_Sequence_Audit_Checkpoint_02.zip': artifact_root / 'preparation/sequence_audit_checkpoint.zip'})
        # The public FASTA-only repack deliberately omits unrelated private Word files.
        # Record its real container hash in the newly generated provenance report.
        ns['EXPECTED'] = {name: digest(path) for name, path in ns['inputs'].items()}
        ns['EXPECTED_MEMBERS'] = {}
        for name, path in ns['inputs'].items():
            with zipfile.ZipFile(path) as archive:
                ns['EXPECTED_MEMBERS'][name] = {
                    member: hashlib.sha256(archive.read(member)).hexdigest()
                    for member in archive.namelist() if not member.endswith('/')}
        # Authenticate each raw file separately; unrelated old Word documents are not needed.
        for cell in [3, 4, 6, 8, 10, 12]:
            execute(nb, cell, ns)
        reference_dir = work / 'reference_comparison'
        reference_dir.mkdir()
        original_manifest_zip = artifact_root / 'phase30_inputs/historical_sources/1_ enhancer_audit_v1_outputs.zip'
        for stage in [1, 2]:
            name = f'stage{stage}_candidate_manifest.csv'
            with zipfile.ZipFile(original_manifest_zip) as archive:
                matches = [n for n in archive.namelist() if Path(n).name == name]
                if len(matches) != 1:
                    raise ValueError('Original candidate-manifest archive is ambiguous')
                expected = reference_dir / name
                expected.write_bytes(archive.read(matches[0]))
            checks['legacy_stage' + str(stage)] = compare_frame(
                work / name, expected)
        checks['fresh_exhaustive_LCS'] = {'passed': True, 'pairs_ge150': len(ns['pairs']),
                                        'pairs_ge160': len(ns['high']), 'gapped_candidates': len(ns['alignment'])}
    if 'activity' in tasks:
        nb = notebook(9)
        base = output / 'activity_work'
        base.mkdir()
        replacements = {'/kaggle/working': str(base), '/kaggle/input': str(artifact_root)}
        old_module = sys.modules.get('py2bit')
        sys.modules['py2bit'] = portable_genome_module()
        ns = {}
        try:
            for cell in range(1, 9):
                execute(nb, cell, ns, replacements)
            work = ns['WORK']
            for file in ['phase9_full_activity_all_cells.csv', 'phase9_frozen_splits.csv']:
                checks[file] = compare_frame(work / file, artifact_root / 'phase34_inputs' / file)
        finally:
            if 'genome' in ns:
                ns['genome'].close()
            if old_module is None:
                sys.modules.pop('py2bit', None)
            else:
                sys.modules['py2bit'] = old_module
    if 'external' in tasks:
        nb = notebook(10)
        work = output / 'external'
        work.mkdir()
        config = json.loads((artifact_root / 'phase34_inputs/phase10_provenance.json').read_text(encoding='utf-8'))['config']
        ns = {'Path': Path, 'np': np, 'pd': pd, 'json': json, 'zipfile': zipfile,
              'shutil': shutil, 'gc': gc, 'os': os, 'urllib': __import__('urllib'),
              'WORK': work, 'ROOT': artifact_root, 'ZIPS': [], 'CONFIG': config,
              'EXTERNAL_HASH': config['source_sha256'], 'SOURCE_URLS': [],
              'fsha': digest, 'sha': lambda b: hashlib.sha256(b).hexdigest(),
              'phase9': pd.read_csv(artifact_root / 'phase34_inputs/phase9_full_activity_all_cells.csv', float_precision='round_trip')}
        execute(nb, 2, ns)
        checks['external_frozen_cohort'] = compare_frame(
            ns['frozen_external'], artifact_root / 'phase34_inputs/phase10_external_frozen_set.csv')
    report = {'passed': True, 'tasks': tasks, 'checks': checks,
              'seconds': time.monotonic() - started, 'new_training_steps': 0,
              'adapter': 'Only filesystem paths and the previously validated read-only TwoBit backend are adapted; original source cells and decisions remain preserved.',
              'private_legacy_documents_included': False}
    (output / 'VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report
