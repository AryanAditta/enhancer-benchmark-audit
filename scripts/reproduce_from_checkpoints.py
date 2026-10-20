"""Regenerate scored predictions from authenticated fitted weights, without training."""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=REPO / 'data/training_workspace')
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=REPO / 'results/reproduced/checkpoint_inference')
    p.add_argument('--task', choices=['classification', 'activity', 'both'], default='both')
    p.add_argument('--dependency-path', type=Path, help='Optional explicitly supplied pinned dependency directory')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print('No inference started. Authenticate the inputs, then add --execute.')
        return
    workspace = a.workspace.resolve()
    artifact_root = a.artifact_root.resolve()
    output = a.output.resolve()
    if not workspace.is_relative_to(REPO) or not output.is_relative_to(REPO):
        raise ValueError('Workspace and generated output must be inside this repository.')
    if output.exists():
        raise ValueError('Use a new output directory: prior prediction caches are never read.')
    for r in json.loads((REPO / 'configs/training_artifacts.json').read_text(encoding='utf-8')):
        f = workspace / r['path']
        if not f.is_file() or sha(f) != r['sha256']:
            raise ValueError('Unauthenticated input: ' + r['path'])
    for f in (REPO / 'src/frozen/analysis_tools').glob('*.py'):
        if sha(workspace / 'analysis_tools' / f.name) != sha(f):
            raise ValueError('Frozen source mismatch: ' + f.name)
    output.mkdir(parents=True)
    runtime = output / '.runtime'
    for key in ['TEMP', 'TMP', 'TMPDIR', 'HF_HOME', 'HF_MODULES_CACHE', 'TRANSFORMERS_CACHE',
                'TORCH_HOME', 'CUDA_CACHE_PATH', 'TORCH_EXTENSIONS_DIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
        target = runtime / key.lower()
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if a.dependency_path:
        sys.path.insert(0, str(a.dependency_path.resolve()))
    sys.path.insert(0, str(workspace / 'analysis_tools'))
    sys.path.insert(0, str(REPO / 'src'))
    import numpy as np
    import pandas as pd
    rows = []
    started = time.monotonic()

    def progress():
        (output / 'progress.json').write_text(json.dumps({
            'status': 'running', 'prediction_sets_checked': len(rows),
            'elapsed_seconds': time.monotonic() - started, 'new_training_steps': 0}, indent=2) + '\n')

    def compare(predicted, expected, name, limit):
        if len(predicted) != len(expected) or not np.isfinite(predicted).all():
            raise ValueError('Invalid predictions: ' + name)
        delta = np.abs(np.asarray(predicted) - np.asarray(expected))
        maximum, mean = float(delta.max()), float(delta.mean())
        if maximum >= limit or mean >= 1e-6:
            raise ValueError(f'Prediction tolerance failed: {name}: max={maximum}, mean={mean}')
        return maximum, mean

    original_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        if a.task in ['classification', 'both']:
            s = importlib.import_module('phase11c_completion')
            torch = s.setup_runtime()
            frames = s.locked_data()
            for f in sorted((REPO / 'results/job_metadata/classification').glob('*.json')):
                m = json.loads(f.read_text(encoding='utf-8'))
                ckfile = artifact_root / m['checkpoint_file']
                if sha(ckfile) != m['checkpoint_sha256']:
                    raise ValueError('Checkpoint hash mismatch: ' + m['job'])
                ck = torch.load(ckfile, map_location='cpu', weights_only=True)
                for key in ['protocol_sha256', 'stage', 'base_seed', 'effective_seed', 'fold']:
                    if ck[key] != m[key]:
                        raise ValueError('Checkpoint identity mismatch: ' + key)
                truth = frames[m['stage']]
                truth = truth[truth.fold_5.eq(m['fold'])].reset_index(drop=True)
                if s.sha('\n'.join(truth.record_id).encode()) != ck['evaluation_record_ids_sha256']:
                    raise ValueError('Evaluation-record mismatch')
                tok, encoder, _ = s.load_backbone(m['model'])
                net = s.new_classifier(encoder).cuda()
                net.load_state_dict(ck['state_dict'], strict=True)
                predicted = np.asarray(s.predict(net, tok, m['model'], truth.sequence.tolist()))
                reference = pd.read_csv(REPO / 'results/predictions/classification_current_all_seeds.csv.gz', float_precision='round_trip')
                reference = reference[reference.stage.eq(m['stage']) & reference.model.eq('finetuned_' + m['model']) & reference.base_seed.eq(m['base_seed'])]
                reference = reference.set_index('record_id').loc[truth.record_id]
                maximum, mean = compare(predicted, reference.score.to_numpy(), m['job'], 2e-5)
                pd.DataFrame({'record_id': truth.record_id, 'score': predicted}).to_csv(output / (m['job'] + '.csv'), index=False, float_format='%.17g')
                rows.append({'task': 'classification', 'job': m['job'], 'n': len(truth),
                             'checkpoint_sha256': m['checkpoint_sha256'], 'max_difference': maximum,
                             'mean_difference': mean, 'passed': True})
                progress()
                del net, encoder, tok, ck
                gc.collect()
                torch.cuda.empty_cache()
        if a.task in ['activity', 'both']:
            s = importlib.import_module('mpra_fairness_training')
            analysis = importlib.import_module('mpra_fairness_analysis')
            torch = s.runtime()
            selection = json.loads((REPO / 'configs/activity_selection_lock.json').read_text(encoding='utf-8'))
            reference = pd.read_csv(REPO / 'results/predictions/activity_all_seed_evaluations.csv.gz', float_precision='round_trip')
            runs = []
            for choice in selection['recipes']:
                for seed in s.C['seeds']:
                    runs.append((choice['cell'], choice['model'], choice['grid_index'], seed, choice['model'] + '_selected', 'best'))
            for cell in s.C['cells']:
                for seed in s.C['seeds']:
                    runs.extend([(cell, 'dnabert2', 1, seed, 'dnabert2_fixedlr_full', 'best'),
                                 (cell, 'dnabert2', 1, seed, 'dnabert2_fixedlr_early2', 'early2')])
            memo = {}
            for cell, model, grid, seed, arm, role in runs:
                key = s.key(cell, model, grid, seed)
                rel = f'phase35_outputs/checkpoints/{key}__{role}.pt'
                expected_hash = selection['checkpoint_sha256'][rel]
                ckfile = artifact_root / rel
                if sha(ckfile) != expected_hash:
                    raise ValueError('Checkpoint hash mismatch: ' + key)
                ck = torch.load(ckfile, map_location='cpu', weights_only=True)
                net = tok = None
                for cohort, frame in [('internal', s.partitions(cell)[2]), ('DHS_natural', analysis.external(cell))]:
                    cachekey = (expected_hash, cohort)
                    if cachekey not in memo:
                        if net is None:
                            net, tok = s.build(model)
                            net.load_state_dict(ck['state_dict'], strict=True)
                        # This in-process memo contains only predictions computed during this fresh invocation.
                        memo[cachekey] = s.predict(net, tok, model, frame.sequence.tolist(), ck['target_mean'], ck['target_scale'])
                    predicted = memo[cachekey]
                    expected = reference[reference.cell.eq(cell) & reference.cohort.eq(cohort) & reference.arm.eq(arm) & reference.seed.eq(seed)].set_index('record_id').loc[frame.record_id]
                    maximum, mean = compare(predicted, expected.prediction.to_numpy(), key + cohort, 2e-5)
                    name = f'{cell}__{arm}__seed{seed}__{cohort}'
                    pd.DataFrame({'record_id': frame.record_id, 'prediction': predicted}).to_csv(output / (name + '.csv'), index=False, float_format='%.17g')
                    rows.append({'task': 'activity', 'job': key, 'arm': arm, 'cohort': cohort, 'n': len(frame),
                                 'checkpoint_sha256': expected_hash, 'max_difference': maximum, 'mean_difference': mean, 'passed': True})
                    progress()
                del net, tok, ck
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        os.chdir(original_cwd)
    pd.DataFrame(rows).to_csv(output / 'checkpoint_checks.csv', index=False, float_format='%.17g')
    report = {'passed': True, 'classification_fold_sets': sum(r['task'] == 'classification' for r in rows),
              'activity_evaluation_sets': sum(r['task'] == 'activity' for r in rows),
              'max_prediction_difference': max(r['max_difference'] for r in rows),
              'mean_difference_limit': 1e-6, 'maximum_difference_limit': 2e-5,
              'new_training_steps': 0, 'prior_prediction_caches_used_for_inference': False,
              'scope': 'Fresh checkpoint inference; saved predictions read only for comparison after inference.',
              'seconds': time.monotonic() - started, 'python': sys.version,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()}
    (output / 'VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
