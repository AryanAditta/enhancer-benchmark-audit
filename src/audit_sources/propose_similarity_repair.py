"""Create a conservative, explicitly provisional grouped split proposal.

The input was already corrected for the 100 exact strong train/test copies.
This tool makes a proposal for inspection, not a frozen benchmark. Its
Group every pair whose exact LCS is at least 160, including those for which
one penalized global alignment scores fewer than 160 matches. Any alignment
with 160 matching bases must have an LCS of at least 160. This deliberately
overgroups uncertain pairs; it does not establish genomic coordinate hygiene.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


FIELDS = ('record_id', 'stage', 'original_split', 'candidate_role',
          'similarity_group_id', 'candidate_fold_5', 'candidate_fold_10',
          'label', 'subtype', 'source_file', 'header', 'sequence_sha256')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1', required=True, type=Path)
    p.add_argument('--stage2', required=True, type=Path)
    p.add_argument('--lcs-candidates', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_rows = {stage: list(csv.DictReader(path.open())) for stage, path in
                  ((1, args.stage1), (2, args.stage2))}
    stage1_by_id = {r['record_id']: r for r in input_rows[1]}
    parent = {r['record_id']: r['record_id'] for r in input_rows[1]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = list(csv.DictReader(args.lcs_candidates.open(), delimiter='\t'))
    for pair in pairs:
        if max(int(pair['lcs_forward']), int(pair['lcs_rc'])) >= 160:
            left, right = pair['id_a'], pair['id_b']
            parent[find(left)] = find(right)
    members = defaultdict(list)
    for record_id in parent:
        members[find(record_id)].append(record_id)
    groups = {member: min(component) for component in members.values()
              for member in component}
    test_groups = {groups[r['record_id']] for r in input_rows[1]
                   if r['split'] == 'test'}

    summary = {'similarity_rule': 'conservative LCS >= 160 of 200, both '
               'orientations; necessary condition for 160 aligned matches; proposal only',
               'candidate_pairs': sum(max(int(p['lcs_forward']),
                                      int(p['lcs_rc'])) >= 160 for p in pairs),
               'input_sha256': {str(stage): hashlib.sha256(path.read_bytes()).hexdigest()
                                for stage, path in ((1, args.stage1), (2, args.stage2))},
               'pair_file_sha256': hashlib.sha256(args.lcs_candidates.read_bytes()).hexdigest(),
               'stages': {}}
    for stage, source in input_rows.items():
        rows = []
        for r in source:
            g = groups[r['record_id']]
            role = ('legacy_test' if r['split'] == 'test' else
                    'excluded_related_to_legacy_test' if g in test_groups else
                    'train_candidate')
            rows.append({'record_id': r['record_id'], 'stage': stage,
                         'original_split': r['split'], 'candidate_role': role,
                         'similarity_group_id': g, 'candidate_fold_5': '',
                         'candidate_fold_10': '', 'label': r['label'],
                         'subtype': r['subtype'], 'source_file': r['source_file'],
                         'header': r['header'], 'sequence_sha256': r['sha256']})
        train_indices = [i for i, r in enumerate(rows)
                         if r['candidate_role'] == 'train_candidate']
        train = [rows[i] for i in train_indices]
        for k in (5, 10):
            cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
            for fold, (_, validation) in enumerate(cv.split(
                    range(len(train)), [r['label'] for r in train],
                    [r['similarity_group_id'] for r in train])):
                for i in validation:
                    rows[train_indices[i]][f'candidate_fold_{k}'] = str(fold)
            assert all(r[f'candidate_fold_{k}'] != '' for r in train)
        for key in ('candidate_role', 'candidate_fold_5', 'candidate_fold_10'):
            for g in set(r['similarity_group_id'] for r in rows):
                grouped = [r for r in rows if r['similarity_group_id'] == g]
                relevant = grouped if key == 'candidate_role' else [r for r in grouped
                    if r['candidate_role'] == 'train_candidate']
                values = {r[key] for r in relevant}
                if key == 'candidate_role':
                    assert len(values) <= 1 or values == {
                        'legacy_test', 'excluded_related_to_legacy_test'}
                else:
                    assert len(values) <= 1
        path = args.output / f'stage{stage}_similarity_proposal.csv'
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        excluded = [r for r in rows if r['candidate_role'] ==
                    'excluded_related_to_legacy_test']
        summary['stages'][str(stage)] = {
            'candidate_roles': dict(Counter(r['candidate_role'] for r in rows)),
            'excluded_subtypes': dict(Counter(r['subtype'] for r in excluded)),
            'excluded_record_ids': [r['record_id'] for r in excluded],
            'train_fold_5_sizes': dict(Counter(r['candidate_fold_5'] for r in train)),
            'train_fold_10_sizes': dict(Counter(r['candidate_fold_10'] for r in train)),
            'candidate_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / 'similarity_proposal_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
