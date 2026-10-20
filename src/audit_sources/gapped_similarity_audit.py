"""Resolve high-LCS pairs with a documented global, affine-gap alignment.

Only LCS >= 160 pairs require this step: if LCS is below 160, no gapped
alignment can have 160 exact base matches across 200 bp. The objective below
is one explicitly defined global alignment; it is not CD-HIT-EST output and
must not be represented as a coordinate or genome-homology audit.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


SCORE_MATCH, SCORE_MISMATCH = 2, -2
GAP_OPEN, GAP_EXTEND = -5, -1
NEG = (-10**9, -10**9, -10**9, -10**9)
RC = str.maketrans("ACGT", "TGCA")


def add(value, score, match=0, mismatch=0, gap=0):
    return (value[0] + score, value[1] + match,
            value[2] - mismatch, value[3] - gap)


def align(a, b):
    """Return (alignment score, matches, mismatches, gap bases)."""
    n = len(b)
    m_prev = [NEG] * (n + 1)
    d_prev = [NEG] * (n + 1)
    i_prev = [NEG] * (n + 1)
    m_prev[0] = (0, 0, 0, 0)
    for j in range(1, n + 1):
        i_prev[j] = add(m_prev[j - 1], GAP_OPEN, gap=1) if j == 1 else add(
            i_prev[j - 1], GAP_EXTEND, gap=1)
    for x in a:
        m = [NEG] * (n + 1)
        d = [NEG] * (n + 1)
        ins = [NEG] * (n + 1)
        d[0] = max(add(m_prev[0], GAP_OPEN, gap=1),
                   add(d_prev[0], GAP_EXTEND, gap=1))
        for j, y in enumerate(b, 1):
            equal = x == y
            m[j] = add(max(m_prev[j - 1], i_prev[j - 1], d_prev[j - 1]),
                       SCORE_MATCH if equal else SCORE_MISMATCH,
                       match=int(equal), mismatch=int(not equal))
            ins[j] = max(add(m[j - 1], GAP_OPEN, gap=1),
                         add(ins[j - 1], GAP_EXTEND, gap=1))
            d[j] = max(add(m_prev[j], GAP_OPEN, gap=1),
                       add(d_prev[j], GAP_EXTEND, gap=1))
        m_prev, i_prev, d_prev = m, ins, d
    score, matches, neg_mismatches, neg_gaps = max(
        m_prev[n], i_prev[n], d_prev[n])
    assert 2 * matches + 2 * (-neg_mismatches) + (-neg_gaps) == len(a) + len(b)
    return score, matches, -neg_mismatches, -neg_gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', type=Path, required=True)
    ap.add_argument('--candidates', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = {r['record_id']: r for r in csv.DictReader(args.records.open())}
    candidates = list(csv.DictReader(args.candidates.open(), delimiter='\t'))
    fieldnames = list(candidates[0]) + [
        'best_orientation', 'alignment_score', 'matches', 'mismatches',
        'gap_bases', 'identity_shorter', 'cross_train_test',
        'cross_fold5_train', 'cross_fold10_train', 'cross_stage1_label']
    counts = Counter()
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in candidates:
            if max(int(p['lcs_forward']), int(p['lcs_rc'])) < 160:
                continue
            a, b = rows[p['id_a']], rows[p['id_b']]
            forward = align(a['sequence'], b['sequence'])
            reverse = align(a['sequence'], b['sequence'].translate(RC)[::-1])
            best = max(forward, reverse)
            output = {**p, 'best_orientation': 'forward' if best == forward else 'reverse',
                      'alignment_score': best[0], 'matches': best[1],
                      'mismatches': best[2], 'gap_bases': best[3],
                      'identity_shorter': f'{best[1] / 200:.4f}',
                      'cross_train_test': a['split'] != b['split'],
                      'cross_fold5_train': a['split'] == b['split'] == 'train'
                        and a['fold_5'] != b['fold_5'],
                      'cross_fold10_train': a['split'] == b['split'] == 'train'
                        and a['fold_10'] != b['fold_10'],
                      'cross_stage1_label': a['label'] != b['label']}
            writer.writerow(output)
            if best[1] >= 160:
                counts['80pct_pairs'] += 1
                for flag in ('cross_train_test', 'cross_fold5_train',
                             'cross_fold10_train', 'cross_stage1_label'):
                    counts[flag] += bool(output[flag])
    print(dict(counts))


if __name__ == '__main__':
    main()
