"""Convert a rebuilt record manifest to the small TSV input for LCS audit."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    with args.manifest.open() as f, args.output.open('w') as out:
        for row in csv.DictReader(f):
            assert len(row['sequence']) == 200
            out.write('\t'.join((row['record_id'], row['split'],
                                 row['fold_5'] or 'NA', row['fold_10'] or 'NA',
                                 row['subtype'], row['sequence'])) + '\n')


if __name__ == '__main__':
    main()
