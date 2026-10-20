"""Rebuild the legacy audit and activity/external datasets from authenticated source inputs."""
from pathlib import Path
import argparse
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from enhancer_benchmark_audit.raw_preparation import rebuild

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact-root', type=Path, required=True)
parser.add_argument('--output', type=Path, default=REPO / 'results/reproduced/raw_preparation')
parser.add_argument('--tasks', default='legacy,activity,external')
args = parser.parse_args()
if not args.output.resolve().is_relative_to(REPO):
    raise ValueError('Generated output must stay inside the repository.')
tasks = args.tasks.split(',')
if not set(tasks).issubset({'legacy', 'activity', 'external'}):
    raise ValueError('Unknown preparation task')
rebuild(args.artifact_root, args.output, tasks)
