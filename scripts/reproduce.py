"""Run a public reporting notebook stage from the command line."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from enhancer_benchmark_audit.reporting import COMMANDS
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage',choices=list(COMMANDS))
parser.add_argument('--full-statistics',action='store_true',help='Repeat all frozen bootstrap and permutation draws for the 14 headline comparisons.')
args=parser.parse_args()
if args.stage=='statistics':COMMANDS[args.stage](full=args.full_statistics)
else:COMMANDS[args.stage]()
