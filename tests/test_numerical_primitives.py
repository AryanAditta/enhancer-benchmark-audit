from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from enhancer_benchmark_audit.statistics import stats_selfcheck
checks=stats_selfcheck()
assert len(checks)>0 and checks.absolute_error.max()<1e-12
print('Weighted metrics agree with literal repeated-row metrics, including tied ranks and zero-weight rows.')
