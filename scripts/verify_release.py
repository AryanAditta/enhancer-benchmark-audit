"""Authenticate tracked package inputs and exact frozen implementation hashes."""
from pathlib import Path
import sys,json
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from enhancer_benchmark_audit import io as s
s.verify_inputs()
lock=json.loads((REPO/'configs/activity_run_lock.json').read_text(encoding='utf-8'))
assert s.sha(REPO/'src/frozen/analysis_tools/mpra_fairness_training.py')==lock['source_sha256']
assert s.sha(REPO/'src/frozen/architectures.py')==lock['architecture_sha256']
assert s.sha(REPO/'configs/activity_protocol.json')==lock['protocol_sha256']
assert s.sha(REPO/'configs/activity_input_manifest.csv')==lock['input_manifest_sha256']
classification=json.loads((REPO/'configs/classification_run_lock.json').read_text(encoding='utf-8'))
assert s.sha(REPO/'src/frozen/analysis_tools/phase11c_completion.py')==classification['source_sha256']
print('Release manifest and frozen MPRA source, architecture, protocol and input-manifest hashes authenticated.')
