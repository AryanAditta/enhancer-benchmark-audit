"""Run the authorized completion sequence; stop honestly on errors, never skip failed jobs."""
from pathlib import Path
import sys,subprocess,json,time
ROOT=Path(__file__).resolve().parents[1]
for number in [31,32,33]:
 status=ROOT/'phase31_outputs/sequence_status.json';status.parent.mkdir(exist_ok=True)
 status.write_text(json.dumps({'status':'running','current_phase':number,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n')
 result=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'analysis_tools/execute_followup.py'),str(number)],cwd=ROOT)
 if result.returncode:
  status.write_text(json.dumps({'status':'stopped_on_error','phase':number,'returncode':result.returncode,'action':'Inspect saved notebook/error; preserve completed job caches and optimizer/RNG resume state.'},indent=2)+'\n')
  raise SystemExit(result.returncode)
status.write_text(json.dumps({'status':'all_notebooks_complete_pending_visual_QA_and_final_archive','phases':[31,32,33]},indent=2)+'\n')
print('Training, independent verification and findings notebooks finished. VisualQA/archive finalization follows.',flush=True)
