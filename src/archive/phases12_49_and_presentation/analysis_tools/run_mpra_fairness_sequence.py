from pathlib import Path
import json,subprocess,sys,datetime
root=Path.cwd();out=root/'phase35_outputs';out.mkdir(exist_ok=True)
status=out/'sequence_status.json'
def write(x):
    p=status.with_suffix('.json.writing');p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');p.replace(status)
for phase in [35,36,37,38]:
    write({'status':'running','current_phase':phase,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    result=subprocess.run([sys.executable,'-B','-X','utf8',str(root/'analysis_tools/execute_followup.py'),str(phase)],cwd=root)
    if result.returncode:
        write({'status':'failed','phase':phase,'exit_code':result.returncode});raise SystemExit(result.returncode)
write({'status':'all_notebooks_complete_pending_visual_QA_and_final_archive','phases':[35,36,37,38]})
