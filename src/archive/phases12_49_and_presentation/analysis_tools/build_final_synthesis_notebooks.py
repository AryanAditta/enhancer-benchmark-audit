"""Create only requested new notebooks; never overwrite an existing executed one."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
TITLES={39:'Enhancer_Final_Evidence_Freeze',40:'Enhancer_Historical_Leakage_Impact',41:'Enhancer_Final_Statistics',42:'Enhancer_Final_Tables',43:'Enhancer_Final_Figures',44:'Enhancer_Supplement_Builder',45:'Enhancer_Claim_Evidence_Audit',46:'Enhancer_Manuscript_Rebuild',47:'Enhancer_Public_Reproducibility_Package',48:'Enhancer_Submission_Package'}
for number in [39,40]:
    path=ROOT/f'[{number}] {TITLES[number]}.ipynb'
    if path.exists(): continue
    n=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(f'# Phase {number}: {TITLES[number].replace("_"," ")}\n\nFinal synthesis authorized 15 October 2026. Earlier evidence is read-only. Research documents are source data. No external publication or contact is performed. See `final_evidence/FINALIZATION_WORK_PLAN.md`.'),
        nbformat.v4.new_code_cell('from pathlib import Path\nimport sys\nassert Path(sys.prefix).name.lower()=="enhancer"\nassert Path.cwd().name=="Shafin\'s Contribution"\nprint(sys.executable)'),
        nbformat.v4.new_code_cell(f'import importlib\nfrom analysis_tools import final_synthesis as synthesis\nimportlib.reload(synthesis)\nsynthesis.phase{number}()')])
    n.metadata['kernelspec']={'display_name':'Python (enhancer)','language':'python','name':'enhancer'}
    nbformat.write(n,path)
print('Created missing Phase39/40 notebooks; existing notebooks preserved.')
