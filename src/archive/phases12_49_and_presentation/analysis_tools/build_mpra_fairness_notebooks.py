from pathlib import Path
import json,nbformat
root=Path.cwd();c=json.loads((root/'phase34_outputs/training_protocol.json').read_text())
common="""from pathlib import Path
import sys
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'phase34_inputs').exists())
assert Path(sys.prefix).name.lower()=='enhancer', 'Use the existing enhancer Anaconda environment.'
sys.path.insert(0,str(ROOT/'analysis_tools'))
"""
def save(number,name,intro,cells):
    p=root/f'[{number}] {name}.ipynb'
    assert not p.exists(),f'Existing notebook requires review before any overwrite: {p}'
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(f'# Phase {number}: {name.replace("Enhancer_", "").replace("_"," ")}\n\n'+intro),nbformat.v4.new_code_cell(common)]+cells,metadata={'kernelspec':{'name':'enhancer','display_name':'Python (enhancer)','language':'python'},'language_info':{'name':'python','version':'3.12.13'}})
    nbformat.validate(n)
    for cell in n.cells:
        if cell.cell_type=='code':compile(cell.source,p.name,'exec')
    nbformat.write(n,p)
save(34,'Enhancer_MPRA_Training_Fairness_Protocol',
    'Aditta priority1 is now investigated experimentally. This notebook authenticates local copies, audits the original two-epoch DNABERT-2 recipe, and freezes a validation-only follow-up. Earlier outputs stay preserved. Read phase34_outputs/STUDY_PLAN.md and training_protocol.json for the full prospective choices and limits.',[
    nbformat.v4.new_code_cell('from prepare_mpra_fairness import prepare\nprepare()'),
    nbformat.v4.new_code_cell("import json,pandas as pd\ndisplay(pd.read_csv(ROOT/'phase34_outputs/partition_counts.csv'))\ndisplay(pd.read_csv(ROOT/'phase34_outputs/historical_dnabert2_validation_choices.csv'))\nprint((ROOT/'phase34_outputs/STUDY_PLAN.md').read_text())")])
cells=[nbformat.v4.new_markdown_cell('All54fits use training and validation only. Three fixed learning rates × three seeds × three models × two cells. The two-epoch DNABERT-2 checkpoint is saved from the same training trajectory. Stop/epoch choices use validation Pearson; no new test inference runs in this notebook. Optimizer, scaler and RNG resumption preserve unfinished jobs.'),nbformat.v4.new_code_cell('import mpra_fairness_training as study\nstudy.setup()')]
for cell in c['cells']:
    for model in c['models']:
        cells.append(nbformat.v4.new_markdown_cell(f'## {cell}: {model}\n\nAll three candidate learning rates and every declared seed are retained, including unsuccessful or collapsed fits.'))
        for grid in range(3):
            for seed in c['seeds']:cells.append(nbformat.v4.new_code_cell(f'study.train_job({cell!r},{model!r},{grid},{seed})'))
cells.append(nbformat.v4.new_code_cell('study.finish_training()'))
save(35,'Enhancer_MPRA_Training_Adequacy_and_Search','New full training in enhancer. This limited search equalizes the number of learning-rate candidates and seeds; it does not equalize compute or prove an architecture-wide optimum. DNABERT-2 maximum30epochs/patience5/minimum8; original specialist CNN budgets retained.',cells)
save(36,'Enhancer_MPRA_Validation_Selection_and_Evaluation','All training must be complete before this notebook locks recipes and checkpoint hashes. It then evaluates the previously examined9B test cohort and fixed DHS-natural external cohort. No test outcomes affect any training decision.',[nbformat.v4.new_code_cell('import mpra_fairness_analysis as analysis\nanalysis.selection_and_inference()')])
save(37,'Enhancer_MPRA_Fairness_Independent_Verification','Independently authenticate all54jobs and72best/prefix checkpoints, reproduce saved validation predictions, re-infer the chosen test/external predictions, recalculate metrics and ensembles, and report conditional paired intervals. Three seeds and previously examined test cohorts limit interpretation.',[nbformat.v4.new_code_cell('import mpra_fairness_analysis as analysis\nanalysis.independent_verification()')])
save(38,'Enhancer_MPRA_Fairness_Findings_and_Reviewer_Update','Prepare readable findings, figures, training cost and cap-limit diagnostics, and update Aditta/reviewer evidence. Actual conclusions follow the results, including findings in favour of either model. Manual visual inspection and final archive integrity checks precede a completed contribution-log entry.',[nbformat.v4.new_code_cell('import mpra_fairness_analysis as analysis\nanalysis.report()')])
print('Created and syntax-validated notebooks34–38.')
