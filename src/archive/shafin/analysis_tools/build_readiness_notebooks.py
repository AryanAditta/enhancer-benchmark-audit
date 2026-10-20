from pathlib import Path
import sys,nbformat
ROOT=Path(__file__).resolve().parents[1]
NAMES={20:'Scope_and_Claim_Freeze',21:'Historical_Classification_Audit',22:'Checkpoint_and_Prediction_Reproduction',23:'Statistical_Reporting_Freeze',24:'Dataset_Filtering_and_Threshold_Reproducibility',25:'Publication_Tables',26:'Publication_Figures_and_Diagrams',27:'Literature_and_Contribution_Assessment',28:'Portable_Reproducibility_Package',29:'Final_PreWriting_Readiness_Review'}
DESCRIPTIONS={20:'Freeze bounded scope; explicitly exclude unsupported historical/long-context claims.',21:'Authenticate record-level classification evidence, recalculate confusion matrices and every metric, and disposition all submitted tables.',22:'Authenticate recovered original checkpoints and reproduce test/external predictions; complete DNABERT-2 removed-population inference.',23:'Assemble one statistical ledger with seeds, ensembles, paired effects, intervals and resampling definitions.',24:'Correct threshold serialization without changing previous artifacts; regenerate authoritative population/subgroup records and counts.',25:'Create traceable main/supplementary publication tables from frozen evidence.',26:'Generate and visually check all retained-study figures and architecture/workflow diagrams.',27:'Integrate focused primary-literature review, exact citations and bounded contribution assessment.',28:'Build a relocatable local-input reproduction package, environment specification, manifests and figure-generation code.',29:'Run the final claim/table/figure/reproducibility readiness gate and update contribution records.'}
for number in [int(x) for x in sys.argv[1:]] or list(NAMES):
 p=ROOT/f'[{number}] Enhancer_{NAMES[number]}.ipynb'
 assert not p.exists(),f'Preserve existing notebook {p.name}'
 cells=[nbformat.v4.new_markdown_cell(f'''# Phase {number} — {NAMES[number].replace('_',' ')}
**Owner: Shafin. Date: 2026-10-13.** {DESCRIPTIONS[number]}

This phase implements priority {number-19} in Aditta's pre-writing checklist. Run all cells in the existing **enhancer** Anaconda environment from this contribution workspace. Every input, cache and output is local to this folder. Previous experiments and original documents are read only. No new neural training or tuning is launched. Missing evidence is reported explicitly.

The portable analysis implementation is included as `analysis_tools/readiness_pipeline.py`. Source hashes, table records and assertions make this notebook reviewable and reproducible. Earlier drafts are not a completed revised paper.'''),
 nbformat.v4.new_code_cell('''from pathlib import Path
import sys,importlib,json
root=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'phase12_outputs').is_dir() and (p/'reference_inputs').is_dir()),None)
assert root is not None, 'Run from the contribution workspace.'
assert Path(sys.prefix).name.lower()=='enhancer', 'Select the enhancer Anaconda environment.'
sys.path.insert(0,str(root/'analysis_tools'))
import readiness_pipeline as readiness
print('Environment:',sys.executable)
print('Workspace:',root)'''),
 nbformat.v4.new_markdown_cell('## Execute the declared checks\nResults are saved only to this phase output folder. Assertions stop the phase if evidence is inconsistent.'),
 nbformat.v4.new_code_cell(f'readiness.phase{number}()'),
 nbformat.v4.new_markdown_cell('## Completion status and saved evidence\nA completed technical phase does not imply journal acceptance. Scientific limitations and excluded claims remain explicit.'),
 nbformat.v4.new_code_cell(f'''summary=json.loads((root/'phase{number}_outputs/summary.json').read_text(encoding='utf8'))
print(json.dumps(summary,indent=2))
assert summary['status'].startswith('completed'), 'Review pending evidence before continuing.'
print('Saved files:')
for path in sorted((root/'phase{number}_outputs').iterdir()):
 if path.is_file():print(path.name)''')]
 n=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (enhancer)','language':'python','name':'enhancer'},'language_info':{'name':'python','version':'3.12'},'contribution':{'owner':'Shafin','phase':number,'priority':number-19,'scope':'Pre-writing readiness'}})
 for i,c in enumerate(n.cells):
  if c.cell_type=='code':compile(c.source,f'{p.name}:cell{i}','exec')
 nbformat.validate(n);nbformat.write(n,p);print('Created',p.name)
