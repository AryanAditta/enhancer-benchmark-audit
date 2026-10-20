"""Local portable-package and final-gate helpers. No research originals are modified."""
from pathlib import Path
import os,sys,json,re,ast,shutil,zipfile,time,subprocess,platform,importlib.metadata as metadata
import numpy as np
import pandas as pd
import readiness_pipeline as r

ROOT=r.ROOT
PACKAGE_NAME='Enhancer_PreWriting_Reproducibility'

def portable_architectures():
 import nbformat
 wanted={'initialize_weights','SELayer','EffBlock','LocalBlock','ResidualConcat','MapperBlock','LegNet','MPRAnn','CNN','Regressor'}
 nodes={};origins=[]
 files=[ROOT/'[16] Enhancer_Filtered_Out_Performance.ipynb',r.SRC/'[9B] Enhancer_Kaggle_Phase9B_Chromosome_Regression.ipynb']
 for p in files:
  for c in nbformat.read(p,as_version=4).cells:
   if c.cell_type!='code':continue
   for node in ast.parse(c.source).body:
    if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in wanted:
     nodes[node.name]=node;origins.append({'name':node.name,'source':p.relative_to(ROOT).as_posix(),'source_sha256':r.fsha(p)})
 assert set(nodes)==wanted
 code='"""Architecture-only extraction; no training or notebook side effects. Source/license index accompanies this file."""\nimport math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n'
 code+='\n\n'.join(ast.unparse(x) for x in nodes.values())+'\n'
 compile(code,'regression_architectures.py','exec')
 (ROOT/'analysis_tools/regression_architectures.py').write_text(code,encoding='utf8')
 r.dump(28,'architecture_source_index.json',origins)

def input_files():
 files=[]
 # Historical notebooks are preserved references; only 20–29 are portable execution entry points.
 files+=list(ROOT.glob('[[]*.ipynb'))
 files += [ROOT/'For Shafin.docx',ROOT/'CONTRIBUTION_LOG.md']
 for folder in ['phase12_inputs','reference_inputs','research_notes','reports']+[f'phase{n}_outputs' for n in range(13,28)]:
  for p in (ROOT/folder).rglob('*'):
   if not p.is_file():continue
   rel=p.relative_to(ROOT).as_posix()
   if any(x in p.parts for x in ['__pycache__','.ipynb_checkpoints','hf_cache','hf_modules','temp']):continue
   if rel.startswith('reference_inputs/readiness_checkpoints/'):continue # Included in authenticated original full-weight archives.
   if p.name.endswith(('.part','.writing','.tmp')):continue
   if 'matplotlib_cache' in p.parts:continue
   files.append(p)
 for p in (ROOT/'phase12_outputs/enhancer_phase12_work').rglob('*'):
  if p.is_file():files.append(p)
 files += [ROOT/'analysis_tools'/name for name in ['readiness_pipeline.py','readiness_packaging.py','regression_architectures.py','execute_followup.py','build_readiness_notebooks.py']]
 return sorted(set(files))

def copy_files(files,destination):
 rows=[]
 for i,p in enumerate(files):
  assert p.is_file(),p
  rel=p.relative_to(ROOT);target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
  # Resume safely; source and target content must agree even when sizes match.
  source_hash=r.fsha(p)
  if not target.exists() or target.stat().st_size!=p.stat().st_size or r.fsha(target)!=source_hash:shutil.copy2(p,target)
  assert r.fsha(target)==source_hash
  rows.append(dict(file=rel.as_posix(),sha256=source_hash,bytes=p.stat().st_size))
  if i%250==0:r.dump(28,'packaging_progress.json',{'operation':'copy and verify','files_finished':i+1,'files_total':len(files)})
 return rows

def package_environment(destination):
 packages=['numpy','pandas','scipy','scikit-learn','matplotlib','nbformat','nbclient','ipykernel','jupyter-client','torch','py2bit','tqdm','requests','packaging','filelock','PyYAML','fsspec']
 versions={}
 for name in packages:
  try:versions[name]=metadata.version(name)
  except metadata.PackageNotFoundError:pass
 versions.update({'transformers':'4.44.2','tokenizers':'0.19.1','huggingface-hub':'0.26.5','safetensors':'0.4.5','regex':'2024.11.6','einops':'0.8.0'})
 # No prefix/absolute installation location in the portable specification.
 req=['--extra-index-url https://download.pytorch.org/whl/cu128']+[f'{name}=={version}' for name,version in versions.items()]
 (destination/'requirements-win-cuda.txt').write_text('\n'.join(req)+'\n',encoding='utf8')
 yaml='name: enhancer\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.12\n  - pip\n  - pip:\n      - --extra-index-url https://download.pytorch.org/whl/cu128\n'+''.join(f'      - {name}=={version}\n' for name,version in versions.items() if name!='torch')+'      - torch=='+versions['torch']+'\n'
 (destination/'environment.yml').write_text(yaml,encoding='utf8')
 (destination/'environment_versions.json').write_text(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'packages':versions,'cuda_wheel_index':'https://download.pytorch.org/whl/cu128','scope':'Snapshot of executed result-reproduction dependencies; historical training runtimes remain separately recorded in provenance.'},indent=2)+'\n',encoding='utf8')
 return versions

def portable_documentation(destination):
 (destination/'README.md').write_text('''# Enhancer pre-writing evidence and reproducibility

This package reproduces the retained **results**, metrics, subgroup definitions, tables and figures for the bounded paper scope. It contains local source data, frozen splits/predictions, pinned model source, 48 neural regression checkpoints (24 older checkpoints inside the two original full-weight ZIPs), the executed notebooks and literature evidence. No neural training is required or launched by notebooks 20–29. Classification result reproduction uses authenticated record-level OOF predictions, config and run metadata. Original classification trained weights and the full original training wrapper were not supplied; the corrected GENA implementation is available. Unknown optimizer/loss details are labelled unavailable in T2. Historical Kaggle notebooks are read-only references, not the portable execution entry points.

## Use

Extract the complete archive into a writable directory. On this Windows machine use the existing Anaconda environment named **enhancer**. On another Windows CUDA12.8-compatible machine, create an enhancer environment with Python3.12 and install `requirements-win-cuda.txt`; `environment.yml` is an additional dependency specification. Do not install a second environment on the original machine. The bundled model vendor wheels target Windows/Python3.12; on other operating systems install equivalent native packages. For CPU-only metric/table/figure reproduction, use the appropriate CPU PyTorch wheel. Fresh Phase22 inference requires a compatible CUDA GPU; cached result verification and Phases21/23–27 are CPU analyses. Numerical tolerances are recorded for cross-backend checkpoint reproduction; the collapsed seed's unstable correlation is explicitly excluded from correlation-based numerical matching while its actual values and failure remain reported.

Run from this directory, e.g. `python -B analysis_tools/execute_followup.py 23`. Notebooks20–29 use relative paths and the local enhancer kernel. Each phase also contains explanations and assertions. Run Phase22 to extract older checkpoints and reproduce cached/fresh neural predictions. Deleting inference caches requests fresh inference and can take tens of minutes; do not delete archived research evidence. No new tuning/training occurs. CPU result verification was tested from this relocated package directory using the existing enhancer environment.

## Read first

`phase29_outputs/READINESS_REPORT.md` is the final ten-priority status. `phase25_outputs/TABLE_CATALOGUE.md` and `MAIN_TABLES_READABLE.md` contain publication tables; `phase26_outputs/FIGURE_CAPTIONS.md` indexes PNG/SVG/PDF figures; `phase27_outputs/LITERATURE_AND_CONTRIBUTION_REVIEW.md` explains closest work and contribution boundaries. `CONTRIBUTION_LOG.md` preserves earlier history and appends the completed readiness work. The old Phase19 manuscript material remains an earlier draft; new evidence does not silently rewrite it.

## Boundaries

Classification OOF records, internal assay test records and external transfer populations are distinct. Training budgets differ. Bootstrap intervals are conditional on fitted models and previously examined partitions; secondary/subgroup analyses are exploratory. Absence of exact/RC/31-mer overlap is not absence of all homology. The failed old K562 seed remains in the original ensemble; its training cause and old per-epoch logs remain unavailable. The chosen scope omits unverified submitted numbers, long-context benefit and incomplete Phase11C completion-dependent claims. Native-locus and broader-cell generalization are not established. This is readiness to start writing within that scope, not a finished revised manuscript, submission or acceptance.

## Integrity and licensing

`FILE_MANIFEST.csv` records SHA256/bytes for packaged files (excluding the manifest itself). `ARCHIVE_VALIDATION.json` is outside the archive and records ZIP CRC verification and archive hash. Prior provenance can contain historical absolute paths; execution code uses local paths. Copied source/license materials remain under reference_inputs and research_notes. No license or permission is granted beyond the original data/model/software terms; redistribution/public deposition requires the authors' normal source/license checks. The full input archives preserve original checkpoint bytes. Temporary partial downloads and duplicate extracted older weights are intentionally absent.
''',encoding='utf8')

def build_portable_package():
 assert r.jread(27,'summary.json')['status']=='completed'
 portable_architectures()
 destination=r.out(28)/PACKAGE_NAME;destination.mkdir(exist_ok=True)
 rows=copy_files(input_files(),destination)
 versions=package_environment(destination);portable_documentation(destination)
 # Static portability checks cover current execution entry points, not archival metadata/source notebooks.
 current=[ROOT/'analysis_tools'/n for n in ['readiness_pipeline.py','readiness_packaging.py','regression_architectures.py','execute_followup.py']]
 for p in current:
  code=p.read_text(encoding='utf8');compile(code,p.name,'exec')
  assert not re.search(r'[CD]:[\\/]',code),p.name
 # Relocated deterministic CPU analysis writes only into the package copy.
 script="import sys;sys.path.insert(0,'analysis_tools');import readiness_pipeline as r;r.phase23();r.phase24();print('RELOCATED_CPU_REPRODUCTION_PASSED')"
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
 proc=subprocess.run([sys.executable,'-B','-X','utf8','-c',script],cwd=destination,env=env,capture_output=True,text=True,encoding='utf8',timeout=600)
 (r.out(28)/'relocated_CPU_validation_log.txt').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf8')
 assert proc.returncode==0,(proc.returncode,proc.stderr[-2000:])
 for number,names in [(23,['internal_metrics_authoritative.csv','external_metrics_authoritative.csv','filtering_metrics.csv','filtering_paired_contrasts.csv']),
  (24,['training_only_thresholds_exact.csv','retained_test_subgroup_metrics_exact.csv','complete_population_subgroup_metrics_exact.csv','reliability_coverage.csv'])]:
  for name in names:assert r.fsha(destination/f'phase{number}_outputs'/name)==r.fsha(r.out(number)/name),name
 checkpoints=r.read(22,'recovered_checkpoint_index.csv').copy();checkpoints['availability']='included in original full-weight ZIP; Phase22 extracts locally'
 modern=r.read(22,'checkpoint_metadata.csv');modern=modern[modern.model.isin(['legnet','mprann'])]
 additional=pd.DataFrame([dict(protocol=x.protocol,name=f'{x.protocol}_{x.cell_line}_{x.model}_seed_{x.seed}_best.pt',path=f'phase12_outputs/enhancer_phase12_work/checkpoints/{x.protocol}_{x.cell_line}_{x.model}_seed_{x.seed}_best.pt',sha256=x.checkpoint_sha256,bytes=(r.P12/'checkpoints'/f'{x.protocol}_{x.cell_line}_{x.model}_seed_{x.seed}_best.pt').stat().st_size,availability='included directly') for x in modern.itertuples()])
 availability=pd.concat([checkpoints,additional],ignore_index=True);assert len(availability)==48
 r.table(28,'checkpoint_availability.csv',availability)
 r.dump(28,'package_staging_manifest.json',{'files':rows,'stage':'validated; final executed notebook/log/report snapshot added after Phase29','package_directory':f'phase28_outputs/{PACKAGE_NAME}'})
 r.dump(28,'summary.json',{'status':'completed_staged_and_relocated_CPU_validated','portable_execution_notebooks':list(range(20,30)),'staged_files':len(rows),'staged_bytes':sum(x['bytes'] for x in rows),'regression_checkpoints_included':48,
  'classification_checkpoint_policy':'Original classification trained weights/full original wrapper unavailable; corrected GENA code available; authenticated OOF/config/run metadata reproduction included; no exact classification retraining claim.',
  'relocated_CPU_reproduction':True,'exact_output_hashes_reproduced':8,'new_runtime_hardcoded_Windows_paths':False,'package_directory':f'phase28_outputs/{PACKAGE_NAME}',
  'final_archive':'Written and independently CRC/hash checked by finalizer after completed Phase29; see ARCHIVE_VALIDATION.json','package_dependency_versions':versions})
 print('Portable package staged, inputs copied and hashed; relocated CPU reproduction passed.')

def current_notebook_check(numbers):
 import nbformat
 rows=[]
 for number in numbers:
  p=next(ROOT.glob(f'[[]{number}]*.ipynb'));n=nbformat.read(p,as_version=4);nbformat.validate(n)
  code=[c for c in n.cells if c.cell_type=='code']
  for c in code:compile(c.source,p.name,'exec')
  errors=[o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
  complete=all(c.get('execution_count') is not None for c in code)
  assert complete and not errors,(number,complete,errors)
  rows.append(dict(phase=number,notebook=p.name,code_cells=len(code),executed=complete,errors=len(errors),sha256=r.fsha(p)))
 return pd.DataFrame(rows)

def final_review_map():
 original=r.read(15,'reviewer_evidence_map.csv');rows=[]
 evidence={
 'R1.1':('evidence_ready_for_writing','Separate actual input length/tokenizers and training settings documented.','phase25_outputs/tables/T2_model_training_settings.csv','Replace obsolete manuscript pseudocode with frozen implementation.'),
 'R1.2':('evidence_ready_for_writing','Every retained classification metric/CM and regression score recalculated. Unauthenticated submitted tables explicitly omitted.','phase21_outputs/submitted_table_dispositions.csv | phase25_outputs/tables/T3_verified_classification.csv','Insert replacement tables; do not revive excluded numbers.'),
 'R1.3':('evidence_ready_with_limits','Regression3-seed mean/SD/ensemble separate; classification5-fold SD is not seed variability. Incomplete11C claims omitted.','phase25_outputs/tables/S4_seed_mean_SD.csv | phase25_outputs/tables/S1_classification_folds.csv','Describe one-seed classification limit.'),
 'R1.4':('evidence_ready_with_limits','Actual selected epochs, train-only scaling and unequal budgets documented. Old per-epoch maxima remain unavailable.','phase25_outputs/tables/T2_model_training_settings.csv | phase22_outputs/checkpoint_metadata.csv','Disclose recipe asymmetry and historical selection limits.'),
 'R1.5':('evidence_ready_for_writing','Focused primary-source review includes current closest work and both reviewer-suggested sources with relevance limits.','phase27_outputs/verified_primary_citations.csv','Use precise relevant citations; no exhaustive novelty claim.'),
 'R4.1':('evidence_ready_with_limits','Novelty reframed as evaluation/population/reliability audit; prior fine-tuned MPRA comparisons acknowledged.','phase27_outputs/closest_work_and_limits.csv','Write bounded contribution; omit first-ever architecture claims.'),
 'R4.2':('evidence_ready_with_limits','Removed-sequence results now available for all3models on held-out chromosomes; native-locus/general-cell claims unsupported.','phase25_outputs/tables/T7_retained_removed_activity.csv','State exact assay/population scope and no intrinsic-sequence-insufficiency proof.'),
 'R4.3':('addressed_by_scope_exclusion','200bp experiments do not test longer flanks; long-context benefit claims excluded.','phase20_outputs/frozen_scope.json','Do not reintroduce long-context claims; flank study is optional future research.'),
 'R4.4':('addressed_by_scope_exclusion','Retained recipes use documented RC policies; unverified mutation/shuffling claims excluded.','phase25_outputs/tables/T2_model_training_settings.csv','Use actual augmentation only and disclose label-preservation assumptions.'),
 'R4.5':('graphics_ready_layout_pending_writing','Accurate architecture/workflow/vector graphics prepared; final manuscript equations/layout follow during writing.','phase26_outputs/FIGURE_CAPTIONS.md','Insert vector diagrams and format equations in final manuscript.'),
 'R7.1':('addressed_by_replacement_and_omission','Original Tables8/14 not silently repaired; unauthenticated values omitted. Authenticated new tables have correct denominators.','phase21_outputs/submitted_table_dispositions.csv | phase25_outputs/tables/T3_verified_classification.csv','Use explicit replacement results.'),
 'R7.2':('evidence_ready_for_writing','Frozen actual architectures, preprocessing and inference code ready.','phase25_outputs/tables/T2_model_training_settings.csv | phase26_outputs/figures/SF6_regression_architectures.svg','Align new algorithm description with implementations.'),
 'R7.3':('evidence_ready_for_writing','Verified ROC/PR/CM, paired regression effects, per-seed plots and readable vector outputs prepared.','phase26_outputs/FIGURE_CAPTIONS.md','Place correct task-specific figures and captions.'),
 'R7.4':('evidence_ready_for_writing','Correct originalDNABERT/ICLR2024DNABERT2 citations; bibliography rebuilt from deduplicated primary sources.','phase27_outputs/verified_scope_references.bib','Use new bibliography and remove obsolete/duplicate references.'),
 'Editor.novelty_limits':('evidence_ready_with_limits','Final bounded claims and closest-work distinctions frozen.','phase27_outputs/CONTRIBUTION_AND_CITATION_DECISIONS.md','Integrate into title, abstract and discussion.'),
 'Editor.graphical_abstract':('graphics_ready','Graphical abstract generated and source-checked.','phase26_outputs/figures/F10_graphical_abstract.svg','Authors select journal layout and accurate highlights while writing.'),
 'Shafin.result_verification':('addressed_with_limits','48neural checkpoints reproduced; all72Phase12 caches audited; collapse remains explicit.','phase22_outputs/summary.json | phase23_outputs/statistical_reporting_rules.json','Missing historical epoch logs/root cause remain disclosed.'),
 'Shafin.filter_population':('addressed_with_limits','Full149632candidate flow and all3model held-out removed performance available; exact thresholds corrected.','phase24_outputs/summary.json | phase25_outputs/tables/T7_retained_removed_activity.csv','No causal filtering/retraining or all-genome claim.')}
 for x in original.itertuples():
  if x.concern_id in evidence:status,finding,files,remaining=evidence[x.concern_id]
  elif x.concern_id in ['R1.6','R7.5']:status,finding,files,remaining='open_manuscript_writing','Evidence preparation does not proofread a revised manuscript.','phase29_outputs/READINESS_REPORT.md','Complete language/notation pass during writing.'
  else:status,finding,files,remaining='open_author_editorial_action','No manuscript submission, author declarations or portal-status assertion made.','reference_inputs/Review From Journal.docx','Authors handle declarations and submission after writing.'
  for file in files.split(' | '):assert (ROOT/file).exists(),file
  rows.append(dict(concern_id=x.concern_id,concern=x.concern,review_paragraphs=x.review_paragraphs,status=status,finding=finding,evidence_files=files,remaining_action=remaining))
 r.table(29,'final_reviewer_evidence_map.csv',pd.DataFrame(rows))
 return pd.DataFrame(rows)

def final_readiness_review():
 for n in range(20,29):assert r.jread(n,'summary.json')['status'].startswith('completed'),n
 checks=[]
 def check(name,condition,evidence):
  assert condition,name;checks.append(dict(check=name,passed=bool(condition),evidence=evidence))
 frozen=r.jread(20,'frozen_scope.json');check('Bounded scope and exclusions explicit',frozen['choice']=='bounded_recipe_comparison','phase20_outputs/frozen_scope.json')
 before=r.jread(20,'protected_before_readiness.json')
 for name,h in before.items():check('Protected unchanged: '+name,(ROOT/name).exists() and r.fsha(ROOT/name)==h,'phase20 protected snapshot')
 nb=current_notebook_check(range(20,29));r.table(29,'notebook_execution_validation.csv',nb)
 check('Nine predecessor notebooks executed without error',len(nb)==9,'phase29 notebook_execution_validation.csv')
 p=r.jread(22,'summary.json');cp=r.read(22,'checkpoint_prediction_reproduction.csv')
 check('48 checkpoints and96test/external partition checks',p['reproduced_neural_checkpoints']==48 and len(cp)==96 and cp.passed.all(),'phase22 checkpoint_prediction_reproduction.csv')
 check('Recovered transformer removed arm complete',p['dnabert2_removed_complete'],'phase22 complete_three_model_filtering_predictions.csv.gz')
 provenance=json.loads((r.P12/'results/phase12_provenance.json').read_text())
 for name,h in provenance['result_sha256'].items():check('Phase12 result hash '+name,r.fsha(r.P12/'results'/name)==h,'Phase12 provenance')
 caches=list((r.P12/'predictions').glob('*.csv'));check('All72Phase12 prediction caches present',len(caches)==72,'Phase12 predictions/')
 cache_rows=[]
 for pth in caches:
  meta=json.loads((r.P12/'cache'/pth.with_suffix('.json').name).read_text());hash_value=r.fsha(pth)
  check('Phase12 cache integrity '+pth.name,hash_value==meta['sha256'] and meta['lock']==provenance['protocol_sha256'],'Phase12 prediction cache metadata')
  cache_rows.append(dict(file=pth.relative_to(ROOT).as_posix(),sha256=hash_value,bytes=pth.stat().st_size,metadata=json.dumps(meta,sort_keys=True)))
 r.table(29,'all72_prediction_cache_index.csv',pd.DataFrame(cache_rows))
 for number,index in [(25,'table_index.csv'),(26,'figure_index.csv')]:
  d=r.read(number,index)
  for x in d.itertuples():check('Published file hash '+x.file,r.fsha(ROOT/x.file)==x.sha256,f'phase{number}_outputs/{index}')
 for x in r.read(26,'figure_source_file_index.csv').itertuples():check('Figure source hash '+x.file,r.fsha(ROOT/x.file)==x.sha256,'phase26 figure_source_file_index.csv')
 check('Main activity table includes all6families',len(r.read(25,'tables/T4_internal_activity.csv'))==24,'T4')
 check('External headline includes all6families',len(r.read(25,'tables/T5_DHS_natural_transfer.csv'))==24,'T5')
 check('Seven main and20supplement tables',r.jread(25,'summary.json')['main_tables']==7 and r.jread(25,'summary.json')['supplementary_tables']==20,'table index')
 visual=r.jread(26,'visual_QA.json');check('All16graphics visually inspected',visual['passed'] and visual['figures_inspected']==16,'phase26_outputs/visual_QA.json')
 threshold=r.read(24,'training_only_thresholds_exact.csv');precision=r.read(24,'threshold_precision_audit.csv')
 check('Threshold precision issue corrected',r.jread(24,'summary.json')['threshold_roundtrip_exact'] and precision[precision.cell_line.eq('K562')].old_export_assignment_changes.sum()==3070,'phase24 threshold precision audit')
 for tag,cell in [('9b','HepG2'),('9b','K562'),('9c','HepG2'),('9c','K562')]:
  sub=r.read(24,'retained_test_subgroup_metrics_exact.csv');sub=sub[sub.protocol.eq(tag)&sub.cell_line.eq(cell)]
  for model,g in sub.groupby('model'):
   counts=g.groupby('family').n.sum();check(f'Subgroup denominators {tag} {cell} {model}',counts.nunique()==1,'phase24 retained_test_subgroup_metrics_exact.csv')
 check('Relocated CPU reproduction passed',r.jread(28,'summary.json')['relocated_CPU_reproduction'],'phase28 relocated_CPU_validation_log.txt')
 check('All retained model identities cited',len(r.read(27,'retained_model_citation_map.csv'))==7,'phase27 retained_model_citation_map.csv')
 dispositions=r.read(21,'submitted_table_dispositions.csv');check('Every15submitted table dispositioned',len(dispositions)==15,'phase21 submitted_table_dispositions.csv')
 r.table(29,'final_gate_checks.csv',pd.DataFrame(checks))
 priorities=r.read(20,'priority_to_phase_map.csv');priorities['status']='completed_for_frozen_scope'
 priorities.loc[priorities.priority.eq(2),'status']='completed_by_audit_replacement_and_explicit_exclusion'
 priorities.loc[priorities.priority.eq(9),'status']='completed_relocated_CPU_validated; final ZIP seal follows this notebook'
 priorities['evidence']=[f'phase{n}_outputs/summary.json' for n in priorities.phase]
 r.table(29,'ten_priority_status.csv',priorities)
 claims=[('C1','Existing recipes are compared, with unequal budgets disclosed.','T2;T4;T5','F4;F5','No universal architecture or pretraining effect'),
 ('C2','Specialist CNNs lead tested overall internal/external metrics.','T4;T5;T6','F4;F5;F6','Conditional on tested cells, partitions and recipes'),
 ('C3','Legacy overlap and filtering affect the evaluation population.','T1;T3;S10','F2;F3;F7','Shared15mer exclusion is not a full duplicate count'),
 ('C4','Removed test-chromosome performance is measured for all3models.','T7;S7;S9','F7;F8','No causal filter/retraining or whole-genome inference'),
 ('C5','Marginal interval coverage hides subgroup differences; seed spread is weak.','S15;S16;S17;S18','SF2','Retrospective diagnostics; no individual confidence guarantee'),
 ('C6','One original K562 seed is nearly constant and remains retained.','S12;S14','SF3','Root cause and old per-epoch logs unknown'),
 ('C7','Verified grouped classification replaces unauthenticated submitted values.','T3;S1;S2;S19;S20','SF4;SF5','Singleseed OOF; dependent folds; not fresh external confirmation')]
 r.table(29,'frozen_claim_evidence_map.csv',pd.DataFrame(claims,columns=['claim_id','supported_claim','tables','figures','limit']))
 r.textfile(29,'READINESS_REPORT.md','''# Ready to start writing within the frozen scope

Aditta's ten pre-writing priorities are complete for the supported bounded comparison. See `ten_priority_status.csv` for each item and `final_gate_checks.csv` for machine-checked evidence. This does not mean every original experiment was finished: unauthenticated submitted tables and incomplete Phase11C completion-dependent claims are explicitly omitted; long-context and equal-budget superiority claims are excluded. These were allowed scope/omission branches in the checklist.

1. **Scope:** frozen task definitions, primary/secondary outcomes and bounded claims.
2. **Classification:** all retained OOF records authenticated; 18 pooled and 90 fold results recomputed, including denominators, confusion matrices, declared augmentation/fold records, overlap and mixed-label sensitivity. Every submitted table has an explicit decision; unsupported values are omitted.
3. **Reproduction:** 24 older weights recovered and authenticated; 48 neural checkpoints reproduced on test/external records in the local enhancer environment; all 72 Phase 12 caches preserved and audited. DNABERT-2 removed-candidate inference is complete. The old nearly constant K562 seed remains visible. Local inference replaces the checklist's Kaggle location because the user explicitly requested execution in enhancer.
4. **Statistics:** seed means/SD and ensembles separate; original and independent intervals separate; sample sizes, resampling-unit counts, conditional resampling and undefined/collapsed cases explicit. No paired split-effect interval across different cohorts.
5. **Data:** 149,632 complete pairs reconcile; 72,799 removed (48.65%). Raw-row/element exclusions and external populations documented. Final subgroup exports use 17 digits with round-trip parsing and integer CpG counts, correcting old reload assignments.
6. **Tables:** 7 main and 20 supplementary numerical tables, source indices and readable main-table views.
7. **Figures:** 16 figures/diagrams, each in PNG/SVG/PDF, with captions, placement and visual QA. Classification and regression graphics remain task-specific.
8. **Literature:** verified primary citations, retained-model identities, closest work and source reading limits. Fine-tuning/CNN MPRA comparison is not claimed as new by itself.
9. **Reproducibility:** local-input portable package, 48 regression checkpoints, exact environment specifications, source/license copies, data/splits/predictions/figure code, SHA manifests and successful relocated CPU result reproduction. Classification trained weights/full original wrapper remain unavailable; authenticated OOF/config/run-metadata result reproduction is complete and no exact classification retraining claim is made.
10. **Final gate:** protected older files unchanged; current tables/figures/claims/source hashes reconciled; reviewer concerns mapped to final evidence, narrower claims or writing/editorial actions.

The next action is to write a new manuscript around these frozen tables, figures and claims, then prepare the point-by-point response. The original manuscript and Phase19 draft remain untouched.

## Findings to carry into writing

Both specialist CNNs exceed DNABERT-2 on the tested overall primary/secondary internal and DHS-natural transfer comparisons. Recovered-weight inference also completes the three-model removed-population comparison on the held-out chromosomes. Removed candidates have slightly higher absolute errors; population activity/composition shifts mean correlation changes alone do not measure robustness. DNABERT-2 removed-minus-retained MAE is +0.00485 for HepG2 and +0.00312 for K562; both conditional 95% intervals include zero. In the final shared bootstrap, K562 LegNet's MAE gap is above zero; K562 MPRAnn's interval just includes zero. The latter is close to the boundary and should not be described as a robust positive effect based on earlier draws. Use the frozen S7 intervals for this three-model comparison; do not average interval versions. Corrected subgroup results and reliability diagnostics preserve exceptions rather than claiming uniform CNN superiority.

## Limitations that remain in the paper

Different model recipes/budgets; previously examined partitions; few seeds and dependent classification folds; selected 200 bp reporter candidates rather than native-locus/all-cell validation; exact/RC/31-mer screens do not eliminate all homology; conditional/exploratory uncertainty; no causal retraining experiment; original collapsed-seed cause and older epoch logs unknown; no long-context experiment. Original classification trained weights and its full original training wrapper are unavailable; corrected GENA code is supplied, while unknown original wrapper/optimizer/loss details are explicitly labelled unavailable in T2. Its authenticated OOF predictions/config support result reproduction rather than exact retraining. These limits are documented, not silently declared resolved. Manuscript language/equation layout, author declarations and submission remain future writing/editorial tasks.

The package archive is sealed after this notebook finishes so it includes the actual executed notebook and final contribution log. `phase28_outputs/ARCHIVE_VALIDATION.json` records the final archive hash, readability and CRC validation.
''')
 final_map=final_review_map()
 r.dump(29,'summary.json',{'status':'completed_ready_to_start_writing_with_frozen_scope','ten_priorities':10,'checks_passed':len(checks),'all_passed':True,
  'manuscript_finished':False,'journal_submission_made':False,'long_context_or_new_training_launched':False,'unsupported_claims_explicitly_excluded':True,'reviewer_map_rows':len(final_map),'archive_seal':'Finalizer runs after notebook29 completes'})
 print('Final ten-priority readiness gate passed; ready to start writing within frozen scope.')

def finalize_archive():
 assert r.jread(29,'summary.json')['all_passed']
 notebooks=current_notebook_check(range(20,30));r.table(29,'notebook_execution_validation.csv',notebooks)
 log=ROOT/'CONTRIBUTION_LOG.md';existing=log.read_text(encoding='utf8');marker='### Completed Phases 20–29: Aditta pre-writing priorities (2026-10-13)'
 if marker not in existing:
  log.write_text(existing.rstrip()+'''\n\n### Completed Phases 20–29: Aditta pre-writing priorities (2026-10-13)

Completed and executed ten new notebooks in enhancer, with all products/inputs/caches contained here. This entry supersedes earlier pending statements while preserving their history. No original research documents, notebooks12–19 or Phase12 outputs were edited. No new neural models trained or tuned; no system settings/global environment changed; no journal or other person contacted.

- **20 — Scope:** froze a bounded comparison of tested recipes. Long-context, universal architecture superiority, DNA-insufficiency and incomplete Phase 11C completion-dependent claims are excluded. This is evidence preparation before further manuscript drafting.
- **21 — Classification:** authenticated 37,926 OOF prediction rows; recalculated 18 pooled/90 fold results and confusion matrices; checked identical records/groups/counts and the six-record mixed-label sensitivity. All 15 submitted tables have explicit replace/omit decisions. Phase 11C remains at 10/40 new jobs, and dependent claims are excluded.
- **22 — Checkpoints:** corrected the earlier local-search mistake: Phase 9B/9C full-weight archives already existed in the supplied parent folder. Copied and authenticated them inside this workspace; recovered 24 older neural weights and reproduced 48 total neural checkpoints across 96 test/external partitions. DNABERT-2 now scores 8,436 HepG2 and 12,702 K562 removed test-chromosome candidates. Both specialist CNNs retain their overall advantage. DNABERT-2 removed/retained MAE is 0.42966/0.42481 for HepG2 and 0.26092/0.25781 for K562, using the same local backend for both populations. Archived benchmark calculations remain authoritative; GPU differences and nearly constant seed correlation instability are documented. The initial 20-minute runner timeout was resumed safely from caches; final execution passed without errors. Pinned model/tokenizer code and dependency wheels are isolated under reference_inputs/readiness_runtime; enhancer itself was not modified.
- **23 — Statistics:** recomputed authoritative internal/external metrics; separated individual seeds, seed means/SD, ensembles, original Phase 12 versus independent Phase 13 intervals, and conditional cluster resampling. Reported sample/resampling-unit counts and extended paired/removed analyses to all three models. DNABERT-2 removed-minus-retained MAE intervals include zero in both cells. In this frozen shared bootstrap, K562 LegNet's interval is above zero while K562 MPRAnn's just includes zero; use S7 rather than treating the earlier boundary-sensitive result as robust. Preserved collapsed seed and labelled sensitivities; no paired split-effect estimate across distinct cohorts.
- **24 — Data/precision:** reconciled 149,632 complete candidates and 72,799 removed (48.65%) to authenticated raw sources. Corrected the 12-digit CpG-density reload issue: 1,844 K562 9B and 1,226 K562 9C assignments differ under the old export. Final definitions use integer CG counts, 17-digit exports, exact round-trip parsing and explicit activity boundaries. Regenerated final subgroup/reliability analyses for all three models; previous Phase 14–18 files remain historical and unchanged. Whole-test scores are unaffected by subgroup relabelling.
- **25/26 — Publication assets:** 7 main and 20 supplementary source-indexed tables, readable main views, and 16 visually inspected figures/diagrams in PNG/SVG/PDF, with captions and placement. No unverified old curve/table copied into final evidence.
- **27 — Literature:** focused primary-source review and precise citations for original DNABERT, DNABERT-2, original GENA-LM, MPRA-LegNet/MPRAnn, LegNet ancestry and relevant closest work. Current status/reading limits included; prior fine-tuned MPRA comparisons acknowledged. No first-ever/exhaustive novelty claim or cross-dataset numerical headline.
- **28 — Portability:** self-contained result-reproduction package with local raw inputs, source archives, 48 regression checkpoints, splits/predictions/provenance, pinned code/license copies, environment specs and source hashes. Relocated CPU execution reproduced eight authoritative output hashes exactly. Historical classification trained weights and full original wrapper are unavailable; corrected GENA code and authenticated OOF/config/run-metadata reproduction are included. Unknown original optimizer/loss details are marked unavailable in T2; no exact classification retraining claim is made.
- **29 — Readiness:** all ten priorities pass for the explicitly frozen scope. Protected older snapshots match; all ten new notebooks fully executed without errors; numerical assets and citations traced; reviewer concerns mapped to final evidence, exclusions or later writing/editorial actions. Final archive is CRC/readability/SHA checked after notebook execution.

**Current stage:** ready to start manuscript writing around Phase 25 tables, Phase 26 figures and frozen claims. The original manuscript and Phase 19 draft are not a finished revision. Remaining scientific limits are asymmetric recipes, previously inspected data, conditional uncertainty, selected 200 bp assays/two cell-design families, incomplete homology exclusion, unavailable original classification training wrapper/weights, and the unknown old collapse cause/missing epoch logs. Long-context, broader-cell/native-locus and causal retraining studies are optional new research, not completed here. Author statements, language/layout pass and submission come after writing.
''',encoding='utf8')
 destination=r.out(28)/PACKAGE_NAME
 # Final notebook/log/status copies replace staging versions only inside this new package.
 files=input_files()+[p for n in [28,29] for p in r.out(n).iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md','.txt'] and p.name not in ['ARCHIVE_VALIDATION.json','package_staging_manifest.json','packaging_progress.json']]
 copy_files(sorted(set(files)),destination)
 portable_documentation(destination)
 manifest=[]
 for p in sorted(destination.rglob('*')):
  if p.is_file() and p.name!='FILE_MANIFEST.csv':manifest.append(dict(file=p.relative_to(destination).as_posix(),sha256=r.fsha(p),bytes=p.stat().st_size))
 pd.DataFrame(manifest).to_csv(destination/'FILE_MANIFEST.csv',index=False)
 final=r.out(28)/'Enhancer_PreWriting_Reproducibility.zip';temporary=final.with_suffix('.zip.writing')
 with zipfile.ZipFile(temporary,'w',allowZip64=True) as z:
  for i,p in enumerate(sorted(destination.rglob('*'))):
   if p.is_file():
    compression=zipfile.ZIP_STORED if p.suffix.lower() in ['.zip','.gz','.pt','.pyd','.dll','.2bit','.png','.pdf'] else zipfile.ZIP_DEFLATED
    z.write(p,arcname=PACKAGE_NAME+'/'+p.relative_to(destination).as_posix(),compress_type=compression,compresslevel=1)
 os.replace(temporary,final)
 # Full ZIP streaming CRC checks plus full-file SHA; not just a directory listing.
 with zipfile.ZipFile(final) as z:
  bad=z.testzip();assert bad is None,bad
  files_in_archive=[x for x in z.infolist() if not x.is_dir()];assert len(files_in_archive)==len(manifest)+1
  packaged=z.read(PACKAGE_NAME+'/FILE_MANIFEST.csv');assert packaged==(destination/'FILE_MANIFEST.csv').read_bytes()
 r.dump(28,'ARCHIVE_VALIDATION.json',{'status':'completed','file':final.name,'sha256':r.fsha(final),'bytes':final.stat().st_size,'crc_all_members_passed':True,'members':len(files_in_archive),'manifest_records':len(manifest),'includes_executed_notebooks_20_through_29':True,'includes_final_contribution_log':True,'archive_readable':True})
 print(json.dumps(r.jread(28,'ARCHIVE_VALIDATION.json'),indent=2))

if __name__=='__main__':
 finalize_archive()
