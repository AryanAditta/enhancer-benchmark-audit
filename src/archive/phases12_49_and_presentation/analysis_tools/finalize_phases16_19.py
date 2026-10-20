from pathlib import Path
import json,hashlib,zipfile,datetime
import numpy as np,pandas as pd,nbformat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'phase19_outputs'
def fsha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

# Embed complete upstream license text in the new notebook; executable source is unchanged.
p=next(ROOT.glob('[[]16]*.ipynb'));n=nbformat.read(p,as_version=4)
license_dir=ROOT/'reference_inputs/phase19_method_sources'
license_text='\n\n'.join((license_dir/name).read_text(encoding='utf8') for name in ['legnet_LICENSE.txt','mprann_LICENSE.txt'])
prefix='\n'.join('# '+line for line in license_text.splitlines())+'\n\n'
for c in n.cells:
 if c.cell_type=='code' and '# ---- MPRA-LegNet' in c.source and 'Permission is hereby granted, free of charge, to any person obtaining a copy' not in c.source:
  c.source=prefix+c.source
nbformat.validate(n);nbformat.write(n,p)

checks=[]
for number in [16,17,18,19]:
 p=next(ROOT.glob(f'[[]{number}]*.ipynb'));n=nbformat.read(p,as_version=4);nbformat.validate(n)
 code=[c for c in n.cells if c.cell_type=='code']
 assert all(c.execution_count is not None for c in code),p.name
 assert not [o for c in code for o in c.outputs if o.output_type=='error'],p.name
 for i,c in enumerate(n.cells):
  if c.cell_type=='code':compile(c.source,f'{p.name}:cell{i}','exec')
 checks.append({'notebook':p.name,'code_cells':len(code),'sha256':fsha(p),'status':'schema, syntax and full execution passed'})

# Preserve previous notebook hashes and all Phase12 protected inputs.
prior=json.loads((ROOT/'analysis_tools/final_validation.json').read_text());prior_notebook_checks=[]
for row in prior['notebooks']:
 p=ROOT/row['notebook'];current=fsha(p);same=current==row['sha256']
 if not same:
  assert p.stat().st_mtime<(ROOT/'phase16_outputs/frozen_protocol.json').stat().st_mtime, 'Concurrent older-notebook change: preserve and inspect.'
 prior_notebook_checks.append({'notebook':p.name,'matches_earlier_validation_snapshot':same,'current_sha256':current,
  'modified':datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
  'note':'Not edited by this follow-up; any older snapshot difference predates Phase16 protocol creation.'})
for rel,h in json.loads((ROOT/'phase16_outputs/protected_input_hashes.json').read_text()).items():assert fsha(ROOT/rel)==h,rel

# Independent check of optimized bootstrap moments against literal repeated rows.
s=pd.read_csv(ROOT/'phase16_outputs/ensemble_predictions_and_metadata.csv.gz')
g=s[s.cell_line.eq('HepG2')&s.model.eq('legnet')&s.population.eq('removed')].copy()
keys=g.resampling_group.unique()[:40];g=g[g.resampling_group.isin(keys)]
y=g.mean_log2_activity.to_numpy();p=g.prediction.to_numpy()
tmp=pd.DataFrame({'group':g.resampling_group,'n':1.,'y':y,'p':p,'yy':y*y,'pp':p*p,'yp':y*p,'ae':abs(p-y),'se':(p-y)**2})
mom=tmp.groupby('group').sum().reindex(keys).to_numpy();rng=np.random.default_rng(191919)
maxerr=0.
for _ in range(10):
 ids=rng.integers(0,len(keys),len(keys));w=np.bincount(ids,minlength=len(keys));v=w@mom
 count,sy,sp,syy,spp,syp,sae,sse=v
 direct=pd.concat([g[g.resampling_group.eq(keys[i])] for i in ids],ignore_index=True)
 dy=direct.mean_log2_activity.to_numpy();dp=direct.prediction.to_numpy()
 quick=np.array([(syp-sy*sp/count)/np.sqrt((syy-sy*sy/count)*(spp-sp*sp/count)),np.sqrt(sse/count),sae/count])
 literal=np.array([np.corrcoef(dy,dp)[0,1],np.sqrt(np.mean((dy-dp)**2)),np.mean(abs(dy-dp))])
 maxerr=max(maxerr,float(np.max(abs(quick-literal))));assert np.allclose(quick,literal,atol=1e-12,rtol=1e-12)

log=ROOT/'CONTRIBUTION_LOG.md';before=log.read_text(encoding='utf8')
marker='### Completed Phases 16–19: excluded-population CNN performance and reliability (2026-10-12)'
entry='''

### Completed Phases 16–19: excluded-population CNN performance and reliability (2026-10-12)

This entry extends the completed Phase14 population reconstruction. Removed-sequence performance is now measured for the two specialist CNNs on the original held-out chromosomes; it remains unavailable for DNABERT-2 because the original fine-tuned regression weights were not found in the supplied/local research inputs.

- **Phase16:** authored and fully executed `[16] Enhancer_Filtered_Out_Performance.ipynb` in enhancer. Used 12 authenticated existing CNN checkpoints, without training. Scored all 8,436 removed HepG2 and 12,702 removed K562 candidates on chr1/12/4/7/9, alongside 9,048 and 13,145 retained candidates. No exact/RC matches to pooled training/validation records; coordinates are chromosome-disjoint. Reproduced archived retained predictions within approximately 5e-12. Cached-inference repeatability was checked; original float32 precision is restored after round-trip CSV parsing.
- Removed-versus-retained primary MAE: HepG2 LegNet 0.40008 vs 0.39294, MPRAnn 0.39931 vs 0.39591; K562 LegNet 0.23876 vs 0.23201, MPRAnn 0.24439 vs 0.23937. Raw K562 MAE contrasts have conditional intervals above zero; HepG2 intervals include zero. K562 correlations are higher on removed candidates despite higher absolute errors. A 31-mer sensitivity excludes 199/230 removed/retained HepG2 candidates and 393/380 K562 candidates; qualitative conclusions persist. These are descriptive population comparisons, not equivalence or the causal effect of retraining without filtering.
- **Phase17:** authored and fully executed `[17] Enhancer_Filtering_Reliability_and_Composition.ipynb`. Shared-composition standardization uses fixed/training-derived activity, GC and homopolymer strata, covering approximately 96–98% of records. K562 MAE gaps shrink to +0.00378 LegNet/+0.00301 MPRAnn, with conditional intervals including zero. Seed-SD versus absolute-error associations are weak (Spearman 0.082–0.155). Evaluated fixed spread cutoffs and nominal 90% intervals using calibration/audit genomic-identity groups assigned before new inference. Retained-calibrated constant-width marginal coverage is 88.1–89.1% HepG2 and 89.8–90.9% K562; seed-scaled intervals are generally much wider without consistent benefit. No audit labels entered interval fitting, but these are retrospective diagnostics with dependence and shift limitations, not formal coverage guarantees.
- **Phase18:** authored and fully executed `[18] Enhancer_Activity_Tails_and_Independent_Audit.ipynb` after inspecting Phase17. Independently reconstructed 24 caches, ensemble means/SDs, 16 metric rows, 16 interval parameter sets, all saved endpoints and 32 coverage summaries. Restored original float32 cache values to explain near-tie Spearman parsing effects. High measured-activity candidates show shared underprediction: mean signed errors about −0.71 to −0.75 log2 HepG2 and −0.33 to −0.36 K562; all three seeds underpredict 71–86% of these examples. Constant-width interval coverage is 57–61% HepG2/69–74% K562 in those outcome-selected groups. Highest predicted-activity quintiles also have about 80–82% coverage. These are exploratory limitations, not a causal biological explanation or proof of a broken marginal conformal guarantee.
- **Phase19:** authored and fully executed `[19] Enhancer_Filtering_Findings_and_Reviewer_Package.ipynb`. Prepared `phase19_outputs/RESEARCH_FINDINGS.md`, `MANUSCRIPT_SECTION_DRAFT.md`, `REVIEWER_RESPONSE_ADDENDUM.md`, `FIGURE_CAPTIONS.md`, exact evidence hashes, follow-up evidence map, focused primary-source notes/references and pending-input instructions. Original manuscript and Phase15 map remain unchanged. Focused literature checking is not represented as an exhaustive review.
- **Validation and containment:** all four new notebooks passed schema/syntax/full execution without notebook errors. Independently checked the optimized bootstrap moments against literal repeated-row samples. Inspected figures. Protected Phase12 inputs and notebooks 14–15 match their earlier hashes. Notebook13 has an earlier modification at 10:01, before Phase16 protocol creation at 10:42, and differs from the 06:51 validation snapshot; it was preserved and not edited by this follow-up. All products, copied source/license material, runtime files and analyses stay inside Shafin's Contribution. No packages were installed, original experiments retrained, system settings changed, or journal/other person contacted.

**Pending:** recover the original Phase9B DNABERT-2 checkpoints (normally `enhancer_phase9b_complete_with_weights.zip`) and pinned compatible tokenizer/model snapshot for the intended three-model removed-population comparison. Historical classification-table repair, the old failed-transformer-run cause, long-context/broader-cell experiments and full manuscript integration remain separate open work. No missing results are fabricated.
'''
if marker not in before:
 log.write_text(before+entry,encoding='utf8');assert log.read_text(encoding='utf8').startswith(before)

readme=ROOT/'FOLLOW_UP_ANALYSES_README.md';old=readme.read_text(encoding='utf8')
if '## Phases 16–19' not in old:
 readme.write_text(old+'''\n\n## Phases 16–19\n\nRun the four new notebooks in numeric order using enhancer. Phase16 reuses existing specialist CNN checkpoints and authenticates cached inference; Phase17 analyses composition and reliability; Phase18 independently verifies calculations and diagnoses activity tails; Phase19 regenerates writing and evidence maps. All dependencies are local inside this folder. No new training is launched. The original DNABERT-2 regression weights are missing, so that arm is explicitly pending.\n\nStart reading `phase19_outputs/RESEARCH_FINDINGS.md`. Manuscript and reviewer-response drafts are in the same folder. See `PENDING_INPUTS_AND_NEXT_ACTIONS.md` for exact missing checkpoint requirements and remaining research/manuscript limits. Original notebooks and Phase12 outputs remain unchanged. The results supplement ZIP excludes original datasets and model weights; rerunning requires this complete local contribution workspace.\n''',encoding='utf8')

result={'status':'completed_CNN_branch_DNABERT2_pending','environment':str(Path(__import__('sys').executable)),
 'notebooks':checks,'bootstrap_literal_reconstruction_max_error':maxerr,
 'previous_notebook_snapshot_checks':prior_notebook_checks,'older_notebooks_not_edited_by_this_followup':True,'protected_inputs_unchanged':True,
 'contribution_log_sha256':fsha(log),'new_training_runs':0,
 'blocking_input':'Original fine-tuned Phase9B DNABERT-2 weights and compatible pinned code/tokenizer snapshot'}
(OUT/'final_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')

# Results supplement only: keep bulky original data and checkpoint dependencies in the workspace.
files=[]
for number in [16,17,18,19]:
 files.append(next(ROOT.glob(f'[[]{number}]*.ipynb')))
 for p in (ROOT/f'phase{number}_outputs').rglob('*'):
  if p.is_file() and not any(x in p.parts for x in ['matplotlib_cache']) and p.suffix!='.zip' and p.name!='package_manifest.json':files.append(p)
for p in (ROOT/'reference_inputs/phase19_method_sources').iterdir():
 if p.is_file():files.append(p)
files.extend([log,readme])
files.extend(ROOT/'analysis_tools'/f'build_phase{i}.py' for i in [16,17,18,19])
files.extend([ROOT/'analysis_tools/execute_followup.py',ROOT/'analysis_tools/finalize_phases16_19.py',ROOT/'analysis_tools/capture_phase19_sources.py'])
files=sorted(set(files));manifest={p.relative_to(ROOT).as_posix():fsha(p) for p in files}
archive=OUT/'Shafin_Phases16_19_CNN_Results_Supplement.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,p.relative_to(ROOT).as_posix())
 z.writestr('SUPPLEMENT_MANIFEST.json',json.dumps({'scope':'Results and new notebooks only; original datasets and weights remain required in local workspace','files':manifest},indent=2))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
(OUT/'package_manifest.json').write_text(json.dumps({'archive':archive.name,'bytes':archive.stat().st_size,'sha256':fsha(archive),'file_count':len(files),'scope':'CNN results only; DNABERT2 pending; upstream inputs/weights excluded'},indent=2),encoding='utf8')
print(json.dumps(result,indent=2));print('Readable authenticated results supplement:',archive.name,archive.stat().st_size,'bytes')
