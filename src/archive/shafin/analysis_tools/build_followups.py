from pathlib import Path
import nbformat as nb, textwrap
ROOT=Path.cwd()
NAMES={13:'[13] Enhancer_Independent_Result_Verification.ipynb',14:'[14] Enhancer_Filtering_and_Subgroup_Analysis.ipynb',15:'[15] Enhancer_Reviewer_Evidence_Map.ipynb'}
def md(s):return nb.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s):return nb.v4.new_code_cell(textwrap.dedent(s).strip())
def save(number,cells):
 p=ROOT/NAMES[number]
 if p.exists():raise FileExistsError('Preserve existing notebook: '+str(p))
 n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (enhancer)','language':'python','name':'enhancer'},'language_info':{'name':'python','version':'3.12'},'contribution':{'author':'Shafin','created':'2026-10-12','scope':'CPU audit; no training; Phase 12 artifacts read only'}})
 for i,c in enumerate(n.cells):
  if c.cell_type=='code':compile(c.source,f'{p.name}:cell{i}','exec')
 nb.validate(n);nb.write(n,p);print('Created',p.name)
BASE=r'''
from pathlib import Path
import os, sys, io, json, zipfile, hashlib, itertools
import numpy as np
import pandas as pd
ROOT=Path(r"D:\Aditta Thesis\For Shafin\Shafin's Contribution").resolve()
assert ROOT.is_dir() and ROOT.name=="Shafin's Contribution"
assert Path(sys.executable).resolve()==Path(r'C:\Users\User\.conda\envs\enhancer\python.exe').resolve(), 'Select the enhancer Anaconda kernel.'
OUT=ROOT/'phaseNUMBER_outputs';OUT.mkdir(exist_ok=True)
P12=ROOT/'phase12_outputs/enhancer_phase12_work'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def filehash(p,algorithm='sha256'):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,algorithm).hexdigest()
def member(name):
 found=[]
 for folder in ['phase12_inputs','reference_inputs']:
  for p in (ROOT/folder).glob('*.zip'):
   with zipfile.ZipFile(p) as z:
    for m in z.namelist():
     if m.split('/')[-1]==name:found.append(z.read(m))
 assert found, 'Missing copied input: '+name
 assert len({digest(b) for b in found})==1, 'Ambiguous input versions: '+name
 return found[0]
def csvmember(name):return pd.read_csv(io.BytesIO(member(name)))
def dump_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def write_table(name,frame):frame.to_csv(OUT/name,index=False,float_format='%.12g')
prov=json.loads((P12/'results/phase12_provenance.json').read_text())
assert prov['smoke_test'] is False
for name,h in prov['config']['input_sha256'].items():assert digest(member(name))==h,name
full=csvmember('phase9_full_activity_all_cells.csv')
splits=csvmember('phase9_frozen_splits.csv')
assert full.record_id.is_unique and splits.record_id.is_unique
DATA=full.merge(splits.drop(columns=[c for c in splits if c in full and c!='record_id']),on='record_id',validate='one_to_one')
assert len(DATA)==76833
SEEDS=[20260930,20260931,20260932]
def corr(y,p):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
 a=y-y.mean();b=p-p.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b))
 return float(np.dot(a,b)/den) if den else float('nan')
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);assert np.isfinite(y).all() and np.isfinite(p).all()
 e=y-p;sst=np.sum((y-y.mean())**2)
 return dict(n=len(y),pearson=corr(y,p),spearman=corr(pd.Series(y).rank().to_numpy(),pd.Series(p).rank().to_numpy()),r2=float(1-np.sum(e*e)/sst) if sst else np.nan,rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))))
def internal_predictions():
 old=[csvmember(f'phase{t}_predictions_validation_test.csv').assign(protocol=t) for t in ['9b','9c']]
 return pd.concat(old+[pd.read_csv(P12/'results/phase12_internal_predictions_new_arms.csv')],ignore_index=True)
print('Environment:',sys.executable,'; full retained records:',len(DATA))
'''
def base(number):return BASE.replace('phaseNUMBER_outputs',f'phase{number}_outputs')
C13=[md('''# Phase 13 — Independent verification of saved enhancer results
**Owner: Shafin. CPU only; no training or model tuning.** Run all cells in the existing Anaconda **enhancer** kernel. Inputs are copied archives and completed Phase 12 artifacts inside this contribution folder. Outputs go only to `phase13_outputs`.

This notebook uses independently written arithmetic on saved predictions. It does not execute Phase 12 source. It checks integrity, all 24 training jobs, records/targets, ensemble arithmetic, metrics, epoch selection, seed variability, and uncertainty. It keeps the earlier collapsed DNABERT-2 seed in the primary ensemble.

The frozen partitions have already been examined in earlier work. Bootstrap intervals are conditional on these trained models and partitions, not a fresh confirmatory experiment and not an estimate of all training uncertainty. External sequence bootstrap can underestimate uncertainty under genomic dependence; the current Phase 12 also supplies a 1 kb region sensitivity analysis, which is checked below.'''),code(base(13)),md('''## 1. Authenticate the completed artifacts and jobs'''),code(r'''
hash_rows=[]
for name,h in prov['result_sha256'].items():
 got=filehash(P12/'results'/name);assert got==h,name
 hash_rows.append(dict(kind='result',file=name,sha256=got))
assert digest(json.dumps(prov['config'],sort_keys=True,separators=(',', ':')).encode())==prov['protocol_sha256']
for name,spec in prov['checkpoints'].items():
 p=P12/'checkpoints'/name;assert p.stat().st_size==spec['bytes'] and filehash(p)==spec['sha256'],name
 hash_rows.append(dict(kind='checkpoint',file=name,sha256=spec['sha256']))
archive_rows=[]
for name in ['enhancer_phase12_strong_baselines_results.zip','enhancer_phase12_complete_with_weights.zip']:
 p=ROOT/'phase12_outputs'/name;before=(p.stat().st_size,p.stat().st_mtime_ns)
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None
  for f,h in {**prov['result_sha256'],'phase12_provenance.json':filehash(P12/'results/phase12_provenance.json')}.items():
   matches=[m for m in z.namelist() if m.split('/')[-1]==f];assert len(matches)==1
   assert digest(z.read(matches[0]))==h
  if 'with_weights' in name:
   for f,spec in prov['checkpoints'].items():
    matches=[m for m in z.namelist() if m.split('/')[-1]==f];assert len(matches)==1
    assert digest(z.read(matches[0]))==spec['sha256']
 assert before==(p.stat().st_size,p.stat().st_mtime_ns), 'Archive still changing'
 archive_rows.append(dict(file=name,bytes=before[0],sha256=filehash(p)))
choices=pd.read_csv(P12/'results/phase12_validation_choices.csv');logs=pd.read_csv(P12/'results/phase12_training_logs.csv')
expected=set(itertools.product(['9b','9c'],['HepG2','K562'],['legnet','mprann'],SEEDS))
assert len(choices)==24 and set(choices[['protocol','cell_line','model','seed']].itertuples(index=False,name=None))==expected
assert not choices.collapsed.any()
cache_rows=[]
for row in choices.itertuples(index=False):
 job=f'{row.protocol}_{row.cell_line}_{row.model}_seed_{row.seed}'
 for kind in ['validation','test','external']:
  meta=json.loads((P12/'cache'/f'{job}_{kind}.json').read_text());p=P12/'predictions'/f'{job}_{kind}.csv'
  assert meta['lock']==prov['protocol_sha256'] and meta['job']==job and meta['kind']==kind
  assert filehash(p)==meta['sha256'];cache_rows.append(dict(job=job,kind=kind,sha256=meta['sha256']))
 l=logs[logs.job.eq(job)].sort_values('epoch');assert len(l)==row.epochs_run
 valid=l[np.isfinite(l.validation_pearson)]
 best=valid.loc[valid.validation_pearson.idxmax()]
 assert int(best.epoch)==row.selected_epoch
 assert abs(best.validation_pearson-row.validation_pearson)<1e-10
write_table('artifact_hash_checks.csv',pd.DataFrame(hash_rows));write_table('archive_hashes.csv',pd.DataFrame(archive_rows));write_table('prediction_cache_hash_checks.csv',pd.DataFrame(cache_rows))
print('Authenticated 7 upstream inputs, 8 result tables, 24 checkpoints, 72 prediction caches and both finished ZIPs.')
print('24/24 expected jobs; selected epochs match validation-only maxima. Recorded job minutes:',choices.minutes.sum())
'''),md('''## 2. Identical evaluation records and independent metrics
Pearson uses centered dot products; Spearman ranks each vector with average ties; R², RMSE and MAE are calculated directly. A strict tolerance is used for ordinary metric comparisons. Spearman can change slightly when 12-digit CSV serialization merges near ties, so this separate tolerance is reported explicitly.'''),code(r'''
pred=internal_predictions();assert not pred.duplicated(['protocol','cell_line','split','model','record_id']).any()
metric_rows=[];record_rows=[];ensemble_rows=[]
for (tag,cell,split),sub in pred.groupby(['protocol','cell_line','split']):
 col={'9b':'split_9b','9c':'split_9a'}[tag]
 ref=DATA[DATA.cell_line.eq(cell)&DATA[col].eq(split)].set_index('record_id').sort_index()
 for model,g in sub.groupby('model'):
  g=g.set_index('record_id').sort_index();assert g.index.equals(ref.index),(tag,cell,split,model)
  assert np.max(abs(g.target.to_numpy()-ref.mean_log2_activity.to_numpy()))<2e-7
  assert g.chrom.equals(ref.chrom) and g.locus_group_id.equals(ref.locus_group_id)
  metric_rows.append(dict(protocol=tag,cell_line=cell,split=split,model=model,**metrics(g.target,g.prediction)))
  record_rows.append(dict(protocol=tag,cell_line=cell,split=split,model=model,n=len(g),sorted_id_sha256=digest('\n'.join(g.index).encode())))
 wide=sub.pivot(index='record_id',columns='model',values='prediction')
 for model in ['cnn','dnabert2','legnet','mprann']:
  error=float(np.max(abs(wide[[f'{model}_seed_{s}' for s in SEEDS]].mean(axis=1)-wide[f'{model}_three_seed_mean'])))
  assert error<5e-7,(tag,cell,split,model,error)
  ensemble_rows.append(dict(protocol=tag,cell_line=cell,split=split,model=model,max_absolute_error=error))
recomputed=pd.DataFrame(metric_rows);reported=pd.read_csv(P12/'results/phase12_internal_metrics_all_arms.csv')
check=recomputed.merge(reported,on=['protocol','cell_line','split','model'],suffixes=('_new','_saved'),validate='one_to_one');assert len(check)==len(reported)==len(recomputed)
errors={}
for m in ['n','pearson','spearman','r2','rmse','mae']:
 error=np.abs(check[m+'_new']-check[m+'_saved']);errors[m]=float(error.max())
 assert error.max()<(1e-5 if m=='spearman' else 2e-8), (m,error.max())
write_table('internal_metrics_recomputed.csv',recomputed);write_table('identical_evaluation_records.csv',pd.DataFrame(record_rows));write_table('ensemble_arithmetic.csv',pd.DataFrame(ensemble_rows))
show=['dnabert2_three_seed_mean','legnet_three_seed_mean','mprann_three_seed_mean','cnn_three_seed_mean','spaced6_4_ridge']
print('Largest metric differences:',errors)
print(recomputed[recomputed.split.eq('test')&recomputed.model.isin(show)][['protocol','cell_line','model','n','pearson','r2','rmse']].round(6).to_string(index=False))
'''),md('''## 3. External cohort, all arms and all origins
All models must score the identical 29,330 frozen records, with targets matching the frozen assay file. The primary external comparison remains the 8,399 DHS natural sequences. Synthetic origins are separate descriptive transfer checks.'''),code(r'''
ext=csvmember('phase10_external_frozen_set.csv').set_index('record_id').sort_index();assert ext.index.is_unique and len(ext)==29330
oldext=csvmember('phase10_external_predictions.csv');newext=pd.read_csv(P12/'results/phase12_external_scores_new_arms.csv').set_index('record_id').sort_index()
assert newext.index.equals(ext.index) and newext.origin.equals(ext.origin)
ext_scores={};ext_rows=[]
for (tag,cell,model),g in oldext.groupby(['protocol','cell_line','model']):
 g=g.set_index('record_id').sort_index();assert g.index.equals(ext.index) and g.origin.equals(ext.origin)
 assert np.max(abs(g.activity-ext[f'{cell}_l2fc']))<1e-9
 ext_scores[tag,cell,model]=g.prediction.to_numpy()
for tag,cell,model in itertools.product(['9b','9c'],['HepG2','K562'],['legnet','mprann']):
 cols=[f'{tag}_{cell}_{model}_seed_{s}' for s in SEEDS];mean=newext[f'{tag}_{cell}_{model}_three_seed_mean'].to_numpy()
 assert np.max(abs(newext[cols].mean(axis=1)-mean))<5e-7
 for s,c in zip(SEEDS,cols):ext_scores[tag,cell,f'{model}_seed_{s}']=newext[c].to_numpy()
 ext_scores[tag,cell,f'{model}_three_seed_mean']=mean
for (tag,cell,model),v in ext_scores.items():
 for origin,g in ext.groupby('origin'):
  loc=ext.index.get_indexer(g.index);m=metrics(g[f'{cell}_l2fc'],v[loc])
  ext_rows.append(dict(protocol=tag,cell_line=cell,origin=origin,model=model,n=len(g),pearson=m['pearson'],spearman=m['spearman']))
external_metrics=pd.DataFrame(ext_rows);reported_ext=pd.read_csv(P12/'results/phase12_external_metrics_all_arms.csv')
ec=external_metrics.merge(reported_ext,on=['protocol','cell_line','origin','model'],suffixes=('_new','_saved'),validate='one_to_one');assert len(ec)==len(reported_ext)==len(external_metrics)
assert (ec.n_new==ec.n_saved).all()
for m in ['pearson','spearman']:assert np.max(abs(ec[m+'_new']-ec[m+'_saved']))<(1e-5 if m=='spearman' else 2e-8)
write_table('external_metrics_recomputed.csv',external_metrics)
print(external_metrics[external_metrics.origin.eq('DHS_natural')&external_metrics.model.isin(show)].round(6).to_string(index=False))
'''),md('''## 4. Seed variability and the collapsed earlier K562 run
A nearly constant predictor may have an unstable correlation. The old Phase 9C K562 DNABERT-2 seed is shown alongside the healthy seeds; it remains in the original three-seed mean. A two-seed sensitivity calculation is labelled explicitly, never substituted for the primary result. Saved predictions can diagnose collapse, but cannot establish its optimizer/gradient cause without the original per-epoch training diagnostics.'''),code(r'''
seed_rows=[]
for (tag,cell,split,model),g in pred[pred.model.str.contains('_seed_')].groupby(['protocol','cell_line','split','model']):
 seed_rows.append(dict(protocol=tag,cell_line=cell,split=split,model=model,prediction_sd=float(g.prediction.std(ddof=0)),target_sd=float(g.target.std(ddof=0)),pearson=corr(g.target,g.prediction)))
seed_diagnostics=pd.DataFrame(seed_rows)
s=seed_diagnostics[seed_diagnostics.split.eq('test')].copy();s['family']=s.model.str.replace(r'_seed_\d+$','',regex=True)
variability=s.groupby(['protocol','cell_line','family']).agg(seeds=('pearson','size'),mean_seed_pearson=('pearson','mean'),sd_seed_pearson=('pearson','std'),min_prediction_sd=('prediction_sd','min')).reset_index()
old=pred[pred.protocol.eq('9c')&pred.cell_line.eq('K562')]
print(seed_diagnostics[seed_diagnostics.protocol.eq('9c')&seed_diagnostics.cell_line.eq('K562')&seed_diagnostics.model.str.startswith('dnabert2')].to_string(index=False))
sensitivity=[]
for split,g in old.groupby('split'):
 w=g.pivot(index='record_id',columns='model',values='prediction');y=g[g.model.eq('dnabert2_three_seed_mean')].set_index('record_id').loc[w.index].target
 for arm,values in [('original_all_three',w.dnabert2_three_seed_mean),('sensitivity_two_healthy_seeds',w[['dnabert2_seed_20260931','dnabert2_seed_20260932']].mean(axis=1))]:
  sensitivity.append(dict(split=split,analysis=arm,**metrics(y,values)))
write_table('all_seed_diagnostics.csv',seed_diagnostics);write_table('test_seed_variability.csv',variability);write_table('old_9c_k562_collapse_sensitivity.csv',pd.DataFrame(sensitivity))
print(pd.DataFrame(sensitivity).round(6).to_string(index=False))
print('Root cause remains unresolved: no new training was launched. SD across 3 seeds is descriptive; bootstrap CIs below do not include that variability.')
'''),md('''## 5. Independently repeat paired uncertainty analysis
Use 1,000 fresh bootstrap draws (fixed independent seed 20261013) for both DNABERT-2 versus CNN contrasts in each protocol/cell. Internal draws resample complete test loci. External draws either resample sequences or nearby regions. Spearman ranks are recomputed **within each draw**, including duplicate sampled observations. These reruns check robustness and approximate reproducibility, not bit-identical RNG replay. Secondary intervals are unadjusted for multiplicity.'''),code(r'''
from scipy.stats import rankdata
B=1000;rng=np.random.default_rng(20261013);boot_rows=[]
def boot_difference(y,a,b,groups,stat):
 members=[np.flatnonzero(groups==g) for g in np.unique(groups)]
 values=[]
 for _ in range(B):
  idx=np.concatenate([members[i] for i in rng.integers(len(members),size=len(members))])
  yy,aa,bb=y[idx],a[idx],b[idx]
  if stat=='spearman':yy,aa,bb=rankdata(yy),rankdata(aa),rankdata(bb)
  values.append(corr(yy,aa)-corr(yy,bb))
 values=np.array(values);assert np.isfinite(values).all()
 return float(np.quantile(values,.025)),float(np.quantile(values,.975))
saved_ib=pd.read_csv(P12/'results/phase12_internal_paired_bootstrap.csv');saved_eb=pd.read_csv(P12/'results/phase12_external_paired_bootstrap.csv')
dhs=ext[ext.origin.eq('DHS_natural')].copy();groups=pd.Series(index=dhs.index,dtype='int64');counter=-1
for chrom,g in dhs.sort_values(['external_chrom','external_start','external_end']).groupby('external_chrom'):
 boundary=-10**15
 for idx,row in g.iterrows():
  if row.external_start>boundary+1000:counter+=1
  boundary=max(boundary,row.external_end);groups.at[idx]=counter
print('External grouping:',len(dhs),'sequences;',groups.nunique(),'regions')
for tag,cell in itertools.product(['9b','9c'],['HepG2','K562']):
 sub=pred[pred.protocol.eq(tag)&pred.cell_line.eq(cell)&pred.split.eq('test')];w=sub.pivot(index='record_id',columns='model',values='prediction')
 truth=sub[sub.model.eq('spaced6_4_ridge')].set_index('record_id').loc[w.index]
 for cnn in ['legnet','mprann']:
  contrast=f'dnabert2_three_seed_mean minus {cnn}_three_seed_mean'
  y=truth.target.to_numpy();a=w.dnabert2_three_seed_mean.to_numpy();b=w[f'{cnn}_three_seed_mean'].to_numpy()
  lo,hi=boot_difference(y,a,b,truth.locus_group_id.to_numpy(),'pearson')
  saved=saved_ib[saved_ib.protocol.eq(tag)&saved_ib.cell_line.eq(cell)&saved_ib.contrast.eq(contrast)].iloc[0]
  observed=corr(y,a)-corr(y,b);assert abs(observed-saved.pearson_difference)<2e-8
  boot_rows.append(dict(protocol=tag,cell_line=cell,contrast=contrast,cohort='internal_test',unit='locus',difference=observed,lower_95=lo,upper_95=hi,saved_lower_95=saved.lower_95,saved_upper_95=saved.upper_95,repeats=B))
  loc=ext.index.get_indexer(dhs.index);y=dhs[f'{cell}_l2fc'].to_numpy();a=ext_scores[tag,cell,'dnabert2_three_seed_mean'][loc];b=ext_scores[tag,cell,f'{cnn}_three_seed_mean'][loc]
  observed=corr(rankdata(y),rankdata(a))-corr(rankdata(y),rankdata(b))
  for unit,gg,analysis in [('sequence',np.arange(len(y)),'primary'),('region_1kb',groups.to_numpy(),'sensitivity')]:
   lo,hi=boot_difference(y,a,b,gg,'spearman');saved=saved_eb[saved_eb.protocol.eq(tag)&saved_eb.cell_line.eq(cell)&saved_eb.contrast.eq(contrast)&saved_eb.analysis.eq(analysis)].iloc[0]
   assert abs(observed-saved.spearman_difference)<1e-5
   assert max(lo,saved.lower_95)<=min(hi,saved.upper_95),'Bootstrap rerun intervals do not overlap; investigate'
   boot_rows.append(dict(protocol=tag,cell_line=cell,contrast=contrast,cohort='DHS_natural',unit=unit,difference=observed,lower_95=lo,upper_95=hi,saved_lower_95=saved.lower_95,saved_upper_95=saved.upper_95,repeats=B))
 print('Independent bootstrap finished:',tag,cell,flush=True)
boots=pd.DataFrame(boot_rows);write_table('independent_paired_bootstrap.csv',boots)
print(boots.round(6).to_string(index=False))
'''),md('''## 6. Record the verified scope
A reproducible CNN advantage is useful evidence even though it reverses the proposed DNABERT-2 superiority claim. These are adaptations to 200 bp inputs, without assay adapters or LegNet shift augmentation. Published repository defaults were used; this is not exhaustive model optimization. The MPRAnn eval parity claim in Phase 12 provenance is reported evidence from the earlier check, not independently rerun here. Three seeds and conditional sampling intervals are different uncertainty summaries.'''),code(r'''
source_nb=next(ROOT.glob('[[]12]*ipynb'));source=json.loads(source_nb.read_text('utf-8'));text='\n'.join(''.join(c['source']) for c in source['cells'])
assert 'clip_grad_norm_' in text and 'region_draws' in text and 'gradient_clip_norm' in text
assert (boots.upper_95<0).all(), 'Do not assume all contrasts favor CNNs; inspect current results'
report=dict(status='completed',environment=sys.executable,phase12_protocol_sha256=prov['protocol_sha256'],phase12_notebook_sha256=filehash(source_nb),expected_jobs=24,authenticated_result_files=8,authenticated_checkpoints=24,internal_metric_max_absolute_differences=errors,bootstrap_repeats=B,bootstrap_seed=20261013,old_9c_k562_collapse='Present in original ensemble; cause unresolved; sensitivity labelled separately',uncertainty_scope='Conditional on trained models and previously examined partitions; secondary intervals unadjusted; external sequence dependence checked by 1kb region sensitivity',conclusion='Both specialist CNN ensembles outperform DNABERT-2 on all four internal Pearson and DHS-natural Spearman comparisons under the tested recipes; no universal architecture or pretraining causal claim.')
dump_json('verification_summary.json',report)
print(json.dumps(report,indent=2))
''')]
save(13,C13)
C14=[md('''# Phase 14 — Filtering and retained-test subgroups
**Owner: Shafin. CPU analysis only; no model fitting, tuning, or test-based model selection.** Use the existing **enhancer** kernel. All original inputs have been copied under `reference_inputs`; raw ENCODE/UCSC bytes are authenticated against the original Phase 9A checksums. This notebook does not download or install anything.

**Main question:** What kinds of sequences did the legacy-similarity filter remove, and how does that limit the paper's conclusions?

We distinguish the complete paired-orientation assay population, the retained filtered population, and prediction performance on retained test records. Removed sequences have no saved model predictions here: their performance cannot be inferred from composition or activity distributions.

**Subgroups fixed before this analysis:** GC bins [0,.35,.45,.55,.65,1]; CpG density below/above the training median; maximum homopolymer <6 versus ≥6; activity below training 20th percentile / middle / above training 80th percentile; chromosomes. Thresholds learned from training use only that protocol/cell's training records. Minimum reporting n=100. These are descriptive, exploratory comparisons on already examined partitions, not calibrated causal tests.'''),code(base(14)),md('''## 1. Reconstruct the complete assay population and ordered filtering reasons
Raw-row exclusions (design family, chromosome, length, count reliability, orientation pairing) are counted separately from sequence filtering among complete 200 bp pairs. Each population record gets exactly one ordered reason: non-ACGT, exact/RC match to legacy, shared 15-mer with legacy, duplicate within cell, or retained. The first matching reason is used, as in Phase 9A.'''),code(r'''
import gzip,re,math,struct
from collections import Counter
AUDIT=json.loads(member('phase9a_audit_and_provenance.json'))
RAW=ROOT/'reference_inputs/raw_sources'
missing=[name for name in AUDIT['sources'] if not (RAW/name).is_file()]
source_status={'status':'pending' if missing else 'available','missing':missing}
if missing:print('PENDING raw population/composition reconstruction; missing copied raw files:',missing)
else:
 for name,checks in AUDIT['sources'].items():
  assert filehash(RAW/name,'md5')==checks['md5'] and filehash(RAW/name)==checks['sha256'], name
 print('All three raw sources match original Phase 9A MD5 and SHA256.')
# A minimal read-only UCSC 2bit reader avoids adding packages.
class TwoBit:
 def __init__(self,path):
  self.f=Path(path).open('rb');head=self.f.read(16)
  self.endian='<' if struct.unpack('<I',head[:4])[0]==0x1A412743 else '>'
  sig,ver,count,res=struct.unpack(self.endian+'4I',head);assert sig==0x1A412743 and ver==0
  self.offsets={};self.cache={}
  for _ in range(count):
   n=self.f.read(1)[0];name=self.f.read(n).decode();self.offsets[name]=self.uints(1)[0]
  self.lut=np.array([[ord('TCAG'[(x>>shift)&3]) for shift in [6,4,2,0]] for x in range(256)],dtype=np.uint8)
 def uints(self,n):return struct.unpack(self.endian+str(n)+'I',self.f.read(4*n)) if n else ()
 def chromosome(self,name):
  if name not in self.cache:
   self.f.seek(self.offsets[name]);size=self.uints(1)[0];n=self.uints(1)[0];starts=self.uints(n);sizes=self.uints(n)
   m=self.uints(1)[0];self.uints(m);self.uints(m);self.uints(1)
   packed=self.f.read((size+3)//4);self.cache[name]=(size,starts,sizes,packed)
  return self.cache[name]
 def sequence(self,name,start,end):
  size,ns,nl,packed=self.chromosome(name);assert 0<=start<end<=size
  b=np.frombuffer(packed,dtype=np.uint8,offset=start//4,count=(end+3)//4-start//4)
  s=bytearray(self.lut[b].ravel()[start%4:start%4+end-start].tobytes())
  for a,l in zip(ns,nl):
   if a>=end:break
   lo=max(start,a);hi=min(end,a+l)
   if lo<hi:s[lo-start:hi-start]=b'N'*(hi-lo)
  return s.decode()
 def close(self):self.f.close()
'''),code(r'''
POP=None;raw_audits={}
if not missing:
 genome=TwoBit(RAW/'hg38.2bit');legacy=csvmember('stage1_candidate_manifest.csv')
 assert digest(member('stage1_candidate_manifest.csv'))==AUDIT['config']['stage1_manifest_sha256'] and len(legacy)==3268
 comp=str.maketrans('ACGT','TGCA')
 def rc(s):return s.translate(comp)[::-1]
 exact={min(s,rc(s)) for s in legacy.sequence};words=set()
 for s in legacy.sequence:
  for strand in [s,rc(s)]:words.update(strand[i:i+15] for i in range(186))
 populations=[]
 for cell,spec in AUDIT['config']['assays'].items():
  count=Counter();rows=[];pat=re.compile(re.escape(spec['prefix'])+r'\d+(?:_Reversed:)?')
  with gzip.open(RAW/spec['file'],'rt') as f:
   for line in f:
    if not line.strip() or line.startswith(('#','track','browser')):continue
    fields=line.rstrip('\n').split('\t');assert len(fields)==11;chrom,st,en,name,score,strand,fc,ic,oc,p,q=fields
    count['rows_total']+=1
    if not pat.fullmatch(name):count['other_design_name']+=1;continue
    if not re.fullmatch(r'chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)',chrom):count['noncanonical_chrom']+=1;continue
    st,en=int(st),int(en)
    if en-st!=200 or st<0 or en>genome.chromosome(chrom)[0]:count['non_200bp']+=1;continue
    if strand not in ['+','-']:count['unknown_strand']+=1;continue
    assert p=='-1' and q=='-1';fc,ic,oc=map(float,[fc,ic,oc])
    if not all(map(math.isfinite,[fc,ic,oc])) or ic<=0 or oc<=0:count['unreliable_counts']+=1;continue
    rows.append((chrom,st,en,re.sub(r'_Reversed:$','',name),strand,fc,ic,oc));count['candidate_rows']+=1
  temp=pd.DataFrame(rows,columns=['chrom','start','end','design_name','strand','activity','input','output'])
  paired=temp.groupby(['chrom','start','end','design_name']).agg(n=('activity','size'),strands=('strand','nunique'),mean_log2_activity=('activity','mean'),min_input_count=('input','min'),min_output_count=('output','min')).reset_index()
  good=paired[paired.n.eq(2)&paired.strands.eq(2)].copy();count['incomplete_or_ambiguous_orientation']=len(paired)-len(good);count['complete_two_orientation_elements']=len(good)
  for key,value in count.items():assert value==AUDIT['assay_audit'][cell].get(key,0),(cell,key,value)
  good['cell_line']=cell;seqs=[];reasons=[];ids=[]
  for row in good.itertuples(index=False):
   seq=genome.sequence(row.chrom,row.start,row.end).upper();seqs.append(seq)
   ids.append(digest(f'{cell}|{row.chrom}|{row.start}|{row.end}|{row.design_name}'.encode())[:20])
   if not re.fullmatch('[ACGT]{200}',seq):reason='non_ACGT_reference'
   elif min(seq,rc(seq)) in exact:reason='exact_or_RC_legacy'
   elif any(seq[i:i+15] in words for i in range(186)):reason='shared_15mer_with_legacy'
   else:reason='retained'
   reasons.append(reason)
  good['record_id']=ids;good['sequence']=seqs;good['sequence_sha256']=[digest(s.encode()) for s in seqs];good['filter_reason']=reasons
  sorted_good=good[good.filter_reason.eq('retained')].sort_values(['sequence_sha256','chrom','start','record_id'])
  duplicates=sorted_good[sorted_good.sequence_sha256.duplicated()].index;good.loc[duplicates,'filter_reason']='within_cell_duplicate_sequences'
  for reason in ['non_ACGT_reference','exact_or_RC_legacy','shared_15mer_with_legacy','within_cell_duplicate_sequences']:
   assert int(good.filter_reason.eq(reason).sum())==AUDIT['assay_audit'][cell].get(reason,0),(cell,reason)
  retained=good[good.filter_reason.eq('retained')].set_index('record_id').sort_index();ref=DATA[DATA.cell_line.eq(cell)].set_index('record_id').sort_index()
  assert retained.index.equals(ref.index)
  assert retained.sequence.equals(ref.sequence) and retained.sequence_sha256.equals(ref.sequence_sha256)
  assert np.max(abs(retained.mean_log2_activity-ref.mean_log2_activity))<1e-10
  populations.append(good);raw_audits[cell]=dict(count)
  print('Reconstructed',cell,':',len(good),'complete pairs;',len(retained),'retained',flush=True)
 genome.close();POP=pd.concat(populations,ignore_index=True)
 assert len(POP)==149632 and POP.record_id.is_unique
 POP['population']=np.where(POP.filter_reason.eq('retained'),'retained','removed')
 write_table('reconstructed_filtering_population.csv.gz',POP)
 source_status={'status':'completed','population_records':len(POP),'retained':int(POP.population.eq('retained').sum()),'removed':int(POP.population.eq('removed').sum()),'source_hashes':AUDIT['sources'],'legacy_manifest_sha256':digest(member('stage1_candidate_manifest.csv'))}
dump_json('population_reconstruction_status.json',source_status);dump_json('raw_row_exclusion_audit.json',raw_audits)
'''),md('''## 2. Activity distributions, DNA composition and cell-type attrition
Differences describe the screen's selection effect. They do not measure how models would perform on the removed DNA. HepG2 and K562 use different candidate-design families, so attrition differences cannot be attributed to cell biology alone. P-values are unnecessary for the central finite-population question; report counts and effect sizes.'''),code(r'''
def sequence_features(frame):
 frame=frame.copy();s=frame.sequence
 frame['gc_fraction']=(s.str.count('G')+s.str.count('C'))/200
 frame['cpg_density']=s.str.count('CG')/199
 probs=np.stack([s.str.count(b).to_numpy()/200 for b in 'ACGT'],axis=1)
 frame['base_entropy_bits']=-np.sum(np.where(probs>0,probs*np.log2(np.maximum(probs,1e-300)),0),axis=1)
 frame['max_homopolymer']=[max(len(m.group()) for m in re.finditer(r'A+|C+|G+|T+',seq)) for seq in s]
 return frame
summary_rows=[];activity_effects=[]
if POP is not None:
 POP=sequence_features(POP)
 attrition=POP.groupby(['cell_line','filter_reason']).size().rename('n').reset_index()
 attrition['complete_pair_population']=attrition.cell_line.map(POP.cell_line.value_counts());attrition['fraction']=attrition.n/attrition.complete_pair_population
 write_table('cell_type_filter_attrition.csv',attrition)
 for (cell,group),g in POP.groupby(['cell_line','population']):
  for feature in ['mean_log2_activity','gc_fraction','cpg_density','base_entropy_bits','max_homopolymer']:
   v=g[feature];summary_rows.append(dict(cell_line=cell,population=group,feature=feature,n=len(g),mean=v.mean(),sd=v.std(),p05=v.quantile(.05),p20=v.quantile(.2),median=v.median(),p80=v.quantile(.8),p95=v.quantile(.95)))
 for cell,g in POP.groupby('cell_line'):
  a=g[g.population.eq('retained')];b=g[g.population.eq('removed')]
  for feature in ['mean_log2_activity','gc_fraction','cpg_density','base_entropy_bits','max_homopolymer']:
   pooled=np.sqrt((a[feature].var()+b[feature].var())/2)
   activity_effects.append(dict(cell_line=cell,feature=feature,retained_minus_removed_mean=a[feature].mean()-b[feature].mean(),standardized_mean_difference=(a[feature].mean()-b[feature].mean())/pooled))
 write_table('retained_removed_distribution_summary.csv',pd.DataFrame(summary_rows));write_table('retained_removed_effect_sizes.csv',pd.DataFrame(activity_effects))
 print(attrition.round(6).to_string(index=False));print(pd.DataFrame(activity_effects).round(6).to_string(index=False))
else:print('Population distributions pending authenticated raw inputs; retained test analysis can still run.')
'''),md('''## 3. Predeclared subgroups on retained test records
Evaluate the same retained test records for DNABERT-2, MPRA-LegNet and MPRAnn ensemble means. Error units are assay log2 activity. Correlations in narrow activity bins may drop from range restriction. These are subgroup summaries, not an uncertainty calibration model. No thresholds or model choices are selected using these test results.'''),code(r'''
features=sequence_features(DATA);allpred=internal_predictions();subgroup_rows=[];paired_rows=[];thresholds=[]
models=['dnabert2_three_seed_mean','legnet_three_seed_mean','mprann_three_seed_mean']
for tag,cell in itertools.product(['9b','9c'],['HepG2','K562']):
 col={'9b':'split_9b','9c':'split_9a'}[tag];tr=features[features.cell_line.eq(cell)&features[col].eq('train')];te=features[features.cell_line.eq(cell)&features[col].eq('test')].copy()
 q20,q80=tr.mean_log2_activity.quantile([.2,.8]);cpg=tr.cpg_density.median()
 thresholds.append(dict(protocol=tag,cell_line=cell,training_n=len(tr),training_activity_q20=q20,training_activity_q80=q80,training_cpg_median=cpg))
 te['gc_bin']=pd.cut(te.gc_fraction,[0,.35,.45,.55,.65,1],include_lowest=True).astype(str)
 te['cpg_group']=np.where(te.cpg_density<=cpg,'at_or_below_training_median','above_training_median')
 te['homopolymer_group']=np.where(te.max_homopolymer>=6,'run_at_least_6','run_below_6')
 te['activity_group']=pd.cut(te.mean_log2_activity,[-np.inf,q20,q80,np.inf],labels=['below_training_q20','middle','above_training_q80']).astype(str)
 pred=allpred[allpred.protocol.eq(tag)&allpred.cell_line.eq(cell)&allpred.split.eq('test')&allpred.model.isin(models)]
 wide=pred.pivot(index='record_id',columns='model',values='prediction');assert set(wide.index)==set(te.record_id)
 te=te.set_index('record_id').join(wide,validate='one_to_one')
 for grouping in ['gc_bin','cpg_group','homopolymer_group','activity_group','chrom']:
  for label,g in te.groupby(grouping):
   if len(g)<100:
    paired_rows.append(dict(protocol=tag,cell_line=cell,subgroup_type=grouping,subgroup=str(label),n=len(g),status='suppressed_n_below_100'));continue
   for model in models:subgroup_rows.append(dict(protocol=tag,cell_line=cell,subgroup_type=grouping,subgroup=str(label),model=model,**metrics(g.mean_log2_activity,g[model])))
   ae=abs(g[models].sub(g.mean_log2_activity,axis=0));paired_rows.append(dict(protocol=tag,cell_line=cell,subgroup_type=grouping,subgroup=str(label),n=len(g),status='descriptive',dnabert2_minus_legnet_mae=float((ae[models[0]]-ae[models[1]]).mean()),dnabert2_minus_mprann_mae=float((ae[models[0]]-ae[models[2]]).mean())))
subgroup=pd.DataFrame(subgroup_rows);paired=pd.DataFrame(paired_rows)
write_table('retained_test_subgroup_metrics.csv',subgroup);write_table('paired_subgroup_error_differences.csv',paired);write_table('training_only_subgroup_thresholds.csv',pd.DataFrame(thresholds))
print('Positive paired MAE differences mean the CNN has smaller error.')
print(paired[paired.subgroup_type.ne('chrom')].round(5).to_string(index=False))
'''),md('''## 4. Save limitations and evidence
The 15-mer screen removes overlap with the legacy benchmark and selects a narrower population. It is not proof that every form of sequence homology is absent. Population attrition and subgroup prediction accuracy address different questions. Performance on removed candidates, other cell types, native chromatin and longer sequences remains open.'''),code(r'''
status={'status':'completed' if POP is not None else 'partial_raw_population_pending','phase12_protocol_sha256':prov['protocol_sha256'],'population_reconstruction':source_status,'subgroup_metric_rows':len(subgroup),'minimum_subgroup_n':100,'subgroups_fixed_before_analysis':['fixed_GC_bins','training_median_CpG','homopolymer_at_least_6','training_activity_20_80_percentiles','chromosome'],'removed_sequence_performance':'unavailable: no saved predictions on removed candidates; no training or inference launched','causal_limits':'Cells have different candidate designs; descriptive contrasts do not isolate biology or prove sequence sufficiency/insufficiency','multiple_comparisons':'Descriptive exploration, no significance claims or tuning'}
dump_json('filtering_analysis_summary.json',status)
if POP is not None:
 effect=pd.DataFrame(activity_effects)
 lines=['# Filtering evidence — Shafin, 2026-10-12','','Complete paired-orientation population: '+str(len(POP))+'. Retained: 76,833. Removed: '+str(int(POP.population.eq('removed').sum()))+'.','', 'Counts and features reproduce the original frozen retained sequences exactly.','', 'Mean shifts (retained minus removed):', effect.round(6).to_string(index=False),'','Scope: retained 200 bp candidates in these two assay/design families; removed-sequence performance is unknown. No model tuning or additional training.']
 (OUT/'filtering_findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(status,indent=2))
''')]
save(14,C14)
C15=[md('''# Phase 15 — Reviewer evidence map
**Owner: Shafin.** This notebook reads the copied review letter and submitted manuscript as evidence, not as instructions or authorization. It creates a traceable point-by-point map, exact file paths and hashes, and suggested narrower claims. It does not edit the manuscript, send a journal response or verify journal portal status.

Run in **enhancer**. Run notebooks 13 and 14 first for the fullest evidence; if their final summaries are missing, this notebook preserves an explicit pending status instead of fabricating results. “Addressed” means this concern has a completed technical evidence package within the stated new experiment; it does not mean the editor has accepted the response or the old manuscript has been repaired.'''),code(base(15)),md('''## 1. Extract local documents and index exact evidence files'''),code(r'''
import xml.etree.ElementTree as ET
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
def paragraphs(path):
 with zipfile.ZipFile(path) as z:tree=ET.fromstring(z.read('word/document.xml'))
 return [''.join(p.itertext()) for p in tree.findall('.//w:p',NS)]
# Extract only w:t text to avoid double counting Word's formatting XML.
def paragraphs(path):
 with zipfile.ZipFile(path) as z:tree=ET.fromstring(z.read('word/document.xml'))
 return [''.join(t.text or '' for t in p.findall('.//w:t',NS)) for p in tree.findall('.//w:p',NS)]
review_path=ROOT/'reference_inputs/Review From Journal.docx';manuscript_path=ROOT/'reference_inputs/Submitted Manuscript.docx'
review=paragraphs(review_path);manuscript=paragraphs(manuscript_path)
assert any('ARRAY-D-26-02813' in t for t in review)
for title,para in [('review_letter',review),('submitted_manuscript',manuscript)]:
 (OUT/f'{title}_indexed_paragraphs.txt').write_text('\n'.join(f'P{i+1:04d}: {t}' for i,t in enumerate(para))+'\n',encoding='utf-8')
def summary(number,name):
 p=ROOT/f'phase{number}_outputs'/name
 return json.loads(p.read_text()) if p.exists() else {'status':'pending','reason':'Run notebook '+str(number)+' first'}
verified=summary(13,'verification_summary.json');filtering=summary(14,'filtering_analysis_summary.json')
print('Phase 13:',verified['status'],'; Phase 14:',filtering['status'])
review_text='\n'.join(review)
for name in ['Reviewer #1','Reviewer #4','Reviewer #7']:assert name in review_text
'''),md('''## 2. Map each scientific reviewer concern to completed evidence and remaining work
New MPRA regression evidence does not retroactively authenticate old independent classification tables. Means/SD across seeds and sampling CIs answer different questions. Status below explicitly distinguishes a technical analysis from manuscript repair.'''),code(r'''
V='phase13_outputs/verification_summary.json';I='phase13_outputs/internal_metrics_recomputed.csv';X='phase13_outputs/external_metrics_recomputed.csv';B='phase13_outputs/independent_paired_bootstrap.csv';S='phase13_outputs/test_seed_variability.csv';F='phase14_outputs/filtering_analysis_summary.json';D='phase14_outputs/retained_removed_distribution_summary.csv';A='phase14_outputs/cell_type_filter_attrition.csv';G='phase14_outputs/retained_test_subgroup_metrics.csv';P='phase12_outputs/enhancer_phase12_work/results/phase12_provenance.json';N='[12] Enhancer_Kaggle_Phase12_Strong_Baselines.ipynb'
rows=[]
def add(id,concern,status,evidence,finding,remaining,anchor):
 anchors=[i+1 for i,t in enumerate(review) if anchor.lower() in t.lower()]
 assert anchors,(id,anchor)
 rows.append(dict(concern_id=id,concern=concern,status=status,evidence_files=' | '.join(evidence),finding=finding,remaining_action=remaining,review_paragraphs=','.join('P'+str(i).zfill(4) for i in anchors)))
add('R1.1','200 bp/6-mer text versus 2000 bp/4-mer pseudocode','partial',[P,N],'Current MPRA comparison uses 200 bp; one-hot CNNs and DNABERT-2 are separate from the original DNABERT 6-mer model.','Rewrite original pseudocode, task definitions, model names and tokenizer descriptions to match each actual experiment.','Algorithm 1 refers')
add('R1.2','Metric consistency, denominators and confusion matrices','partial',[V,I,X],'New continuous regression metrics are independently recomputed on identical records; confusion matrices are not applicable to these regression targets.','Recover predictions and confusion matrices for every retained historical classification result or remove unauthenticated tables; reconcile actual class counts.','confusion matrices')
add('R1.3','Missing mean/SD and single-run versus repeated results','partial',[S,I,B],'New seed-level scores, descriptive seed SD and three-seed ensemble scores are separated from conditional bootstrap CIs.','Revise tables and legends; old classification tables still need their own run-level accounting.','mean and standard deviation')
add('R1.4','Hyperparameter and epoch selection','partial',[P,'phase12_outputs/enhancer_phase12_work/results/phase12_validation_choices.csv',V],'Phase 12 records repository defaults, train-only target scaling and validation-only epoch maxima; 24 jobs authenticated.','Explain all older experiment searches and selection rules; disclose asymmetric model recipes and no exhaustive tuning.','hyperparameter selection')
add('R1.5','Relevant literature and suggested papers','open',['reference_inputs/Review From Journal.docx'],'Local experimental results cannot establish literature completeness.','Read the suggested primary papers and current MPRA/CNN/genomic foundation-model literature; correct citations based on relevance.','provided references')
add('R1.6','Typos','open',['reference_inputs/Submitted Manuscript.docx'],'New analysis notebooks do not repair the submitted manuscript.','Proofread the revised manuscript, tables, pseudocode and captions.','6=')
add('R4.1','Novelty and incomplete recent research','partial',[I,X,B,F],'Credible contribution is a reproducible evaluation and selection-bias audit; both specialist CNNs beat DNABERT-2 in this tested setting.','Reframe novelty as evidence, not a new architecture or universal transformer advantage; complete literature comparison.','innovation of this work')
add('R4.2','Sequence-insufficiency/generalizability conclusion is unsupported','partial',[I,X,B,F,D,G],'Strong sequence-only CNN results contradict claiming that these experiments prove sequence intrinsically insufficient. Filtering narrows the target population.','Replace sufficiency/insufficiency proof with bounded empirical claims; removed-sequence performance, additional cells and native context remain unknown.','information in the sequence')
add('R4.3','Longformer long-range claims from only 200 bp; suggested 1–2 kb context','open',[P],'Phase 12 tests 200 bp sequence only; external 200 bp transfer is not a flank-length experiment.','Narrow long-context claims or design a separate future flank study; no extra training launched here.','more flanking context')
add('R4.4','Mutation and k-mer shuffling label noise and reproducibility','partial',[N,P],'Current Phase 12 uses random reverse complement in training and forward/RC averaging in inference. It does not validate earlier mutation or shuffle claims.','Inspect older implemented augmentation and revise prose; remove unimplemented operations, or document their label-preservation limits.','limited base substitution')
add('R4.5','Architecture diagrams and equation layout','open',['reference_inputs/Submitted Manuscript.docx'],'Analytical evidence does not resolve figure and equation formatting.','Redraw architecture/pipeline diagrams and standardize equations after final scope is agreed.','messy layout')
add('R7.1','Recalculate Tables 8/14 and reconcile arithmetic','partial',[V,I,X],'Current saved regression metrics reconcile independently; no claim that historical Tables 8/14 have been repaired.','Recalculate historical classification tables from original predictions; do not copy reviewer arithmetic without verifying actual denominators.','Table 8 and Table')
add('R7.2','Algorithm mismatches actual pipeline','partial',[N,P],'Current data length, strand policy and training recipes are traceable.','Replace obsolete pseudocode and make all main-text/table/diagram settings consistent.','Mismatched algorithm')
add('R7.3','Comparative ROC curves, unreadable figures and conflicting settings','partial',[I,X,S,P],'New comparisons are paired regression results with seed diagnostics; ROC curves belong to the old classification task.','Provide appropriate comparative figures for each task, readable labels and consistent scheduler/epoch settings throughout.','Non-standard and problematic')
add('R7.4','Incorrect DNABERT-2 citation and duplicates','open',['reference_inputs/Review From Journal.docx','reference_inputs/Submitted Manuscript.docx'],'No bibliography edits were made.','Check original DNABERT and DNABERT-2 primary citations and remove duplicate Hu entries.','Non-standard reference citations')
add('R7.5','Spelling and typographical errors','open',['reference_inputs/Submitted Manuscript.docx'],'Original manuscript remains preserved.','Comprehensive language and notation pass.','Widespread spelling')
add('Editor.novelty_limits','Explain novelty and limitations','partial',[V,F,B],'Comparison reverses DNABERT-2 superiority; filter audit defines narrower population, uncertainty is conditional.','Rewrite title/abstract/discussion to reflect final evidence and disclose remaining limitations.','novel aspects and limitations')
add('Editor.graphical_abstract','Graphical abstract and highlights','open',['reference_inputs/Review From Journal.docx'],'Analysis notebooks are not publication graphics or highlights.','Prepare graphics and 3–5 accurate highlights after final evidence review.','graphical abstract is required')
add('Editor.submission_materials','Statements, rebuttal, clean/highlighted files and cover letter','open',['reference_inputs/Review From Journal.docx'],'No journal submission or manuscript revision was made. Letter date is historical; current portal status is unverified.','Authors prepare declarations and submission files, and check current editorial status directly.','Data availability statement')
add('Shafin.result_verification','Integrity, record alignment, metrics, ensembles and conditional uncertainty','addressed' if verified.get('status')=='completed' else 'open',[V,I,X,B,S],'Completed CPU verification if summary status is completed; no model retraining. Old Phase 9C K562 collapsed seed retained and diagnosed, root cause unresolved.','Carry verified tables and limitations into revised manuscript; investigate optimizer cause only in a separately authorized future experiment.','systematic validation')
add('Shafin.filter_population','Retained-versus-removed population and error subgroups','addressed' if filtering.get('status')=='completed' else 'partial',[F,D,A,G],'Raw population reconstructed and retained records matched exactly if Phase 14 summary is completed; removed-sequence prediction performance remains unavailable.','Report selection effects and bounded population claims; do not impute removed-sequence accuracy.','generalizability of the results')
map_df=pd.DataFrame(rows)
# A missing evidence file always downgrades an addressed row and is explicitly listed.
map_df['missing_evidence']=map_df.evidence_files.map(lambda s:' | '.join(p for p in s.split(' | ') if not (ROOT/p).is_file()))
map_df.loc[map_df.missing_evidence.ne('')&map_df.status.eq('addressed'),'status']='partial'
write_table('reviewer_evidence_map.csv',map_df)
print(map_df[['concern_id','status','concern','missing_evidence']].to_string(index=False))
'''),md('''## 3. Claims requiring narrower wording
The suggested statements below are evidence-bound drafts for discussion. They must be integrated with the actual revised manuscript; this notebook does not assume an unpublished revised manuscript exists.'''),code(r'''
claims=[
 ('DNABERT-2 outperforms strong specialist CNNs.','Under the tested 200 bp recipes and frozen splits, MPRA-LegNet and MPRAnn ensembles outperform DNABERT-2 on within-assay Pearson and DHS-natural transfer Spearman.',I+' | '+X),
 ('Primary sequence is insufficient for robust enhancer strength prediction.','Sequence-only models predict part of the measured activity variation in these assays. These experiments do not establish an intrinsic information limit or the benefit of adding epigenomic context.',I),
 ('These results generalize broadly to enhancers.','Results concern filtered 200 bp assayed candidate elements from two cell/design families, with external transfer examined separately. Performance on sequences removed by the legacy-similarity screen is unknown.',F+' | '+A+' | '+X),
 ('The confidence intervals express total model uncertainty.','Paired bootstrap intervals are conditional on trained models and previously examined partitions. Seed variability is reported separately; sequence-wise external resampling is supplemented by a 1 kb region sensitivity analysis.',B+' | '+S),
 ('Longformer captures useful long-range regulatory context here.','The experiments using 200 bp inputs do not test the value of kilobase-scale flanking context.',P),
 ('The old failed seed was fixed.','The earlier Phase 9C K562 DNABERT-2 seed remains nearly constant and is included in the archived three-seed ensemble; the cause is unresolved, with healthy-seed sensitivity reported separately.','phase13_outputs/old_9c_k562_collapse_sensitivity.csv')]
claim_df=pd.DataFrame(claims,columns=['claim_to_avoid','suggested_bounded_wording','evidence_files']);write_table('manuscript_claim_revisions.csv',claim_df)
index=[]
for rel in sorted(set([p for s in map_df.evidence_files for p in s.split(' | ')]+['reference_inputs/Review From Journal.docx','reference_inputs/Submitted Manuscript.docx'])):
 p=ROOT/rel;index.append(dict(relative_path=rel,absolute_path=str(p),exists=p.is_file(),sha256=filehash(p) if p.is_file() else '',bytes=p.stat().st_size if p.is_file() else 0))
write_table('evidence_file_index.csv',pd.DataFrame(index))
lines=['# Reviewer evidence map — Shafin, 2026-10-12','','Status meanings: addressed = technical evidence completed; partial = evidence helps but scientific/manuscript work remains; open = not resolved. No editorial acceptance is implied.','']
for row in map_df.itertuples(index=False):lines.extend([f'## {row.concern_id} — {row.status}',row.concern,'',row.finding,'Remaining: '+row.remaining_action,'Review source paragraphs: '+row.review_paragraphs,'Evidence: '+row.evidence_files,''])
lines.extend(['## Suggested claim revisions',''])
for old,new,ev in claims:lines.extend(['Avoid: '+old,'Suggested: '+new,'Evidence: '+ev,''])
(OUT/'reviewer_evidence_map.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
dump_json('reviewer_map_summary.json',dict(status='completed',phase13_status=verified['status'],phase14_status=filtering['status'],concerns=len(map_df),counts=map_df.status.value_counts().to_dict(),missing_evidence_rows=int(map_df.missing_evidence.ne('').sum()),manuscript_edited=False,journal_contacted=False))
print('Evidence map saved. Remaining scientific and manuscript issues are explicitly open/partial.')
''')]
save(15,C15)
