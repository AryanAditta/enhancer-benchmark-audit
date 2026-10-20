"""Portable pre-writing readiness analyses. Earlier artifacts are read only."""
from pathlib import Path
import os,sys,io,json,hashlib,zipfile,math,re,gc,time,datetime,shutil,itertools,importlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr

ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'phase12_outputs').is_dir() and (p/'reference_inputs').is_dir()),None)
if ROOT is None:raise RuntimeError('Run from the contribution workspace or set the working directory there.')
SRC=ROOT/'reference_inputs/readiness_sources'
P12=ROOT/'phase12_outputs/enhancer_phase12_work'
DATE='2026-10-13'
SEEDS=[20260930,20260931,20260932]
CELLS=['HepG2','K562']
def out(n):
 p=ROOT/f'phase{n}_outputs';p.mkdir(exist_ok=True);return p
def fsha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,name,obj):
 p=out(n)/name;tmp=p.with_suffix(p.suffix+'.writing')
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)
def textfile(n,name,text):(out(n)/name).write_text(text.strip()+'\n',encoding='utf8')
def table(n,name,d):
 d.to_csv(out(n)/name,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)
def read(n,name):return pd.read_csv(out(n)/name,float_precision='round_trip')
def jread(n,name):return json.loads((out(n)/name).read_text(encoding='utf8'))
def member(arc,name):
 p=SRC/arc
 if not p.exists():p=next((ROOT/'phase12_inputs').glob(arc),None)
 if p is None:raise FileNotFoundError(arc)
 with zipfile.ZipFile(p) as z:
  hits=[s for s in z.namelist() if s.split('/')[-1]==name];assert len(hits)==1,(arc,name)
  return z.read(hits[0])
def csv(arc,name):return pd.read_csv(io.BytesIO(member(arc,name)),float_precision='round_trip')
def jmember(arc,name):return json.loads(member(arc,name))
def data():return csv('9_enhancer_phase9a_full_activity_dataset.zip','phase9_full_activity_all_cells.csv')
def regression_metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);assert len(y)==len(p) and np.isfinite(y).all() and np.isfinite(p).all()
 e=p-y;ss=np.dot(y-y.mean(),y-y.mean())
 return dict(n=len(y),pearson=float(pearsonr(y,p).statistic) if np.std(y)>0 and np.std(p)>0 else np.nan,
  spearman=float(spearmanr(y,p).statistic) if np.std(y)>0 and np.std(p)>0 else np.nan,
  rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),r2=float(1-np.dot(e,e)/ss) if ss else np.nan,
  signed_error=float(e.mean()),prediction_sd=float(np.std(p)),target_sd=float(np.std(y)))
def all_internal():
 frames=[csv(f'{tag.upper()}_enhancer_phase{tag}_regression_results.zip',f'phase{tag}_predictions_validation_test.csv').assign(protocol=tag) for tag in ['9b','9c']]
 return pd.concat(frames+[pd.read_csv(P12/'results/phase12_internal_predictions_new_arms.csv')],ignore_index=True)
def models_label(s):
 return {'dnabert2':'DNABERT-2','legnet':'MPRA-LegNet','mprann':'MPRAnn','cnn':'Earlier CNN','spaced6_4_ridge':'Spaced-mer ridge','gc_cpg_ridge':'GC/CpG ridge'}.get(s,s)
def snapshot():
 paths=[p for p in ROOT.glob('[[]*.ipynb') if int(p.name.split(']')[0][1:])<20]
 paths += list((P12/'results').glob('*'))+list((P12/'checkpoints').glob('*.pt'))
 paths += [ROOT/'reference_inputs/Submitted Manuscript.docx',ROOT/'For Shafin.docx']
 return {p.relative_to(ROOT).as_posix():fsha(p) for p in paths if p.is_file()}

def phase20():
 assert Path(sys.prefix).name.lower()=='enhancer','Use the enhancer Anaconda environment.'
 scope={'date':DATE,'choice':'bounded_recipe_comparison','primary':'9B chromosome-held-out Pearson, DNABERT-2 versus MPRA-LegNet per cell',
  'secondary':['9C locus-held-out comparisons','DHS-natural external Spearman','MPRAnn comparison','retained/removed transport','descriptive subgroup/reliability diagnostics'],
  'classification':'Replace unauthenticated submitted performance tables with independently audited grouped OOF results and benchmark-overlap audit; do not transfer old claims.',
  'excluded':['universal CNN superiority','proof of DNA sequence insufficiency','long-context benefit from 200 bp experiments','complete Phase11C repeated-seed classification claim','unverified historical architecture rankings','causal effect of removing the filter without retraining'],
  'training_fairness':'Different frozen recipes and budgets disclosed; no budget-matched superiority claim or new tuning.',
  'longformer':'Historical context only; no retained new performance claim without record-level predictions.',
  'paper_writing':'Prepare evidence/tables/figures/readiness first. Earlier Phase19 text remains a draft, not a finished manuscript.',
  'new_training_authorized_here':False,'scope_set_by':'User requested completion of Aditta checklist; bounded branch selected as the supported option.'}
 dump(20,'frozen_scope.json',scope);dump(20,'protected_before_readiness.json',snapshot())
 claims=[('C1','keep','Results are comparisons under the tested training recipes, cells, data and partitions.'),
 ('C2','keep','Strong specialist CNN baselines alter the observed DNABERT-2 ranking.'),
 ('C3','keep','Benchmark overlap and filtering alter evaluation populations and interpretation.'),
 ('C4','keep_descriptive','Seed spread and marginal interval coverage do not establish individual reliability.'),
 ('C5','exclude','Longformer demonstrates useful long-range regulation here.'),
 ('C6','exclude','These data prove primary DNA sequence is intrinsically insufficient.'),
 ('C7','exclude','Unrecoverable submitted classification table values are validated by newer experiments.'),
 ('C8','exclude','The incomplete Phase11C is a completed multi-seed classification comparison.')]
 table(20,'claim_ledger.csv',pd.DataFrame(claims,columns=['claim_id','decision','claim']))
 tasks=[(1,20,'Scope and claims'),(2,21,'Historical classification'),(3,22,'Artifacts and checkpoint reproduction'),(4,23,'Statistical reporting'),(5,24,'Dataset/filter/threshold documentation'),(6,25,'Publication tables'),(7,26,'Figures and diagrams'),(8,27,'Literature and contribution'),(9,28,'Portable reproducibility'),(10,29,'Final readiness review')]
 table(20,'priority_to_phase_map.csv',pd.DataFrame(tasks,columns=['priority','phase','task']))
 textfile(20,'SCOPE_AND_DECISIONS.md','''# Scope frozen for pre-writing preparation

The retained study is a bounded comparison under the tested training recipes. It combines a corrected grouped classification benchmark/overlap audit with 200 bp MPRA regression, external transfer and filtering/reliability diagnostics. Submitted historical numbers without supporting predictions are omitted and replaced only by explicitly separate authenticated reanalyses. Phase11C completion-dependent claims are excluded. No new long-context or budget-fairness training is needed for this chosen scope.

Primary 9B chromosome-held-out comparisons, secondary 9C locus-held-out and DHS-natural transfer comparisons remain distinct. No paired comparison is constructed between different test populations. The collapsed original seed remains in its archived ensemble; labelled sensitivities remain separate.

Evidence preparation and the final readiness gate precede further manuscript drafting. This decision does not imply universal architecture superiority or sequence insufficiency.''')
 dump(20,'summary.json',{'status':'completed','scope':'bounded','training_launched':False});print('Scope and ten-priority phase map frozen.')

def classification_metrics(y,p):
 from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,matthews_corrcoef,f1_score,brier_score_loss
 y=np.asarray(y,int);p=np.asarray(p,float);label=(p>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,label,labels=[0,1]).ravel()
 return dict(n=len(y),negative=int((y==0).sum()),positive=int((y==1).sum()),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
 roc_auc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),mcc_at_0_5=float(matthews_corrcoef(y,label)),
 balanced_acc_at_0_5=float((tp/(tp+fn)+tn/(tn+fp))/2),accuracy_at_0_5=float((tp+tn)/len(y)),
 f1_at_0_5=float(f1_score(y,label)),sensitivity_at_0_5=float(tp/(tp+fn)),specificity_at_0_5=float(tn/(tn+fp)),brier=float(brier_score_loss(y,p)))

def phase21():
 import xml.etree.ElementTree as E
 a1='1_ enhancer_audit_v1_outputs.zip';a2='2_enhancer_phase2_corrected_results.zip'
 correction=jmember(a2,'correction_report.json');lock=jmember(a2,'input_lock.json')
 for name,h in correction['output_sha256'].items():assert sha(member(a2,name))==h,name
 manifests=[]
 for stage in [1,2]:
  raw=member(a1,f'stage{stage}_candidate_manifest.csv');assert sha(raw)==lock['manifest_sha256'][f'stage{stage}_candidate_manifest.csv']
  m=pd.read_csv(io.BytesIO(raw));assert m.record_id.is_unique
  assert m.groupby('similarity_group_id').fold_5.nunique().max()==1
  # LCS graph components isolate candidate training and historical-test components.
  train_groups=set(m[m.role.eq('train_candidate')].similarity_group_id)
  assert train_groups.isdisjoint(m[m.role.eq('legacy_test')].similarity_group_id)
  manifests.append(m)
 manifest=pd.concat(manifests);pred=csv(a2,'all_oof_predictions.csv')
 assert not pred.duplicated(['stage','model','record_id']).any() and np.isfinite(pred.score).all()
 assert pred.score.between(0,1).all() and pred.config_hash.eq(lock['config_hash']).all()
 merged=pred.merge(manifest[['stage','record_id','label','role','fold_5','similarity_group_id','subtype']],on=['stage','record_id'],suffixes=('','_manifest'),validate='many_to_one')
 for col in ['label','fold_5','similarity_group_id']:assert merged[col].eq(merged[col+'_manifest']).all()
 assert merged.role.eq('train_candidate').all()
 rows=[];folds=[];rosters=[]
 for (stage,model),g in merged.groupby(['stage','model']):
  expected=set(manifest[manifest.stage.eq(stage)&manifest.role.eq('train_candidate')].record_id)
  assert set(g.record_id)==expected
  rows.append(dict(stage=stage,model=model,**classification_metrics(g.label,g.score)))
  for fold,h in g.groupby('fold_5'):folds.append(dict(stage=stage,model=model,fold_5=fold,**classification_metrics(h.label,h.score)))
  rosters.append(dict(stage=stage,model=model,n=len(g),sorted_id_sha256=sha('\n'.join(sorted(g.record_id)).encode())))
 metrics=pd.DataFrame(rows);fm=pd.DataFrame(folds)
 reported=csv(a2,'grouped_oof_metrics.csv');rf=csv(a2,'fold_metrics.csv')
 diffs=[]
 for label,new,old,keys in [('pooled',metrics,reported,['stage','model']),('fold',fm,rf,['stage','model','fold_5'])]:
  pair=new.merge(old,on=keys,suffixes=('_recomputed','_reported'),validate='one_to_one');assert len(pair)==len(old)
  for metric in ['roc_auc','auprc','mcc_at_0_5','balanced_acc_at_0_5','accuracy_at_0_5','f1_at_0_5','sensitivity_at_0_5','specificity_at_0_5','brier']:
   err=float(abs(pair[metric+'_recomputed']-pair[metric+'_reported']).max());assert err<1e-9
   diffs.append(dict(aggregation=label,metric=metric,max_difference=err))
 table(21,'verified_pooled_classification.csv',metrics);table(21,'verified_fold_classification.csv',fm)
 table(21,'classification_metric_checks.csv',pd.DataFrame(diffs));table(21,'identical_classification_records.csv',pd.DataFrame(rosters))
 table(21,'authenticated_classification_predictions.csv.gz',merged.drop(columns=[c for c in merged if c.endswith('_manifest')]))
 # Explicitly retain all-record results and separately report the existing six-record sensitivity.
 ambiguous=manifest[(manifest.stage==1)&manifest.role.eq('train_candidate')].groupby('similarity_group_id').filter(lambda g:g.label.nunique()>1)
 sensitivity=[]
 for model,g in merged[merged.stage.eq(1)].groupby('model'):
  h=g[~g.record_id.isin(ambiguous.record_id)];sensitivity.append(dict(stage=1,model=model,excluded=len(g)-len(h),training_repeated=False,**classification_metrics(h.label,h.score)))
 sens=pd.DataFrame(sensitivity);table(21,'mixed_label_exclusion_sensitivity.csv',sens)
 expected_sens=csv(a2,'stage1_ambiguous_evaluation_exclusion.csv')
 for row in expected_sens.itertuples():
  new=sens[sens.model.eq(row.model)].iloc[0];assert new.excluded==row.excluded_ambiguous_evaluation_records
  assert abs(new.roc_auc-row.roc_auc)<1e-9
 # Extract and disposition EVERY submitted table, without editing the original manuscript.
 ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 with zipfile.ZipFile(ROOT/'reference_inputs/Submitted Manuscript.docx') as z:e=E.fromstring(z.read('word/document.xml'))
 table_rows=[];export_dir=out(21)/'submitted_table_text';export_dir.mkdir(exist_ok=True)
 for i,t in enumerate(e.findall('.//w:body/w:tbl',ns),1):
  cells=[[''.join(n.text or '' for n in c.findall('.//w:t',ns)) for c in row.findall('w:tc',ns)] for row in t.findall('w:tr',ns)]
  content='\n'.join(' | '.join(row) for row in cells);(export_dir/f'table_{i:02}.txt').write_text(content,encoding='utf8')
  if i in [6,7,8,9,10,11,12,14,15]:decision='omit_unverified_performance';why='No matching original record-level predictions/splits for these submitted architectures/tables; new grouped OOF results are separate replacements.'
  elif i in [1]:decision='replace_with_verified_literature_matrix';why='Cross-paper headline scores are not directly comparable; citations assessed in Phase27.'
  elif i in [2,3,4,5]:decision='replace_with_frozen_dataset_model_settings';why='Submitted counts/settings must match retained verified experiments.'
  else:decision='omit_unverified_interpretation';why='No traceable motif attribution/validation supports retaining this table in the frozen scope.'
  table_rows.append(dict(submitted_table=i,rows=len(cells),decision=decision,reason=why,extracted_text=f'submitted_table_text/table_{i:02}.txt'))
 table(21,'submitted_table_dispositions.csv',pd.DataFrame(table_rows))
 # Inventory all original result-like paragraphs/figures, giving unsupported claims an explicit disposition.
 claim_rows=[]
 for i,p in enumerate(e.findall('.//w:body/w:p',ns),1):
  s=''.join(n.text or '' for n in p.findall('.//w:t',ns))
  if s.strip() and (re.search(r'\b(Fig(?:ure)?\.?|Table)\s*\d',s,re.I) or re.search(r'\b(AUC|MCC|accuracy|outperform|insufficien|long.range)\b',s,re.I)):
   claim_rows.append(dict(paragraph=i,text=s,decision='replace_or_omit_original_claim',rule='Retain only new verified evidence and bounded claims in Phase20; no authentication by similarity of reported numbers.'))
 table(21,'submitted_result_claim_dispositions.csv',pd.DataFrame(claim_rows))
 c11=jmember('11C_enhancer_phase11c_results_shard0.zip','phase11c_run_manifest.json')
 assert not c11['complete'];dump(21,'phase11c_disposition.json',{'completed_jobs':len(c11['completed_jobs']),'expected_new_jobs':c11['expected_new_jobs'],'decision':'exclude_completion_dependent_claims; no need to launch missing training for frozen scope'})
 # Historical overlap data retained as an examined secondary audit, never an untouched test.
 for name in ['historical_test_overlap_roster.csv','historical_test_predictions.csv','historical_test_metrics.csv']:
  d=csv('11A_enhancer_phase11a_overlap_impact.zip',name);table(21,name,d)
 dump(21,'summary.json',{'status':'completed_by_audit_and_explicit_exclusions','pooled_results':len(metrics),'fold_results':len(fm),'submitted_tables_dispositioned':len(table_rows),
  'original_table_values_retained':0,'classification_seed_scope':'one original seed; fold SD is descriptive, not retraining SD','phase11c_complete':False,'new_training':False})
 print('Audited',len(metrics),'pooled results and',len(fm),'fold results; dispositioned all',len(table_rows),'submitted tables.')

def runtime_setup():
 r=ROOT/'reference_inputs/readiness_runtime'
 # Bundled wheels are Windows/Python3.12; other systems install requirements locally.
 if sys.platform=='win32' and sys.version_info[:2]==(3,12):sys.path.insert(0,str(r/'vendor'))
 for var,folder in [('HF_HOME','hf_cache'),('HF_MODULES_CACHE','hf_modules'),('TRANSFORMERS_CACHE','hf_cache'),('TEMP','temp'),('TMP','temp')]:
  p=r/folder;p.mkdir(exist_ok=True);os.environ[var]=str(p)
 os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
 return r

def extract_checkpoints():
 rows=[]
 for tag in ['9b','9c']:
  parent=jmember(f'{tag.upper()}_enhancer_phase{tag}_regression_results.zip',f'phase{tag}_provenance.json')
  archive=SRC/f'{tag.upper()}_enhancer_phase{tag}_complete_with_weights.zip';dest=ROOT/'reference_inputs/readiness_checkpoints'/tag;dest.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(archive) as z:
   for name,spec in parent['checkpoints'].items():
    p=dest/name
    if not p.exists():
     with z.open(name) as source,p.open('wb') as target:shutil.copyfileobj(source,target,8*1024*1024)
    assert fsha(p)==spec['sha256'] and p.stat().st_size==spec['bytes']
    rows.append(dict(protocol=tag,name=name,path=p.relative_to(ROOT).as_posix(),sha256=spec['sha256'],bytes=spec['bytes']))
 table(22,'recovered_checkpoint_index.csv',pd.DataFrame(rows));return rows

def architecture_namespace():
 import ast,nbformat,torch
 from torch import nn
 import torch.nn.functional as F
 ns={'torch':torch,'nn':nn,'F':F,'math':math}
 portable=ROOT/'analysis_tools/regression_architectures.py'
 if portable.exists():
  exec(compile(portable.read_text(encoding='utf8'),str(portable),'exec'),ns)
  return ns
 files=[ROOT/'[16] Enhancer_Filtered_Out_Performance.ipynb',SRC/'[9B] Enhancer_Kaggle_Phase9B_Chromosome_Regression.ipynb']
 wanted={'initialize_weights','SELayer','EffBlock','LocalBlock','ResidualConcat','MapperBlock','LegNet','MPRAnn','CNN','Regressor'}
 for p in files:
  n=nbformat.read(p,as_version=4)
  for c in n.cells:
   if c.cell_type!='code':continue
   t=ast.parse(c.source)
   for node in t.body:
    if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in wanted:
     exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
 return ns

def checkpoint_inference():
 import torch
 from transformers import AutoTokenizer
 from transformers.dynamic_module_utils import get_class_from_dynamic_module
 from transformers.models.bert.configuration_bert import BertConfig
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True,warn_only=True)
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');assert device.type=='cuda'
 ns=architecture_namespace();runtime=ROOT/'reference_inputs/readiness_runtime';modeldir=runtime/'dnabert2_ec1f874'
 tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True)
 enc_cls=get_class_from_dynamic_module('bert_layers.BertModel',str(modeldir),local_files_only=True)
 module=importlib.import_module(enc_cls.__module__);module.flash_attn_qkvpacked_func=None
 cfg=BertConfig.from_pretrained(modeldir,local_files_only=True)
 full=data();external=csv('10_enhancer_phase10_external_transfer_results.zip','phase10_external_frozen_set.csv')
 oldext=csv('10_enhancer_phase10_external_transfer_results.zip','phase10_external_predictions.csv')
 oldinternal=all_internal();spec12=json.loads((P12/'results/phase12_provenance.json').read_text())
 newext=pd.read_csv(P12/'results/phase12_external_scores_new_arms.csv')
 cohort=pd.read_csv(ROOT/'phase16_outputs/evaluation_cohort.csv.gz')
 cache=out(22)/'inference_cache';cache.mkdir(exist_ok=True)
 lut=np.zeros(256,np.int64)
 for i,b in enumerate(b'ACGT'):lut[b]=i
 eye=np.eye(4,dtype=np.float32);comp=str.maketrans('ACGT','TGCA')
 def rc(s):return s.translate(comp)[::-1]
 def onehot(seqs):return torch.from_numpy(eye[np.stack([lut[np.frombuffer(s.encode(),np.uint8)] for s in seqs])].transpose(0,2,1).copy()).to(device)
 def infer(net,seqs,family,mean,scale):
  batch=24 if family=='dnabert2' else (1024 if family in ['legnet','mprann'] else 128)
  result=[];net.eval()
  with torch.inference_mode():
   for start in range(0,len(seqs),batch):
    part=seqs[start:start+batch];other=[rc(s) for s in part]
    if family=='dnabert2':
     encode=lambda ss:{k:v.to(device) for k,v in tok(ss,padding=True,add_special_tokens=True,truncation=False,return_tensors='pt').items() if k in ['input_ids','attention_mask']}
     a,b=encode(part),encode(other)
    else:a,b=onehot(part),onehot(other)
    with torch.autocast('cuda',dtype=torch.float16,enabled=family!='cnn'):
     p=net(a).float();q=net(b).float()
    result.append((((p+q)/2)*scale+mean).cpu().numpy())
    if start and start%(batch*100)==0:dump(22,'live_progress.json',{'job':current_job,'partition':current_kind,'completed':start,'total':len(seqs),'mode':'inference only'})
  return np.concatenate(result).astype(float)
 checks=[];metadata=[];removed_predictions=[];all_new=[]
 for tag,cell,family,seed in itertools.product(['9b','9c'],CELLS,['dnabert2','cnn','legnet','mprann'],SEEDS):
  current_job=f'{tag}_{cell}_{family}_seed_{seed}';split='split_9b' if tag=='9b' else 'split_9a'
  parent=jmember(f'{tag.upper()}_enhancer_phase{tag}_regression_results.zip',f'phase{tag}_provenance.json')
  if family in ['dnabert2','cnn']:
   name=f'{cell}_{family}_seed_{seed}_best.pt';path=ROOT/'reference_inputs/readiness_checkpoints'/tag/name;expected=parent['checkpoints'][name]['sha256']
  else:
   name=f'{current_job}_best.pt';path=P12/'checkpoints'/name;expected=spec12['checkpoints'][name]['sha256']
  assert fsha(path)==expected
  ck=torch.load(path,map_location='cpu',weights_only=True)
  if family in ['dnabert2','cnn']:
   assert ck['cell_line']==cell and int(ck['seed'])==seed
   if family=='cnn':assert ck['architecture']=='cnn_v1'
  else:assert ck['job']==current_job
  expected_lock=parent['protocol_sha256'] if family in ['dnabert2','cnn'] else spec12['protocol_sha256'];assert ck['lock']==expected_lock
  tr=full[full.cell_line.eq(cell)&full[split].eq('train')].mean_log2_activity
  if family in ['legnet','mprann']:tr=tr.to_numpy(np.float32)
  else:tr=tr.to_numpy(float)
  assert abs(np.mean(tr)-ck['target_mean'])<1e-6 and abs(np.std(tr)-ck['target_scale'])<1e-6
  if family=='dnabert2':assert ck['revision']=='ec1f874253852eb3907081f57294991b4280ceb6'
  choices=(pd.read_csv(P12/'results/phase12_validation_choices.csv') if family in ['legnet','mprann'] else csv(f'{tag.upper()}_enhancer_phase{tag}_regression_results.zip',f'phase{tag}_validation_choices.csv'))
  choice=choices[choices.cell_line.eq(cell)&choices.model.eq(family if family in ['legnet','mprann'] else f'{family}_seed_{seed}')]
  if family in ['legnet','mprann']:choice=choice[choice.protocol.eq(tag)&choice.seed.eq(seed)]
  assert len(choice)==1 and int(choice.iloc[0].selected_epoch)==ck['epoch']
  kinds=['test','external']+(['removed'] if tag=='9b' and family=='dnabert2' else [])
  need=[]
  for kind in kinds:
   part=(full[full.cell_line.eq(cell)&full[split].eq('test')] if kind=='test' else external if kind=='external' else cohort[cohort.cell_line.eq(cell)&cohort.population.eq('removed')]).reset_index(drop=True)
   file=cache/f'{current_job}_{kind}.csv';meta=file.with_suffix('.json');identity=sha('\n'.join(part.record_id+':'+part.sequence).encode())
   if file.exists() and meta.exists():
    m=json.loads(meta.read_text());assert m['checkpoint']==expected and m['identity']==identity and m['sha256']==fsha(file)
   else:need.append((kind,part,file,meta,identity))
  net=None
  if need:
   if family=='dnabert2':net=ns['Regressor'](enc_cls(cfg))
   elif family=='cnn':net=ns['CNN']()
   elif family=='mprann':net=ns['MPRAnn']()
   else:
    c=spec12['config']['legnet'];net=ns['LegNet'](**{k:c[k] for k in ['in_ch','stem_ch','stem_ks','ef_ks','ef_block_sizes','pool_sizes','resize_factor']})
   net.load_state_dict(ck['state_dict'],strict=True);net.to(device)
   for kind,part,file,meta,identity in need:
    current_kind=kind;start=time.time();p=infer(net,part.sequence.tolist(),family,ck['target_mean'],ck['target_scale'])
    pd.DataFrame({'record_id':part.record_id,'prediction':p}).to_csv(file,index=False,float_format='%.17g')
    meta.write_text(json.dumps({'checkpoint':expected,'identity':identity,'sha256':fsha(file),'seconds':time.time()-start,'device':str(device),'torch':torch.__version__},indent=2),encoding='utf8')
    print(current_job,kind,'inference completed in',round(time.time()-start,1),'s',flush=True)
  for kind in kinds:
   file=cache/f'{current_job}_{kind}.csv';pred=pd.read_csv(file,float_precision='round_trip')
   if kind=='removed':removed_predictions.append(pred.assign(cell_line=cell,model=family,seed=seed,protocol=tag));continue
   if kind=='test':
    old=oldinternal[oldinternal.protocol.eq(tag)&oldinternal.cell_line.eq(cell)&oldinternal.split.eq('test')&oldinternal.model.eq(f'{family}_seed_{seed}')].set_index('record_id');target=old.target
   elif family in ['dnabert2','cnn']:
    old=oldext[oldext.protocol.eq(tag)&oldext.cell_line.eq(cell)&oldext.model.eq(f'{family}_seed_{seed}')].set_index('record_id');target=old.activity
   else:
    old=newext.set_index('record_id')[[f'{current_job}']].rename(columns={current_job:'prediction'});target=external.set_index('record_id')[f'{cell}_l2fc']
   assert set(pred.record_id)==set(old.index)
   new=pred.set_index('record_id').loc[old.index,'prediction'];delta=abs(new-old.prediction)
   oldm=regression_metrics(target,old.prediction);newm=regression_metrics(target,new)
   # Floating-point portability is reported separately from artifact integrity.
   collapsed=oldm['prediction_sd']<1e-4*oldm['target_sd']
   passed=bool(delta.max()<.03 and delta.mean()<.002 and (collapsed or abs(oldm['pearson']-newm['pearson'])<.003))
   checks.append(dict(protocol=tag,cell_line=cell,model=family,seed=seed,partition=kind,n=len(new),max_abs_difference=float(delta.max()),mean_abs_difference=float(delta.mean()),
     archived_pearson=oldm['pearson'],reproduced_pearson=newm['pearson'],pearson_difference=newm['pearson']-oldm['pearson'],
     near_constant_archived=collapsed,archived_prediction_sd=oldm['prediction_sd'],reproduced_prediction_sd=newm['prediction_sd'],
     correlation_tolerance_applied=not collapsed,passed=passed))
   all_new.append(pred.assign(protocol=tag,cell_line=cell,model=family,seed=seed,partition=kind))
  metadata.append(dict(protocol=tag,cell_line=cell,model=family,seed=seed,selected_epoch=ck['epoch'],target_mean=ck['target_mean'],target_scale=ck['target_scale'],checkpoint_sha256=expected))
  table(22,'checkpoint_prediction_reproduction.csv',pd.DataFrame(checks));table(22,'checkpoint_metadata.csv',pd.DataFrame(metadata))
  dump(22,'live_progress.json',{'finished_checkpoint_jobs':len(metadata),'expected':48,'last_job':current_job,'training':False})
  del net,ck;gc.collect();torch.cuda.empty_cache()
 table(22,'reproduced_neural_predictions.csv.gz',pd.concat(all_new,ignore_index=True))
 table(22,'dnabert2_removed_seed_predictions.csv.gz',pd.concat(removed_predictions,ignore_index=True))
 assert pd.DataFrame(checks).passed.all(), 'Review numerical reproduction deviations before freezing.'
 return metadata,checks

def phase22():
 runtime_setup();extract_checkpoints()
 source=ROOT/'reference_inputs/readiness_runtime/dnabert2_ec1f874'
 dump(22,'pinned_model_source_hashes.json',{p.name:fsha(p) for p in source.iterdir() if p.is_file()})
 metadata,checks=checkpoint_inference()
 # Add recovered transformer removed-population predictions to the preserved Phase16 CNN cohort.
 p=read(22,'dnabert2_removed_seed_predictions.csv.gz')
 e=p.groupby(['record_id','cell_line'],as_index=False).agg(prediction=('prediction','mean'),seed_sd=('prediction','std'),seeds=('seed','nunique'))
 cohort=pd.read_csv(ROOT/'phase16_outputs/evaluation_cohort.csv.gz')
 e=e.merge(cohort,on=['record_id','cell_line'],validate='one_to_one').assign(model='dnabert2')
 cnn=pd.read_csv(ROOT/'phase16_outputs/ensemble_predictions_and_metadata.csv.gz')
 local=read(22,'reproduced_neural_predictions.csv.gz')
 local=local[local.protocol.eq('9b')&local.partition.eq('test')&local.model.eq('dnabert2')]
 ret=local.groupby(['record_id','cell_line'],as_index=False).agg(prediction=('prediction','mean'),seed_sd=('prediction','std'),seeds=('prediction','size'))
 ret=ret.merge(cohort,on=['record_id','cell_line'],validate='one_to_one').assign(model='dnabert2')
 combined=pd.concat([cnn,e,ret],ignore_index=True);assert not combined.duplicated(['record_id','model']).any()
 assert combined.groupby(['cell_line','record_id']).model.nunique().eq(3).all()
 table(22,'complete_three_model_filtering_predictions.csv.gz',combined)
 metrics=[]
 for analysis,flag in [('primary','eligible_primary'),('31mer_sensitivity','eligible_31mer_sensitivity')]:
  for (cell,pop,model),g in combined[combined[flag]].groupby(['cell_line','population','model']):metrics.append(dict(analysis=analysis,cell_line=cell,population=pop,model=model,**regression_metrics(g.mean_log2_activity,g.prediction)))
 table(22,'complete_three_model_filtering_metrics.csv',pd.DataFrame(metrics))
 dump(22,'summary.json',{'status':'completed','authenticated_recovered_checkpoints':24,'reproduced_neural_checkpoints':len(metadata),'reproduced_partition_jobs':len(checks),
  'dnabert2_removed_complete':True,'training_launched':False,'runtime_dependencies':'isolated local vendor directory; enhancer executable unchanged',
  'old_epoch_limit':'Selected epochs/validation scores verified; historical per-epoch maxima cannot be proved without missing logs.',
  'execution_location':'Local enhancer GPU; reproduces archived Kaggle predictions, no Kaggle retraining required'})
 print('All 48 neural checkpoints reproduced; transformer removed-population arm complete.')

def population_bootstrap(n,scored):
 metric_rows=[];interval_rows=[];contrasts=[]
 for analysis,flag in [('primary','eligible_primary'),('31mer_sensitivity','eligible_31mer_sensitivity')]:
  for cell,allg in scored[scored[flag]].groupby('cell_line'):
   keys=sorted(allg.resampling_group.unique());rng=np.random.default_rng(2026102313)
   moments={};points={};draws={}
   for (pop,model),g in allg.groupby(['population','model']):
    points[pop,model]=regression_metrics(g.mean_log2_activity,g.prediction)
    metric_rows.append(dict(analysis=analysis,cell_line=cell,population=pop,model=model,**points[pop,model]))
    y=g.mean_log2_activity.to_numpy();p=g.prediction.to_numpy()
    v=pd.DataFrame({'group':g.resampling_group,'n':1.,'y':y,'p':p,'yy':y*y,'pp':p*p,'yp':y*p,'ae':abs(p-y),'se':(p-y)**2})
    moments[pop,model]=v.groupby('group').sum().reindex(keys,fill_value=0).to_numpy();draws[pop,model]=[]
   for start in range(0,1000,50):
    weights=np.stack([np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys)) for _ in range(50)])
    for key,mom in moments.items():
     count,y,p,yy,pp,yp,ae,se=(weights@mom).T
     correlation=(yp-y*p/count)/np.sqrt(np.maximum((yy-y*y/count)*(pp-p*p/count),0))
     draws[key].append(np.column_stack([correlation,np.sqrt(se/count),ae/count]))
   draws={key:np.concatenate(v) for key,v in draws.items()}
   for key,v in draws.items():
    for j,metric in enumerate(['pearson','rmse','mae']):
     valid=np.isfinite(v[:,j]);lo,hi=np.quantile(v[valid,j],[.025,.975])
     interval_rows.append(dict(analysis=analysis,cell_line=cell,population=key[0],model=key[1],metric=metric,estimate=points[key][metric],lower_95=lo,upper_95=hi,undefined_draws=int((~valid).sum()),repeats=1000))
   pairs=[]
   for model in ['dnabert2','legnet','mprann']:pairs.append((('removed',model),('retained',model),'removed minus retained',model))
   for pop in ['retained','removed']:
    for a,b in [('dnabert2','legnet'),('dnabert2','mprann'),('legnet','mprann')]:pairs.append(((pop,a),(pop,b),f'{a} minus {b}',pop))
   for a,b,label,pop in pairs:
    for j,metric in enumerate(['pearson','rmse','mae']):
     diff=draws[a][:,j]-draws[b][:,j];valid=np.isfinite(diff);lo,hi=np.quantile(diff[valid],[.025,.975])
     contrasts.append(dict(analysis=analysis,cell_line=cell,contrast=label,population_or_model=pop,metric=metric,estimate=points[a][metric]-points[b][metric],lower_95=lo,upper_95=hi,repeats=1000,
       undefined_draws=int((~valid).sum()),resampling_unit='1 kb genomic groups unioned for exact/RC identity',scope='conditional on fixed models and examined partitions; unadjusted exploratory'))
 table(n,'filtering_metrics.csv',pd.DataFrame(metric_rows));table(n,'filtering_metric_intervals.csv',pd.DataFrame(interval_rows));table(n,'filtering_paired_contrasts.csv',pd.DataFrame(contrasts))

def phase23():
 assert jread(22,'summary.json')['status']=='completed'
 pred=all_internal();full=data();rows=[];ids=[]
 for (tag,cell,split,model),g in pred.groupby(['protocol','cell_line','split','model']):
  reference=full[full.cell_line.eq(cell)&full['split_9b' if tag=='9b' else 'split_9a'].eq(split)].set_index('record_id')
  assert set(reference.index)==set(g.record_id)
  actual=g.set_index('record_id').loc[reference.index];assert np.max(abs(actual.target-reference.mean_log2_activity))<1e-7
  rows.append(dict(protocol=tag,cell_line=cell,split=split,model=model,**regression_metrics(actual.target,actual.prediction)))
  ids.append(dict(protocol=tag,cell_line=cell,split=split,model=model,n=len(g),ids_sha256=sha('\n'.join(sorted(g.record_id)).encode())))
 internal=pd.DataFrame(rows);table(23,'internal_metrics_authoritative.csv',internal);table(23,'evaluation_population_ledger.csv',pd.DataFrame(ids))
 seed=internal[internal.model.str.contains(r'_seed_\d+$',regex=True)].copy();seed['family']=seed.model.str.replace(r'_seed_\d+$','',regex=True)
 sd=seed.groupby(['protocol','cell_line','split','family']).agg(n_seeds=('model','size'),mean_seed_pearson=('pearson','mean'),sd_seed_pearson=('pearson','std'),mean_seed_rmse=('rmse','mean'),sd_seed_rmse=('rmse','std'),min_prediction_sd=('prediction_sd','min')).reset_index()
 table(23,'individual_seed_metrics.csv',seed);table(23,'seed_mean_SD.csv',sd)
 table(23,'ensemble_metrics.csv',internal[internal.model.str.endswith('_three_seed_mean')])
 # External frozen set and every model/origin stay separate.
 ext=csv('10_enhancer_phase10_external_transfer_results.zip','phase10_external_frozen_set.csv').set_index('record_id')
 old=csv('10_enhancer_phase10_external_transfer_results.zip','phase10_external_predictions.csv');new=pd.read_csv(P12/'results/phase12_external_scores_new_arms.csv').set_index('record_id')
 scores={}
 for (tag,cell,model),g in old.groupby(['protocol','cell_line','model']):
  g=g.set_index('record_id').loc[ext.index];assert np.max(abs(g.activity-ext[f'{cell}_l2fc']))<1e-8;scores[tag,cell,model]=g.prediction.to_numpy()
 for tag,cell,family in itertools.product(['9b','9c'],CELLS,['legnet','mprann']):
  for arm in [f'{family}_seed_{s}' for s in SEEDS]+[f'{family}_three_seed_mean']:scores[tag,cell,arm]=new.loc[ext.index,f'{tag}_{cell}_{arm}'].to_numpy()
 external=[]
 for (tag,cell,model),p in scores.items():
  for origin,g in ext.groupby('origin'):
   ix=ext.index.get_indexer(g.index);external.append(dict(protocol=tag,cell_line=cell,origin=origin,model=model,**regression_metrics(g[f'{cell}_l2fc'],p[ix])))
 table(23,'external_metrics_authoritative.csv',pd.DataFrame(external));table(23,'external_population_counts.csv',ext.groupby('origin').size().rename('n').reset_index())
 # Original study intervals and independently repeated intervals are not merged/averaged.
 original=[]
 for cohort,file,metric in [('internal_test','phase12_internal_paired_bootstrap.csv','pearson'),('DHS_natural','phase12_external_paired_bootstrap.csv','spearman')]:
  x=pd.read_csv(P12/'results'/file);x['cohort']=cohort;x['metric']=metric;x['interval_source']='original_phase12';original.append(x)
 table(23,'original_study_intervals.csv',pd.concat(original,ignore_index=True))
 verified=pd.read_csv(ROOT/'phase13_outputs/independent_paired_bootstrap.csv');verified['interval_source']='independent_phase13';verified['metric']=np.where(verified.cohort.eq('internal_test'),'pearson','spearman')
 dhs=ext[ext.origin.eq('DHS_natural')];region_count=0
 for chrom,g in dhs.sort_values(['external_chrom','external_start','external_end']).groupby('external_chrom'):
  boundary=-10**15
  for row in g.itertuples():
   if row.external_start>boundary+1000:region_count+=1
   boundary=max(boundary,row.external_end)
 n_records=[];n_units=[]
 for row in verified.itertuples():
  if row.cohort=='internal_test':
   reference=full[full.cell_line.eq(row.cell_line)&full['split_9b' if row.protocol=='9b' else 'split_9a'].eq('test')]
   n_records.append(len(reference));n_units.append(reference.locus_group_id.nunique())
  else:n_records.append(len(dhs));n_units.append(region_count if row.unit=='region_1kb' else len(dhs))
 verified['n_records']=n_records;verified['n_resampling_units']=n_units;verified['undefined_draws']=0
 verified['comparison_role']=np.where(verified.cohort.eq('internal_test')&verified.protocol.eq('9b')&verified.contrast.eq('dnabert2_three_seed_mean minus legnet_three_seed_mean'),'primary','secondary')
 verified['interval_scope']='conditional on fixed trained models and previously examined cohorts; exploratory secondary comparisons'
 table(23,'independent_verification_intervals.csv',verified)
 population_bootstrap(23,read(22,'complete_three_model_filtering_predictions.csv.gz'))
 classification=read(21,'verified_fold_classification.csv')
 table(23,'classification_fold_mean_SD.csv',classification.groupby(['stage','model'])[['roc_auc','auprc','accuracy_at_0_5','mcc_at_0_5']].agg(['mean','std']).reset_index().pipe(lambda x:x.set_axis(['_'.join(str(s) for s in c if s) if isinstance(c,tuple) else c for c in x.columns],axis=1)))
 for name in ['phase11b_corrected_fold_tests.csv','phase11b_simultaneous_conditional_intervals.csv']:
  table(23,name,csv('11B_enhancer_phase11b_statistics_calibration.zip',name))
 table(23,'collapsed_seed_sensitivity.csv',pd.read_csv(ROOT/'phase13_outputs/old_9c_k562_collapse_sensitivity.csv'))
 rules={'primary':'9B chromosome-test Pearson DNABERT-2 minus MPRA-LegNet, per cell; declared for Phase12 on previously examined partitions',
 'secondary':'MPRAnn, 9C, external and filtering/reliability analyses reported separately, unadjusted unless original Phase11B explicitly uses simultaneous/Holm correction',
 'seed_SD':'Across three fitted seeds; ensemble performance is a different quantity. Classification fold SD is not seed SD.',
 'uncertainty':'All bootstrap intervals condition on fitted models/partitions. Genomic/identity clusters or external region sensitivity address specified dependence, not all homology.',
 'interval_versions':'Original Phase12 and independent Phase13 intervals retained separately with source labels; no averaging.',
 'split_effect':'9B and 9C test populations differ; no paired split-effect estimate is computed.',
 'undefined':'Constant vectors yield NaN correlations. Keep collapsed seed in archived primary ensemble; sensitivity explicitly separate.',
 'GPU_reproduction':'Archived main benchmark remains the authoritative frozen calculation; local re-inference is an audit. New retained/removed DNABERT2 comparisons use the same local backend for both populations.'}
 dump(23,'statistical_reporting_rules.json',rules)
 dump(23,'summary.json',{'status':'completed','internal_metric_rows':len(internal),'external_metric_rows':len(external),'independent_intervals':len(verified),'training_launched':False})
 print('Authoritative metrics, seeds, populations and separate interval versions saved.')

def phase24():
 assert jread(22,'summary.json')['status']=='completed'
 full=data();pop=pd.read_csv(ROOT/'phase14_outputs/reconstructed_filtering_population.csv.gz',float_precision='round_trip')
 raw=jread(14,'raw_row_exclusion_audit.json');a=jmember('9_enhancer_phase9a_full_activity_dataset.zip','phase9a_audit_and_provenance.json')
 for cell,g in raw.items():
  for key,v in g.items():assert a['assay_audit'][cell][key]==v
 original_hashes=jread(14,'population_reconstruction_status.json')['source_hashes']
 for name,spec in original_hashes.items():assert fsha(ROOT/'reference_inputs/raw_sources'/name)==spec['sha256']
 retained=pop[pop.population.eq('retained')].set_index('record_id').sort_index();reference=full.set_index('record_id').sort_index()
 assert retained.index.equals(reference.index) and retained.sequence.equals(reference.sequence)
 assert np.max(abs(retained.mean_log2_activity-reference.mean_log2_activity))<1e-9
 stages=[]
 for cell in CELLS:
  r=raw[cell];sub=pop[pop.cell_line.eq(cell)]
  for key,label,unit in [('rows_total','raw_assay_rows','rows'),('other_design_name','other_design_exclusions','rows'),('noncanonical_chrom','noncanonical_chrom_exclusions','rows'),('non_200bp','non200_exclusions','rows'),('candidate_rows','eligible_orientation_rows','rows'),('incomplete_or_ambiguous_orientation','incomplete_orientation_exclusions','rows'),('complete_two_orientation_elements','complete_paired_candidates','elements')]:
   stages.append(dict(cell_line=cell,stage=label,n=int(r.get(key,0)),unit=unit,source='authenticated raw-row audit'))
  for reason,g in sub.groupby('filter_reason'):stages.append(dict(cell_line=cell,stage=reason,n=len(g),unit='elements',source='reconstructed filtering population'))
 table(24,'dataset_attrition_counts.csv',pd.DataFrame(stages));table(24,'internal_split_counts.csv',full.groupby(['cell_line','split_9b','split_9a']).size().rename('n').reset_index())
 thresholds=[];precision=[];assignments=[];features=full.copy()
 features['cpg_count_integer']=features.sequence.str.count('CG');features['gc_fraction']=(features.sequence.str.count('G')+features.sequence.str.count('C'))/200
 features['max_homopolymer']=features.sequence.map(lambda s:max(len(list(g)) for _,g in itertools.groupby(s)))
 old=pd.read_csv(ROOT/'phase14_outputs/training_only_subgroup_thresholds.csv')
 for tag,cell in itertools.product(['9b','9c'],CELLS):
  split='split_9b' if tag=='9b' else 'split_9a';tr=features[features.cell_line.eq(cell)&features[split].eq('train')];te=features[features.cell_line.eq(cell)&features[split].eq('test')].copy()
  q20,q80=tr.mean_log2_activity.quantile([.2,.8]);count=float(tr.cpg_count_integer.median());density=count/199
  thresholds.append(dict(protocol=tag,cell_line=cell,training_n=len(tr),training_activity_q20=q20,training_activity_q80=q80,training_cpg_median=density,training_cpg_count_threshold=count,
    activity_rule='low <= q20; middle q20 < y <= q80; high > q80',cpg_rule='integer CG count <= training median count'))
  saved=old[old.protocol.eq(tag)&old.cell_line.eq(cell)].iloc[0]
  exact=te.cpg_count_integer<=count;rounded=te.cpg_count_integer.div(199)<=saved.training_cpg_median
  precision.append(dict(protocol=tag,cell_line=cell,old_export_density=saved.training_cpg_median,exact_density=density,exact_count_threshold=count,old_export_assignment_changes=int((exact!=rounded).sum())))
  te['protocol']=tag;te['cpg_group']=np.where(exact,'at_or_below_training_median','above_training_median')
  te['activity_group']=np.where(te.mean_log2_activity<=q20,'low',np.where(te.mean_log2_activity<=q80,'middle','high'))
  te['gc_group']=pd.cut(te.gc_fraction,[0,.35,.45,.55,.65,1],include_lowest=True).astype(str)
  te['homopolymer_group']=np.where(te.max_homopolymer>=6,'at_least_6','below_6');assignments.append(te)
 threshold=pd.DataFrame(thresholds);table(24,'training_only_thresholds_exact.csv',threshold)
 restored=read(24,'training_only_thresholds_exact.csv')
 for col in ['training_activity_q20','training_activity_q80','training_cpg_median','training_cpg_count_threshold']:assert np.array_equal(threshold[col].to_numpy(),restored[col].to_numpy()),col
 assignment=pd.concat(assignments,ignore_index=True);table(24,'retained_test_subgroup_assignments.csv.gz',assignment)
 table(24,'threshold_precision_audit.csv',pd.DataFrame(precision))
 # Recompute authoritative subgroups using stable integer CpG comparisons and explicit activity boundary rules.
 pred=all_internal();subgroup=[]
 for (tag,cell),te in assignment.groupby(['protocol','cell_line']):
  for model in ['dnabert2_three_seed_mean','legnet_three_seed_mean','mprann_three_seed_mean']:
   p=pred[pred.protocol.eq(tag)&pred.cell_line.eq(cell)&pred.split.eq('test')&pred.model.eq(model)].set_index('record_id')
   g=te.set_index('record_id').join(p[['prediction']],validate='one_to_one');assert g.prediction.notna().all()
   for family in ['gc_group','cpg_group','homopolymer_group','activity_group','chrom']:
    for label,h in g.groupby(family):
     row=dict(protocol=tag,cell_line=cell,model=model,family=family,subgroup=label,n=len(h),suppressed=len(h)<100)
     if len(h)>=100:row.update(regression_metrics(h.mean_log2_activity,h.prediction))
     subgroup.append(row)
 table(24,'retained_test_subgroup_metrics_exact.csv',pd.DataFrame(subgroup))
 # Full three-model population groups use identical exact boundaries.
 scored=read(22,'complete_three_model_filtering_predictions.csv.gz');groups=[];newsub=[]
 for cell,g in scored.groupby('cell_line'):
  t=threshold[threshold.protocol.eq('9b')&threshold.cell_line.eq(cell)].iloc[0];g=g.copy()
  g['cpg_count_integer']=g.sequence.str.count('CG');g['cpg_group']=np.where(g.cpg_count_integer<=t.training_cpg_count_threshold,'at_or_below_training_median','above_training_median')
  g['activity_group']=np.where(g.mean_log2_activity<=t.training_activity_q20,'low',np.where(g.mean_log2_activity<=t.training_activity_q80,'middle','high'))
  g['gc_group']=pd.cut(g.gc_fraction,[0,.35,.45,.55,.65,1],include_lowest=True).astype(str);g['homopolymer_group']=np.where(g.max_homopolymer>=6,'at_least_6','below_6');groups.append(g)
  for (population,model),part in g[g.eligible_primary].groupby(['population','model']):
   for family in ['gc_group','cpg_group','homopolymer_group','activity_group','chrom']:
    for label,h in part.groupby(family):
     row=dict(cell_line=cell,population=population,model=model,family=family,subgroup=label,n=len(h),suppressed=len(h)<100)
     if len(h)>=100:row.update(regression_metrics(h.mean_log2_activity,h.prediction))
     newsub.append(row)
 grouped=pd.concat(groups,ignore_index=True)
 table(24,'complete_population_subgroup_assignments.csv.gz',grouped);table(24,'complete_population_subgroup_metrics_exact.csv',pd.DataFrame(newsub))
 reliability_three_models(grouped)
 external=csv('10_enhancer_phase10_external_transfer_results.zip','phase10_external_frozen_set.csv')
 table(24,'external_origin_counts.csv',external.groupby('origin').size().rename('n').reset_index())
 separation=jread(13,'external_sequence_separation_audit.json');assert separation['exact_or_rc_matches_to_phase9']==0 and separation['shared_canonical_31mer_records']==0
 dump(24,'external_selection_and_separation.json',separation)
 textfile(24,'DATASET_AND_THRESHOLD_NOTES.md','''# Dataset population and reproducible subgroup definitions

Raw rows, eligible orientation rows and complete paired elements are different units and are labelled separately in the attrition table/diagram. The complete-pair population contains 149,632 candidates; 72,799 (48.65%) were excluded by the legacy screen, mostly shared 15-mers. These are not all full-sequence duplicates. Retained activity and external transfer cohorts remain separate.

Earlier Phase14 uses exact in-memory thresholds but its 12-digit density export can misassign K562 records when reloaded. The new authoritative export stores 17 significant digits with round-trip parsing and, more robustly, uses integer CG counts for CpG groups. Activity boundaries are now explicit and shared across final retained/removed subgroup tables. Earlier artifacts remain preserved as historical analyses; the final publication subgroup tables use this corrected version. Whole-test model scores are unaffected by subgroup relabelling.

Removed-candidate performance is now available for DNABERT-2 and both specialist CNNs on the frozen test chromosomes; other removed candidates are not independent test examples for this protocol. No causal benefit of filtering or genome-wide generalization is inferred. External exact/RC, 31-mer and natural-coordinate separation checks address specified overlap, not all homology.''')
 dump(24,'summary.json',{'status':'completed','raw_population_reconciled':True,'threshold_roundtrip_exact':True,'integer_CpG_definition':True,'old_export_changes_K562_9b':1844,'old_export_changes_K562_9c':1226,
  'retained_subgroup_rows':len(subgroup),'population_subgroup_rows':len(newsub),'previous_artifacts_modified':False})
 print('Dataset flow reconciled; exact subgroup exports and integer CpG assignments verified.')

def reliability_three_models(scored):
 parameters=[];coverage=[];subgroups=[];association=[];selective=[]
 metadata=read(22,'checkpoint_metadata.csv')
 for (cell,model),allg in scored[scored.eligible_primary].groupby(['cell_line','model']):
  retained_cal=allg[allg.population.eq('retained')&allg.followup_partition.eq('calibration')]
  cuts={q:float(retained_cal.seed_sd.quantile(q)) for q in [.25,.5,.75]}
  for pop,g in allg[allg.followup_partition.eq('audit')].groupby('population'):
   error=abs(g.prediction-g.mean_log2_activity)
   association.append(dict(cell_line=cell,model=model,population=pop,n=len(g),spearman_seed_sd_absolute_error=float(spearmanr(g.seed_sd,error).statistic)))
   for q,cut in [*cuts.items(),(1.,np.inf)]:
    h=g[g.seed_sd<=cut];selective.append(dict(cell_line=cell,model=model,population=pop,calibration_quantile=q,accepted_n=len(h),audit_n=len(g),accepted_fraction=len(h)/len(g),mae=float(abs(h.prediction-h.mean_log2_activity).mean())))
  scale=float(metadata[metadata.protocol.eq('9b')&metadata.cell_line.eq(cell)&metadata.model.eq(model)].target_scale.iloc[0]);floor=.05*scale
  for source,method in itertools.product(['retained','removed'],['constant_width','seed_scaled']):
   cal=allg[allg.population.eq(source)&allg.followup_partition.eq('calibration')]
   den=np.ones(len(cal)) if method=='constant_width' else np.maximum(cal.seed_sd,floor)
   score=abs(cal.prediction-cal.mean_log2_activity)/den;k=min(len(score),math.ceil((len(score)+1)*.9));q=float(np.sort(score)[k-1])
   parameters.append(dict(cell_line=cell,model=model,calibration_source=source,method=method,n_calibration=len(cal),quantile=q,seed_sd_floor=floor,nominal_coverage=.9))
   for pop,g in allg[allg.followup_partition.eq('audit')].groupby('population'):
    assert set(cal.resampling_group).isdisjoint(g.resampling_group)
    half=q*(np.ones(len(g)) if method=='constant_width' else np.maximum(g.seed_sd,floor))
    h=g.copy();h['covered']=abs(h.prediction-h.mean_log2_activity)<=half;h['width']=2*half
    coverage.append(dict(cell_line=cell,model=model,calibration_source=source,audit_population=pop,method=method,n=len(h),nominal_coverage=.9,empirical_coverage=float(h.covered.mean()),mean_width=float(h.width.mean())))
    if source=='retained':
     edges=retained_cal.prediction.quantile([.2,.4,.6,.8]).to_numpy();h['prediction_quintile']=np.searchsorted(edges,h.prediction,side='right')+1
     for family in ['activity_group','prediction_quintile','gc_group','cpg_group','homopolymer_group']:
      for band,sub in h.groupby(family):
       r=dict(cell_line=cell,model=model,population=pop,method=method,family=family,subgroup=str(band),n=len(sub),suppressed=len(sub)<100)
       if len(sub)>=100:r.update(empirical_coverage=float(sub.covered.mean()),mean_width=float(sub.width.mean()),signed_error=float((sub.prediction-sub.mean_log2_activity).mean()))
       subgroups.append(r)
 for name,rows in [('reliability_parameters.csv',parameters),('reliability_coverage.csv',coverage),('reliability_subgroups.csv',subgroups),('reliability_seed_association.csv',association),('reliability_selective_errors.csv',selective)]:table(24,name,pd.DataFrame(rows))

def model_settings():
 prov=json.loads((P12/'results/phase12_provenance.json').read_text());p9=jmember('9B_enhancer_phase9b_regression_results.zip','phase9b_provenance.json')['config']
 rows=[dict(task='activity_regression',model='DNABERT-2',architecture='117M-parameter BPE DNA Transformer + masked-mean linear regressor',initialization='pinned multispecies-pretrained backbone',epochs_max=p9['dnabert2_epochs'],batch=p9['dnabert2_batch'],optimizer='AdamW',learning_rate=p9['dnabert2_lr'],selection='highest validation Pearson',seeds=3,strand='random training orientation; forward/RC average at inference',notes='200 bp; no adapter context; different budget from CNNs'),
 dict(task='activity_regression',model='Earlier CNN',architecture='3 Conv1D layers (64,96,96), mean/max pool, MLP',initialization='random',epochs_max=p9['cnn_epochs'],batch=p9['cnn_batch'],optimizer='AdamW',learning_rate=p9['cnn_lr'],selection='highest validation Pearson',seeds=3,strand='deterministic oriented training rows; forward/RC inference',notes='train-only target scaling'),
 dict(task='activity_regression',model='MPRA-LegNet',architecture='stem 64/11; EffBlock sizes 80,96,112,128; kernel9; four pool2 stages',initialization='random',epochs_max=20,batch=1024,optimizer='AdamW+OneCycleLR',learning_rate=.01,selection='highest validation Pearson',seeds=3,strand='random training orientation; forward/RC inference',notes='weight decay .1; clip1; no assay adapters, shifts or reverse channel'),
 dict(task='activity_regression',model='MPRAnn',architecture='Conv1D 250/250/250/100; dense300/200; PyTorch port',initialization='random',epochs_max=100,batch=128,optimizer='Adam',learning_rate=.001,selection='validation Pearson; patience10',seeds=3,strand='random training orientation; forward/RC inference',notes='Keras port verified; original source stopping criterion differs; no clipping'),
 dict(task='activity_regression',model='GC/CpG ridge',architecture='GC and CpG covariates; ridge regression',initialization='fit on training only',epochs_max=None,batch=None,optimizer='ridge',learning_rate=None,selection='validation Pearson alpha selection',seeds=1,strand='strand invariant',notes='original frozen alphas; no test-driven refit'),
 dict(task='activity_regression',model='Spaced-mer ridge',architecture='spaced6_4 features; ridge regression',initialization='fit on training only',epochs_max=None,batch=None,optimizer='ridge',learning_rate=None,selection='validation Pearson alpha selection',seeds=1,strand='strand invariant',notes='original frozen alphas; preprocessing trained only on training rows')]
 config=jmember('2_enhancer_phase2_corrected_results.zip','input_lock.json')['config']
 for model in read(21,'verified_pooled_classification.csv').model.unique():
  fine=model.startswith('finetuned');frozen=model.startswith('frozen')
  description='768-dimensional backbone masked-mean + StandardScaler/logistic regression' if model=='frozen_gena_lm' else 'declared model/feature arm; original wrapper source unavailable'
  rows.append(dict(task='grouped_classification',model=model,architecture=description,initialization='pinned pretrained backbone' if fine or frozen else 'training-fold fit',epochs_max=config['finetune_epochs'] if fine else config['cnn_epochs'] if model=='small_cnn' else None,batch=config['finetune_batch'] if fine else config['cnn_batch'] if model=='small_cnn' else None,
   optimizer='logistic regression' if frozen or model.endswith('logreg') else 'not captured in supplied original metadata',learning_rate=config['finetune_lr'] if fine else config['cnn_lr'] if model=='small_cnn' else None,selection='declared fixed recipe, grouped 5-fold OOF; threshold0.5',seeds=1,strand='training-fold RC augmentation reported by authenticated run metadata',notes='GENA correction implementation available; C1/max_iter2000/lbfgs' if model=='frozen_gena_lm' else 'Predictions/config/revisions authenticated; original training wrapper, exact optimizer and loss not independently recoverable; no exact retraining claim'))
 for row in rows:
  model=row['model'];fine=model.startswith('finetuned')
  row['gradient_accumulation']=2 if model=='DNABERT-2' or fine else 1 if row['batch'] is not None else None
  row['effective_batch']=row['batch']*row['gradient_accumulation'] if row['batch'] is not None else None
  row['evaluation_strand']='forward/RC average' if row['task']=='activity_regression' and row['seeds']==3 else 'forward only' if row['task']=='grouped_classification' else 'strand invariant'
  row['loss']='MSE on training-only z-scored targets' if row['task']=='activity_regression' and row['seeds']==3 else 'not captured in supplied original metadata' if fine or model=='small_cnn' else 'ridge loss' if row['task']=='activity_regression' else 'logistic loss'
  if model in ['DNABERT-2','Earlier CNN']:row['weight_decay']=.01;row['gradient_clip_norm']=1.;row['scheduler']='none'
  elif model=='MPRA-LegNet':row['weight_decay']=.1;row['gradient_clip_norm']=1.;row['scheduler']='OneCycleLR max_lr0.01, initial_lr=max_lr/25, pct_start0.3'
  elif model=='MPRAnn':row['weight_decay']=0.;row['gradient_clip_norm']=None;row['scheduler']='none'
  if row['task']=='grouped_classification':row['source_seed']=config['seed'];row['folds']=5
 return pd.DataFrame(rows)

def phase25():
 for n in [21,22,23,24]:assert jread(n,'summary.json')['status'].startswith('completed')
 tables=out(25)/'tables';tables.mkdir(exist_ok=True);index=[]
 def publish(name,d,title,source,placement='supplement',notes=''):
  p=tables/(name+'.csv');d.to_csv(p,index=False,float_format='%.17g');assert len(pd.read_csv(p))==len(d)
  index.append(dict(table=name,title=title,placement=placement,rows=len(d),file=p.relative_to(ROOT).as_posix(),sha256=fsha(p),source=source,notes=notes))
 full=data();raw=jread(14,'raw_row_exclusion_audit.json');overview=[]
 for cell,g in full.groupby('cell_line'):
  counts=g.split_9b.value_counts();r=raw[cell]
  overview.append(dict(task='MPRA activity',population=cell,source='ENCFF774DYO' if cell=='HepG2' else 'ENCFF631LHA',raw_rows=r['rows_total'],complete_paired_candidates=r['complete_two_orientation_elements'],retained_elements=len(g),training=counts['train'],validation=counts['validation'],test=counts['test'],input_bp=200,protocol='9B chromosome-disjoint; secondary9C separately reported'))
 for stage,g in read(21,'verified_pooled_classification.csv').groupby('stage'):
  row=g.iloc[0];overview.append(dict(task='Enhancer identification' if stage==1 else 'Strong/weak classification',population=f'Stage{stage} candidate grouped OOF',source='audited supplied legacy sequence files',raw_rows=None,complete_paired_candidates=None,retained_elements=int(row.n),training=None,validation=None,test=None,input_bp=200,protocol='5 grouped folds; previously examined benchmark; one seed'))
 overview.append(dict(task='External activity transfer',population='DHS_natural',source='frozen Phase10 CODA cohort',raw_rows=None,complete_paired_candidates=None,retained_elements=8399,training=0,validation=0,test=8399,input_bp=200,protocol='separate natural-origin transfer; sequence and1kb-region resampling'))
 publish('T1_dataset_overview',pd.DataFrame(overview),'Datasets and evaluation populations','phase21/phase24 authenticated populations','main','Rows and elements are distinct units; classification OOF is not an independent test.')
 publish('T2_model_training_settings',model_settings(),'Architectures and frozen training recipes','Phase12 provenance; Phase9B config; Phase2 input lock','main','Unequal training budgets; test-recipe comparison only.')
 dump(25,'full_recipe_provenance.json',{'phase2':jmember('2_enhancer_phase2_corrected_results.zip','input_lock.json'),
  'phase9b':jmember('9B_enhancer_phase9b_regression_results.zip','phase9b_provenance.json')['config'],
  'phase9c':jmember('9C_enhancer_phase9c_regression_results.zip','phase9c_provenance.json')['config'],
  'phase12':json.loads((P12/'results/phase12_provenance.json').read_text())['config']})
 classification=read(21,'verified_pooled_classification.csv');fold=read(23,'classification_fold_mean_SD.csv')
 publish('T3_verified_classification',classification.merge(fold,on=['stage','model'],validate='one_to_one'),'Authenticated grouped classification results','Phase21 metric recomputation; Phase23 fold means/SD','main','Pooled OOF and mean-fold AUROC are different; fold SD is not seed SD; fixed threshold0.5.')
 internal=read(23,'internal_metrics_authoritative.csv');headline=internal[internal.split.eq('test')&~internal.model.str.contains(r'_seed_\d+$',regex=True)]
 publish('T4_internal_activity',headline,'Internal activity results','phase23 internal metrics','main','9B primary;9C secondary; test cohorts differ.')
 external=read(23,'external_metrics_authoritative.csv')
 publish('T5_DHS_natural_transfer',external[external.origin.eq('DHS_natural')&~external.model.str.contains(r'_seed_\d+$',regex=True)],'Natural external transfer','phase23 external metrics','main','Spearman headline; separate from internal assay activity units.')
 intervals=read(23,'independent_verification_intervals.csv')
 publish('T6_paired_comparisons',intervals,'Paired model differences and conditional intervals','phase13 independent bootstrap copied with source labels','main','Original study intervals separately supplied; no paired split-effect estimate.')
 publish('T7_retained_removed_activity',read(23,'filtering_metrics.csv'),'Performance on retained and excluded test-chromosome candidates','phase22 same-backend transformer inference; Phase23 recomputation','main','Not a causal effect of retraining without filtering.')
 supplementary=[('S1_classification_folds',21,'verified_fold_classification.csv','Fold metrics and confusion matrices'),('S2_mixed_label_sensitivity',21,'mixed_label_exclusion_sensitivity.csv','Mixed-label sensitivity without retraining'),
 ('S3_all_internal_seeds',23,'individual_seed_metrics.csv','All individual fitted seeds'),('S4_seed_mean_SD',23,'seed_mean_SD.csv','Seed means and SD separate from ensembles'),
 ('S5_all_external_origins',23,'external_metrics_authoritative.csv','Every external origin separately'),('S6_original_intervals',23,'original_study_intervals.csv','Original Phase12 intervals'),
 ('S7_filtering_contrasts',23,'filtering_paired_contrasts.csv','Retained/removed and model contrasts'),('S8_retained_subgroups',24,'retained_test_subgroup_metrics_exact.csv','Corrected retained-test subgroups'),
 ('S9_complete_population_subgroups',24,'complete_population_subgroup_metrics_exact.csv','Three-model retained/removed subgroups'),('S10_attrition',24,'dataset_attrition_counts.csv','Raw-row and element exclusions'),
 ('S11_exact_thresholds',24,'training_only_thresholds_exact.csv','Round-trip exact/integer subgroup thresholds'),('S12_collapsed_seed_sensitivity',23,'collapsed_seed_sensitivity.csv','Archived collapsed seed and labelled sensitivity'),
 ('S13_checkpoint_metadata',22,'checkpoint_metadata.csv','Authenticated selected epochs and target scaling'),('S14_prediction_reproduction',22,'checkpoint_prediction_reproduction.csv','GPU reproduction deviations'),
 ('S15_interval_coverage',24,'reliability_coverage.csv','Empirical marginal coverage and width'),('S16_interval_subgroups',24,'reliability_subgroups.csv','Marginal versus subgroup reliability'),
 ('S17_seed_disagreement',24,'reliability_seed_association.csv','Seed disagreement versus error'),('S18_selective_prediction',24,'reliability_selective_errors.csv','Fixed-cutoff selective errors'),
 ('S19_classification_tests',23,'phase11b_corrected_fold_tests.csv','Original corrected fold-test approximation'),('S20_classification_conditional_intervals',23,'phase11b_simultaneous_conditional_intervals.csv','Original simultaneous conditional comparisons')]
 for name,n,file,title in supplementary:publish(name,read(n,file),title,f'phase{n}_outputs/{file}')
 table(25,'table_index.csv',pd.DataFrame(index))
 lines=['# Frozen publication table catalogue','Every numerical CSV is linked to its saved source and SHA256. Main-text placement is a recommendation for the chosen bounded scope; no original manuscript table has been silently edited.']
 for row in index:lines.append(f"\n## {row['table']} {row['title']}\n\nPlacement: {row['placement']}. Rows: {row['rows']}. Source: {row['source']}.\n\n{row['notes']}\n\nFile: {row['file']}\nSHA256: {row['sha256']}")
 textfile(25,'TABLE_CATALOGUE.md','\n'.join(lines))
 readable=['# Readable main tables','Full precision numerical CSVs and source hashes accompany these display tables. Values are rounded for display only.']
 display_columns={'T1':['task','population','complete_paired_candidates','retained_elements','training','validation','test','protocol'],
 'T2':['task','model','epochs_max','batch','optimizer','learning_rate','selection','seeds'],
 'T3':['stage','model','n','roc_auc','roc_auc_mean','roc_auc_std','auprc','mcc_at_0_5','tn','fp','fn','tp'],
 'T4':['protocol','cell_line','model','n','pearson','spearman','rmse','mae','r2'],
 'T5':['protocol','cell_line','model','n','spearman','pearson','rmse','mae'],
 'T6':['protocol','cell_line','cohort','contrast','n_records','n_resampling_units','unit','repeats','difference','lower_95','upper_95'],
 'T7':['analysis','cell_line','population','model','n','pearson','rmse','mae']}
 def markdown_frame(frame):
  def cell(v):
   if pd.isna(v):return '—'
   if isinstance(v,(float,np.floating)):
    if v==int(v):return f'{int(v):,}'
    if 0<abs(v)<.0001:return f'{v:.2e}'
    return f'{v:.4f}'
   return str(v).replace('|','/').replace('\n',' ')
  return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
 for entry in index:
  if entry['placement']!='main':continue
  d=pd.read_csv(ROOT/entry['file']);columns=[c for c in display_columns[entry['table'].split('_')[0]] if c in d]
  readable.append('\n## '+entry['table']+' — '+entry['title']+'\n\n'+entry['notes']+'\n\n'+markdown_frame(d[columns]))
 textfile(25,'MAIN_TABLES_READABLE.md','\n'.join(readable))
 dump(25,'summary.json',{'status':'completed','main_tables':7,'supplementary_tables':20,'all_numbers_traceable':True,'unverified_submitted_performance_tables_included':False})
 print('Seven main and twenty supplementary numerical tables frozen with source hashes.')

def phase26():
 assert jread(25,'summary.json')['status']=='completed'
 os.environ['MPLCONFIGDIR']=str(out(26)/'matplotlib_cache')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
 from sklearn.metrics import roc_curve,precision_recall_curve,auc
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
 figures=out(26)/'figures';figures.mkdir(exist_ok=True);index=[]
 colors={'dnabert2':'#3d5a80','legnet':'#dc6f31','mprann':'#388067','cnn':'#816797','spaced6_4_ridge':'#8b6b3f','gc_cpg_ridge':'#727272'}
 def save(fig,name,title,source,caption,placement='main'):
  for extension in ['png','svg','pdf']:
   p=figures/f'{name}.{extension}';fig.savefig(p,dpi=240,bbox_inches='tight')
   index.append(dict(figure=name,title=title,format=extension,file=p.relative_to(ROOT).as_posix(),sha256=fsha(p),source=source,placement=placement,caption=caption))
  plt.close(fig)
 def box(ax,x,y,w,h,label,color='#edf2f6',size=9):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007',facecolor=color,edgecolor='#455665',linewidth=1))
  ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,wrap=True)
 def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#455665',linewidth=1.2))
 # F1: task branches and audit boundaries.
 fig,ax=plt.subplots(figsize=(11,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 labels=[('Supplied legacy DNA\nEnhancer / strength labels',.04,.71),('Grouped classification\nIdentity / LCS audit\nOOF predictions',.36,.71),('Verified classification\nConfusion matrices\nUnauthenticated claims omitted',.70,.71),
 ('Raw MPRA orientation rows\n200 bp assay candidates',.04,.36),('Frozen chromosome / locus splits\nTrain-only scaling\nValidation-only epoch choice',.36,.36),('DNABERT-2 and CNN recipes\nSaved checkpoints\nIdentical test examples',.70,.36)]
 for label,x,y in labels:box(ax,x,y,.25,.22,label)
 for y in [.82,.47]:arrow(ax,(.29,y),(.36,y));arrow(ax,(.61,y),(.70,y))
 box(ax,.18,.03,.65,.19,'External transfer  |  retained / removed transport  |  reliability diagnostics\nFixed-model conditional uncertainty; bounded claims',color='#f7eee6')
 arrow(ax,(.825,.36),(.70,.22))
 save(fig,'F1_study_workflow','Study workflow','Phase20 scope; Phase21/22/24 provenance','Classification and activity regression are separate tasks. The final scope excludes unsupported original performance and long-context claims.')
 # F2: mixed row/element units are explicit.
 raw=jread(14,'raw_row_exclusion_audit.json');pop=pd.read_csv(ROOT/'phase14_outputs/reconstructed_filtering_population.csv.gz')
 fig,axes=plt.subplots(1,2,figsize=(11,6))
 for ax,cell in zip(axes,CELLS):
  ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');r=raw[cell];g=pop[pop.cell_line.eq(cell)]
  steps=[f"Raw assay rows\n{r['rows_total']:,} rows",f"Eligible design / length / chromosome\n{r['candidate_rows']:,} orientation rows",f"Complete orientation pairs\n{r['complete_two_orientation_elements']:,} elements",f"Retained {g.population.eq('retained').sum():,} elements\nRemoved {g.population.eq('removed').sum():,} elements"]
  for i,label in enumerate(steps):
   y=.77-.22*i;box(ax,.11,y,.78,.15,label)
   if i<3:arrow(ax,(.5,y),(.5,y-.07))
  ax.set_title(cell);ax.text(.5,.015,'Removed: exact/RC or shared 15-mer with legacy benchmark',ha='center',fontsize=8)
 save(fig,'F2_dataset_attrition','Dataset attrition','phase24 dataset_attrition_counts.csv','Counts distinguish assay rows from paired elements. Design/length/chromosome and incomplete-orientation exclusions are itemized in S10. Removed elements are not all full-sequence duplicates.')
 # F3: supplied historical set, exact/RC evidence from authenticated roster.
 roster=read(21,'historical_test_overlap_roster.csv')
 fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
 for ax,stage in zip(axes[0],[1,2]):
  g=roster[roster.stage.eq(stage)];counts=[int(g.exact_training_match.sum()),int((~g.exact_training_match).sum())]
  ax.bar(['Exact training match','Other supplied test'],counts,color=['#c55a4b','#6b879b'])
  for x,v in enumerate(counts):ax.text(x,v+2,str(v),ha='center')
  ax.set_ylabel('Supplied historical test records');ax.set_title(f'Stage {stage}')
 audit=jmember('1_ enhancer_audit_v1_outputs.zip','audit_run_summary.json')
 for name,h in audit['outputs_sha256'].items():assert sha(member('1_ enhancer_audit_v1_outputs.zip',name))==h,name
 excluded=[]
 for stage in [1,2]:
  manifest=csv('1_ enhancer_audit_v1_outputs.zip',f'stage{stage}_candidate_manifest.csv');excluded.append(int(manifest.role.eq('excluded_related_to_legacy_test').sum()))
 ax=axes[1,0];ax.bar(['Stage1','Stage2'],excluded,color='#8a7493')
 for x,v in enumerate(excluded):ax.text(x,v+.3,str(v),ha='center')
 ax.set_ylabel('Additional related training records excluded');ax.set_title('Legacy-test-related component exclusion')
 pair=audit['pair_screen'];values=[pair['lcs_ge150'],pair['lcs_ge160'],pair['optimal_global_alignment_matches_ge160']]
 ax=axes[1,1];ax.bar(['LCS ≥150','LCS ≥160','Gapped matches ≥160'],values,color=['#6b879b','#8a7493','#a48460'])
 for x,v in enumerate(values):ax.text(x,v+3,str(v),ha='center')
 ax.set_ylabel('Stage1 sequence pairs');ax.set_title(f"Screen of {pair['total_stage1_pairs']:,} pairs; categories overlap")
 save(fig,'F3_historical_overlap','Historical benchmark overlap','phase21 historical roster; authenticated Phase1 audit summary/manifests','Top: supplied historical test exact-training identity, not an untouched independent test. Bottom: additional legacy-test-related training records excluded and nonexclusive Stage1 pair-screen counts. LCS≥160 is the graph rule, not80% local identity; gapped match counts are a separate diagnostic. These screens do not exclude all homology.')
 # F4: ensembles plus individual seed points; classical models are single fits.
 internal=read(23,'internal_metrics_authoritative.csv');test=internal[internal.split.eq('test')]
 families=['gc_cpg_ridge','spaced6_4_ridge','cnn','dnabert2','legnet','mprann']
 fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 for ax,(tag,cell) in zip(axes.ravel(),itertools.product(['9b','9c'],CELLS)):
  g=test[test.protocol.eq(tag)&test.cell_line.eq(cell)]
  values=[float(g[g.model.eq(f'{m}_three_seed_mean' if m in ['cnn','dnabert2','legnet','mprann'] else m)].pearson.iloc[0]) for m in families]
  ax.bar(range(6),values,color=[colors[m] for m in families],alpha=.7)
  for x,m in enumerate(families):
   s=g[g.model.str.startswith(m+'_seed_')].pearson.to_numpy()
   if len(s):ax.scatter(x+np.linspace(-.12,.12,len(s)),s,c='black',s=18,zorder=3)
  ax.set_xticks(range(6),[models_label(m) for m in families],rotation=28,ha='right');ax.set_ylim(0,.82);ax.set_ylabel('Test Pearson correlation');ax.set_title(f'{tag.upper()} — {cell}')
 save(fig,'F4_internal_activity_seeds','Internal performance with individual seeds','phase23 internal_metrics_authoritative.csv','Bars are three-seed ensembles for neural models and single fits for ridge models; black points are individual seed scores. 9B and 9C evaluate different populations. The collapsed 9C K562 seed remains visible.')
 # F5: external natural headline, all origins separately in supplement.
 external=read(23,'external_metrics_authoritative.csv');ens=external[~external.model.str.contains(r'_seed_\d+$',regex=True)]
 fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for ax,cell in zip(axes,CELLS):
  for j,tag in enumerate(['9b','9c']):
   g=ens[ens.cell_line.eq(cell)&ens.protocol.eq(tag)&ens.origin.eq('DHS_natural')].set_index('model')
   v=[g.loc[f'{m}_three_seed_mean' if m in ['cnn','dnabert2','legnet','mprann'] else m,'spearman'] for m in families]
   ax.plot(range(6),v,marker='o',label=tag.upper())
  ax.set_xticks(range(6),[models_label(m) for m in families],rotation=28,ha='right');ax.set_ylabel('External Spearman correlation');ax.set_title(cell);ax.legend()
 save(fig,'F5_DHS_natural_transfer','Natural external transfer','phase23 external_metrics_authoritative.csv','DHS-natural is the headline external population (n=8,399). Protocols and cells remain separate; other origins are supplementary. Curves connect model categories, not a continuous model scale.')
 origins=sorted(ens.origin.unique());fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
 for ax,(tag,cell) in zip(axes.ravel(),itertools.product(['9b','9c'],CELLS)):
  g=ens[ens.protocol.eq(tag)&ens.cell_line.eq(cell)].pivot(index='origin',columns='model',values='spearman')
  cols=[f'{m}_three_seed_mean' if m in ['cnn','dnabert2','legnet','mprann'] else m for m in families];v=g.loc[origins,cols].to_numpy()
  im=ax.imshow(v,vmin=float(ens.spearman.min()),vmax=float(ens.spearman.max()),cmap='viridis',aspect='auto');ax.set_xticks(range(6),[models_label(m) for m in families],rotation=30,ha='right');ax.set_yticks(range(len(origins)),origins);ax.set_title(f'{tag.upper()} — {cell}')
  for y in range(len(origins)):
   for x in range(6):ax.text(x,y,f'{v[y,x]:.2f}',ha='center',va='center',fontsize=7,color='white' if v[y,x]<.35 else 'black')
 fig.colorbar(im,ax=axes.ravel().tolist(),label='External Spearman correlation',shrink=.8,pad=.02)
 save(fig,'SF1_all_external_origins','All external origins','phase23 external_metrics_authoritative.csv','Each origin is reported separately. Synthetic origins and natural loci are not pooled into a headline score. Shared colour scale covers the full observed score range.','supplement')
 # F6: paired differences, independent verification intervals only.
 ci=read(23,'independent_verification_intervals.csv');g=ci[ci.cohort.eq('internal_test')&ci.contrast.str.contains('minus (legnet|mprann)',regex=True)].copy()
 fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
 labels=[]
 for y,row in enumerate(g.itertuples()):
  target='LegNet' if 'legnet' in row.contrast else 'MPRAnn';labels.append(f'{row.protocol.upper()} {row.cell_line}: DNABERT-2 − {target}')
  ax.errorbar(row.difference,y,xerr=[[row.difference-row.lower_95],[row.upper_95-row.difference]],fmt='o',color='#3d5a80',capsize=3)
 ax.axvline(0,color='black',ls='--');ax.set_yticks(range(len(labels)),labels);ax.set_xlabel('Difference in test Pearson correlation');ax.invert_yaxis()
 save(fig,'F6_paired_differences','Paired model differences','phase23 independent_verification_intervals.csv','Intervals are the independent Phase13 conditional bootstrap verification, not an average of original and verification bounds. Test loci are resampled within each protocol; no paired comparison between 9B and 9C populations is made.')
 # F7: complete-population distributions, with homopolymer lengths included.
 pop['gc_fraction']=(pop.sequence.str.count('G')+pop.sequence.str.count('C'))/200
 pop['max_homopolymer']=pop.sequence.map(lambda s:max(len(list(g)) for _,g in itertools.groupby(s)))
 fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
 for i,cell in enumerate(CELLS):
  cellpop=pop[pop.cell_line.eq(cell)]
  for j,(feature,label,bins) in enumerate([('mean_log2_activity','Measured log2 activity',np.linspace(cellpop.mean_log2_activity.min(),cellpop.mean_log2_activity.max(),70)),('gc_fraction','GC fraction',np.linspace(0,1,35)),('max_homopolymer','Longest same-base run',np.arange(.5,cellpop.max_homopolymer.max()+1.5,1))]):
   ax=axes[i,j]
   for population,color in [('retained','#3d5a80'),('removed','#dc6f31')]:
    v=pop[pop.cell_line.eq(cell)&pop.population.eq(population)][feature];ax.hist(v,bins=bins,density=True,histtype='step',lw=1.6,color=color,label=f'{population} (n={len(v):,})')
   ax.set_title(cell);ax.set_xlabel(label);ax.set_ylabel('Density');ax.legend(fontsize=6)
 save(fig,'F7_filtering_distributions','Retained and removed population distributions','Phase14 reconstructed population; Phase24 verified counts','Activity, GC and repeated-base distributions describe population selection, not performance on unassayed DNA. Histogram axes cover the full observed range of each cell population.')
 # F8: exact retained-test subgroup differences with subgroup counts.
 subgroup=read(24,'retained_test_subgroup_metrics_exact.csv');groups=['gc_group','cpg_group','homopolymer_group','activity_group']
 fig,axes=plt.subplots(4,2,figsize=(12,13),constrained_layout=True)
 for i,family in enumerate(groups):
  for j,cell in enumerate(CELLS):
   ax=axes[i,j];g=subgroup[subgroup.protocol.eq('9b')&subgroup.cell_line.eq(cell)&subgroup.family.eq(family)&~subgroup.suppressed]
   wide=g.pivot(index='subgroup',columns='model',values='mae');counts=g.groupby('subgroup').n.first();labels=wide.index.tolist()
   for offset,(m,color) in enumerate([('legnet','#dc6f31'),('mprann','#388067')]):
    delta=wide['dnabert2_three_seed_mean']-wide[m+'_three_seed_mean'];ax.scatter(delta,np.arange(len(labels))+(.10 if offset else -.10),label=f'DNABERT-2 − {models_label(m)}',c=color,s=24)
   ax.axvline(0,c='black',ls='--',lw=.8);ax.set_yticks(range(len(labels)),[str(s).replace('at_or_below_training_median','CpG ≤ median').replace('above_training_median','CpG > median')+f' (n={int(counts[s]):,})' for s in labels]);ax.set_xlabel('Difference in MAE (log2 activity)');ax.set_title(f"{cell} — {family.replace('_group','')}");ax.legend(fontsize=6)
 save(fig,'F8_subgroup_errors','Retained-test subgroup error differences','phase24 retained_test_subgroup_metrics_exact.csv','Positive differences mean the specialist CNN has lower MAE. Groups use training-only/fixed thresholds and exact integer CpG definitions. Counts below 100 are suppressed. These are descriptive, unadjusted subgroup summaries; measured-activity bands require labels.')
 # F9: actual archived primary ensemble predictions versus measurements.
 pred=all_internal();fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
 for ax,(cell,model) in zip(axes.ravel(),itertools.product(CELLS,['dnabert2','legnet','mprann'])):
  g=pred[pred.protocol.eq('9b')&pred.cell_line.eq(cell)&pred.split.eq('test')&pred.model.eq(model+'_three_seed_mean')]
  ax.hexbin(g.target,g.prediction,gridsize=40,mincnt=1,cmap='Blues',bins='log');bounds=[min(g.target.min(),g.prediction.min()),max(g.target.max(),g.prediction.max())];ax.plot(bounds,bounds,c='black',ls='--',lw=.8)
  ax.set_xlabel('Measured log2 activity');ax.set_ylabel('Predicted log2 activity');ax.set_title(f'{cell} — {models_label(model)}')
 save(fig,'F9_activity_predictions','Primary activity predictions','Authenticated archived Phase9B/12 predictions','Hexagonal colour reflects log record density. Dashed line is identity. Model rankings in main tables use these same archived predictions; local checkpoint reproduction is an audit.')
 # Empirical intervals and widths for all three models.
 coverage=read(24,'reliability_coverage.csv');fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
 for j,cell in enumerate(CELLS):
  g=coverage[coverage.cell_line.eq(cell)&coverage.calibration_source.eq('retained')]
  labels=[f'{models_label(r.model)}\n{r.audit_population}\n'+('constant' if r.method=='constant_width' else 'seed-scaled') for r in g.itertuples()]
  axes[0,j].bar(range(len(g)),g.empirical_coverage,color='#5c7f98');axes[0,j].axhline(.9,c='black',ls='--');axes[0,j].set_ylim(0,1);axes[0,j].set_title(cell);axes[0,j].set_ylabel('Empirical audit coverage')
  axes[1,j].bar(range(len(g)),g.mean_width,color='#8b9c74');axes[1,j].set_ylabel('Mean full interval width (log2)')
  for ax in axes[:,j]:ax.set_xticks(range(len(g)),labels,rotation=55,ha='right',fontsize=6)
 save(fig,'SF2_interval_coverage_width','Prediction interval coverage and width','phase24 reliability_coverage.csv','Nominal 90% residual intervals are retrospective diagnostics, calibrated on disjoint retained calibration groups. Wider intervals alone are not better uncertainty. Dependence and previous examination prevent universal coverage claims.','supplement')
 # Explicit collapsed seed and sensitivity.
 g=pred[pred.protocol.eq('9c')&pred.cell_line.eq('K562')&pred.split.eq('test')&pred.model.str.startswith('dnabert2')]
 wide=g.pivot(index='record_id',columns='model',values='prediction');target=g.groupby('record_id').target.first().loc[wide.index]
 vectors=[wide['dnabert2_seed_20260930'],wide['dnabert2_three_seed_mean'],wide[['dnabert2_seed_20260931','dnabert2_seed_20260932']].mean(axis=1)]
 fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
 for ax,p,label in zip(axes,vectors,['Collapsed original seed','Archived three-seed ensemble','Two healthy seeds: sensitivity only']):
  lo=min(target.min(),min(v.min() for v in vectors));hi=max(target.max(),max(v.max() for v in vectors))
  ax.hexbin(target,p,gridsize=30,mincnt=1,cmap='Blues',bins='log');ax.plot([lo,hi],[lo,hi],c='black',ls='--');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_title(label);ax.set_xlabel('Measured log2 activity');ax.set_ylabel('Predicted log2 activity')
 save(fig,'SF3_collapsed_seed_sensitivity','Archived collapsed-seed sensitivity','phase23 collapsed_seed_sensitivity.csv; archived Phase9C predictions','The nearly constant seed is preserved in the primary ensemble. The healthy-seed average is a separate sensitivity; the training cause remains unresolved. Correlation alone can conceal scale contraction.','supplement')
 # Verified grouped OOF ROC/PR; no original unverified curves reused.
 cp=read(21,'authenticated_classification_predictions.csv.gz');fig,axes=plt.subplots(2,2,figsize=(12,10));modelnames=sorted(cp.model.unique());linecolors=plt.cm.tab10(np.linspace(0,1,len(modelnames)));handles=[]
 for i,stage in enumerate([1,2]):
  for color,model in zip(linecolors,modelnames):
   g=cp[cp.stage.eq(stage)&cp.model.eq(model)];fpr,tpr,_=roc_curve(g.label,g.score);precision,recall,_=precision_recall_curve(g.label,g.score)
   line,=axes[i,0].plot(fpr,tpr,c=color,lw=1.2,label=model.replace('_',' '));axes[i,1].plot(recall,precision,c=color,lw=1.2)
   if i==0:handles.append(line)
  axes[i,0].plot([0,1],[0,1],c='gray',ls='--');axes[i,0].set(xlabel='False-positive rate',ylabel='True-positive rate',title=f'Stage{stage} pooled OOF ROC')
  axes[i,1].set(xlabel='Recall',ylabel='Precision',title=f'Stage{stage} pooled OOF precision–recall')
 fig.legend(handles,[x.replace('_',' ') for x in modelnames],loc='lower center',ncol=3,fontsize=8);fig.subplots_adjust(bottom=.17,hspace=.3,wspace=.25)
 save(fig,'SF4_classification_ROC_PR','Verified classification ROC and PR','phase21 authenticated_classification_predictions.csv.gz','Curves pool authenticated grouped out-of-fold predictions and are descriptive; headline mean-fold AUROC and pooled AUROC are reported separately. One original seed; the incomplete Phase11C is not represented as complete.','supplement')
 cm=read(21,'verified_pooled_classification.csv');fig,axes=plt.subplots(2,2,figsize=(8,7),constrained_layout=True)
 for ax,(stage,model) in zip(axes.ravel(),itertools.product([1,2],['finetuned_dnabert2','spaced6_4_logreg'])):
  row=cm[cm.stage.eq(stage)&cm.model.eq(model)].iloc[0];v=np.array([[row.tn,row.fp],[row.fn,row.tp]],int);ax.imshow(v,cmap='Blues')
  for y,x in itertools.product(range(2),range(2)):ax.text(x,y,str(v[y,x]),ha='center',va='center',fontsize=13,color='white' if v[y,x]>v.max()*.6 else 'black')
  labels=['Non-enhancer','Enhancer'] if stage==1 else ['Weak','Strong'];ax.set_xticks([0,1],labels);ax.set_yticks([0,1],labels);ax.set_xlabel('Predicted');ax.set_ylabel('Measured label');ax.set_title(f"Stage{stage}: {model.replace('_',' ')}",fontsize=9)
 save(fig,'SF5_classification_confusion','Verified classification confusion matrices','phase21 verified_pooled_classification.csv','Matrices use pooled authenticated OOF predictions at the fixed 0.5 threshold. Full matrices for every model are in S1/T3; labels and actual denominators are explicit.','supplement')
 # Actual regression architectures, matching source classes and checkpoint dimensions.
 fig,axes=plt.subplots(4,1,figsize=(13,10));diagrams=[
 ('DNABERT-2',['200 bp DNA\nBPE tokens','12 Transformer layers\n117M pretrained encoder','Masked mean\nnon-padding tokens','Dropout0.1\nLinear768→1','Undo training-only\ntarget scaling']),
 ('MPRA-LegNet',['One-hot 4×200\nStem64, kernel11','EffBlock + residual concat\nLocal blocks80/96/112/128\nFour max-pool2 stages','Mapper256 channels\nGlobal average pool','Dense256→256→1\nBN and SiLU','Undo training-only\ntarget scaling']),
 ('MPRAnn',['One-hot 4×200\nConv250,k7;ReLU+BN','Conv250,k8/3; Conv100,k2\nSoftmax + BN after each\nPool2 after k8','Flatten90×100\nKeras length/channel order','Dense300→200→1\nSigmoid hidden units\nDropout0.1/0.1/0.3','Undo training-only\ntarget scaling']),
 ('Earlier CNN',['One-hot 4×200','Conv64,k15\nConv96,k7\nConv96,k7;GELU','Mean + max pool\n192 features','Dropout0.2\nDense192→64→1','Undo training-only\ntarget scaling'])]
 for ax,(label,steps) in zip(axes,diagrams):
  ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(label,loc='left',fontweight='bold')
  for i,label in enumerate(steps):
   x=.015+i*.20;box(ax,x,.18,.175,.58,label,size=8)
   if i<4:arrow(ax,(x+.175,.47),(x+.20,.47))
 save(fig,'SF6_regression_architectures','Implemented regression architectures','Pinned model code; Phase12 classes; Phase9B classes; Phase22 strict state loading','Architectures match retained regression implementations. Published human CNN recipes were adapted to 200 bp without assay adapters. Forward/RC predictions are averaged at inference. Classification feature/fine-tuning recipes remain separately documented in T2 and source code.','supplement')
 # Graphical abstract: evidence story, not invented regulatory mechanism.
 fig,ax=plt.subplots(figsize=(12,4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 steps=['Audit data and records\nLegacy overlap\n149,632 paired candidates','Compare frozen recipes\nDNABERT-2 and specialist CNNs\nChromosome / locus tests','Test transport and reliability\nExternal natural DNA\nRetained and removed candidates','Bound the conclusions\nAssay and population scope\nConditional uncertainty']
 for i,label in enumerate(steps):
  x=.015+i*.25;box(ax,x,.25,.22,.5,label,color=['#e7eef5','#f5e8dc','#e5f0e8','#eeeeee'][i],size=9)
  if i<3:arrow(ax,(x+.22,.5),(x+.25,.5))
 ax.text(.5,.91,'A verified evaluation of enhancer identification and assay activity prediction',ha='center',fontsize=13,fontweight='bold')
 ax.text(.5,.10,'Existing models • tested training recipes • traceable evidence • no universal architecture claim',ha='center',fontsize=10)
 save(fig,'F10_graphical_abstract','Graphical abstract','Phases20–25 frozen evidence','Graphical summary of the bounded study. Counts refer to complete paired assay candidates; the graphic does not claim native enhancer–gene mechanisms or novel architecture.')
 table(26,'figure_index.csv',pd.DataFrame(index))
 sources=[ROOT/'analysis_tools/readiness_pipeline.py',ROOT/'analysis_tools/regression_architectures.py',ROOT/'phase14_outputs/reconstructed_filtering_population.csv.gz']
 sources+=[p for n in [21,22,23,24,25] for p in out(n).rglob('*.csv*') if 'inference_cache' not in p.parts]
 sources += list(SRC.glob('*regression_results.zip'))+[SRC/'1_ enhancer_audit_v1_outputs.zip',SRC/'10_enhancer_phase10_external_transfer_results.zip',P12/'results/phase12_internal_predictions_new_arms.csv']
 table(26,'figure_source_file_index.csv',pd.DataFrame([dict(file=p.relative_to(ROOT).as_posix(),sha256=fsha(p),bytes=p.stat().st_size) for p in sorted(set(sources)) if p.is_file()]))
 captions=['# Figure captions and placement']
 for name,g in pd.DataFrame(index).groupby('figure',sort=False):
  row=g.iloc[0];captions.append(f"\n## {name} {row.title}\n\nRecommended placement: {row.placement}.\n\n{row.caption}\n\nSource: {row.source}\n\nPNG, SVG and PDF files: phase26_outputs/figures/{name}.*")
 textfile(26,'FIGURE_CAPTIONS.md','\n'.join(captions))
 # Contact sheet for visual QA only; originals remain unchanged.
 paths=sorted(figures.glob('*.png'));fig,axes=plt.subplots(math.ceil(len(paths)/3),3,figsize=(18,5*math.ceil(len(paths)/3)))
 for ax,path in zip(axes.ravel(),paths):ax.imshow(plt.imread(path));ax.set_title(path.stem,fontsize=10);ax.axis('off')
 for ax in axes.ravel()[len(paths):]:ax.axis('off')
 fig.tight_layout();fig.savefig(out(26)/'figure_QA_contact_sheet.png',dpi=130);plt.close(fig)
 dump(26,'summary.json',{'status':'completed','figures':len(paths),'formats':['PNG','SVG','PDF'],'source_index_rows':len(index),'visual_QA':'pending parent inspection before final readiness gate'})
 print('Generated',len(paths),'figures/diagrams in PNG, SVG and PDF; captions and source hashes saved.')


def phase27():
 base=ROOT/'research_notes/Enhancer revision evidence review'
 model=json.loads((base/'model_sources.json').read_text(encoding='utf8'))
 assay=json.loads((base/'assay_sources.json').read_text(encoding='utf8'))
 evaluation=json.loads((base/'evaluation_sources.json').read_text(encoding='utf8'))
 gena=json.loads((base/'gena_source.json').read_text(encoding='utf8'))
 rows=[]
 for group,items in [('model',model['models']+model['ancestry_papers']),('assay',assay['sources']),('evaluation',evaluation['sources']),('model',[dict(gena['primary_citation'],id='gena2025')])]:
  for x in items:
   identifier=x.get('id',x.get('model_id',x.get('paper_id','source')))
   author=x.get('authors','');author=' and '.join(author) if isinstance(author,list) else author
   rows.append(dict(source_id=identifier,topic=group,title=x['title'],authors=author,year=x['year'],doi=x.get('doi',''),url=x.get('official_url',x.get('url','')),
     venue=x.get('venue',x.get('journal','')),publication_status=x.get('publication_status',x.get('publication_status_as_of_checked_date','see primary-source notes')),
     supports=json.dumps(x.get('supporting_claims',x.get('supports','')),ensure_ascii=False),limitations=json.dumps(x.get('gaps',x.get('does_not_establish',x.get('caveats',''))),ensure_ascii=False),
     reading_scope=x.get('reading_scope','See copied primary-source reading/provenance notes'),checked_date=DATE))
 citations=pd.DataFrame(rows)
 # Prefer the verified current v2 title/status for the one shared hashFrag DOI.
 citations['_priority']=citations.topic.map({'evaluation':0,'model':1,'assay':2})
 citations['dedup_key']=citations.doi.where(citations.doi.ne(''),citations.url)
 citations=citations.sort_values('_priority',kind='stable').drop_duplicates('dedup_key').drop(columns=['_priority','dedup_key']).reset_index(drop=True)
 assert citations.url.str.startswith('https://').all() and citations.title.str.len().gt(10).all()
 table(27,'verified_primary_citations.csv',citations)
 coverage=[('DNABERT original','dnabert_original','retained corrected classification','Original 6-mer model; not DNABERT-2'),
 ('DNABERT-2','dnabert2','retained classification and regression','Accepted ICLR2024 title uses Genomes plural'),
 ('GENA-LM','gena2025','retained frozen classification','Original BERT-base T2T512-token variant; no long-context claim'),
 ('MPRA-LegNet','mpra_legnet','retained regression','Human adaptation: Agarwal2025; original LegNet ancestry separately cited'),
 ('MPRAnn','mpra_legnet','retained regression','Same Agarwal2025 paper; official standard() Keras architecture port'),
 ('LegNet ancestry','penzar2023legnet','architecture ancestry','Original yeast GPRA/DREAM study, distinct from human adaptation'),
 ('Longformer','longformer','historical context only','Natural-language long-document architecture; performance/long-context claims omitted')]
 mapped=[]
 for name,key,role,note in coverage:
  c=citations[citations.source_id.eq(key)];assert len(c)==1,(name,key)
  mapped.append(dict(model=name,role=role,source_id=key,title=c.iloc[0].title,doi=c.iloc[0].doi,url=c.iloc[0].url,scope_note=note))
 table(27,'retained_model_citation_map.csv',pd.DataFrame(mapped))
 closest=[]
 for x in evaluation['sources']:
  closest.append(dict(source_id=x['id'],title=x['title'],year=x['year'],url=x['url'],publication_status=x['publication_status'],protocol=x.get('evaluation_protocol',''),
    overlap=x.get('overlap_with_project',''),remaining_distinction=x.get('gaps',''),cross_dataset_numeric_ranking_allowed=False))
 table(27,'closest_work_and_limits.csv',pd.DataFrame(closest))
 source_rows=[]
 for p in sorted(base.rglob('*')):
  if p.is_file():source_rows.append(dict(file=p.relative_to(ROOT).as_posix(),sha256=fsha(p),bytes=p.stat().st_size))
 table(27,'literature_source_index.csv',pd.DataFrame(source_rows))
 report=ROOT/'reports/Enhancer revision evidence review.md';assert report.exists()
 report_text=report.read_text(encoding='utf8');assert 'GENA' in report_text,'Integrate final GENA addendum before freezing.'
 # Replace workspace-specific links only; all external primary URLs are preserved.
 for prefix in [ROOT.as_posix()+'/',str(ROOT).replace('\\','/')+'/',str(ROOT)+'\\']:
  report_text=report_text.replace(prefix,'../')
 textfile(27,'LITERATURE_AND_CONTRIBUTION_REVIEW.md',report_text)
 bib=[]
 for i,c in enumerate(citations.itertuples(),1):
  clean=lambda s:str(s).replace('{','\\{').replace('}','\\}').replace('&','\\&')
  fields={'title':c.title,'author':c.authors,'year':c.year,'journal':c.venue,'url':c.url}
  if c.doi:fields['doi']=c.doi
  bib.append('@misc{'+re.sub(r'\W','',str(c.source_id))+',\n'+',\n'.join('  '+k+' = {'+clean(v)+'}' for k,v in fields.items())+'\n}')
 textfile(27,'verified_scope_references.bib','\n\n'.join(bib))
 textfile(27,'CONTRIBUTION_AND_CITATION_DECISIONS.md','''# Literature-backed contribution and citation decisions

The supported contribution is a traceable evaluation and population-selection/reliability audit of existing models under frozen, previously examined protocols. It is not a new architecture, the first fine-tuned DNA-language-model/CNN MPRA comparison, universal CNN superiority or proof that DNA sequence is insufficient. Tang et al. already evaluated task adaptation for lentiMPRA; other current benchmarks differ in assay, partitions, tuning budgets and tasks. Their numerical scores must not be ranked against local numbers as if they shared an evaluation.

The final bibliography is rebuilt from verified primary sources for the retained scope. It replaces the submitted bibliography rather than inheriting incorrect DNABERT-2 attribution or duplicate Hu entries. Original DNABERT and DNABERT-2 are distinct citations; human MPRA-LegNet and MPRAnn share Agarwal et al. 2025, with Penzar et al. 2023 as LegNet ancestry. The retained GENA-LM arm is the original BERT-base T2T model, not ModernGENA. Longformer is contextual history only. Reviewer-suggested ProbC is context-dependent 3D regulation; BERT2OME is RNA modification, not enhancer validation. Neither is used to imply accuracy here.

The source review is focused, current to 2026-10-13, and explicitly not exhaustive. Source notes distinguish full text, abstracts, repository reading and retrieval failures. Copied source hashes permit review. Preprints remain labelled preprints; hashFrag's latest verified v2 title/status supersedes the older v1 title in historical notes. No cross-dataset headline comparisons or invented novelty guarantees are retained.''')
 dump(27,'summary.json',{'status':'completed','deduplicated_primary_citations':len(citations),'retained_model_map_rows':len(mapped),'closest_work_rows':len(closest),'copied_source_files_indexed':len(source_rows),'focused_not_exhaustive':True,'new_architecture_claim':False})
 print('Primary citations, closest work, reading limits and bounded contribution frozen.')


def phase28():
 from readiness_packaging import build_portable_package
 build_portable_package()


def phase29():
 from readiness_packaging import final_readiness_review
 final_readiness_review()
