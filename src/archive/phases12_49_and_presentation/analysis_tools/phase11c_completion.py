"""Phase11C audit/completion. New work stays local; historical experiments are read only."""
from pathlib import Path
import os,sys,json,io,zipfile,hashlib,shutil,random,time,gc,math,itertools,platform,ast
import numpy as np
import pandas as pd

ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'phase30_inputs').is_dir() and (p/'reference_inputs').is_dir()),None)
if ROOT is None:raise RuntimeError('Run in the contribution workspace or complete local reproduction package.')
INPUT=ROOT/'phase30_inputs';SOURCES=INPUT/'historical_sources'
SEEDS=[20260928,20261016,20261017]
MODELS=['dnabert2','dnabert_legacy'];STAGES=[1,2];FOLDS=list(range(5))
REVISIONS={'dnabert2':'ec1f874253852eb3907081f57294991b4280ceb6','dnabert_legacy':'c56e67ea5827e0ddc67ef059addcf71569b1216e'}
CONFIG={'protocol':'phase31_consistent_local_classification_three_seeds_v1','historical_parent_protocol':'phase11c_multiseed_grouped5_v1',
 'models':MODELS,'stages':STAGES,'base_seeds':SEEDS,'folds':FOLDS,'epochs':2,'batch_size':8,'gradient_accumulation':2,
 'lr':2e-5,'weight_decay':.01,'gradient_clip':1.,'dropout':.1,'rc_augment_train':True,'max_tokens':256,
 'transformers':'4.44.2','model_revisions':REVISIONS,'score_orientation':'original sequence only',
 'selection':'final fixed epoch2; outer OOF predictions never select settings, thresholds or epochs',
 'head':'attention-mask mean including unpadded special tokens; dropout0.1; Linear768to2',
 'loss':'cross_entropy; fp16 autocast; AdamW; accumulation2; divide all minibatch losses by2 including terminal partial accumulation',
 'comparison_policy':'All60fits are fresh local runs. Original20 and historical10 remain separate; no selective reuse by outcome.',
 'local_runtime':'enhancer; torch2.11.0+cu128; Windows; RTX5060Ti16GB; same frozen software/backend for all60jobs',
 'attention':'DNABERT2 pinned PyTorch fallback; optional Triton disabled as in historical code',
 'purpose':'Complete classification seed study; MPRA activity-training adequacy is a separate open question'}

def out(n):
 p=ROOT/f'phase{n}_outputs';p.mkdir(exist_ok=True);return p
def fsha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sha(b):return hashlib.sha256(b).hexdigest()
def atomic_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.writing')
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)
def dump(n,name,obj):atomic_json(out(n)/name,obj)
def jread(n,name):return json.loads((out(n)/name).read_text(encoding='utf8'))
def table(n,name,d):d.to_csv(out(n)/name,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)
def read(n,name):return pd.read_csv(out(n)/name,float_precision='round_trip')
def member(archive,name):
 with zipfile.ZipFile(SOURCES/archive) as z:
  names=[name] if name in z.namelist() else [s for s in z.namelist() if s.split('/')[-1]==Path(name).name];assert len(names)==1,(archive,name)
  return z.read(names[0])
def csv(archive,name):return pd.read_csv(io.BytesIO(member(archive,name)),float_precision='round_trip')
def jmember(archive,name):return json.loads(member(archive,name))
def snapshot():
 files=list(ROOT.glob('[[]*.ipynb'))
 files=[p for p in files if int(p.name.split(']')[0][1:])<=29]
 for n in range(12,30):
  d=ROOT/f'phase{n}_outputs'
  files += [p for p in d.rglob('*') if p.is_file() and not any(x in p.parts for x in ['Enhancer_PreWriting_Reproducibility','inference_cache','matplotlib_cache']) and p.suffix in ['.json','.csv','.md','.ipynb','.pt']]
 return {p.relative_to(ROOT).as_posix():fsha(p) for p in sorted(set(files))}
def locked_data():
 frames={}
 for stage in STAGES:
  d=csv('1_ enhancer_audit_v1_outputs.zip',f'stage{stage}_candidate_manifest.csv')
  d=d[d.role.eq('train_candidate')].reset_index(drop=True)
  d['fold_5']=d.fold_5.astype(int);d['label']=d.label.astype(int)
  assert len(d)=={1:2838,2:1376}[stage] and d.record_id.is_unique
  assert d.sequence.str.fullmatch('[ACGT]{200}').all()
  assert d.groupby('similarity_group_id').fold_5.nunique().max()==1
  frames[stage]=d
 return frames
def metrics(y,p):
 from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix,matthews_corrcoef,f1_score
 y=np.asarray(y,int);p=np.asarray(p,float);assert set(y)=={0,1} and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
 return dict(n=len(y),negative=int((y==0).sum()),positive=int((y==1).sum()),auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),
  brier=float(brier_score_loss(y,p)),accuracy=float((tn+tp)/len(y)),balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2),
  mcc=float(matthews_corrcoef(y,p>=.5)),f1=float(f1_score(y,p>=.5)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def phase30():
 import nbformat
 assert Path(sys.prefix).name.lower()=='enhancer'
 SOURCES.mkdir(parents=True,exist_ok=True)
 names=['1_ enhancer_audit_v1_outputs.zip','2_enhancer_phase2_corrected_results.zip','11C_enhancer_phase11c_results_shard0.zip',
  '[11C] Enhancer_Kaggle_Phase11C_Transformer_Seed_Robustness.ipynb','[11C2] Enhancer_Kaggle_Phase11C_Transformer_Seed_Robustness.ipynb',
  '[11C3] Enhancer_Kaggle_Phase11C_Transformer_Seed_Robustness.ipynb','[2] Enhancer_Kaggle_Phase2_GENA_Correction.ipynb']
 copied=[]
 for name in names:
  source=ROOT/'reference_inputs/readiness_sources'/name;target=SOURCES/name
  if source.exists():
   if target.exists():assert fsha(source)==fsha(target)
   else:shutil.copy2(source,target)
   assert fsha(source)==fsha(target)
  else:
   # A relocated supplement already contains authenticated input copies.
   prior=pd.read_csv(out(30)/'copied_input_manifest.csv')
   expected=prior[prior['copy'].eq(target.relative_to(ROOT).as_posix())].iloc[0].sha256
   assert target.exists() and fsha(target)==expected,name
  copied.append(dict(source=source.relative_to(ROOT).as_posix(),copy=target.relative_to(ROOT).as_posix(),sha256=fsha(target),bytes=target.stat().st_size))
 table(30,'copied_input_manifest.csv',pd.DataFrame(copied))
 audit=jmember(names[0],'audit_run_summary.json');corr=jmember(names[1],'correction_report.json');old=jmember(names[2],'phase11c_run_manifest.json')
 for name,h in audit['outputs_sha256'].items():assert sha(member(names[0],name))==h,name
 for name,h in corr['output_sha256'].items():assert sha(member(names[1],name))==h,name
 base=csv(names[1],'all_oof_predictions.csv');frames=locked_data();checks=[];historical=[]
 for (stage,model),g in base[base.model.isin(['finetuned_dnabert2','finetuned_dnabert_legacy'])].groupby(['stage','model']):
  truth=frames[stage].set_index('record_id').loc[g.record_id]
  assert len(g)==len(frames[stage]) and set(g.record_id)==set(frames[stage].record_id)
  assert np.array_equal(g.label,truth.label) and np.array_equal(g.fold_5,truth.fold_5)
  checks.append(dict(source='original_seed',model=model,stage=stage,base_seed=20260928,fold='all',n=len(g),record_label_fold_checks=True,source_code_available=False,checkpoint_available=False))
  historical.append(dict(source='original_seed',stage=stage,model=model,base_seed=20260928,**metrics(g.label,g.score)))
 oldframes=[]
 for key,spec in old['completed_jobs'].items():
  raw=member(names[2],spec['prediction_filename']);assert sha(raw)==spec['prediction_sha256']
  p=pd.read_csv(io.BytesIO(raw),float_precision='round_trip');truth=frames[spec['stage']];truth=truth[truth.fold_5.eq(spec['fold'])]
  assert p.record_id.tolist()==truth.record_id.tolist()
  for col in ['label','fold_5','similarity_group_id']:assert p[col].tolist()==truth[col].tolist(),(key,col)
  assert p.model.eq('finetuned_'+spec['model_key']).all() and p.stage.eq(spec['stage']).all()
  assert p.base_seed.eq(spec['base_seed']).all() and p.effective_seed.eq(spec['base_seed']+100*spec['stage']+spec['fold']).all()
  assert p.score.between(0,1).all() and np.isfinite(p.score).all()
  checks.append(dict(source='historical_additional_seed',model=spec['model_key'],stage=spec['stage'],base_seed=spec['base_seed'],fold=spec['fold'],n=len(p),record_label_fold_checks=True,source_code_available='templates agree; executed-code hash absent',checkpoint_available=False))
  oldframes.append(p)
 p=pd.concat(oldframes,ignore_index=True)
 for (stage,model,seed),g in p.groupby(['stage','model','base_seed']):historical.append(dict(source='historical_additional_seed',stage=stage,model=model,base_seed=seed,**metrics(g.label,g.score)))
 table(30,'historical_artifact_audit.csv',pd.DataFrame(checks));table(30,'historical_seed_metrics.csv',pd.DataFrame(historical))
 table(30,'historical_additional_seed_predictions.csv.gz',p)
 source=[]
 for name in names:
  if not name.endswith('.ipynb'):continue
  n=nbformat.read(SOURCES/name,as_version=4)
  for i,c in enumerate(n.cells):
   if c.cell_type=='code':source.append(dict(notebook=name,cell=i,source_sha256=sha(c.source.encode()),bytes=len(c.source.encode()),kind='model definition' if i==4 and '11C' in name else 'training' if i==5 and '11C' in name else 'other'))
 table(30,'source_cell_audit.csv',pd.DataFrame(source))
 original_lock=jmember(names[1],'input_lock.json')
 shared={'epochs':(original_lock['config']['finetune_epochs'],old['config']['epochs']),
  'batch':(original_lock['config']['finetune_batch'],old['config']['batch_size']),
  'accumulation':(original_lock['config']['finetune_accum'],old['config']['gradient_accumulation']),
  'lr':(original_lock['config']['finetune_lr'],old['config']['lr'])}
 assert all(a==b for a,b in shared.values())
 settings=[dict(setting=k,original_declared=a,additional_declared=b,matching=a==b) for k,(a,b) in shared.items()]
 table(30,'declared_setting_alignment.csv',pd.DataFrame(settings))
 inputs={f'stage{s}_candidate_manifest.csv':audit['outputs_sha256'][f'stage{s}_candidate_manifest.csv'] for s in STAGES}
 protocol=dict(CONFIG,input_member_sha256=inputs)
 lock=sha(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode());dump(30,'training_protocol.json',dict(protocol=protocol,protocol_sha256=lock))
 dump(30,'statistical_reporting_plan.json',{'declared_before_new_training':True,'previously_examined_dataset':True,
  'primary_purpose':'classification seed stability: mean-fold AUROC per seed; sampleSD across3seeds; fold variation separately',
  'metrics':['AUROC','AUPRC','Brier','accuracy','balanced_accuracy','MCC','F1','sensitivity','specificity','confusion_counts'],
  'decision_threshold':.5,'ensembles':'arithmetic mean of3seed OOF probabilities on identical IDs; separate from mean seed/fold metrics',
  'secondary_contrasts':'DNABERT2 ensemble minus originalDNABERT ensemble, separately by stage; archived baselines labelled single-run',
  'bootstrap_repeats':1000,'bootstrap_seed_formula':'2026101300+stage','bootstrap_unit':'frozen similarity_group_id',
  'interval_scope':'fixed trained OOF model predictions on previously examined candidates; exploratory unadjusted; not algorithm-level confidence',
  'mixed_label_sensitivity':'exclude the known6Stage1records at evaluation only; no retraining',
  'collapse_policy':'retain all runs; flag scoreSD<1e-5; no selective exclusion, tuning or stopping based on OOF scores',
  'historical_results':'descriptive separate comparisons; no mixing into local seed SD',
  'calibration_plot':'10fixed equal-width probability bins; n<30 suppressed; descriptive only'})
 dump(30,'compatibility_decision.json',{'decision':'fresh_consistent_60_job_study','historical_prediction_files_valid':True,'historical_new_jobs':10,'historical_planned_additional_jobs':40,
  'data_labels_folds_declared_primary_settings_match':True,'original_training_implementation_verified':False,'historical_checkpoint_files_available':False,
  'reason':'Original seed complete optimizer/head/initialization/shuffle implementation and checkpoints unavailable; supplied notebook2 is only GENA correction. Matching config and template names do not prove identical executed implementations. Do not mix implementation/environment differences with seed variability.',
  'new_additional_seed_jobs':40,'original_seed_jobs_rerun':20,'total_local_jobs':60,'historical_results_policy':'Preserve separately; descriptive comparisons only, never selective reuse based on new outcomes.',
  'MPRA_fairness_status':'open; classification completion does not address activity-regression training adequacy'})
 protected=out(30)/'protected_before.json'
 if not protected.exists():atomic_json(protected,snapshot())
 priorlog=out(30)/'contribution_log_before.md'
 if not priorlog.exists():shutil.copy2(ROOT/'CONTRIBUTION_LOG.md',priorlog)
 (out(30)/'AUDIT_AND_DECISION.md').write_text('''# Compatibility audit and chosen completion route

All ten historical prediction hashes, IDs, labels, folds, similarity groups and effective-seed values pass. Both stages use the same authenticated Phase1 candidate manifests. Original and additional declared epochs, batch, accumulation and learning rate agree. Historical results are usable as archived evidence.

Full training comparability cannot be established. The original Phase2 training wrapper/checkpoints are unavailable; the supplied Phase2 notebook corrects GENA only. The Phase11C notebook templates agree on their model implementation but the result manifest does not authenticate the exact executed source. Later variants changed checkpoint storage/recovery. The saved older runtime also differs from current enhancer. No corrupted scores or failed model performance are inferred from these provenance gaps.

The chosen route is a consistent local three-seed study: rerun all40additional-seed jobs plus20original-seed jobs, using the locked data/folds and declared fixed2-epoch Phase11C recipe. All60jobs have the same implementation/runtime, auditable logs and saved full-precision model weights. This resolves the baseline comparability gap without mixing old and new settings or selecting runs by outcomes. The original20 and historical10 are preserved as separate historical results. It does not claim bitwise reproduction of missing historical training.

No outer OOF score selects training settings, epochs or thresholds. FoldSD and seedSD remain separate; overlapping folds and three seeds limit uncertainty interpretation. Activity-regression training fairness remains a separate open item. Earlier Phase20–29 packages remain historical snapshots and are not overwritten.
''',encoding='utf8')
 dump(30,'summary.json',{'status':'completed','artifact_audit_rows':len(checks),'completed_historical_jobs_validated':10,'original_seed_combinations_validated':4,
  'decision':'fresh_consistent_60_job_study','additional_seed_jobs':40,'original_seed_reruns':20,'total_jobs_to_run':60,'training_started':False})
 print('Audit passed; consistent60-job completion route frozen before new outcomes.')

def setup_runtime():
 assert Path(sys.prefix).name.lower()=='enhancer'
 vendor=ROOT/'reference_inputs/readiness_runtime/vendor';sys.path.insert(0,str(vendor))
 runtime=out(31)/'runtime';runtime.mkdir(exist_ok=True)
 for name,folder in [('HF_HOME','hf_home'),('HF_MODULES_CACHE','hf_modules'),('TRANSFORMERS_CACHE','hf_cache'),('TEMP','temp'),('TMP','temp'),('MPLCONFIGDIR','matplotlib_cache')]:
  p=runtime/folder;p.mkdir(exist_ok=True);os.environ[name]=str(p)
 for name in ['CUDA_CACHE_PATH','TORCH_HOME','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME']:
  p=runtime/name.lower();p.mkdir(exist_ok=True);os.environ[name]=str(p)
 os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 import torch,transformers
 assert transformers.__version__=='4.44.2' and torch.__version__=='2.11.0+cu128'
 assert torch.cuda.is_available()
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True,warn_only=True)
 return torch

def authenticate_models():
 rows=[]
 for model in MODELS:
  folder=INPUT/'models'/model;api=json.loads((folder/'model_api_revision.json').read_text())
  assert api['sha']==REVISIONS[model]
  for spec in api['siblings']:
   if spec['rfilename']=='.gitattributes':continue
   p=folder/spec['rfilename'];assert p.is_file() and p.stat().st_size==spec['size'],p
   h=fsha(p)
   if spec.get('lfs'):assert h==spec['lfs']['sha256'],p
   elif spec.get('blobId'):
    raw=p.read_bytes();git=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();assert git==spec['blobId'],p
   rows.append(dict(model=model,revision=REVISIONS[model],file=p.relative_to(ROOT).as_posix(),sha256=h,bytes=p.stat().st_size))
 table(30,'pinned_pretrained_model_files.csv',pd.DataFrame(rows));return rows

def load_backbone(model):
 from transformers import AutoTokenizer,AutoModel
 from transformers.models.bert.configuration_bert import BertConfig
 from transformers.dynamic_module_utils import get_class_from_dynamic_module
 folder=INPUT/'models'/model;tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
 cfg=BertConfig.from_pretrained(folder,local_files_only=True)
 if model=='dnabert2':
  cls=get_class_from_dynamic_module('bert_layers.BertModel',str(folder),local_files_only=True)
  import importlib
  module=importlib.import_module(cls.__module__);module.flash_attn_qkvpacked_func=None
  enc,loading=cls.from_pretrained(folder,config=cfg,local_files_only=True,output_loading_info=True)
 else:enc,loading=AutoModel.from_pretrained(folder,config=cfg,trust_remote_code=True,local_files_only=True,output_loading_info=True)
 missing=loading['missing_keys'];unexpected=loading['unexpected_keys']
 assert not loading.get('mismatched_keys') and not loading.get('error_msgs'),loading
 assert all('pooler' in k for k in missing),loading
 assert all(k.startswith('cls.') or k.startswith('bert.cls.') for k in unexpected),loading
 # Unused pretrained MLM heads/unused pooling weights are distinct from our random classification head.
 return tok,enc,{'missing_unused_pooler':missing,'unexpected_unused_MLM':unexpected}

def new_classifier(encoder):
 import torch
 from torch import nn
 class BinaryEncoder(nn.Module):
  def __init__(self,enc):
   super().__init__();self.encoder=enc;self.dropout=nn.Dropout(.1);self.head=nn.Linear(int(enc.config.hidden_size),2)
  def forward(self,b):
   output=self.encoder(**b);h=getattr(output,'last_hidden_state',None)
   if h is None:h=output[0]
   mask=b['attention_mask'].to(h.dtype).unsqueeze(-1);pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
   return self.head(self.dropout(pooled))
 return BinaryEncoder(encoder)

def tokenize(tok,model,seqs,device):
 text=[' '.join(s[i:i+6] for i in range(195)) for s in seqs] if model=='dnabert_legacy' else seqs
 b=tok(text,padding=True,add_special_tokens=True,truncation=False,return_tensors='pt')
 assert b['input_ids'].shape[1]<=256
 return {k:v.to(device) for k,v in b.items() if k in ['input_ids','attention_mask']}

def seed_everything(seed):
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def atomic_torch_save(path,value):
 import torch
 path=Path(path);tmp=path.with_suffix(path.suffix+'.writing');torch.save(value,tmp);os.replace(tmp,path)
 return fsha(path)

def phase31_setup():
 torch=setup_runtime();assert jread(30,'summary.json')['status']=='completed'
 model_files=authenticate_models()
 for folder in ['predictions','checkpoints','jobs','training_logs']:(out(31)/folder).mkdir(exist_ok=True)
 records=[]
 for model in MODELS:
  seed_everything(12345);tok,enc,info=load_backbone(model);net=new_classifier(enc).cuda()
  b=tokenize(tok,model,['ACGT'*50,'CG'*100],'cuda')
  net.eval()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):p=net(b).float()
  assert p.shape==(2,2) and torch.isfinite(p).all()
  net.train();net.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(net(b),torch.tensor([0,1],device='cuda'))
  loss.backward();assert torch.isfinite(loss) and net.head.weight.grad is not None and torch.isfinite(net.head.weight.grad).all()
  assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.encoder.parameters())
  records.append(dict(model=model,revision=REVISIONS[model],tokens=b['input_ids'].shape[1],parameters=sum(p.numel() for p in net.parameters()),loading=info,finite_forward_backward=True))
  del net,enc,tok,b,p,loss;gc.collect();torch.cuda.empty_cache()
 dump(31,'runtime_validation.json',{'python':platform.python_version(),'torch':torch.__version__,'transformers':'4.44.2','numpy':np.__version__,
  'gpu':torch.cuda.get_device_name(0),'models':records,'neural_training_runs_launched_by_probe':0,'probe':'two synthetic sequences; one backward pass; no optimizer step; all probe weights discarded'})
 # Freeze the exact code used for all actual jobs; continuation checks it before training.
 source=ROOT/'analysis_tools/phase11c_completion.py';lock=jread(30,'training_protocol.json')['protocol_sha256']
 run={'protocol_sha256':lock,'source_sha256':fsha(source),'statistical_reporting_plan_sha256':fsha(out(30)/'statistical_reporting_plan.json'),
  'model_weight_sha256':{m:next(x['sha256'] for x in model_files if x['model']==m and x['file'].endswith('pytorch_model.bin')) for m in MODELS}}
 saved=out(31)/'run_lock.json'
 if saved.exists():assert json.loads(saved.read_text())==run,'Do not change code/protocol during training; create a versioned follow-up for amendments.'
 else:atomic_json(saved,run)
 dump(31,'summary.json',{'status':'running','completed_jobs':0,'expected_jobs':60,'additional_seed_jobs':40,'original_seed_reruns':20})
 print('Both pinned pretrained models pass finite forward/backward checks;60-job run ready.')

def training_progress(**kwargs):
 value={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs};dump(31,'live_progress.json',value)

def predict(net,tok,model,seqs):
 import torch
 result=[];net.eval()
 with torch.inference_mode():
  for start in range(0,len(seqs),8):
   b=tokenize(tok,model,seqs[start:start+8],'cuda')
   with torch.autocast('cuda',dtype=torch.float16):p=torch.softmax(net(b).float(),dim=1)[:,1]
   result.extend(p.cpu().numpy().tolist())
 return np.asarray(result,float)

def rng_state():
 import torch
 n=np.random.get_state()
 return {'python':random.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
  'numpy':{'name':n[0],'keys':torch.from_numpy(n[1].astype(np.int64)),'pos':n[2],'has_gauss':n[3],'cached_gaussian':n[4]}}
def restore_rng(state):
 import torch
 random.setstate(state['python']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
 n=state['numpy'];np.random.set_state((n['name'],n['keys'].numpy().astype(np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))

def train_job(model,stage,base_seed,fold):
 torch=setup_runtime();from torch.utils.data import Dataset,DataLoader
 lock=jread(31,'run_lock.json');assert fsha(ROOT/'analysis_tools/phase11c_completion.py')==lock['source_sha256']
 protocol=jread(30,'training_protocol.json')['protocol_sha256'];assert lock['protocol_sha256']==protocol
 key=f'{model}__s{stage}__seed{base_seed}__fold{fold}';jobfile=out(31)/'jobs'/f'{key}.json'
 if jobfile.exists():
  saved=json.loads(jobfile.read_text());assert saved['protocol_sha256']==protocol
  for kind in ['prediction','checkpoint']:assert fsha(ROOT/saved[kind+'_file'])==saved[kind+'_sha256']
  print('REUSED completed local job',key,flush=True);return
 full=locked_data()[stage];tr=full[full.fold_5.ne(fold)].reset_index(drop=True);va=full[full.fold_5.eq(fold)].reset_index(drop=True)
 assert set(tr.similarity_group_id).isdisjoint(va.similarity_group_id)
 effective_seed=base_seed+stage*100+fold;seed_everything(effective_seed)
 tok,enc,loading=load_backbone(model);net=new_classifier(enc).cuda()
 opt=torch.optim.AdamW(net.parameters(),lr=2e-5,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
 sequences=tr.sequence.tolist();labels=tr.label.tolist();comp=str.maketrans('ACGT','TGCA')
 sequences+= [s.translate(comp)[::-1] for s in tr.sequence];labels+=tr.label.tolist()
 class Rows(Dataset):
  def __len__(self):return len(labels)
  def __getitem__(self,i):return sequences[i],labels[i]
 def collate(items):return tokenize(tok,model,[s for s,_ in items],'cuda'),torch.tensor([y for _,y in items],dtype=torch.long,device='cuda')
 start_epoch=0;start_batch=0;loss_sum=0.;loss_count=0;epoch_rows=[];resumed=False;started=time.time();overflow_steps=0;consecutive_overflows=0
 resume_path=out(31)/'current_training_resume.pt';resume_meta=out(31)/'current_training_resume.json'
 if resume_path.exists() and resume_meta.exists():
  meta=json.loads(resume_meta.read_text())
  if meta['job']==key:
   assert fsha(resume_path)==meta['sha256'] and meta['protocol_sha256']==protocol
   state=torch.load(resume_path,map_location='cpu',weights_only=True)
   assert state['job']==key and state['protocol_sha256']==protocol
   net.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
   start_epoch=state['next_epoch'];start_batch=state['next_batch'];loss_sum=state['loss_sum'];loss_count=state['loss_count'];epoch_rows=state['epoch_rows'];overflow_steps=state.get('overflow_steps',0)
   restore_rng(state['rng']);del state;resumed=True
   print('RESUMED',key,'epoch',start_epoch+1,'batch',start_batch,flush=True)
 def save_resume(next_epoch,next_batch,ls,lc):
  state={'job':key,'protocol_sha256':protocol,'model':{k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},
   'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'next_epoch':next_epoch,'next_batch':next_batch,
   'loss_sum':ls,'loss_count':lc,'epoch_rows':epoch_rows,'overflow_steps':overflow_steps,'rng':rng_state()}
  h=atomic_torch_save(resume_path,state);atomic_json(resume_meta,{'job':key,'protocol_sha256':protocol,'sha256':h,'next_epoch':next_epoch,'next_batch':next_batch});del state
 last_saved=time.monotonic()
 for epoch in range(start_epoch,2):
  generator=torch.Generator().manual_seed(effective_seed+epoch)
  loader=DataLoader(Rows(),batch_size=8,shuffle=True,generator=generator,collate_fn=collate,num_workers=0)
  net.train();opt.zero_grad(set_to_none=True)
  if epoch!=start_epoch:loss_sum=0.;loss_count=0
  for i,(b,y) in enumerate(loader):
   if epoch==start_epoch and i<start_batch:continue
   with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(net(b),y)
   assert torch.isfinite(loss),(key,epoch,i)
   scaler.scale(loss/2).backward();step=(i+1)%2==0 or i+1==len(loader)
   if step:
    scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
    # GradScaler legitimately skips overflowing fp16 steps and reduces its scale.
    before_scale=scaler.get_scale();scaler.step(opt);scaler.update();after_scale=scaler.get_scale()
    skipped=after_scale<before_scale;overflow_steps+=int(skipped);consecutive_overflows=consecutive_overflows+1 if skipped else 0
    assert skipped or torch.isfinite(norm),(key,epoch,i,'unhandled nonfinite gradient')
    assert consecutive_overflows<20,(key,'persistent nonfinite gradients')
    opt.zero_grad(set_to_none=True)
   loss_sum+=float(loss.detach().cpu());loss_count+=1
   if i%50==0 or i+1==len(loader):training_progress(job=key,model=model,stage=stage,seed=base_seed,fold=fold,epoch=epoch+1,batches_done=i+1,batches_total=len(loader),mean_training_loss=loss_sum/loss_count,elapsed_job_seconds=time.time()-started)
   if step and time.monotonic()-last_saved>=120:
    save_resume(epoch,i+1,loss_sum,loss_count);last_saved=time.monotonic()
  epoch_rows.append(dict(job=key,epoch=epoch+1,training_loss=loss_sum/loss_count,batches=loss_count,cumulative_AMP_overflow_skips=overflow_steps))
  pd.DataFrame(epoch_rows).to_csv(out(31)/'training_logs'/f'{key}.csv',index=False,float_format='%.17g')
  print(key,'epoch',epoch+1,'training_loss',round(loss_sum/loss_count,6),flush=True)
  save_resume(epoch+1,0,0.,0);last_saved=time.monotonic();start_batch=0
 p=predict(net,tok,model,va.sequence.tolist());assert len(p)==len(va) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 result=va[['record_id','similarity_group_id','label','fold_5']].copy();result['stage']=stage;result['model']='finetuned_'+model
 result['base_seed']=base_seed;result['effective_seed']=effective_seed;result['score']=p
 prediction=out(31)/'predictions'/f'{key}.csv';tmp=prediction.with_suffix('.csv.writing');result.to_csv(tmp,index=False,float_format='%.17g');os.replace(tmp,prediction)
 checkpoint=out(31)/'checkpoints'/f'{key}.pt'
 state={'state_dict':{k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},'protocol_sha256':protocol,'model_key':model,
  'model_revision':REVISIONS[model],'pretrained_weight_sha256':lock['model_weight_sha256'][model],'stage':stage,'base_seed':base_seed,
  'effective_seed':effective_seed,'fold':fold,'epochs':2,'training_record_ids_sha256':sha('\n'.join(tr.record_id).encode()),
  'evaluation_record_ids_sha256':sha('\n'.join(va.record_id).encode())}
 h=atomic_torch_save(checkpoint,state);del state
 meta={'job':key,'protocol_sha256':protocol,'source_sha256':lock['source_sha256'],'model':model,'stage':stage,'base_seed':base_seed,'effective_seed':effective_seed,'fold':fold,
  'training_n':len(tr),'augmented_training_n':len(sequences),'evaluation_n':len(va),'epochs':2,'prediction_file':prediction.relative_to(ROOT).as_posix(),
  'prediction_sha256':fsha(prediction),'checkpoint_file':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':h,
  'seconds_this_execution':time.time()-started,'resumed':resumed,'AMP_overflow_skipped_steps':overflow_steps,'loading_info':loading,'metrics':metrics(va.label,p)}
 atomic_json(jobfile,meta);completed=len(list((out(31)/'jobs').glob('*.json')))
 dump(31,'summary.json',{'status':'running' if completed<60 else 'training_complete_pending_aggregation','completed_jobs':completed,'expected_jobs':60,'additional_seed_jobs':40,'original_seed_reruns':20})
 training_progress(job=key,status='job_completed',completed_jobs=completed,expected_jobs=60,seconds_this_execution=meta['seconds_this_execution'])
 print('SAVED',completed,'/60',key,'AUROC',round(meta['metrics']['auroc'],6),'seconds',round(meta['seconds_this_execution'],1),flush=True)
 del net,enc,tok,opt,scaler,loader;gc.collect();torch.cuda.empty_cache()

def phase31_finish():
 rows=[];frames=[]
 for model,stage,seed,fold in itertools.product(MODELS,STAGES,SEEDS,FOLDS):
  key=f'{model}__s{stage}__seed{seed}__fold{fold}';meta=json.loads((out(31)/'jobs'/f'{key}.json').read_text())
  for kind in ['prediction','checkpoint']:assert fsha(ROOT/meta[kind+'_file'])==meta[kind+'_sha256']
  frames.append(pd.read_csv(ROOT/meta['prediction_file'],float_precision='round_trip'))
  rows.append({k:v for k,v in meta.items() if k not in ['loading_info','metrics']}|meta['metrics'])
 assert len(rows)==60
 allp=pd.concat(frames,ignore_index=True);assert len(allp)==25284 and not allp.duplicated(['stage','model','base_seed','record_id']).any()
 table(31,'all_local_oof_predictions.csv.gz',allp);table(31,'job_and_fold_metrics.csv',pd.DataFrame(rows))
 dump(31,'summary.json',{'status':'completed','completed_jobs':60,'expected_jobs':60,'additional_seed_jobs':40,'original_seed_reruns':20,'oof_prediction_rows':len(allp),
  'selection':'fixed epoch2; no outcome-driven hyperparameter/epoch/threshold selection','historical_results_mixed_into_primary':False,'independent_verification':'Phase32 required'})
 print('Completed all40additional-seed jobs plus20original-seed reruns;60consistent local fits saved.')

def phase32():
 from phase11c_verification import independent_verification
 independent_verification()

def phase33():
 from phase11c_verification import findings_and_evidence_update
 findings_and_evidence_update()
