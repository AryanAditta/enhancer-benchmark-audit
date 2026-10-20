"""Independent verification and new evidence supplement for the completed local seed study."""
from pathlib import Path
import os,sys,json,hashlib,zipfile,gc,time,math,itertools,shutil,platform
import numpy as np
import pandas as pd
import phase11c_completion as s

def protected_history_check():
 rows=[];original_workspace=(s.ROOT/'phase12_outputs').exists()
 for name,h in s.jread(30,'protected_before.json').items():
  p=s.ROOT/name
  if p.exists():assert s.fsha(p)==h,name;status='verified_unchanged'
  else:assert not original_workspace,name;status='not_in_standalone_supplement; original_workspace_check_retained'
  rows.append(dict(file=name,sha256=h,status=status))
 return rows

def independent_metrics(y,p):
 from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,matthews_corrcoef,f1_score
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);assert set(y)=={0,1} and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 hard=p>=.5;positive=y==1;negative=~positive
 tp=int(np.sum(hard&positive));tn=int(np.sum(~hard&negative));fp=int(np.sum(hard&negative));fn=int(np.sum(~hard&positive))
 return dict(n=len(y),negative=int(negative.sum()),positive=int(positive.sum()),auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),
  brier=float(brier_score_loss(y,p)),accuracy=(tn+tp)/len(y),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
  mcc=float(matthews_corrcoef(y,hard)),f1=float(f1_score(y,hard)),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),tn=tn,fp=fp,fn=fn,tp=tp)

def independent_verification():
 assert s.jread(31,'summary.json')['status']=='completed'
 torch=s.setup_runtime();frames=s.locked_data();run=s.jread(31,'run_lock.json')
 assert s.fsha(s.ROOT/'analysis_tools/phase11c_completion.py')==run['source_sha256']
 assert s.fsha(s.out(30)/'statistical_reporting_plan.json')==run['statistical_reporting_plan_sha256']
 checks=[];allpred=[];foldrows=[];profiles=[]
 cache=s.out(32)/'inference_cache';cache.mkdir(exist_ok=True)
 for model,stage,seed,fold in itertools.product(s.MODELS,s.STAGES,s.SEEDS,s.FOLDS):
  key=f'{model}__s{stage}__seed{seed}__fold{fold}';meta=json.loads((s.out(31)/'jobs'/f'{key}.json').read_text())
  assert meta['source_sha256']==run['source_sha256'] and meta['protocol_sha256']==run['protocol_sha256']
  for kind in ['prediction','checkpoint']:assert s.fsha(s.ROOT/meta[kind+'_file'])==meta[kind+'_sha256']
  d=frames[stage];tr=d[d.fold_5.ne(fold)];va=d[d.fold_5.eq(fold)]
  assert set(tr.similarity_group_id).isdisjoint(va.similarity_group_id)
  assert meta['training_n']==len(tr) and meta['augmented_training_n']==2*len(tr) and meta['evaluation_n']==len(va)
  p=pd.read_csv(s.ROOT/meta['prediction_file'],float_precision='round_trip')
  for col in ['record_id','similarity_group_id','label','fold_5']:assert p[col].tolist()==va[col].tolist(),(key,col)
  assert p.model.eq('finetuned_'+model).all() and p.base_seed.eq(seed).all() and p.stage.eq(stage).all()
  checkpoint=s.ROOT/meta['checkpoint_file'];ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
  for field,value in [('protocol_sha256',run['protocol_sha256']),('model_key',model),('model_revision',s.REVISIONS[model]),('base_seed',seed),('stage',stage),('fold',fold),('epochs',2),('effective_seed',seed+stage*100+fold),('pretrained_weight_sha256',run['model_weight_sha256'][model])]:assert ck[field]==value,(key,field)
  assert ck['training_record_ids_sha256']==s.sha('\n'.join(tr.record_id).encode())
  assert ck['evaluation_record_ids_sha256']==s.sha('\n'.join(va.record_id).encode())
  assert ck['state_dict']['head.weight'].shape==(2,768) and ck['state_dict']['head.bias'].shape==(2,)
  file=cache/f'{key}.csv';mfile=file.with_suffix('.json')
  if file.exists() and mfile.exists():
   cm=json.loads(mfile.read_text());assert cm['checkpoint_sha256']==meta['checkpoint_sha256'] and cm['prediction_sha256']==s.fsha(file)
   reproduced=pd.read_csv(file,float_precision='round_trip');q=reproduced.score.to_numpy()
   assert reproduced.record_id.tolist()==va.record_id.tolist()
  else:
   tok,encoder,loading=s.load_backbone(model);net=s.new_classifier(encoder).cuda();net.load_state_dict(ck['state_dict'],strict=True)
   q=s.predict(net,tok,model,va.sequence.tolist())
   reproduced=pd.DataFrame({'record_id':va.record_id,'score':q});reproduced.to_csv(file,index=False,float_format='%.17g')
   s.atomic_json(mfile,{'checkpoint_sha256':meta['checkpoint_sha256'],'prediction_sha256':s.fsha(file),'runtime':'same enhancer implementation and GPU; fixed8record inference batches'})
   del net,encoder,tok;gc.collect();torch.cuda.empty_cache()
  delta=np.abs(q-p.score.to_numpy());assert delta.max()<2e-6 and delta.mean()<1e-7,(key,delta.max(),delta.mean())
  independent=independent_metrics(p.label,p.score)
  for metric,value in independent.items():assert abs(value-meta['metrics'][metric])<1e-12,(key,metric)
  foldrows.append(dict(stage=stage,model='finetuned_'+model,base_seed=seed,fold_5=fold,**independent))
  checks.append(dict(job=key,model=model,stage=stage,base_seed=seed,fold=fold,n=len(p),checkpoint_sha256=meta['checkpoint_sha256'],max_prediction_difference=float(delta.max()),mean_prediction_difference=float(delta.mean()),metrics_recomputed=True,passed=True))
  profiles.append(dict(job=key,stage=stage,model=model,base_seed=seed,fold=fold,n=len(p),score_sd=float(p.score.std(ddof=0)),score_min=float(p.score.min()),score_max=float(p.score.max()),predicted_positive_fraction=float((p.score>=.5).mean()),near_constant=bool(p.score.std(ddof=0)<1e-5),AMP_overflow_skipped_steps=meta.get('AMP_overflow_skipped_steps',0)))
  allpred.append(p.assign(reproduced_score=q));del ck;gc.collect()
  s.table(32,'checkpoint_reproduction_checks.csv',pd.DataFrame(checks));s.dump(32,'live_progress.json',{'verified_jobs':len(checks),'expected':60,'last':key})
  print('Verified',len(checks),'/60',key,'maxdelta',float(delta.max()),flush=True)
 actual=pd.concat(allpred,ignore_index=True);assert len(actual)==25284
 schedule=actual[['model','stage','base_seed','fold_5','effective_seed']].drop_duplicates()
 assert len(schedule)==60 and schedule.groupby(['model','stage','fold_5']).effective_seed.nunique().eq(3).all()
 schedule['effective_seed_reused_across_folds']=schedule.duplicated(['model','stage','effective_seed'],keep=False)
 s.table(32,'effective_seed_schedule_and_dependence.csv',schedule)
 saved=s.read(31,'all_local_oof_predictions.csv.gz').sort_values(['stage','model','base_seed','record_id']).reset_index(drop=True)
 comp=actual.sort_values(['stage','model','base_seed','record_id']).reset_index(drop=True)
 for col in saved.columns:assert np.array_equal(saved[col],comp[col]),col
 s.table(32,'independently_verified_predictions.csv.gz',actual);s.table(32,'fold_metrics_independent.csv',pd.DataFrame(foldrows));s.table(32,'prediction_and_AMP_profiles.csv',pd.DataFrame(profiles))
 seedrows=[];ensemble=[];ensemble_rows=[];variability=[]
 for (stage,model,seed),g in actual.groupby(['stage','model','base_seed']):
  assert set(g.fold_5)==set(s.FOLDS) and set(g.record_id)==set(frames[stage].record_id)
  fm=pd.DataFrame(foldrows);fm=fm[fm.stage.eq(stage)&fm.model.eq(model)&fm.base_seed.eq(seed)]
  seedrows.append(dict(stage=stage,model=model,base_seed=seed,mean_fold_auroc=float(fm.auroc.mean()),fold_sd_auroc=float(fm.auroc.std(ddof=1)),mean_fold_auprc=float(fm.auprc.mean()),**independent_metrics(g.label,g.score)))
 for (stage,model),g in actual.groupby(['stage','model']):
  w=g.pivot(index='record_id',columns='base_seed',values='score');assert w.shape==(len(frames[stage]),3) and set(w.columns)==set(s.SEEDS)
  t=frames[stage].set_index('record_id').loc[w.index];p=w.mean(axis=1)
  reference=np.asarray([math.fsum(row)/3 for row in w[s.SEEDS].to_numpy()])
  assert np.max(abs(reference-p.to_numpy()))<1e-15
  reference_sd=np.sqrt(np.sum((w[s.SEEDS].to_numpy()-reference[:,None])**2,axis=1)/2)
  assert np.max(abs(reference_sd-w.std(axis=1,ddof=1).to_numpy()))<1e-15
  e=t[['label','similarity_group_id','fold_5']].copy();e['stage']=stage;e['model']=model;e['score']=p;e['seed_sd']=w.std(axis=1,ddof=1)
  ensemble.append(e.reset_index());ensemble_rows.append(dict(stage=stage,model=model,training_runs_per_fold=3,**independent_metrics(t.label,p)))
  fm=pd.DataFrame(foldrows);fm=fm[fm.stage.eq(stage)&fm.model.eq(model)]
  seed_means=fm.groupby('base_seed').auroc.mean();fold_means=fm.groupby('fold_5').auroc.mean()
  pooled=pd.DataFrame(seedrows);pooled=pooled[pooled.stage.eq(stage)&pooled.model.eq(model)]
  variability.append(dict(stage=stage,model=model,n_seeds=3,n_folds=5,mean_fold_auroc_across_seeds=float(seed_means.mean()),seed_sd_of_mean_fold_auroc=float(seed_means.std(ddof=1)),
   fold_sd_of_seed_mean_auroc=float(fold_means.std(ddof=1)),min_seed_mean_fold_auroc=float(seed_means.min()),max_seed_mean_fold_auroc=float(seed_means.max()),
   mean_seed_pooled_auroc=float(pooled.auroc.mean()),sd_seed_pooled_auroc=float(pooled.auroc.std(ddof=1)),ensemble_pooled_auroc=ensemble_rows[-1]['auroc']))
 e=pd.concat(ensemble,ignore_index=True);s.table(32,'three_seed_ensemble_oof_predictions.csv.gz',e);s.table(32,'seed_metrics_pooled_and_fold.csv',pd.DataFrame(seedrows))
 s.table(32,'seed_vs_fold_variability.csv',pd.DataFrame(variability));s.table(32,'ensemble_metrics.csv',pd.DataFrame(ensemble_rows))
 fm=pd.DataFrame(foldrows);paired=fm.pivot(index=['stage','base_seed','fold_5'],columns='model',values='auroc').reset_index()
 paired['dnabert2_minus_original_dnabert']=paired.finetuned_dnabert2-paired.finetuned_dnabert_legacy
 s.table(32,'paired_seed_fold_differences.csv',paired)
 # Separate comparison with frozen single-run legacy baselines; do not pretend equal seed/tuning budgets.
 old=s.csv('2_enhancer_phase2_corrected_results.zip','all_oof_predictions.csv');rows=[]
 for (stage,model),g in old[~old.model.str.startswith('finetuned_')].groupby(['stage','model']):rows.append(dict(stage=stage,model=model,source='archived_Phase2_single_run',training_runs_per_fold=1,**independent_metrics(g.label,g.score)))
 for row in ensemble_rows:rows.append(dict(source='new_local_three_seed_ensemble',**row))
 s.table(32,'updated_classification_comparison.csv',pd.DataFrame(rows))
 sensitivity=[]
 ambiguous=frames[1].groupby('similarity_group_id').filter(lambda g:g.label.nunique()>1);assert len(ambiguous)==6
 for model,g in e[e.stage.eq(1)].groupby('model'):
  h=g[~g.record_id.isin(ambiguous.record_id)];sensitivity.append(dict(stage=1,model=model,excluded_records=6,training_repeated=False,**independent_metrics(h.label,h.score)))
 s.table(32,'mixed_label_sensitivity.csv',pd.DataFrame(sensitivity))
 historical=[]
 original=old[old.model.str.startswith('finetuned_')].copy();original['base_seed']=20260928;original['source']='original_Phase2'
 additional=s.read(30,'historical_additional_seed_predictions.csv.gz');additional['source']='archived_Phase11C'
 for (source,stage,model,seed),g in pd.concat([original,additional],ignore_index=True).groupby(['source','stage','model','base_seed']):
  new=actual[actual.stage.eq(stage)&actual.model.eq(model)&actual.base_seed.eq(seed)]
  pair=g[['record_id','score','label']].merge(new[['record_id','score']],on='record_id',suffixes=('_historical','_new'),validate='one_to_one')
  assert len(pair)==len(frames[stage]);before=independent_metrics(pair.label,pair.score_historical);after=independent_metrics(pair.label,pair.score_new)
  historical.append(dict(source=source,stage=stage,model=model,base_seed=seed,n=len(pair),historical_auroc=before['auroc'],new_auroc=after['auroc'],difference=after['auroc']-before['auroc'],
   score_correlation=float(np.corrcoef(pair.score_historical,pair.score_new)[0,1]),interpretation='Descriptive only; historical training implementation/runtime comparability unverified. Not bitwise training reproduction.'))
 s.table(32,'historical_comparison_separate.csv',pd.DataFrame(historical))
 paired_ensemble_intervals(e)
 s.table(32,'protected_history_checks.csv',pd.DataFrame(protected_history_check()))
 s.dump(32,'summary.json',{'status':'completed','checkpoints_reproduced':60,'all_prediction_checks_passed':True,'oof_rows_verified':25284,'seed_metric_rows':12,'ensemble_rows':4,
  'seed_fold_metric_rows':60,'paired_seed_fold_rows':30,'historical_runs_in_primary':False,'near_constant_fold_runs':int(pd.DataFrame(profiles).near_constant.sum()),
  'older_outputs_unchanged':True,'uncertainty_scope':'Fixed trained models and previously examined grouped OOF cohorts; fold dependence and only3seeds disclosed; no algorithm-level or fresh-test guarantee.'})
 print('All60weights independently reproduce saved predictions; metrics/ensembles/variability verified.')

def paired_ensemble_intervals(e):
 from sklearn.metrics import roc_auc_score,average_precision_score
 rows=[]
 for stage,g in e.groupby('stage'):
  w=g.pivot(index='record_id',columns='model',values='score');truth=g.drop_duplicates('record_id').set_index('record_id').loc[w.index]
  y=truth.label.to_numpy();codes=pd.factorize(truth.similarity_group_id,sort=True)[0];n_groups=int(codes.max()+1);rng=np.random.default_rng(2026101300+stage)
  p=w[['finetuned_dnabert2','finetuned_dnabert_legacy']].to_numpy();samples={'auroc':[],'auprc':[]};undefined=0
  for draw in range(1000):
   picked=rng.integers(0,n_groups,n_groups);weights=np.bincount(picked,minlength=n_groups)[codes]
   if not weights[y==0].sum() or not weights[y==1].sum():undefined+=1;continue
   for name,fn in [('auroc',roc_auc_score),('auprc',average_precision_score)]:samples[name].append(fn(y,p[:,0],sample_weight=weights)-fn(y,p[:,1],sample_weight=weights))
   if draw==0:
    ix=np.repeat(np.arange(len(y)),weights)
    for fn in [roc_auc_score,average_precision_score]:assert abs(fn(y,p[:,0],sample_weight=weights)-fn(y[ix],p[ix,0]))<1e-12
  for name,fn in [('auroc',roc_auc_score),('auprc',average_precision_score)]:
   lo,hi=np.quantile(samples[name],[.025,.975]);rows.append(dict(stage=stage,contrast='DNABERT2three_seed_ensemble minus originalDNABERTthree_seed_ensemble',metric=name,
    n_records=len(y),n_similarity_groups=n_groups,estimate=fn(y,p[:,0])-fn(y,p[:,1]),lower_95=lo,upper_95=hi,repeats=1000,undefined_draws=undefined,
    resampling_unit='frozen LCS160/exactRC similarity group',scope='conditional on fitted OOF models and previously examined candidates; not training-algorithm uncertainty; secondary unadjusted comparison'))
 s.table(32,'conditional_paired_ensemble_intervals.csv',pd.DataFrame(rows))

def generate_figures():
 os.environ['MPLCONFIGDIR']=str(s.out(33)/'matplotlib_cache')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from sklearn.metrics import roc_curve,precision_recall_curve
 plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
 folder=s.out(33)/'figures';folder.mkdir(exist_ok=True);index=[]
 color={'finetuned_dnabert2':'#3d5a80','finetuned_dnabert_legacy':'#cc7735'}
 label={'finetuned_dnabert2':'DNABERT-2','finetuned_dnabert_legacy':'Original DNABERT'}
 def save(fig,name,caption,source):
  for ext in ['png','svg','pdf']:
   p=folder/f'{name}.{ext}';fig.savefig(p,dpi=220,bbox_inches='tight');index.append(dict(figure=name,format=ext,file=p.relative_to(s.ROOT).as_posix(),sha256=s.fsha(p),caption=caption,source=source,placement='classification section or supplement; avoid merging with MPRA activity figures'))
  plt.close(fig)
 seed=s.read(32,'seed_metrics_pooled_and_fold.csv');fold=s.read(32,'fold_metrics_independent.csv');ens=s.read(32,'ensemble_metrics.csv');p=s.read(32,'three_seed_ensemble_oof_predictions.csv.gz')
 fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for ax,stage in zip(axes,[1,2]):
  for x,model in enumerate(color):
   values=seed[seed.stage.eq(stage)&seed.model.eq(model)].mean_fold_auroc.to_numpy();ax.scatter(x+np.linspace(-.12,.12,3),values,c=color[model],s=35)
   ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),fmt='s',c='black',capsize=5)
  ax.set_xticks([0,1],[label[m] for m in color]);ax.set_ylabel('Mean five-fold AUROC');ax.set_title(f'Stage{stage}: three declared base-seed runs')
 save(fig,'C1_seed_variability','Coloured points are per-seed mean-fold AUROC; black square/error bar is mean±sampleSD across3seeds. Fold SD is reported separately. No fresh-test or algorithm-level CI.','phase32 seed_metrics_pooled_and_fold.csv')
 fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
 for i,stage in enumerate([1,2]):
  for model in color:
   g=p[p.stage.eq(stage)&p.model.eq(model)];fpr,tpr,_=roc_curve(g.label,g.score);precision,recall,_=precision_recall_curve(g.label,g.score)
   axes[i,0].plot(fpr,tpr,c=color[model],label=label[model]);axes[i,1].plot(recall,precision,c=color[model],label=label[model])
  axes[i,0].plot([0,1],[0,1],c='gray',ls='--');axes[i,0].set(xlabel='False-positive rate',ylabel='True-positive rate',title=f'Stage{stage} pooled OOF ROC')
  axes[i,1].set(xlabel='Recall',ylabel='Precision',title=f'Stage{stage} pooled OOF precision–recall')
  for ax in axes[i]:ax.legend()
 save(fig,'C2_three_seed_ROC_PR','Curves use average OOF probabilities across3localseeds. Pooled ensemble AUROC differs from mean-fold or mean-seed metrics. Previously examined grouped candidates.','phase32 three_seed_ensemble_oof_predictions.csv.gz')
 fig,axes=plt.subplots(2,2,figsize=(8,7),constrained_layout=True)
 for ax,(stage,model) in zip(axes.ravel(),itertools.product([1,2],color)):
  row=ens[ens.stage.eq(stage)&ens.model.eq(model)].iloc[0];cm=np.array([[row.tn,row.fp],[row.fn,row.tp]],int);ax.imshow(cm,cmap='Blues')
  for y,x in itertools.product([0,1],[0,1]):ax.text(x,y,str(cm[y,x]),ha='center',va='center',fontsize=13,color='white' if cm[y,x]>.55*cm.max() else 'black')
  labels=['Non-enhancer','Enhancer'] if stage==1 else ['Weak','Strong'];ax.set_xticks([0,1],labels);ax.set_yticks([0,1],labels);ax.set(xlabel='Predicted',ylabel='Measured label',title=f'Stage{stage} — {label[model]}')
 save(fig,'C3_ensemble_confusion_matrices','Pooled three-seed OOF confusion matrices at the predeclared0.5threshold. Stage2positive=strong. Counts are independently verified.','phase32 ensemble_metrics.csv')
 fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);calibration=[]
 for ax,stage in zip(axes,[1,2]):
  ax.plot([0,1],[0,1],c='gray',ls='--')
  for model in color:
   g=p[p.stage.eq(stage)&p.model.eq(model)].copy();g['bin']=np.minimum((g.score*10).astype(int),9)
   bins=g.groupby('bin').agg(n=('label','size'),mean_probability=('score','mean'),positive_fraction=('label','mean')).reset_index()
   for row in bins.itertuples():calibration.append(dict(stage=stage,model=model,bin=row.bin,n=row.n,mean_probability=row.mean_probability,positive_fraction=row.positive_fraction,suppressed=row.n<30))
   bins=bins[bins.n>=30];ax.plot(bins.mean_probability,bins.positive_fraction,'o-',c=color[model],label=label[model])
  ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title=f'Stage{stage}: descriptive reliability');ax.legend()
 save(fig,'C4_probability_reliability','Fixed10equal-width probability bins; n<30bins suppressed, full counts saved. This is descriptive OOF reliability, not fitted calibration or a deployment guarantee.','phase32 verified ensemble probabilities')
 s.table(33,'reliability_bin_counts.csv',pd.DataFrame(calibration))
 fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);delta=s.read(32,'paired_seed_fold_differences.csv')
 for ax,stage in zip(axes,[1,2]):
  g=delta[delta.stage.eq(stage)]
  for i,seednum in enumerate(s.SEEDS):
   h=g[g.base_seed.eq(seednum)];ax.scatter(h.fold_5+i*.05,h.dnabert2_minus_original_dnabert,label=str(seednum),s=30)
  ax.axhline(0,c='black',ls='--');ax.set(xticks=list(range(5)),xlabel='Frozen grouped fold',ylabel='DNABERT-2 minus original DNABERT AUROC',title=f'Stage{stage} paired seed/fold differences');ax.legend(title='Base seed')
 save(fig,'C5_paired_fold_differences','Each point compares models on the same fold and base seed. Overlapping fold training sets and shared candidates prevent treating15points as independent replicates.','phase32 paired_seed_fold_differences.csv')
 fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True);logs=[]
 for file in (s.out(31)/'training_logs').glob('*.csv'):
  d=pd.read_csv(file);meta=json.loads((s.out(31)/'jobs'/file.with_suffix('.json').name).read_text());logs.append(d.assign(stage=meta['stage'],model=meta['model'],base_seed=meta['base_seed'],fold=meta['fold']))
 logs=pd.concat(logs,ignore_index=True)
 for ax,(stage,model) in zip(axes.ravel(),itertools.product([1,2],s.MODELS)):
  g=logs[logs.stage.eq(stage)&logs.model.eq(model)]
  for (seednum,foldnum),h in g.groupby(['base_seed','fold']):ax.plot(h.epoch,h.training_loss,alpha=.25,c=color['finetuned_'+model])
  mean=g.groupby('epoch').training_loss.mean();ax.plot(mean.index,mean.values,c='black',marker='o',lw=2)
  ax.set(xticks=[1,2],xlabel='Fixed training epoch',ylabel='Training cross-entropy',title=f'Stage{stage} — {label["finetuned_"+model]}')
 save(fig,'C6_training_loss_diagnostics','All15seed/fold training curves per model/task; black line is descriptive mean. Two recorded epochs do not establish convergence or MPRA training adequacy.','phase31 training_logs; job metadata')
 s.table(33,'training_loss_all_jobs.csv',logs);s.table(33,'figure_index.csv',pd.DataFrame(index))
 captions=['# Classification completion figure captions']
 for name,g in pd.DataFrame(index).groupby('figure',sort=False):captions.append('\n## '+name+'\n\n'+g.iloc[0].caption+'\n\nSource: '+g.iloc[0].source)
 (s.out(33)/'FIGURE_CAPTIONS.md').write_text('\n'.join(captions)+'\n',encoding='utf8')
 paths=sorted(folder.glob('*.png'));fig,axes=plt.subplots(3,2,figsize=(15,15))
 for ax,path in zip(axes.ravel(),paths):ax.imshow(plt.imread(path));ax.set_title(path.stem);ax.axis('off')
 fig.tight_layout();fig.savefig(s.out(33)/'figure_contact_sheet.png',dpi=130);plt.close(fig)

def findings_and_evidence_update():
 assert s.jread(32,'summary.json')['status']=='completed'
 generate_figures()
 sources=[(32,'seed_metrics_pooled_and_fold.csv'),(32,'seed_vs_fold_variability.csv'),(32,'ensemble_metrics.csv'),(32,'updated_classification_comparison.csv'),
  (32,'fold_metrics_independent.csv'),(32,'conditional_paired_ensemble_intervals.csv'),(32,'paired_seed_fold_differences.csv'),(32,'mixed_label_sensitivity.csv'),
  (32,'historical_comparison_separate.csv'),(32,'checkpoint_reproduction_checks.csv'),(32,'prediction_and_AMP_profiles.csv')]
 tabledir=s.out(33)/'tables';tabledir.mkdir(exist_ok=True);index=[]
 for number,name in sources:
  source=s.out(number)/name;target=tabledir/name;shutil.copy2(source,target);index.append(dict(file=target.relative_to(s.ROOT).as_posix(),sha256=s.fsha(target),source=source.relative_to(s.ROOT).as_posix(),rows=len(pd.read_csv(target))))
 settings=[];probe=s.jread(31,'runtime_validation.json')
 for model in s.MODELS:
  record=next(x for x in probe['models'] if x['model']==model)
  settings.append(dict(model=model,model_revision=s.REVISIONS[model],parameters=record['parameters'],input_bp=200,
   tokenization='overlapping6mers;195DNAwords plus CLS/SEP' if model=='dnabert_legacy' else 'pinned byte-pair tokenizer; variable token length',
   head='masked mean of unpadded hidden states including special tokens; dropout0.1; Linear768to2',optimizer='AdamW',learning_rate=2e-5,weight_decay=.01,
   epochs=2,batch_size=8,gradient_accumulation=2,effective_full_batch=16,gradient_clip_norm=1.,AMP='fp16 with GradScaler; skips recorded',
   training_augmentation='original and RC rows after grouped split',scoring_orientation='original only',threshold=.5,seed_count=3,fold_count=5,
   epoch_selection='fixed final epoch2; no OOF selection',folds='authenticated fixed LCS160/exactRC groups; shared by all jobs'))
 file=tabledir/'classification_training_settings.csv';pd.DataFrame(settings).to_csv(file,index=False,float_format='%.17g')
 index.append(dict(file=file.relative_to(s.ROOT).as_posix(),sha256=s.fsha(file),source='phase30 training_protocol.json; phase31 runtime_validation.json and frozen source',rows=2))
 s.table(33,'table_index.csv',pd.DataFrame(index))
 values=s.read(32,'seed_vs_fold_variability.csv');ensembles=s.read(32,'ensemble_metrics.csv');ci=s.read(32,'conditional_paired_ensemble_intervals.csv')
 lines=['# Completed classification seed study','This supplement supersedes the earlier exclusion of incomplete Phase11C completion-dependent claims for the two fine-tuned classification arms. Earlier packages remain preserved historical snapshots. The MPRA fairness decision remains open.']
 for row in values.itertuples():
  name='DNABERT-2' if row.model=='finetuned_dnabert2' else 'original DNABERT'
  en=ensembles[ensembles.stage.eq(row.stage)&ensembles.model.eq(row.model)].iloc[0]
  lines.append(f'\n- Stage{row.stage}, {name}: mean five-fold AUROC across three seeds **{row.mean_fold_auroc_across_seeds:.4f} ± {row.seed_sd_of_mean_fold_auroc:.4f}** (seed SD); three-seed pooled OOF ensemble AUROC **{en.auroc:.4f}**, AUPRC **{en.auprc:.4f}**, accuracy **{en.accuracy:.4f}**, MCC **{en.mcc:.4f}**, n={int(en.n):,}. Fold SD is separately {row.fold_sd_of_seed_mean_auroc:.4f}.')
 for row in ci[ci.metric.eq('auroc')].itertuples():lines.append(f'\n- Stage{row.stage} ensemble AUROC difference, DNABERT-2 minus original DNABERT: {row.estimate:+.4f}, conditional95% similarity-group bootstrap interval [{row.lower_95:+.4f}, {row.upper_95:+.4f}]. This is conditional on these fitted OOF models and examined candidates.')
 lines += ['''
## What was completed

All40declared additional-seed jobs plus20original-seed reruns were executed with one fixed local implementation, the original candidate manifests/folds and the declared two-epoch classification recipe. All60checkpoints and25,284OOFprediction rows were independently checked; saved predictions reproduce from the weights. Historical20original-seed and10additional-seed fits remain separate because their exact implementation comparability could not be established. The old predictions were not corrupted and were not selected for replacement based on outcomes.

The new results provide separate seed variation, fold variation, seed-pooled metrics, ensemble metrics, confusion matrices, ROC/PR, descriptive probability reliability and paired model comparisons. The prior seven non-fine-tuned classification arms are retained only as labelled single-run archived comparisons; they are not presented as equally replicated or compute matched.

## Remaining limits

Only three declared base-seed runs and five fixed folds; overlapping training sets; previously examined candidates; unknown pretraining membership; LCS160graph separation is not chromosome-disjoint validation or proof of absence of homology. The inherited effective-seed formula uses base_seed+100*stage+fold: the two adjacent additional base seeds reuse some initialization seeds across different folds, although every same-fold comparison has three distinct effective seeds. The complete schedule is saved; across-seed SD is descriptive and the15seed/fold points are not independent replicates. Mixed-label sensitivity is an evaluation exclusion without retraining. The historical original implementation remains unavailable, so this is a documented consistent rerun rather than exact reproduction of old training. Two-epoch classification completion does not establish training convergence or resolve MPRA-regression undertraining/fairness. Longer context/native-locus/broader-cell claims remain excluded.

## Manuscript integration

Use the new Phase33classification supplement for the two fine-tuned arms and clearly separate its three-seed metrics from earlier single-run results. Earlier Phase25T3/S1/S2/S19/S20 and Phase26classification graphics remain historical; do not silently combine their claims/intervals with the new study. Regression/filtering evidence remains as previously frozen. Update the Methods to the supplied actual training code, and state the fixed epoch/threshold/augmentation policy. Manuscript writing, final formatting and submission remain separate tasks.
''']
 (s.out(33)/'CLASSIFICATION_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 changes=[('Aditta2_Phase11C','completed_consistent_rerun','All40additional jobs plus20baseline reruns complete;60saved/reproduced checkpoints and three-seed classification evidence.','phase32_outputs/summary.json'),
  ('Aditta1_MPRA_fairness','open_scientific_choice','Classification seed study does not test activity-regression training adequacy; bounded claims still required until separate fair-training investigation.','phase30_outputs/compatibility_decision.json'),
  ('R1.3_classification_seed_variation','addressed_with_limits','New seed/fold SD separated; allfolds retained; three seeds only.','phase32_outputs/seed_vs_fold_variability.csv'),
  ('R1.4_classification_training_implementation','new_implementation_documented','Actual new source, fixed settings, checkpoints and logs available; historical wrapper still unavailable.','phase31_outputs/run_lock.json'),
  ('historical_performance_claims','retain_exclusions','Unverified original submitted tables and long-context claims remain excluded.','phase21_outputs/submitted_table_dispositions.csv')]
 s.table(33,'updated_readiness_and_reviewer_map.csv',pd.DataFrame(changes,columns=['item','status','finding','evidence']))
 (s.out(33)/'UPDATED_READINESS_STATUS.md').write_text('''# Updated readiness after Phase11C completion

The repeated-seed classification study is now completed through a consistent local rerun:40additional-seed fits and20rerun original-seed fits. The older incomplete study remains archived, not retroactively marked complete. New evidence and actual training code replace the earlier exclusion for these completed classification claims.

Aditta's training-fairness question for MPRA activity regression remains open. The earlier readiness report closed that item by choosing bounded claims; that remains a scope option, not a completed fairness experiment. Do not say every scientific issue is solved. The new classifier seed study and MPRA training adequacy are distinct.

The verified regression/filtering package remains available and unchanged. Manuscript writing can use the frozen bounded scope and this classification supplement, with stated limitations. Visual QA and final archive sealing are recorded after notebook33finishes; see FINAL_VALIDATION.json and the supplemental ZIP.
''',encoding='utf8')
 for name in ['environment.yml','requirements-win-cuda.txt','environment_versions.json']:
  previous=s.ROOT/'phase28_outputs/Enhancer_PreWriting_Reproducibility'/name
  if previous.exists():shutil.copy2(previous,s.out(33)/name)
  else:assert (s.out(33)/name).exists(),'Environment specification missing from relocated supplement.'
 citations=s.ROOT/'phase27_outputs/retained_model_citation_map.csv'
 if citations.exists():
  c=pd.read_csv(citations);c=c[c.model.isin(['DNABERT original','DNABERT-2'])]
  s.table(33,'model_primary_citations.csv',c)
 (s.out(33)/'README_REPRODUCTION.md').write_text('''# Classification completion supplement

Extract into a writable directory and use the existing enhancer Anaconda environment. The original run used Windows/Python3.12/PyTorch2.11.0+cu128/RTX5060Ti16GB, Transformers4.44.2 and the bundled isolated vendor dependencies. Exact specifications are supplied in environment.yml, requirements-win-cuda.txt and environment_versions.json. No global installation is needed on the original machine. Native requirements are needed if reproducing on another OS; bundled vendor wheels target Windows/Python3.12.

Run current notebooks30–33 from the extracted root with `python -B analysis_tools/execute_followup.py 32`, or open them in enhancer. Notebook30 uses authenticated local input copies; notebook31 checks completed-job hashes and skips already completed training. Fresh training or checkpoint re-inference requires a compatible CUDA GPU; figures/tables are ordinary CPU analyses. `python -B -c "import sys;sys.path.insert(0,'analysis_tools');import phase11c_verification as v;v.generate_figures()"` regenerates figures from the saved verified probabilities. All scientific inputs are local; no model download is needed after extraction.

All60full-precision classifier checkpoints are included, together with the two original pinned pretrained model snapshots, both frozen candidate manifests, historical OOF archives, new predictions, training logs, source code, indices and licenses. Historical experiments outside this supplement are not needed for new classification calculations; the recorded original-workspace protection audit remains evidence of preservation. Existing earlier regression/filtering packages stay separate. Old-source notebooks in phase30_inputs are references, not execution entry points.

The study uses3seeds×2models×2tasks×5folds. Forty are the declared additional-seed fits; twenty rerun the original seed for consistent implementation/runtime. Historical predictions are separate and not pooled. Fixed two epochs and threshold0.5 remain unchanged by results. Seed/fold/ensemble metrics are distinct, intervals are conditional on trained OOF models, and MPRA activity-regression training fairness remains open. No bitwise reproduction of missing historical training is claimed.

The supplemental archive is sealed after notebook33and visualQAfinish. FINAL_VALIDATION.json is external to the ZIP because an archive cannot contain its own final hash. package_manifest.csv contains file SHA256/size records. Copied software/model licenses retain their original terms.
''',encoding='utf8')
 # Source/evidence hashes, including current implementation and every numerical source.
 files=[s.ROOT/'analysis_tools/phase11c_completion.py',s.ROOT/'analysis_tools/phase11c_verification.py']
 files += [p for n in [30,31,32,33] for p in s.out(n).rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.md'] and 'inference_cache' not in p.parts and 'runtime' not in p.parts]
 s.table(33,'evidence_file_index.csv',pd.DataFrame([dict(file=p.relative_to(s.ROOT).as_posix(),sha256=s.fsha(p),bytes=p.stat().st_size) for p in sorted(set(files))]))
 s.dump(33,'summary.json',{'status':'completed_evidence_prepared_pending_visual_QA_and_archive_seal','classification_training_jobs':60,'additional_seed_jobs':40,'original_seed_reruns':20,
  'verified_checkpoints':60,'publication_tables':len(index),'figures':6,'historical_outputs_preserved':True,'MPRA_fairness':'open; no activity-training fairness experiment run','manuscript_finished':False})
 print('New classification findings, tables, figures and readiness/reviewer updates prepared.')

def finalize():
 import nbformat
 assert s.jread(32,'summary.json')['status']=='completed'
 qa=s.jread(33,'visual_QA.json');assert qa['passed'] and qa['figures_inspected']==6
 notebooks=[]
 for n in range(30,34):
  p=next(s.ROOT.glob(f'[[]{n}]*.ipynb'));nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
  codes=[c for c in nb.cells if c.cell_type=='code']
  assert all(c.execution_count is not None for c in codes)
  assert not any(o.output_type=='error' for c in codes for o in c.get('outputs',[]))
  notebooks.append(dict(phase=n,file=p.relative_to(s.ROOT).as_posix(),sha256=s.fsha(p),code_cells=len(codes),fully_executed=True,errors=0))
 s.table(33,'notebook_execution_validation.csv',pd.DataFrame(notebooks))
 protected_history_check()
 oldlog=(s.out(30)/'contribution_log_before.md').read_text(encoding='utf8');log=s.ROOT/'CONTRIBUTION_LOG.md';text=log.read_text(encoding='utf8');assert text.startswith(oldlog)
 marker='### Completed Phases 30–33: classification seed-study completion'
 if marker not in text:
  rows=s.read(32,'seed_vs_fold_variability.csv');findings=[]
  for row in rows.itertuples():findings.append(f'- Stage {row.stage}, {row.model}: mean five-fold AUROC {row.mean_fold_auroc_across_seeds:.6f}; across-seed SD {row.seed_sd_of_mean_fold_auroc:.6f}; pooled three-seed ensemble AUROC {row.ensemble_pooled_auroc:.6f}.')
  addition='''\n\n### Completed Phases 30–33: classification seed-study completion (2026-10-13)

This entry supersedes the earlier decision to exclude completion-dependent Phase11C claims for the two fine-tuned classifier arms. The old10-job study and all earlier outputs remain archived unchanged. The user explicitly authorized the audit and completion/rerun routes.

- **Phase30:** authenticated all10historical prediction files, IDs, labels, folds/groups and effective seeds; original-seed data/config also match. Exact original training wrapper/checkpoints are unavailable, so config equality alone cannot establish comparability. Selected a consistent local60-job study before seeing new results:40additional-seed fits plus20original-seed reruns. Historical runs were preserved separately, not discarded as invalid or selected by outcomes.
- **Phase31:** all60jobs executed in enhancer with pinned pretrained model snapshots, original fixed folds, declared seeds, fixed2epochs, training-fold-only RC augmentation, AdamW2e-5, batch8/accumulation2 and fixed0.5classification threshold. No OOF outcomes tune settings/epochs/thresholds. Full-precision checkpoints, epoch-loss/AMP logs, per-job manifests and resumable optimizer/RNG state saved inside this contribution folder. Initial/pretrained weights are SHAauthenticated and distinct from existing fine-tuned regression weights. Local vendor dependencies are reused; no global package/system changes.
- **Phase32:** independently reproduced every checkpoint's OOF predictions and recalculated all60fold metrics,12seed-pooled summaries,4ensembles, seed-versus-fold SD,30paired seed/fold effects, conditional similarity-group intervals and mixed-label sensitivity. Verified25,284prediction rows and identical evaluation records. Any nearconstant run remains visible; no failed seed is silently omitted. Archived single-run non-fine-tuned baselines remain labelled separately.
- **Phase33:** prepared source-indexed numerical/settings tables and six figures in PNG/SVG/PDF, findings/captions, exact evidence index and an updated readiness/reviewer map. Visual QA, notebook execution checks, protected historical hashes and the final archive are verified by the finalizer. Original research documents and the earlier reproducibility ZIP are unchanged; no manuscript submission or external message was sent.

New seed-study results:
'''+ '\n'.join(findings)+'''

**Scope:** this completes classification seed-stability evidence under the declared two-epoch recipe. MPRA activity-regression training fairness/undertraining remains open and was not tested by this study. Three seeds, dependent fixed folds, previously examined cohorts, unknown pretraining membership and incomplete homology exclusion limit generalization. This is a consistent documented rerun, not claimed exact reproduction of missing historical training. All copied inputs, downloads, runtime files, training/verification outputs and logs stay inside Shafin's Contribution.
'''
  log.write_text(text.rstrip()+addition,encoding='utf8')
 files=[]
 for n in range(30,34):
  files += [p for p in s.out(n).rglob('*') if p.is_file() and not any(x in p.parts for x in ['runtime','inference_cache','matplotlib_cache']) and p.name not in ['current_training_resume.pt','FINAL_VALIDATION.json','package_manifest.csv'] and not p.name.endswith(('.zip','.writing'))]
 files += [p for p in s.INPUT.rglob('*') if p.is_file() and 'weight_ranges' not in p.parts and not any(x in p.name for x in ['.downloading','.alternative','.headers','probe'])]
 files += [s.ROOT/x['file'] for x in notebooks]+[s.ROOT/'CONTRIBUTION_LOG.md',s.ROOT/'analysis_tools/phase11c_completion.py',s.ROOT/'analysis_tools/phase11c_verification.py',s.ROOT/'analysis_tools/execute_followup.py',s.ROOT/'analysis_tools/download_phase11c_models.py']
 # Include the isolated dependency wheels/source metadata needed for model loading.
 files += [p for p in (s.ROOT/'reference_inputs/readiness_runtime/vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 files=sorted(set(files));manifest=[dict(file=p.relative_to(s.ROOT).as_posix(),sha256=s.fsha(p),bytes=p.stat().st_size) for p in files]
 s.table(33,'package_manifest.csv',pd.DataFrame(manifest));files.append(s.out(33)/'package_manifest.csv')
 final=s.out(33)/'Shafin_Phase11C_Completed_Classification_Study.zip';temp=final.with_suffix('.zip.writing')
 with zipfile.ZipFile(temp,'w',allowZip64=True) as z:
  for p in files:
   mode=zipfile.ZIP_STORED if p.suffix in ['.pt','.bin','.zip','.gz','.pyd','.dll','.png','.pdf'] else zipfile.ZIP_DEFLATED
   z.write(p,p.relative_to(s.ROOT).as_posix(),compress_type=mode,compresslevel=1)
 os.replace(temp,final)
 with zipfile.ZipFile(final) as z:assert z.testzip() is None;count=len(z.infolist())
 s.dump(33,'FINAL_VALIDATION.json',{'status':'completed','notebooks_fully_executed':[30,31,32,33],'checks_passed':True,'historical_protected_files_unchanged':True,
  'visual_QA_passed':True,'archive':final.name,'archive_sha256':s.fsha(final),'archive_bytes':final.stat().st_size,'archive_members':count,'all_archive_CRC_checks_passed':True,
  'classification_jobs_complete':60,'additional_seed_jobs_complete':40,'original_seed_reruns_complete':20,'MPRA_fairness':'still open'})
 print(json.dumps(s.jread(33,'FINAL_VALIDATION.json'),indent=2))

if __name__=='__main__':finalize()
