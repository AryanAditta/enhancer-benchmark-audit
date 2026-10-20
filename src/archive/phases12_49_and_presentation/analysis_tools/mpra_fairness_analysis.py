"""Evaluation/verification/reporting; never changes the frozen training implementation."""
from pathlib import Path
import json,sys,hashlib,gc,time,math,zipfile,itertools,os,shutil,io
import numpy as np
import pandas as pd
import mpra_fairness_training as s
ROOT=s.ROOT
def out(n):
    p=ROOT/f'phase{n}_outputs';p.mkdir(exist_ok=True);return p
def read(n,name):return pd.read_csv(out(n)/name,float_precision='round_trip')
def table(n,name,d):
    d.to_csv(out(n)/name,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)
def dump(n,name,obj):s.jout(out(n)/name,obj)
def measure(y,p):
    from scipy.stats import spearmanr
    y=np.asarray(y,float);p=np.asarray(p,float);assert np.isfinite(y).all() and np.isfinite(p).all() and len(y)==len(p)
    r=s.safe_r(y,p);error=p-y
    return {'n':len(y),'pearson':r,'spearman':float(spearmanr(y,p).statistic) if r is not None else None,'r2':float(1-np.sum(error**2)/np.sum((y-y.mean())**2)),'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(abs(error))),'mean_error':float(error.mean()),'prediction_sd':float(p.std())}
def independent_measure(y,p):
    from scipy.stats import pearsonr,spearmanr
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    y=np.asarray(y,float);p=np.asarray(p,float);valid=np.std(p)>=1e-8
    return {'n':len(y),'pearson':float(pearsonr(y,p).statistic) if valid else None,'spearman':float(spearmanr(y,p).statistic) if valid else None,'r2':float(r2_score(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'mae':float(mean_absolute_error(y,p)),'mean_error':float(np.average(p-y)),'prediction_sd':float(np.std(p))}
def external(cell):
    d=pd.read_csv(s.INPUT/'phase10_external_frozen_set.csv',float_precision='round_trip');d=d[d.origin.eq('DHS_natural')].reset_index(drop=True)
    d['mean_log2_activity']=d['HepG2_l2fc' if cell=='HepG2' else 'K562_l2fc']
    assert len(d)==8399 and np.isfinite(d.mean_log2_activity).all()
    return d
def selected_runs():
    choices=s.jread(out(36)/'selection_lock.json');result=[]
    for choice in choices['recipes']:
        for seed in s.C['seeds']:
            result.append((choice['cell'],choice['model'],choice['grid_index'],seed,choice['model']+'_selected','best'))
    for cell in s.C['cells']:
        for seed in s.C['seeds']:
            result += [(cell,'dnabert2',1,seed,'dnabert2_fixedlr_full','best'),(cell,'dnabert2',1,seed,'dnabert2_fixedlr_early2','early2')]
    return result
def selection_and_inference():
    assert s.jread(s.P35/'summary.json')['status']=='completed'
    torch=s.runtime();run=s.jread(s.P35/'run_lock.json')
    assert s.sha(ROOT/'analysis_tools/mpra_fairness_training.py')==run['source_sha256']
    records=[];scores=[]
    for args in s.jobs():
        m=s.jread(s.P35/'jobs'/f'{s.key(*args)}.json')
        for a in m['artifacts']:assert s.sha(ROOT/a['file'])==a['sha256']
        records.append(m)
    for cell,model,grid in itertools.product(s.C['cells'],s.C['models'],range(3)):
        rows=[m for m in records if m['cell_line']==cell and m['model']==model and m['grid_index']==grid]
        assert len(rows)==3 and {m['seed'] for m in rows}==set(s.C['seeds'])
        valid=all(m['validation_pearson'] is not None for m in rows)
        scores.append({'cell':cell,'model':model,'grid_index':grid,'learning_rate':s.C['lr_grid'][model][grid],'eligible':valid,'mean_validation_pearson':float(np.mean([m['validation_pearson'] for m in rows])) if valid else None,'seed_sd_validation_pearson':float(np.std([m['validation_pearson'] for m in rows],ddof=1)) if valid else None,'seeds':3,'possibly_budget_limited_runs':sum(m['possibly_budget_limited'] for m in rows)})
    gridtable=pd.DataFrame(scores);table(36,'validation_candidate_scores.csv',gridtable);recipes=[]
    for (cell,model),g in gridtable.groupby(['cell','model']):
        eligible=g[g.eligible];assert len(eligible),f'No valid three-seed candidate for {cell}/{model}; retain failed fits and investigate.'
        winner=eligible.sort_values(['mean_validation_pearson','grid_index'],ascending=[False,True]).iloc[0]
        recipes.append({'cell':cell,'model':model,'grid_index':int(winner.grid_index),'learning_rate':float(winner.learning_rate),'mean_validation_pearson':float(winner.mean_validation_pearson)})
    assert len(recipes)==6
    selection={'training_run_lock':run,'recipes':recipes,'training_job_metadata_sha256':{m['job']:s.sha(s.P35/'jobs'/f"{m['job']}.json") for m in records},'checkpoint_sha256':{a['file']:a['sha256'] for m in records for a in m['artifacts'] if a['file'].endswith('.pt')},'policy':'All54fits completed. Only validation Pearson used. Frozen before new test or external inference.','test_cohorts_previously_examined':True}
    path=out(36)/'selection_lock.json'
    if path.exists():assert s.jread(path)==selection,'Frozen validation choices changed.'
    else:s.jout(path,selection)
    lockhash=s.sha(path);folder=out(36)/'predictions';folder.mkdir(exist_ok=True);cache=out(36)/'inference_cache';cache.mkdir(exist_ok=True)
    result=[];metric=[]
    for cell,model,grid,seed,arm,role in selected_runs():
        name=s.key(cell,model,grid,seed);ckfile=s.P35/'checkpoints'/f'{name}__{role}.pt';h=s.sha(ckfile)
        assert h==selection['checkpoint_sha256'][ckfile.relative_to(ROOT).as_posix()]
        ck=torch.load(ckfile,map_location='cpu',weights_only=True);net=None;tok=None
        for cohort,frame in [('internal',s.partitions(cell)[2]),('DHS_natural',external(cell))]:
            cachefile=cache/f'{h[:20]}__{cohort}.csv';side=cachefile.with_suffix('.json')
            if cachefile.exists() and side.exists():
                cm=s.jread(side);assert cm['checkpoint_sha256']==h and cm['sha256']==s.sha(cachefile)
                q=pd.read_csv(cachefile,float_precision='round_trip');assert q.record_id.tolist()==frame.record_id.tolist();p=q.prediction.to_numpy()
            else:
                if net is None:net,tok=s.build(model);net.load_state_dict(ck['state_dict'],strict=True)
                p=s.predict(net,tok,model,frame.sequence.tolist(),ck['target_mean'],ck['target_scale'])
                s.csvout(cachefile,pd.DataFrame({'record_id':frame.record_id,'prediction':p}));s.jout(side,{'checkpoint_sha256':h,'sha256':s.sha(cachefile),'selection_lock_sha256':lockhash})
            group=frame.locus_group_id if cohort=='internal' else frame.canonical
            d=pd.DataFrame({'record_id':frame.record_id,'activity':frame.mean_log2_activity,'group_id':group,'prediction':p,'cell':cell,'cohort':cohort,'arm':arm,'seed':seed,'checkpoint_sha256':h,'selected_epoch':ck['epoch']})
            file=folder/f'{cell}__{arm}__seed{seed}__{cohort}.csv';s.csvout(file,d);result.append(d)
            metric.append(dict(cell=cell,cohort=cohort,arm=arm,seed=seed,selected_epoch=ck['epoch'],**measure(d.activity,d.prediction)))
            dump(36,'live_progress.json',{'evaluation_caches':len(result),'expected':60,'last':file.name})
        del ck,net,tok;gc.collect();torch.cuda.empty_cache()
    d=pd.concat(result,ignore_index=True);table(36,'all_evaluation_predictions.csv.gz',d);table(36,'seed_metrics.csv',pd.DataFrame(metric))
    dump(36,'summary.json',{'status':'completed','training_fits_before_selection':54,'recipes_locked':6,'selection_lock_sha256':lockhash,'evaluation_sets':60,'test_tuning_performed':False,'historical_outputs_preserved':True})
def independent_verification():
    assert s.jread(out(36)/'summary.json')['status']=='completed'
    torch=s.runtime();run=s.jread(s.P35/'run_lock.json');selection=s.jread(out(36)/'selection_lock.json')
    assert s.sha(ROOT/'analysis_tools/mpra_fairness_training.py')==run['source_sha256']
    assert s.sha(s.P34/'training_protocol.json')==run['protocol_sha256']
    assert selection['training_run_lock']==run
    for item in pd.read_csv(s.P34/'copied_input_manifest.csv').itertuples():assert s.sha(ROOT/item.copy)==item.sha256
    checks=[];cache=out(37)/'inference_cache';cache.mkdir(exist_ok=True)
    def infer_checked(ckfile,model,frame,expected,cohort):
        h=s.sha(ckfile);file=cache/f'{h}__{cohort}.csv';side=file.with_suffix('.json')
        if file.exists() and side.exists():
            m=s.jread(side);assert m['checkpoint_sha256']==h and m['sha256']==s.sha(file);d=pd.read_csv(file,float_precision='round_trip');assert d.record_id.tolist()==frame.record_id.tolist();p=d.prediction.to_numpy()
        else:
            ck=torch.load(ckfile,map_location='cpu',weights_only=True);net,tok=s.build(model);net.load_state_dict(ck['state_dict'],strict=True)
            p=s.predict(net,tok,model,frame.sequence.tolist(),ck['target_mean'],ck['target_scale']);s.csvout(file,pd.DataFrame({'record_id':frame.record_id,'prediction':p}));s.jout(side,{'checkpoint_sha256':h,'sha256':s.sha(file)})
            del net,tok,ck;gc.collect();torch.cuda.empty_cache()
        delta=np.abs(p-np.asarray(expected));assert delta.max()<2e-5 and delta.mean()<1e-6,(ckfile.name,cohort,delta.max(),delta.mean())
        return float(delta.max())
    for cell,model,grid,seed in s.jobs():
        name=s.key(cell,model,grid,seed);m=s.jread(s.P35/'jobs'/f'{name}.json')
        assert s.sha(s.P35/'jobs'/f'{name}.json')==selection['training_job_metadata_sha256'][name]
        tr,va,te=s.partitions(cell);assert set(tr.chrom).isdisjoint(te.chrom) and set(tr.chrom).isdisjoint(va.chrom)
        assert m['train_record_ids_sha256']==s.digest('\n'.join(tr.record_id)) and m['validation_record_ids_sha256']==s.digest('\n'.join(va.record_id))
        assert abs(m['target_mean']-tr.mean_log2_activity.mean())<1e-12 and abs(m['target_scale']-tr.mean_log2_activity.std(ddof=0))<1e-12
        log=pd.read_csv(s.P35/'training_logs'/f'{name}.csv',float_precision='round_trip');assert len(log)==m['epochs_run'] and log.epoch.tolist()==list(range(1,len(log)+1))
        for a in m['artifacts']:assert s.sha(ROOT/a['file'])==a['sha256']
        for role in ['best']+(['early2'] if model=='dnabert2' else []):
            file=s.P35/'checkpoints'/f'{name}__{role}.pt';ck=torch.load(file,map_location='cpu',weights_only=True)
            for field in ['job','model','seed','cell_line','protocol_sha256','source_sha256','architecture_sha256','target_mean','target_scale']:assert ck[field]==m[field]
            available=log if role=='best' else log[log.epoch.le(2)]
            valid=available[available.validation_pearson.notna()]
            expected_epoch=int(valid.sort_values(['validation_pearson','epoch'],ascending=[False,True]).iloc[0].epoch) if len(valid) else int(available.epoch.iloc[-1])
            assert ck['epoch']==expected_epoch
            p=pd.read_csv(s.P35/'validation'/f'{name}__{role}.csv',float_precision='round_trip');assert p.record_id.tolist()==va.record_id.tolist()
            assert np.array_equal(p.prediction,np.asarray(ck['validation_predictions']))
            r=independent_measure(va.mean_log2_activity,p.prediction)['pearson']
            if r is None:assert ck['validation_pearson'] is None
            else:assert abs(r-ck['validation_pearson'])<1e-12
            delta=infer_checked(file,model,va,p.prediction,'validation');checks.append({'job':name,'role':role,'cohort':'validation','checkpoint_sha256':s.sha(file),'max_prediction_difference':delta,'passed':True})
            del ck
        table(37,'checkpoint_reproduction_checks.csv',pd.DataFrame(checks));dump(37,'live_progress.json',{'validation_checkpoints_verified':len(checks),'expected_validation_checkpoints':72})
    fresh=[]
    for cell,model,grid in itertools.product(s.C['cells'],s.C['models'],range(3)):
        scores=[s.jread(s.P35/'jobs'/f'{s.key(cell,model,grid,seed)}.json')['validation_pearson'] for seed in s.C['seeds']]
        fresh.append({'cell':cell,'model':model,'grid_index':grid,'score':float(np.mean(scores)) if all(x is not None for x in scores) else None})
    for recipe in selection['recipes']:
        candidates=[x for x in fresh if x['cell']==recipe['cell'] and x['model']==recipe['model'] and x['score'] is not None]
        winner=sorted(candidates,key=lambda x:(-x['score'],x['grid_index']))[0]
        assert recipe['grid_index']==winner['grid_index'] and abs(recipe['mean_validation_pearson']-winner['score'])<1e-12
    p=read(36,'all_evaluation_predictions.csv.gz');assert len(p.groupby(['cell','cohort','arm','seed']))==60
    metrics=[]
    for cell,model,grid,seed,arm,role in selected_runs():
        ckfile=s.P35/'checkpoints'/f'{s.key(cell,model,grid,seed)}__{role}.pt'
        for cohort,frame in [('internal',s.partitions(cell)[2]),('DHS_natural',external(cell))]:
            d=p[p.cell.eq(cell)&p.cohort.eq(cohort)&p.arm.eq(arm)&p.seed.eq(seed)]
            assert d.record_id.tolist()==frame.record_id.tolist() and np.array_equal(d.activity,frame.mean_log2_activity)
            assert d.checkpoint_sha256.eq(s.sha(ckfile)).all()
            delta=infer_checked(ckfile,model,frame,d.prediction,cohort)
            checks.append({'job':s.key(cell,model,grid,seed),'role':role,'cohort':cohort,'arm':arm,'max_prediction_difference':delta,'passed':True})
            metrics.append(dict(cell=cell,cohort=cohort,arm=arm,seed=seed,**independent_measure(d.activity,d.prediction)))
    independent=pd.DataFrame(metrics);prior=read(36,'seed_metrics.csv')
    joined=prior.merge(independent,on=['cell','cohort','arm','seed'],suffixes=('_saved','_verified'),validate='one_to_one')
    for metric in ['n','pearson','spearman','r2','rmse','mae','mean_error','prediction_sd']:
        assert np.allclose(joined[metric+'_saved'],joined[metric+'_verified'],atol=1e-12,rtol=0,equal_nan=True),metric
    table(37,'seed_metrics_independent.csv',independent);table(37,'checkpoint_reproduction_checks.csv',pd.DataFrame(checks))
    ensembles=[];em=[];variability=[]
    for (cell,cohort,arm),g in p.groupby(['cell','cohort','arm']):
        w=g.pivot(index='record_id',columns='seed',values='prediction');assert set(w.columns)==set(s.C['seeds']) and w.notna().all().all()
        ref=g.drop_duplicates('record_id').set_index('record_id').loc[w.index]
        pred=w.mean(axis=1).to_numpy();manual=np.array([math.fsum(row)/3 for row in w[s.C['seeds']].to_numpy()]);assert np.max(abs(pred-manual))<1e-12
        sd=w.std(axis=1,ddof=1).to_numpy();manualsd=np.sqrt(np.sum((w[s.C['seeds']].to_numpy()-manual[:,None])**2,axis=1)/2);assert np.max(abs(sd-manualsd))<1e-12
        e=pd.DataFrame({'record_id':w.index,'cell':cell,'cohort':cohort,'arm':arm,'activity':ref.activity.to_numpy(),'group_id':ref.group_id.to_numpy(),'prediction':pred,'seed_sd':sd});ensembles.append(e)
        em.append(dict(cell=cell,cohort=cohort,arm=arm,**independent_measure(e.activity,e.prediction)))
        seedmetrics=independent[independent.cell.eq(cell)&independent.cohort.eq(cohort)&independent.arm.eq(arm)]
        for metric in ['pearson','spearman','rmse','mae']:variability.append({'cell':cell,'cohort':cohort,'arm':arm,'metric':metric,'n_seeds':3,'defined_seeds':int(seedmetrics[metric].notna().sum()),'mean':float(seedmetrics[metric].mean()),'sample_sd':float(seedmetrics[metric].std(ddof=1))})
    e=pd.concat(ensembles,ignore_index=True);table(37,'verified_ensemble_predictions.csv.gz',e);table(37,'ensemble_metrics.csv',pd.DataFrame(em));table(37,'seed_variability.csv',pd.DataFrame(variability))
    intervals(e);table(37,'protected_history_checks.csv',pd.DataFrame(s.protected_check()))
    dump(37,'summary.json',{'status':'completed','training_jobs_authenticated':54,'validation_checkpoints_reproduced':72,'evaluation_sets_reproduced':60,'ensemble_metric_rows':20,'all_checks_passed':True,'max_prediction_difference':max(x['max_prediction_difference'] for x in checks),'older_outputs_unchanged':True,'uncertainty_scope':s.C['bootstrap']['scope']})
def intervals(e):
    from scipy.stats import rankdata
    rows=[];rng=np.random.default_rng(s.C['bootstrap']['seed'])
    contrasts=[('duration_fixed_lr','dnabert2_fixedlr_full','dnabert2_fixedlr_early2'),('selected_vs_legnet','dnabert2_selected','legnet_selected'),('selected_vs_mprann','dnabert2_selected','mprann_selected')]
    def corr(y,p,w):
        total=w.sum();my=np.sum(w*y)/total;mp=np.sum(w*p)/total
        den=np.sqrt(np.sum(w*(y-my)**2)*np.sum(w*(p-mp)**2))
        return np.sum(w*(y-my)*(p-mp))/den if den>0 else np.nan
    for (cell,cohort),g in e.groupby(['cell','cohort']):
        w=g.pivot(index='record_id',columns='arm',values='prediction');ref=g.drop_duplicates('record_id').set_index('record_id').loc[w.index];y=ref.activity.to_numpy()
        units={'locus_group_id' if cohort=='internal' else 'canonical_DNA_sequence':ref.group_id.to_numpy()}
        if cohort=='DHS_natural':
            x=external(cell).set_index('record_id').loc[w.index].reset_index();groups={};num=0
            for chrom,d in x.sort_values(['external_chrom','external_start','external_end']).groupby('external_chrom',sort=True):
                end=None
                for r in d.itertuples():
                    if end is None or r.external_start>end+1000:num+=1
                    end=r.external_end if end is None else max(end,r.external_end);groups[r.record_id]=num
            units['external_intervals_1kb']=np.array([groups[i] for i in w.index])
        for unit,labels in units.items():
            codes=pd.factorize(labels,sort=True)[0];ng=int(codes.max()+1)
            for contrast,a,b in contrasts:
                p=w[a].to_numpy();q=w[b].to_numpy()
                for metric in (['pearson','mae'] if cohort=='internal' else ['spearman']):
                    yy=rankdata(y) if metric=='spearman' else y;pp=rankdata(p) if metric=='spearman' else p;qq=rankdata(q) if metric=='spearman' else q
                    one=np.ones(len(y));estimate=(np.mean(abs(p-y))-np.mean(abs(q-y))) if metric=='mae' else corr(yy,pp,one)-corr(yy,qq,one)
                    samples=[]
                    for draw in range(1000):
                        weights=np.bincount(rng.integers(0,ng,ng),minlength=ng)[codes]
                        if metric=='mae':value=np.sum(weights*(abs(p-y)-abs(q-y)))/weights.sum()
                        elif metric=='spearman':
                            # Re-rank the actual resampled observations. Fixed original ranks would be incorrect.
                            ix=np.repeat(np.arange(len(y)),weights);yr=rankdata(y[ix]);pr=rankdata(p[ix]);qr=rankdata(q[ix]);value=corr(yr,pr,np.ones(len(ix)))-corr(yr,qr,np.ones(len(ix)))
                        else:value=corr(yy,pp,weights)-corr(yy,qq,weights)
                        if np.isfinite(value):samples.append(value)
                    if not samples:lo=hi=np.nan
                    else:lo,hi=np.quantile(samples,[.025,.975])
                    rows.append({'cell':cell,'cohort':cohort,'contrast':contrast,'arm_a':a,'arm_b':b,'metric':metric,'estimate_a_minus_b':estimate,'lower_95':lo,'upper_95':hi,'n_records':len(y),'n_groups':ng,'resampling_unit':unit,'draws':1000,'defined_draws':len(samples),'scope':s.C['bootstrap']['scope']})
    table(37,'conditional_paired_intervals.csv',pd.DataFrame(rows))
def report():
    assert s.jread(out(37)/'summary.json')['status']=='completed'
    os.environ['MPLCONFIGDIR']=str(out(38)/'matplotlib_cache')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    figures=out(38)/'figures';figures.mkdir(exist_ok=True);index=[]
    def save(fig,name,caption):
        for ext in ['png','svg','pdf']:
            p=figures/f'{name}.{ext}';fig.savefig(p,dpi=210,bbox_inches='tight');index.append({'figure':name,'format':ext,'file':p.relative_to(ROOT).as_posix(),'sha256':s.sha(p),'caption':caption})
        plt.close(fig)
    jobs=read(35,'training_job_summary.csv');candidate=read(36,'validation_candidate_scores.csv');metrics=read(37,'ensemble_metrics.csv');seed=read(37,'seed_metrics_independent.csv');ci=read(37,'conditional_paired_intervals.csv')
    selection=s.jread(out(36)/'selection_lock.json');chosen={(x['cell'],x['model']):x['grid_index'] for x in selection['recipes']}
    logs=[]
    for row in jobs.itertuples():logs.append(pd.read_csv(s.P35/'training_logs'/f'{row.job}.csv').assign(cell=row.cell_line,model=row.model,grid=row.grid_index,seed=row.seed))
    logs=pd.concat(logs,ignore_index=True);table(38,'all_training_curves.csv',logs)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,(cell,model) in zip(axes.ravel(),itertools.product(s.C['cells'],s.C['models'])):
        for grid in range(3):
            g=logs[logs.cell.eq(cell)&logs.model.eq(model)&logs.grid.eq(grid)]
            for _,h in g.groupby('seed'):ax.plot(h.epoch,h.validation_pearson,alpha=.25,c=f'C{grid}')
            mean=g.groupby('epoch').validation_pearson.mean();ax.plot(mean.index,mean.values,c=f'C{grid}',label=f"lr {s.C['lr_grid'][model][grid]:g}")
        if model=='dnabert2':ax.axvline(2,c='black',ls='--',lw=1)
        ax.set(title=f'{cell}: {model}',xlabel='Epoch',ylabel='Validation Pearson');ax.legend(fontsize=7)
    save(fig,'A1_validation_learning_curves','All54validation-only trajectories; faint individual seeds and mean per learning rate. Mean at late epochs can use fewer active seeds after early stopping. Dashed line marks the two-epoch reference. No test-based tuning.')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    armnames=['dnabert2_fixedlr_early2','dnabert2_fixedlr_full','dnabert2_selected','legnet_selected','mprann_selected'];short=['DB2 two epochs','DB2 longer, fixed LR','DB2 selected LR','LegNet selected','MPRAnn selected']
    for ax,cell in zip(axes,s.C['cells']):
        for i,arm in enumerate(armnames):
            row=metrics[metrics.cell.eq(cell)&metrics.cohort.eq('internal')&metrics.arm.eq(arm)].iloc[0]
            ax.scatter(i,row.pearson,c='black',marker='s',s=45)
            q=seed[seed.cell.eq(cell)&seed.cohort.eq('internal')&seed.arm.eq(arm)].pearson;ax.scatter(i+np.linspace(-.12,.12,3),q,s=20)
        ax.set(xticks=range(5),xticklabels=short,title=cell,ylabel='9B test Pearson');ax.tick_params(axis='x',rotation=25)
        ax.legend(handles=[Line2D([],[],color='black',marker='s',ls='None',label='Three-seed ensemble'),Line2D([],[],color='gray',marker='o',ls='None',label='Individual seed')],loc='lower right',fontsize=7)
    save(fig,'A2_primary_test_performance','Black squares: three-seed ensemble; coloured dots: individual seeds. Two-epoch and longer fixed-LR DNABERT-2 share the identical initial training trajectory. Other selected recipes use validation-only candidate selection.')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    contrast_labels={'duration_fixed_lr':'DB2 longer − two epochs\n(fixed LR)','selected_vs_legnet':'DB2 − LegNet\n(selected LR)','selected_vs_mprann':'DB2 − MPRAnn\n(selected LR)'}
    for ax,cell in zip(axes,s.C['cells']):
        q=ci[ci.cell.eq(cell)&ci.cohort.eq('internal')&ci.metric.eq('pearson')]
        for i,r in enumerate(q.itertuples()):ax.errorbar(r.estimate_a_minus_b,i,xerr=[[r.estimate_a_minus_b-r.lower_95],[r.upper_95-r.estimate_a_minus_b]],fmt='o',capsize=4)
        ax.axvline(0,c='black',ls='--');ax.set(yticks=range(len(q)),yticklabels=[contrast_labels[x] for x in q.contrast],title=cell,xlabel='Pearson difference (A − B); conditional 95% CI')
    save(fig,'A3_paired_primary_differences','Locus-group paired bootstrap intervals conditional on the selected fitted models and previously examined test cohort. Unadjusted comparisons; seed/search uncertainty is not represented by these intervals.')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,cell in zip(axes,s.C['cells']):
        q=metrics[metrics.cell.eq(cell)&metrics.cohort.eq('DHS_natural')].set_index('arm').loc[armnames]
        ax.scatter(range(5),q.spearman);ax.set(xticks=range(5),xticklabels=short,title=cell,ylabel='DHS-natural external Spearman');ax.tick_params(axis='x',rotation=25)
    save(fig,'A4_external_transfer','Frozen DHS-natural external sequences; all checkpoints selected before external inference. Neither external scores nor intervals tune recipes.')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,cell in zip(axes,s.C['cells']):
        q=jobs[jobs.cell_line.eq(cell)].groupby('model').elapsed_seconds.sum().reindex(s.C['models'])/3600
        bars=ax.bar(range(3),q);ax.set(xticks=range(3),xticklabels=s.C['models'],title=cell,ylabel='Recorded fit hours across 9 fits');ax.bar_label(bars,labels=[f'{v:.2f} h' if v>=1 else f'{v*60:.1f} min' for v in q],padding=3,fontsize=8);ax.set_ylim(0,max(q)*1.13)
    save(fig,'A5_training_cost','Actual elapsed time across all three rates and three seeds, including validation and checkpointing within fits. Equal candidate counts are not equal GPU-time or pretraining cost.')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,cell in zip(axes,s.C['cells']):
        for i,model in enumerate(s.C['models']):
            g=jobs[jobs.cell_line.eq(cell)&jobs.model.eq(model)&jobs.grid_index.eq(chosen[(cell,model)])]
            ax.scatter(i+np.linspace(-.1,.1,3),g.selected_epoch,s=35,label=model)
        ax.set(xticks=range(3),xticklabels=s.C['models'],ylabel='Selected epoch for chosen learning rate',title=cell)
    save(fig,'A6_selected_training_epochs','Each declared seed remains visible. Selected epochs are validation choices; maximum-budget and late-improvement flags are saved separately and do not prove convergence.')
    table(38,'figure_index.csv',pd.DataFrame(index));paths=sorted(figures.glob('*.png'));fig,axes=plt.subplots(3,2,figsize=(15,15))
    for ax,p in zip(axes.ravel(),paths):ax.imshow(plt.imread(p));ax.set_title(p.stem);ax.axis('off')
    fig.tight_layout();fig.savefig(out(38)/'figure_contact_sheet.png',dpi=130);plt.close(fig)
    tables=out(38)/'tables';tables.mkdir(exist_ok=True)
    for n,file in [(35,'training_job_summary.csv'),(36,'validation_candidate_scores.csv'),(37,'seed_metrics_independent.csv'),(37,'ensemble_metrics.csv'),(37,'seed_variability.csv'),(37,'conditional_paired_intervals.csv'),(37,'checkpoint_reproduction_checks.csv')]:shutil.copy2(out(n)/file,tables/file)
    lines=['# MPRA training adequacy: verified findings','This follow-up tests Aditta priority1 on the primary9B chromosome-held-out protocol. It uses54fits with the same candidate-count and seed budget per model/cell; model-specific optimizers/batch sizes/epoch caps and actual computation differ.']
    for cell in s.C['cells']:
        lines.append('\n## '+cell)
        for arm in armnames:
            row=metrics[metrics.cell.eq(cell)&metrics.cohort.eq('internal')&metrics.arm.eq(arm)].iloc[0]
            lines.append(f'- {arm}: ensemble test Pearson {row.pearson:.6f}; RMSE {row.rmse:.6f}; MAE {row.mae:.6f}; n={int(row.n):,}.')
        for row in ci[ci.cell.eq(cell)&ci.cohort.eq('internal')&ci.metric.eq('pearson')].itertuples():lines.append(f'- {row.contrast}: Pearson difference {row.estimate_a_minus_b:+.6f}, conditional95% interval [{row.lower_95:+.6f}, {row.upper_95:+.6f}].')
    selected=jobs[[chosen[(r.cell_line,r.model)]==r.grid_index for r in jobs.itertuples()]]
    selected=selected.copy();selected['selected_grid_boundary']=selected.grid_index.isin([0,2])
    selected['validation_gain_over_early2']=selected.validation_pearson-selected.early2_validation_pearson
    table(38,'selected_recipe_adequacy_flags.csv',selected)
    flags=int(selected.possibly_budget_limited.sum());collapsed=int(jobs.collapsed.sum())
    lines += [f'\n## Training adequacy checks\n\nSelected candidate fits flagged as potentially budget-limited: {flags}/18. Collapsed validation fits across the full study: {collapsed}/54. All remain in the audit. The grid, stopping policy and every selection hash are saved. No outcome-driven changes were made after test evaluation.',
      '\n## How to interpret the experiment\n\nUse the fixed-lr2e-5 paired trajectory comparison to assess the specific two-epoch concern. Use the validation-selected three-model comparison to assess performance after the same limited number of learning-rate candidates. These are distinct contrasts. A validation-selected longer checkpoint can improve validation without improving test generalization; report either outcome. Inspect the exact intervals rather than claiming equivalence when they include zero.',
      '\n## Remaining boundaries\n\nThis is a limited training-adequacy investigation, not proof of exhaustive tuning or equal compute. Three seeds and previously examined9B/DHS cohorts limit generalization. Conditional bootstrap intervals omit model/search/selection uncertainty. Late best epochs, selected boundary learning rates or improving curves leave residual budget/grid concerns. No fresh9C training, historical collapsed-seed root-cause repair, longer context, broader cell experiment or universal CNN/transformer conclusion is supplied. Earlier packages are preserved historical snapshots; this new evidence must be integrated explicitly into the revised manuscript.']
    details=['\n## Observed selected-recipe limits']
    for (cell,model),g in selected.groupby(['cell_line','model'],sort=False):
        details.append(f'- {cell}, {model}: LR {g.learning_rate.iloc[0]:g}; selected epochs {", ".join(str(int(x)) for x in g.selected_epoch)}; {int(g.possibly_budget_limited.sum())}/3 fits potentially budget-limited; selected LR at grid boundary: {bool(g.selected_grid_boundary.iloc[0])}.')
    details.append('\nLegNet selected the upper LR boundary and MPRAnn the lower LR boundary in both cells. Two selected HepG2 DNABERT2 fits reached the 30-epoch cap. These observations prevent a claim of exhaustive tuning or established convergence.')
    details.append('\n## External ranking and calibration')
    for cell in s.C['cells']:
        ext=metrics[metrics.cell.eq(cell)&metrics.cohort.eq('DHS_natural')].set_index('arm')
        early=ext.loc['dnabert2_fixedlr_early2'];full=ext.loc['dnabert2_fixedlr_full']
        details.append(f'- {cell}, n={int(full.n):,}: fixed-LR DNABERT2 external Spearman changed from {early.spearman:.6f} to {full.spearman:.6f}, while MAE changed from {early.mae:.6f} to {full.mae:.6f} and RMSE from {early.rmse:.6f} to {full.rmse:.6f}.')
        details.append('- Validation-selected external Spearman: '+', '.join(f'{name} {ext.loc[arm,"spearman"]:.6f}' for name,arm in [('DNABERT2','dnabert2_selected'),('LegNet','legnet_selected'),('MPRAnn','mprann_selected')])+'.')
    details.append('\nLonger DNABERT2 training improved external rank correlation but increased absolute prediction error in both cells. All external ensemble R² values are negative and mean errors are negative; external ranking performance does not establish calibrated activity prediction on this shifted cohort.')
    lines += details
    (out(38)/'MPRA_FAIRNESS_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    evidence=pd.DataFrame([{'concern':'Aditta1_training_fairness','status':'investigated_with_explicit_limits','evidence':'phase37_outputs/ensemble_metrics.csv; phase37_outputs/conditional_paired_intervals.csv','remaining':'Limited3learning-rate grid; model-specific compute; previously examined cohorts; cap/grid flags must qualify conclusions.'},{'concern':'DNABERT2_two_epoch_duration','status':'paired_trajectory_control_completed','evidence':'phase35_outputs/checkpoints/*dnabert2*early2.pt; phase37_outputs/conditional_paired_intervals.csv','remaining':'Training duration measured at fixed lr2e-5; does not prove every possible recipe converges.'},{'concern':'old9C_collapsed_seed','status':'open_historical_root_cause','evidence':'phase13_outputs/old_9c_k562_collapse_sensitivity.csv','remaining':'This new primary9B training study does not repair unavailable historical9C epoch logs.'}])
    table(38,'reviewer_evidence_update.csv',evidence)
    (out(38)/'README_REPRODUCTION.md').write_text('''# MPRA training adequacy supplement

Use the existing enhancer environment on Windows/Python3.12/PyTorch2.11.0+cu128 and a compatible CUDA GPU. Bundled isolated Transformers4.44.2 dependencies are under reference_inputs/readiness_runtime/vendor. All inputs and pretrained weights are local. Run notebooks34–38 from the package root using analysis_tools/execute_followup.py. Training verifies and skips authenticated completed fits. Checkpoint re-inference needs CUDA; figure generation from saved outputs uses ordinary CPU libraries.

The new study has54training fits and72saved best/early-prefix checkpoints. Selection is validation-only and locked before new test/external inference. Existing test populations were previously examined. Conditional inference and the equal-number candidate search do not establish universal architecture superiority or equal compute. Historical outputs stay separate.

The finalizer appends the completed contribution-log entry only after all notebooks and visual checks pass. FINAL_VALIDATION.json contains the sealed archive SHA256 and remains outside the archive. package_manifest.csv authenticates all included files. Do not execute copied old-source notebooks as new-study entry points.
''',encoding='utf-8')
    for name in ['environment.yml','requirements-win-cuda.txt','environment_versions.json']:shutil.copy2(ROOT/'phase33_outputs'/name,out(38)/name)
    dump(38,'summary.json',{'status':'completed_findings_pending_visual_QA_and_archive','training_fits':54,'validation_checkpoints_verified':72,'evaluation_sets_verified':60,'figures':6,'selected_fits_possibly_budget_limited':flags,'collapsed_validation_fits':collapsed,'MPRA_fairness':'experimentally_investigated_on_primary9B_with_limited_search; qualifications_retained','manuscript_finished':False})
def finalize():
    import nbformat
    assert s.jread(out(37)/'summary.json')['status']=='completed'
    qa=s.jread(out(38)/'visual_QA.json');assert qa['passed'] and qa['figures_inspected']==6
    s.protected_check();notebooks=[]
    for n in range(34,39):
        p=next(ROOT.glob(f'[[]{n}]*.ipynb'));nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        codes=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in codes)
        assert not any(o.output_type=='error' for c in codes for o in c.get('outputs',[]))
        notebooks.append(p)
    log=ROOT/'CONTRIBUTION_LOG.md';text=log.read_text(encoding='utf-8');assert text.startswith((s.P34/'contribution_log_before.md').read_text(encoding='utf-8'))
    marker='### Completed Phases34–38: MPRA training adequacy and limited search fairness'
    if marker not in text:
        from datetime import datetime,timezone,timedelta
        completed_date=datetime.now(timezone(timedelta(hours=6))).date().isoformat()
        addition='\n\n'+marker+' ('+completed_date+')\n\nThe user explicitly chose to investigate Aditta priority1 experimentally. Authenticated local inputs and froze the prospective protocol in Phase34. Executed54validation-only fits: three models × two cells × three learning rates × three seeds. Saved72best/two-epoch checkpoints and optimizer/scaler/RNG recovery. Fixed all recipe and checkpoint choices before new test/external inference. Phase37independently reproduced validation/test/external predictions, recalculated metrics/ensembles/seed SD and conditional paired intervals. Phase38prepared six figures, source tables, findings and reviewer evidence.\n\n'+(out(38)/'MPRA_FAIRNESS_FINDINGS.md').read_text(encoding='utf-8')+'\n\nAll earlier research snapshots and original notebooks are preserved. This investigates training adequacy under a limited equal-candidate-count search; it does not prove compute-matched or exhaustive architecture superiority. Budget/grid flags, previously examined cohorts, three-seed uncertainty and unavailable historical9C logs remain explicit. All work products are inside Shafin\'s Contribution; no journal contact or manuscript submission occurred.\n'
        log.write_text(text.rstrip()+addition,encoding='utf-8')
    files=[]
    for n in range(34,39):
        files += [p for p in out(n).rglob('*') if p.is_file() and not any(x in p.parts for x in ['runtime','inference_cache','matplotlib_cache']) and p.name not in ['current_training_resume.pt','FINAL_VALIDATION.json','package_manifest.csv'] and not p.name.endswith(('.zip','.writing'))]
    files += [p for p in s.INPUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [p for p in (ROOT/'reference_inputs/readiness_runtime/vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += notebooks+[log]+[ROOT/'analysis_tools'/name for name in ['prepare_mpra_fairness.py','mpra_fairness_training.py','mpra_fairness_analysis.py','build_mpra_fairness_notebooks.py','run_mpra_fairness_sequence.py','execute_followup.py']]
    files=sorted(set(files));manifest=pd.DataFrame([{'file':p.relative_to(ROOT).as_posix(),'sha256':s.sha(p),'bytes':p.stat().st_size} for p in files]);table(38,'package_manifest.csv',manifest);files.append(out(38)/'package_manifest.csv')
    final=out(38)/'Shafin_MPRA_Training_Fairness_Study.zip';temp=final.with_suffix('.zip.writing')
    assert shutil.disk_usage(ROOT).free>sum(p.stat().st_size for p in files)*1.05,'Insufficient space for a separate final archive; preserve all study outputs.'
    with zipfile.ZipFile(temp,'w',allowZip64=True) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in ['.pt','.bin','.gz','.png','.pyd','.dll','.zip','.pdf'] else zipfile.ZIP_DEFLATED,compresslevel=1)
    os.replace(temp,final)
    with zipfile.ZipFile(final) as z:assert z.testzip() is None;count=len(z.infolist())
    dump(38,'FINAL_VALIDATION.json',{'status':'completed','notebooks_fully_executed':[34,35,36,37,38],'training_jobs_complete':54,'validation_checkpoints_reproduced':72,'all_archive_CRC_checks_passed':True,'historical_outputs_preserved':True,'archive':final.name,'archive_sha256':s.sha(final),'archive_bytes':final.stat().st_size,'archive_members':count,'MPRA_fairness':'limited equal-candidate-count investigation complete; broader claims remain bounded'})
    print(json.dumps(s.jread(out(38)/'FINAL_VALIDATION.json'),indent=2))
if __name__=='__main__':finalize()
