import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from enhancer_benchmark_audit import io as s
from enhancer_benchmark_audit.statistics import *
s.verify_inputs();c=s.read('final_statistics/classification_predictions.csv.gz');m=s.read('phase37_outputs/verified_ensemble_predictions.csv.gz')
expected=s.read('final_statistics/classification_summary.csv').set_index(['stage','model']);rows=[];errors=[]
for (stage,model),g in c.groupby(['stage','model']):
    got=classification_metrics(g.label,g.score)
    for key,value in got.items():errors.append(abs(value-expected.loc[(stage,model),key]))
    rows.append(dict(stage=stage,model=model,**got))
s.table('classification_metrics_recomputed.csv',pd.DataFrame(rows))
expected=pd.concat([s.read('final_statistics/mpra_internal_summary.csv'),s.read('final_statistics/external_transfer_summary.csv')]).set_index(['cell','cohort','arm']);rows=[]
for (cell,cohort,arm),g in m.groupby(['cell','cohort','arm']):
    got=regression_metrics(g.activity,g.prediction)
    for key,value in got.items():errors.append(abs(value-expected.loc[(cell,cohort,arm),key]))
    rows.append(dict(cell=cell,cohort=cohort,arm=arm,**got))
s.table('MPRA_metrics_recomputed.csv',pd.DataFrame(rows));assert max(errors)<1e-12
checks=stats_selfcheck();s.table('weighted_vs_literal.csv',checks)
contrasts=s.read('final_statistics/adjusted_tests.csv');point_errors=[];stat_errors=[];results=[]
policy=s.jd('final_statistics/statistical_methods.json')
for row in contrasts.to_dict('records'):
    y,a,b,gi,ng=paired_arrays(row,c,m);w0=np.ones(len(y))
    obs=weighted_metric(y,a,w0,row['metric'])-weighted_metric(y,b,w0,row['metric'])
    point_errors.append(abs(obs-row['estimate_a_minus_b']))
    if '--full-statistics' not in sys.argv:continue
    seed=int.from_bytes(hashlib.sha256(row['comparison_id'].encode()).digest()[:4],'little')^policy['random_seed'];rng=np.random.default_rng(seed);boot=[]
    for i in range(policy['bootstrap_draws']):
        w=np.bincount(rng.integers(0,ng,ng),minlength=ng)[gi].astype(float)
        boot.append(weighted_metric(y,a,w,row['metric'])-weighted_metric(y,b,w,row['metric']))
    defined=np.asarray(boot);defined=defined[np.isfinite(defined)];lo,hi=np.quantile(defined,[.025,.975]);extreme=0
    for i in range(policy['permutation_draws']):
        flip=rng.integers(0,2,ng,dtype=np.int8)[gi].astype(bool);pa=np.where(flip,b,a);pb=np.where(flip,a,b)
        value=weighted_metric(y,pa,w0,row['metric'])-weighted_metric(y,pb,w0,row['metric']);assert np.isfinite(value)
        extreme+=abs(value)>=abs(obs)-1e-14
    p=(extreme+1)/(policy['permutation_draws']+1)
    stat_errors.extend([abs(lo-row['lower_95']),abs(hi-row['upper_95']),abs(p-row['p_raw'])]);results.append(dict(comparison_id=row['comparison_id'],estimate=obs,lower_95=lo,upper_95=hi,p_raw=p))
    print('Reproduced',len(results),'/14:',row['comparison_id'],flush=True)
assert max(point_errors)<1e-12
if results:
    f=pd.DataFrame(results);order=np.argsort(f.p_raw.to_numpy(),kind='stable');p=f.p_raw.to_numpy()[order];adj=np.minimum(1,np.maximum.accumulate(p*(len(p)-np.arange(len(p)))));out=np.empty(len(p));out[order]=adj;f['p_holm']=out
    stat_errors.extend(abs(f.p_holm.to_numpy()-contrasts.p_holm.to_numpy()));assert max(stat_errors)<1e-12;s.table('headline_statistics_recomputed.csv',f)
s.dump('metric_validation.json',dict(passed=True,classification_rows=18,MPRA_ensemble_rows=20,weighted_literal_checks=len(checks),point_contrasts=14,max_metric_error=float(max(errors)),max_point_difference_error=float(max(point_errors)),full_statistics_contrasts=len(results),max_interval_pvalue_error=float(max(stat_errors)) if stat_errors else None,bootstrap_draws=policy['bootstrap_draws'] if results else 0,permutation_draws=policy['permutation_draws'] if results else 0,scope=policy['scope']))
print('Metric reproduction passed.',flush=True)
