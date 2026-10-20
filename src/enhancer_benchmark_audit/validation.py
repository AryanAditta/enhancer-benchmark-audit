"""Validate all supplied MPRA validation and seed evaluation predictions."""
import json
import numpy as np
import pandas as pd
from . import io as s
from .statistics import regression_metrics

_DONE=False
def validate_activity_artifacts():
    global _DONE
    if _DONE:return
    truth=pd.read_csv(s.PACKAGE/'data/manifests/activity_validation_truth.csv.gz',float_precision='round_trip')
    errors=[];validation_rows=[]
    for p in sorted((s.PACKAGE/'results/validation_predictions').glob('*.csv.gz')):
        stem=p.name[:-7];job,role=stem.rsplit('__',1)
        meta=json.loads((s.PACKAGE/'results/job_metadata/activity'/(job+'.json')).read_text(encoding='utf-8'))
        d=pd.read_csv(p,float_precision='round_trip')
        t=truth[truth.cell_line.eq(meta['cell_line'])].set_index('record_id')
        assert d.record_id.is_unique and set(d.record_id)==set(t.index)
        y=t.loc[d.record_id,'mean_log2_activity'].to_numpy()
        prediction=d.prediction.to_numpy()
        got=float(np.corrcoef(y,prediction)[0,1]) if np.std(prediction)>=1e-8 and np.std(y)>0 else None
        expected=meta['validation_pearson'] if role=='best' else meta['early2_validation_pearson']
        assert (got is None)==(expected is None)
        if got is not None:errors.append(abs(got-expected));assert errors[-1]<1e-12
        validation_rows.append({'job':job,'role':role,'n':len(d),'pearson':got})
    assert len(validation_rows)==72
    all_seeds=pd.read_csv(s.PACKAGE/'results/predictions/activity_all_seed_evaluations.csv.gz',float_precision='round_trip')
    expected=s.read('phase37_outputs/seed_metrics_independent.csv').set_index(['cell','cohort','arm','seed'])
    count=0
    for key,d in all_seeds.groupby(['cell','cohort','arm','seed']):
        values=regression_metrics(d.activity,d.prediction)
        for metric in ['pearson','spearman','r2','rmse','mae','prediction_sd']:
            error=abs(values[metric]-expected.loc[key,metric]);errors.append(error);assert error<1e-12,(key,metric,error)
        errors.append(abs(values['mean_signed_error']-expected.loc[key,'mean_error']));assert errors[-1]<1e-12
        count+=1
    assert count==60
    ensembles=s.read('phase37_outputs/verified_ensemble_predictions.csv.gz')
    for key,d in all_seeds.groupby(['cell','cohort','arm']):
        pivot=d.pivot(index='record_id',columns='seed',values='prediction').sort_index()
        expected_ensemble=ensembles[ensembles.cell.eq(key[0])&ensembles.cohort.eq(key[1])&ensembles.arm.eq(key[2])].set_index('record_id').sort_index()
        assert pivot.index.equals(expected_ensemble.index) and pivot.shape[1]==3
        for got,want in [(pivot.mean(axis=1),expected_ensemble.prediction),(pivot.std(axis=1,ddof=1),expected_ensemble.seed_sd)]:
            error=float(np.max(np.abs(got.to_numpy()-want.to_numpy())));errors.append(error);assert error<1e-12
    s.table('all_validation_metrics_recomputed.csv',pd.DataFrame(validation_rows))
    s.dump('activity_prediction_artifact_validation.json',{'passed':True,'validation_checkpoints':72,'seed_evaluation_sets':60,'ensemble_sets':20,'max_numeric_error':float(max(errors)),'new_training':False,'new_checkpoint_inference':False})
    _DONE=True
