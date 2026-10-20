"""Public reporting adapters; no training, inference, recipe changes or input writes."""
import json,sys,runpy,shutil,itertools
import numpy as np
import pandas as pd
from . import io as s
from .statistics import classification_metrics,regression_metrics,stats_selfcheck

def audit():
    s.verify_inputs()
    roster=s.read('phase21_outputs/historical_test_overlap_roster.csv')
    expected=s.read('final_evidence/benchmark_audit.csv').set_index('stage')
    checks=[]
    for stage,m in zip([1,2],s.manifests()):
        test=roster[roster.stage.eq(stage)]
        train=pd.concat([m[~m.role.eq('legacy_test')],m[m.record_id.isin(test[test.exact_training_match].record_id)]])
        exact=test.sequence_sha256.isin(set(train.sequence_sha256))
        reverse=test.reverse_complement_sha256.isin(set(train.sequence_sha256))
        candidate=m[m.role.eq('train_candidate')]
        assert candidate.groupby('similarity_group_id').fold_5.nunique().max()==1
        assert candidate.groupby('canonical_sequence_sha256').fold_5.nunique().max()==1
        row=dict(stage=stage,train=len(train),test=len(test),exact=int(exact.sum()),reverse_complement=int(reverse.sum()),candidates=len(candidate),groups=candidate.similarity_group_id.nunique())
        for a,b in [('train','supplied_train_n'),('test','supplied_test_n'),('exact','exact_train_test_overlap'),('reverse_complement','reverse_complement_train_test_overlap'),('candidates','candidate_n'),('groups','candidate_groups')]:assert row[a]==expected.loc[stage,b],(stage,a)
        checks.append(row)
    s.dump('audit_validation.json',dict(passed=True,counts=checks,fixed_group_isolation=True,scope='Authenticated sequence/RC hashes and supplied graph; not a fresh exhaustive raw-DNA LCS rescan'))
    s.table('benchmark_audit_recomputed.csv',pd.DataFrame(checks))
    print(pd.DataFrame(checks).to_string(index=False),flush=True)
    return checks

def classification():
    s.verify_inputs();d=s.read('final_statistics/classification_predictions.csv.gz')
    expected=s.read('final_statistics/classification_summary.csv').set_index(['stage','model'])
    rows=[];errors=[]
    for (stage,model),g in d.groupby(['stage','model']):
        values=classification_metrics(g.label,g.score)
        for key,value in values.items():errors.append(abs(value-expected.loc[(stage,model),key]))
        rows.append(dict(stage=stage,model=model,**values))
    assert len(rows)==18 and max(errors)<1e-12
    result=pd.DataFrame(rows);s.table('classification_metrics_recomputed.csv',result)
    seeds=pd.read_csv(s.PACKAGE/'results/predictions/classification_current_all_seeds.csv.gz',float_precision='round_trip')
    reference=s.read('phase32_outputs/seed_metrics_pooled_and_fold.csv').set_index(['stage','model','base_seed'])
    seed_checks=0
    for (stage,model,seed),g in seeds.groupby(['stage','model','base_seed']):
        values=classification_metrics(g.label,g.score)
        assert abs(values['auroc']-reference.loc[(stage,model,seed),'auroc'])<1e-12
        seed_checks+=1
    assert seed_checks==12
    s.dump('classification_validation.json',dict(passed=True,rows=18,individual_seed_sets=seed_checks,max_error=float(max(errors)),seed_and_fold_variability_distinct=True))
    print(result[['stage','model','auroc']].to_string(index=False),flush=True)
    return result

def validate_selection():
    jobs=s.read('phase35_outputs/training_job_summary.csv')
    assert len(jobs)==54 and jobs.job.is_unique
    expected=s.read('phase36_outputs/validation_candidate_scores.csv')
    rows=[]
    for (cell,model,grid),d in jobs.groupby(['cell_line','model','grid_index']):
        assert len(d)==3 and d.seed.nunique()==3
        finite=np.isfinite(d.validation_pearson).all()
        mean=float(d.validation_pearson.mean()) if finite else None
        r=expected[expected.cell.eq(cell)&expected.model.eq(model)&expected.grid_index.eq(grid)].iloc[0]
        assert bool(r.eligible)==bool(finite)
        if finite:assert abs(mean-r.mean_validation_pearson)<1e-12
        rows.append(dict(cell=cell,model=model,grid_index=int(grid),eligible=bool(finite),mean_validation_pearson=mean))
    candidates=pd.DataFrame(rows);chosen=[]
    lock=json.loads((s.PACKAGE/'configs/activity_selection_lock.json').read_text(encoding='utf-8'))
    for (cell,model),d in candidates.groupby(['cell','model']):
        d=d[d.eligible].sort_values(['mean_validation_pearson','grid_index'],ascending=[False,True],kind='stable')
        assert len(d)>0
        choice=d.iloc[0].to_dict()
        original=next(r for r in lock['recipes'] if r['cell']==cell and r['model']==model)
        assert choice['grid_index']==original['grid_index']
        chosen.append(choice)
    s.table('validation_candidates_recomputed.csv',candidates)
    s.table('validation_selected_recipes_recomputed.csv',pd.DataFrame(chosen))
    s.dump('selection_validation.json',dict(passed=True,all_fit_records=54,candidates=18,selected_model_cell_recipes=6,uses_test_outcomes=False))
    return chosen

def activity(cohort='internal'):
    s.verify_inputs()
    if cohort=='external':cohort='DHS_natural'
    choices=validate_selection()
    from .validation import validate_activity_artifacts
    validate_activity_artifacts()
    d=s.read('phase37_outputs/verified_ensemble_predictions.csv.gz')
    expected=s.read('final_statistics/mpra_internal_summary.csv' if cohort=='internal' else 'final_statistics/external_transfer_summary.csv').set_index(['cell','cohort','arm'])
    rows=[];errors=[]
    for (cell,group,arm),g in d[d.cohort.eq(cohort)].groupby(['cell','cohort','arm']):
        values=regression_metrics(g.activity,g.prediction)
        for key,value in values.items():errors.append(abs(value-expected.loc[(cell,group,arm),key]))
        rows.append(dict(cell=cell,cohort=group,arm=arm,**values))
    assert len(rows)==10 and max(errors)<1e-12
    result=pd.DataFrame(rows);s.table(cohort+'_activity_metrics_recomputed.csv',result)
    s.dump(cohort+'_validation.json',dict(passed=True,ensemble_sets=len(rows),max_error=float(max(errors)),selection_locked_before_inference=True))
    print(result[['cell','cohort','arm','pearson','spearman','r2']].to_string(index=False),flush=True)
    return result

def statistics(full=False):
    args=sys.argv[:]
    try:
        sys.argv=['metric_runner.py']+(['--full-statistics'] if full else [])
        runpy.run_path(str(s.PACKAGE/'src/enhancer_benchmark_audit/_metric_runner.py'),run_name='__main__')
    finally:sys.argv=args
    return json.loads((s.ROOT/'metric_validation.json').read_text(encoding='utf-8'))

def figures_and_tables():
    from .figures import build
    s.verify_inputs();report=build()
    for name in ['main_tables','supplementary_tables']:
        target=s.ROOT/'tables'/name
        target.mkdir(parents=True,exist_ok=True)
        for p in (s.PACKAGE/'results'/name).glob('*.csv'):
            shutil.copy2(p,target/p.name);assert s.sha(p)==s.sha(target/p.name)
    s.dump('table_assembly_validation.json',dict(passed=True,main_tables=7,supplementary_tables=19,literature_records='Preserved cited source rows; no new literature scraping'))
    return report

COMMANDS={'audit':audit,'classification':classification,'activity':activity,'external':lambda:activity('external'),'statistics':statistics,'figures-and-tables':figures_and_tables}
