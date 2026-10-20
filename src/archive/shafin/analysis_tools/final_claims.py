"""Pre-writing numerical claim registry and final document claim checks."""
import json,io,zipfile,re,shutil
import pandas as pd
import numpy as np
from analysis_tools import final_synthesis as s
ROOT=s.ROOT

def clean(value):
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,)):return None if not np.isfinite(value) else float(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def phase45():
    s.verify_inputs();assert s.jd('phase44_outputs/summary.json')['status']=='completed'
    rows=[];values={}
    def add(cid,section,claim,path,columns,record,status='VERIFIED',caveat=''):
        data={k:clean(record[k]) for k in columns}
        values[cid]=data
        rows.append(dict(claim_id=cid,manuscript_section=section,claim_text=claim,source_file=path,source_columns=';'.join(columns),exact_value=json.dumps(data,ensure_ascii=False,allow_nan=False),status=status,caveat=caveat,source_sha256=s.sha(ROOT/path)))
    for file,prefix,section,keys,comment in [
      ('final_tables/main/T1_Datasets.csv','D','Results audit / activity',['retained','removed_or_excluded','groups','train','validation','internal_test','external_DHS_natural'],'Different source rows, candidate elements and held-out populations.'),
      ('final_tables/main/T2_Benchmark_audit.csv','A','Results benchmark audit',['supplied_train_n','supplied_test_n','exact_train_test_overlap','reverse_complement_train_test_overlap','duplicated_strong_records','duplicated_weak_records','candidate_n','candidate_groups','mixed_label_candidate_groups','mixed_label_candidate_records'],'Supplied files, not a claim about unavailable publisher or original author provenance.')]:
        d=s.read(file)
        for i,r in d.iterrows():
            label=str(r['task']) if 'task' in r else f'Stage{int(r.stage)}'
            add(prefix+'_'+re.sub(r'\W+','_',label),section,label+' dataset counts',file,keys,r,caveat=comment)
    file='final_tables/main/T3_Classification.csv'
    for r in s.read(file).to_dict('records'):
        add(f'C_S{int(r["stage"])}_{r["model"]}','Results grouped classification and composition',f'{r["model_label"]} Stage{int(r["stage"])} performance',file,['n','n_seeds','mean_fold_auroc','fold_sd_of_seed_mean_auroc','seed_sd_of_mean_fold_auroc','auroc','auprc','mcc','exact_GC_mean_fold_AUROC'],r,caveat=r['fit_scope']+'; current GC diagnostic uses fixed ensemble mean-fold scores; table mean-fold primary averages seed metrics; no confusion of aggregation or uncertainty types.')
    for file,prefix,section,keys in [
      ('final_tables/main/T4_MPRA_internal.csv','M','Results direct MPRA prediction',['n','pearson','spearman','rmse','mae','r2']),
      ('final_tables/main/T5_External_transfer.csv','E','Results external transfer',['n','spearman','pearson','mae','rmse','r2','mean_signed_error'])]:
        for r in s.read(file).to_dict('records'):
            add(f'{prefix}_{r["cell"]}_{r["arm"]}',section,f'{r["cell"]} {r["model_label"]} metrics',file,keys,r,caveat='Frozen previously examined cohort; conditional on selected fitted models. Archived spaced-mer baseline outside54-fit neural search where present.')
    file='final_statistics/external_transfer_summary.csv'
    for r in s.read(file).query('arm in ["dnabert2_fixedlr_full","dnabert2_fixedlr_early2"]').to_dict('records'):
        add(f'X_{r["cell"]}_{r["arm"]}','Results external duration','External own-prefix duration rank/error metrics',file,['n','spearman','pearson','mae','rmse','r2','mean_signed_error'],r,caveat='Secondary descriptive duration outcome; same fixed-lr trajectories, previously examined assay-shift cohort.')
    file='final_statistics/adjusted_tests.csv'
    for r in s.read(file).to_dict('records'):
        add('T_'+r['comparison_id'],'Results paired model and duration comparisons',r['arm_a']+' minus '+r['arm_b'],file,['estimate_a_minus_b','lower_95','upper_95','p_raw','p_holm','n_records','n_groups'],r,caveat='Fourteen-comparison Holm family chosen for final synthesis after earlier outcomes; pointwise conditional intervals and sharp prediction-swap-null p-values differ; no algorithm/search uncertainty.')
    file='final_tables/supplement/S17_Compute.csv'
    for r in s.read(file).to_dict('records'):
        add(f'K_{r["cell_line"]}_{r["model"]}','Results training adequacy and cost',r['cell_line']+' '+r['model']+' fit cost',file,['fits','elapsed_hours','optimizer_updates','training_examples_seen'],r,caveat=r['scope'])
    jobs=s.read('phase35_outputs/training_job_summary.csv');flags=s.read('phase38_outputs/selected_recipe_adequacy_flags.csv')
    countrecord=dict(training_fits=len(jobs),frozen_undefined_Pearson_flags=int(jobs.collapsed.sum()),practical_nearconstant_fits=int(jobs.validation_prediction_sd.lt(1e-4).sum()),selected_fits=len(flags),selected_potentially_budget_limited=int(flags.possibly_budget_limited.sum()),classification_checkpoints=s.jd('phase32_outputs/summary.json')['checkpoints_reproduced'],MPRA_validation_checkpoints=s.jd('phase37_outputs/summary.json')['validation_checkpoints_reproduced'],MPRA_evaluation_sets=s.jd('phase37_outputs/summary.json')['evaluation_sets_reproduced'])
    assert countrecord==dict(training_fits=54,frozen_undefined_Pearson_flags=0,practical_nearconstant_fits=2,selected_fits=18,selected_potentially_budget_limited=6,classification_checkpoints=60,MPRA_validation_checkpoints=72,MPRA_evaluation_sets=60)
    taskcounts=s.read('final_tables/main/T1_Datasets.csv').query('task in ["HepG2","K562"]')
    countrecord.update(complete_assay_candidates=int(taskcounts.complete_assay_candidates.sum()),retained_assay_candidates=int(taskcounts.retained.sum()),removed_assay_candidates=int(taskcounts.removed_or_excluded.sum()))
    s.dump('final_evidence/derived_study_counts.json',countrecord)
    add('RUNTIME_COUNTS','Results adequacy / Methods reproducibility','Completed fits, reproductions and practical collapse accounting','final_evidence/derived_study_counts.json',list(countrecord),countrecord,caveat='Practical collapse is a post hoc descriptive screen, never a new exclusion or eligibility rule. Six selected budget flags and boundary choices prevent a convergence claim.')
    cp=s.jd('phase30_outputs/training_protocol.json')['protocol']
    add('P_CLASS','Methods classification','Frozen current classification protocol','phase30_outputs/training_protocol.json',['base_seeds','epochs','batch_size','gradient_accumulation','lr','weight_decay','dropout','gradient_clip','max_tokens','model_revisions'],cp,caveat='Current consistent implementation; not authenticated reproduction of the unavailable original historical training wrapper.')
    mp=s.jd('phase34_outputs/training_protocol.json')
    add('P_MPRA','Methods activity','Frozen current activity protocol','phase34_outputs/training_protocol.json',['seeds','input_bp','lr_grid','epochs','patience','minimum_epochs','batch','accumulation','early_stopping_delta','revision'],mp,caveat='Equal candidates, unequal budgets; stopping and selection frozen before new test inference.')
    sp=s.jd('final_statistics/statistical_methods.json')
    add('P_STATS','Methods statistical analysis','Frozen final-synthesis statistical settings','final_statistics/statistical_methods.json',['family_size','bootstrap_draws','permutation_draws','random_seed','threshold','alpha'],sp,caveat=sp['scope'])
    # Extend the input generation explicitly for the feedback-requested archived zero-shot subsection.
    archive='5_enhancer_phase5_external_transfer_results.zip';provenance=json.loads(s.member(archive,'phase5_provenance.json'))
    for name in ['phase5_external_metrics.csv','phase5_external_predictions.csv']:
        raw=s.member(archive,name);assert s.digest(raw)==provenance['output_sha256'][name]
        if name.endswith('metrics.csv'):s.text('final_evidence/inputs/archived_'+name,raw.decode('utf-8'))
    s.dump('final_evidence/inputs/archived_phase5_provenance.json',provenance)
    lock=dict(reason='Additional locked source for feedback-requested Results2.5 legacy-label transfer; not a change to Phase39/41 primary statistical inputs or outcomes',archive='reference_inputs/readiness_sources/'+archive,archive_sha256=s.sha(s.SRC/archive),member_sha256=provenance['output_sha256'],scope='Archived Stage1 legacy classifiers on top/bottom activity cohort; not current repeated-seed classifier fits or same population as primary9B test',utc=s.NOW())
    s.dump('final_evidence/phase45_additional_source_lock.json',lock)
    file='final_evidence/inputs/archived_phase5_external_metrics.csv';zero=s.read(file)
    for r in zero[zero.model.str.startswith('finetuned')].to_dict('records'):
        add(f'Z_{r["cell_line"]}_{r["model"]}','Results legacy-label transfer','Archived legacy classifier score transfer to activity',file,['n','auroc','spearman_activity','exact_gc_pair_auc'],r,caveat='Older full-legacy-training classifier fits; top/bottom MPRA activity cohort differs from new regression test. No matched causal effect of changing task or training is estimated.')
    file='phase23_outputs/filtering_paired_contrasts.csv'
    for r in s.read(file).query('analysis=="primary" and contrast=="removed minus retained" and metric=="mae"').to_dict('records'):
        add(f'F_{r["cell_line"]}_{r["population_or_model"]}','Results filtering transport','Older frozen-model removed minus retained MAE',file,['estimate','lower_95','upper_95'],r,caveat='Previously examined, differently filtered populations; older fitted models; unadjusted exploratory intervals; not a filter retraining intervention.')
    file='phase18_outputs/observed_activity_tail_diagnostics.csv'
    for r in s.read(file).query('population=="retained" and band=="high"').to_dict('records'):
        add(f'H_{r["cell_line"]}_{r["model"]}','Results high activity','Older frozen CNN high-activity underprediction',file,['n','signed_error','all_seeds_underpredict_fraction','empirical_coverage'],r,caveat='Measured-activity diagnostic, older fitted CNNs; not a prospective selection procedure or current selected-model calibration.')
    conceptual=[('I_GROUPING','Discussion benchmark validity','Similarity-aware fixed evaluation exposes dependence in the supplied benchmark.','SUPPORTED_WITH_LIMIT','Cannot quantify the causal amount by which exact duplicates inflated the unavailable original neural results.'),('I_MODELS','Discussion task dependence','Specialist CNNs can outperform the evaluated DNABERT-2 model under the tested MPRA recipes.','SUPPORTED_WITH_LIMIT','Unequal compute/pretraining; finite LR grid, boundary and budget flags, two cells, short sequences.'),('I_TRANSFER','Discussion assay transfer','External rank information persists while absolute calibration shifts.','SUPPORTED_WITH_LIMIT','All current external ensemble R2 values are negative; no genome-wide or clinical validity claim.'),('I_EQUAL','Discussion classification','DNABERT generations are equivalent or tied.','REMOVE','Intervals including zero do not prove equivalence.'),('I_BEST','Discussion ranking','DNABERT-2 is the best MPRA model.','REMOVE','LegNet has higher current internal Pearson under the tested recipes.'),('I_SEQUENCE','Discussion biological scope','Sequence alone cannot predict enhancer strength.','REMOVE','Direct MPRA training recovers substantial signal; no intrinsic impossibility has been demonstrated.'),('I_CALIBRATED','Discussion external interpretation','External transfer validates absolute activity prediction.','REMOVE','Negative R2 and signed bias contradict this interpretation.'),('I_OLD','Results current ranking','Earlier Phase25 neural rankings represent the current selected-model study.','SUPERSEDED','Use Phase32 classification and Phase37 current MPRA; older filtering diagnostics remain separately valid.')]
    for cid,section,claim,status,caveat in conceptual:
        rows.append(dict(claim_id=cid,manuscript_section=section,claim_text=claim,source_file='final_evidence/authoritative_sources.csv',source_columns='',exact_value='',status=status,caveat=caveat,source_sha256=s.sha(ROOT/'final_evidence/authoritative_sources.csv')))
    allowed={'VERIFIED','SUPPORTED_WITH_LIMIT','SUPERSEDED','REMOVE','NEEDS_CITATION'}
    ledger=pd.DataFrame(rows);assert ledger.claim_id.is_unique and set(ledger.status)<=allowed
    assert ledger[ledger.exact_value.ne('')].status.eq('VERIFIED').all()
    s.table('claim_audit/claim_evidence_ledger.csv',ledger)
    s.dump('claim_audit/verified_values.json',values)
    immutable=[]
    for folder in ['final_evidence','final_statistics','final_tables','final_figures','final_supplement']:
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file() or any(x in p.parts for x in ['source_at_freeze','source_at_plan','contrast_cache']):continue
            if p.name.endswith('.inspect.ndjson'):continue
            immutable.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=s.sha(p)))
    s.dump('claim_audit/prewriting_evidence_lock.json',immutable)
    s.dump('phase45_outputs/summary.json',dict(status='completed_prewrite_numeric_gate_passed',ledger_rows=len(ledger),numerical_claims_verified=len(values),supported_with_limits=int(ledger.status.eq('SUPPORTED_WITH_LIMIT').sum()),removed_or_superseded=int(ledger.status.isin(['REMOVE','SUPERSEDED']).sum()),new_training=False,manuscript_started=False,actual_manuscript_audit_required_after_drafting=True,source_sha256=s.sha(__file__)))
    print('Phase45 prewriting gate passed:',len(values),'numerical claims verified;',len(ledger),'total ledger rows.')
