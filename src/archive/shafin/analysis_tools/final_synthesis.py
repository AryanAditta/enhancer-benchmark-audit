"""Versioned final synthesis. Phases 12--38 and historical sources are read-only.

Run only with the enhancer interpreter, -B -X utf8, from the contribution root.
This helper does not train or change selection, folds, thresholds, or old results.
"""
from pathlib import Path
import os, sys, json, hashlib, io, zipfile, shutil, datetime, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
assert Path.cwd().resolve() == ROOT
assert Path(sys.prefix).name.lower() == 'enhancer'
SRC = ROOT / 'reference_inputs/readiness_sources'
NOW = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def jd(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def dump(path, obj):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def text(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + '\n', encoding='utf-8')

def read(path):
    return pd.read_csv(ROOT / path, float_precision='round_trip')

def table(path, frame):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g',
                 compression={'method': 'gzip', 'mtime': 0} if path.name.endswith('.gz') else None)

def member(archive, filename):
    with zipfile.ZipFile(SRC / archive) as z:
        hits = [n for n in z.namelist() if n.split('/')[-1] == filename]
        assert len(hits) == 1, (archive, filename)
        return z.read(hits[0])

def manifests():
    return [read(f'final_evidence/inputs/stage{s}_candidate_manifest.csv') for s in (1, 2)]

def verify_inputs():
    lock = jd('final_evidence/phase39_provenance.json')
    for item in lock['input_files']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    for item in jd('final_evidence/protected_history_before.json'):
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    return lock

def protected_snapshot():
    files = []
    for number in range(12, 39):
        folder = ROOT / f'phase{number}_outputs'
        if not folder.exists():
            continue
        for p in folder.rglob('*'):
            if not p.is_file() or p.suffix.lower() in {'.zip', '.pt', '.bin', '.safetensors'}:
                continue
            if any(x.lower() in {'runtime', 'runtime_validation', '__pycache__', 'fit_cache', 'cache'} for x in p.parts):
                continue
            if p.suffix.lower() in {'.csv', '.gz', '.tsv', '.json', '.md', '.txt', '.py'}:
                files.append(p)
    files += [p for p in ROOT.glob('[[]*.ipynb') if int(p.name.split(']')[0][1:]) <= 38]
    files += [p for p in (ROOT / 'analysis_tools').glob('*.py') if p.name != Path(__file__).name]
    return [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(set(files))]

def phase39():
    output = ROOT / 'final_evidence'
    output.mkdir(exist_ok=True)
    if (output / 'phase39_provenance.json').exists():
        verify_inputs()
        print('Reused authenticated Phase 39 evidence freeze.'); return
    (output / 'inputs').mkdir(exist_ok=True)
    archive = '1_ enhancer_audit_v1_outputs.zip'
    audit = json.loads(member(archive, 'audit_run_summary.json'))
    extracted = []
    for name, expected in audit['outputs_sha256'].items():
        raw = member(archive, name)
        assert digest(raw) == expected, name
        dest = output / 'inputs' / name
        dest.write_bytes(raw)
        extracted.append({'path': dest.relative_to(ROOT).as_posix(), 'sha256': expected,
                          'bytes': len(raw), 'source': f'reference_inputs/readiness_sources/{archive}!{name}'})
    for name in ['phase3_model_robustness.csv', 'phase3_fold_gc_matched.csv', 'phase3_matching_coverage.csv']:
        raw = member('3_enhancer_phase3_robustness.zip', name)
        dest = output / 'inputs' / name; dest.write_bytes(raw)
        extracted.append({'path': dest.relative_to(ROOT).as_posix(), 'sha256': digest(raw), 'bytes': len(raw),
                          'source': 'reference_inputs/readiness_sources/3_enhancer_phase3_robustness.zip!' + name})
    dump('final_evidence/inputs/audit_run_summary.json', audit)
    assert sha(ROOT / 'phase34_outputs/training_protocol.json') == '2d204c4d565b46ebbb1833a00141d045201d59b79b503d43c3e6d9214d1101a9'
    assert sha(ROOT / 'analysis_tools/mpra_fairness_training.py') == '83a1ae9a9e03361e4fe517044ee7b3cc76f580db04878e4a12f1f7873c60d311'
    assert jd('phase32_outputs/summary.json')['all_prediction_checks_passed']
    assert jd('phase37_outputs/summary.json')['status'] == 'completed'
    authorities = [
        ('Legacy benchmark audit', 'identities, grouping and folds', '1', 'final_evidence/inputs/stage1_candidate_manifest.csv', 'submitted benchmark counts', 'Authenticated original manifest, fixed LCS160 graph'),
        ('Legacy Stage 2 audit', 'identities, grouping and folds', '1', 'final_evidence/inputs/stage2_candidate_manifest.csv', 'submitted benchmark counts', 'Same locked graph; no chromosome-disjoint claim'),
        ('Seven archived classification baselines', 'single-run grouped OOF', '2/21', 'phase21_outputs/authenticated_classification_predictions.csv.gz', 'unverified submitted tables', 'Record-level archived predictions; never manufacture repeated-seed estimates'),
        ('Composition sensitivity', 'archived exact-GC matching', '3', 'final_evidence/inputs/phase3_model_robustness.csv', 'naive composition independence claim', 'Original single-run predictions; distinct from current repeated transformer fits'),
        ('Fine-tuned classification', 'three-seed grouped OOF predictions', '30-33', 'phase32_outputs/independently_verified_predictions.csv.gz', 'Phase21/25 fine-tuned values', 'Fresh consistent 60-job implementation; historical fits separate'),
        ('Fine-tuned classification variability', 'seed and fold variability', '32', 'phase32_outputs/seed_vs_fold_variability.csv', 'single-run uncertainty', 'Three seeds; dependent folds do not provide fifteen independent replicates'),
        ('Primary MPRA comparisons', '9B and DHS-natural ensemble predictions', '34-38', 'phase37_outputs/verified_ensemble_predictions.csv.gz', 'Phase12/25 current neural ranking', 'Validation-only finite grid selection; exact 72+60 reproduction'),
        ('MPRA candidate search and duration', '54 fits, epoch and time records', '35', 'phase35_outputs/training_job_summary.csv', 'two-epoch historical duration interpretation', 'New-run own two-epoch prefix; equal candidate counts, unequal compute'),
        ('MPRA selection', 'validation-only recipe lock', '36', 'phase36_outputs/selection_lock.json', 'test-driven recipe selection', 'Three-seed candidate means and all checkpoint hashes locked before inference'),
        ('MPRA practical collapse', 'post-seal descriptive erratum', '38 addendum', 'phase38_outputs/ERRATUM_collapse_count.md', 'unqualified zero collapse claim', 'Zero frozen undefined-Pearson flags and two practical near-constant fits; no exclusions'),
        ('Filtering attrition', 'assay population construction', '14/24', 'phase24_outputs/dataset_attrition_counts.csv', 'retained population equals all candidates', 'Complete-pair, retained and removed counts are different populations'),
        ('Filtering transport', 'retained/removed frozen-model error', '16', 'phase16_outputs/retained_removed_metrics.csv', 'causal benefit of filtering', 'Older fitted recipes; evaluate transport, not a filter retraining intervention'),
        ('Filtering uncertainty', 'conditional contrasts', '23', 'phase23_outputs/population_contrasts.csv', 'uncertain earlier error-gap claims', 'Final shared bootstrap where available; source checked before use'),
        ('Reliability', 'interval and subgroup coverage', '17/24', 'phase17_outputs/interval_audit_coverage.csv', 'seed spread guarantees reliability', 'Marginal calibration; disjoint audit population; historical model versions'),
        ('High activity', 'tail underprediction', '18', 'phase18_outputs/observed_activity_tail_diagnostics.csv', 'high-activity seed agreement implies accuracy', 'Measured-activity diagnostic, not prospective selection'),
        ('External construction', 'origin and sequence separation', '10/24', 'phase24_outputs/external_selection_and_separation.json', 'fresh confirmatory test claim', 'Previously examined assay-shift cohort; current rankings use Phase37'),
        ('Subgroup diagnostics', 'training-only thresholds', '24', 'phase24_outputs/training_only_thresholds_exact.csv', 'rounded post hoc cutoffs', 'Full precision, model/population versions explicit')]
    # Resolve the actual Phase23 filename rather than creating a phantom source.
    possible = list((ROOT / 'phase23_outputs').glob('*contrast*.csv'))
    authorities = [tuple((str(possible[0].relative_to(ROOT)).replace('\\', '/') if j == 3 and row[3] == 'phase23_outputs/population_contrasts.csv' and possible else value)
                        for j, value in enumerate(row)) for row in authorities]
    cols = ['scientific_question', 'result_type', 'authoritative_phase', 'authoritative_file', 'supersedes', 'reason']
    sources = pd.DataFrame(authorities, columns=cols)
    for path in sources.authoritative_file: assert (ROOT / path).is_file(), path
    table('final_evidence/authoritative_sources.csv', sources)
    table('final_evidence/superseded_results.csv', sources[['supersedes', 'authoritative_file', 'reason']].rename(columns={'supersedes':'historical_result'}))
    roster = read('phase21_outputs/historical_test_overlap_roster.csv')
    counts = []; splits = []; audits = []
    rc = lambda s: s.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    for stage, m in zip((1, 2), manifests()):
        candidate = m[m.role.eq('train_candidate')]
        test = roster[roster.stage.eq(stage)]
        train = pd.concat([m[~m.role.eq('legacy_test')], m[m.record_id.isin(test[test.exact_training_match].record_id)]])
        assert len(train) == {1:2968, 2:1484}[stage]
        ts = set(train.sequence)
        exact = test.sequence.isin(ts); reverse = test.sequence.map(rc).isin(ts)
        ambiguous = candidate.groupby('similarity_group_id').label.nunique().gt(1)
        for population, frame in [('supplied_training',train),('supplied_test',test),('grouped_candidates',candidate),('cleaned_supplied_test',test[~exact])]:
            counts.append(dict(task=f'classification_stage{stage}',population=population,n=len(frame),positive=int(frame.label.sum()),negative=int(frame.label.eq(0).sum()),unit='unique records within source split',source='locked Phase1 manifest and Phase11A roster'))
        audits.append(dict(stage=stage,supplied_train_n=len(train),supplied_test_n=len(test),exact_train_test_overlap=int(exact.sum()),reverse_complement_train_test_overlap=int(reverse.sum()),exact_or_rc_overlap=int((exact|reverse).sum()),duplicated_strong_records=int((exact & test.label.eq(1)).sum()) if stage==2 else int(test[exact].sequence.isin(set(train[train.subtype.eq('strong')].sequence)).sum()),duplicated_weak_records=int(test[exact].sequence.isin(set(train[train.subtype.eq('weak')].sequence)).sum()),additional_test_related_training_excluded=int(m.role.eq('excluded_related_to_legacy_test').sum()),candidate_n=len(candidate),candidate_groups=int(candidate.similarity_group_id.nunique()),mixed_label_candidate_groups=int(ambiguous.sum()),mixed_label_candidate_records=int(candidate.similarity_group_id.isin(ambiguous[ambiguous].index).sum()),group_rule='connected components of exact LCS>=160 in either orientation; not percent local identity'))
        for fold, g in candidate.groupby('fold_5'):
            splits.append(dict(task=f'classification_stage{stage}',partition=f'OOF_fold_{int(fold)}',n=len(g),groups=g.similarity_group_id.nunique(),positive=int(g.label.sum()),source='Phase1 manifest'))
    for r in read('phase34_outputs/partition_counts.csv').itertuples():
        splits.append(dict(task=f'MPRA_{r.cell_line}',partition=r.split_9b,n=r.n,groups=None,positive=None,source='Phase34 locked primary9B partition counts'))
        counts.append(dict(task=f'MPRA_{r.cell_line}',population='retained_'+r.split_9b,n=r.n,unit='assay elements',source='Phase34 partition_counts.csv'))
    for r in read('phase24_outputs/dataset_attrition_counts.csv').itertuples():
        counts.append(dict(task=f'MPRA_{r.cell_line}',population=r.stage,n=r.n,unit=r.unit,source='Phase24 attrition audit'))
    e = read('phase37_outputs/verified_ensemble_predictions.csv.gz')
    for cell, g in e[e.cohort.eq('DHS_natural')].groupby('cell'):
        counts.append(dict(task=f'MPRA_{cell}',population='external_DHS_natural',n=g.record_id.nunique(),unit='assay elements',source='Phase37 authenticated ensemble predictions'))
    table('final_evidence/final_dataset_counts.csv',pd.DataFrame(counts))
    table('final_evidence/final_split_inventory.csv',pd.DataFrame(splits))
    table('final_evidence/benchmark_audit.csv',pd.DataFrame(audits))
    classpred=read('phase21_outputs/authenticated_classification_predictions.csv.gz')
    models=[]
    for model in sorted(classpred.model.unique()):
        repeated=model in ['finetuned_dnabert2','finetuned_dnabert_legacy']
        models.append(dict(task='classification',model=model,current_source='Phase32' if repeated else 'Phase2/21',seeds_per_fold=3 if repeated else 1,folds=5,scope='fixed grouped folds; 3 current seeds' if repeated else 'archived single run; no invented seed SD'))
    for model in ['spaced6_4_ridge','dnabert2','legnet','mprann']:
        models.append(dict(task='MPRA',model=model,current_source='Phase9B archived baseline' if model=='spaced6_4_ridge' else 'Phase35-37',seeds_per_fold=1 if model=='spaced6_4_ridge' else 3,folds=None,scope='separate single baseline recipe; outside 54-fit grid' if model=='spaced6_4_ridge' else '3 declared learning rates x 3 seeds per cell; unequal compute'))
    table('final_evidence/final_model_inventory.csv',pd.DataFrame(models))
    seedclass=read('phase32_outputs/effective_seed_schedule_and_dependence.csv')
    table('final_evidence/final_seed_inventory.csv',pd.concat([seedclass.assign(task='classification'),read('phase35_outputs/training_job_summary.csv')[['cell_line','model','seed','job']].assign(task='MPRA')],ignore_index=True))
    metricrows=[dict(task='classification',metric=m,aggregation='pooled fixed-model OOF; fold/seed variability separate',primary=m=='AUROC') for m in ['AUROC','AUPRC','Brier','sensitivity_at_0.5','specificity_at_0.5','MCC_at_0.5','balanced_accuracy_at_0.5']]
    metricrows += [dict(task=t,metric=m,aggregation='three-seed ensemble; per-seed metrics also retained',primary=m==('Pearson' if t=='MPRA_internal' else 'Spearman')) for t in ['MPRA_internal','MPRA_external'] for m in ['Pearson','Spearman','RMSE','MAE','R2','mean_signed_error']]
    table('final_evidence/final_metric_inventory.csv',pd.DataFrame(metricrows))
    claims=[('audit','The supplied benchmark test contains training-identical strong records.','benchmark_audit.csv','VERIFIED','Do not infer why publisher and supplied files differ.'),('classification','Current fine-tuned classification uses three seeds on five fixed grouped folds.','phase32_outputs/summary.json','VERIFIED','Folds/seeds do not establish algorithm-level uncertainty.'),('duration','Longer DNABERT-2 is compared with its own exact first-two-epoch prefix.','phase37_outputs/ensemble_metrics.csv','VERIFIED','A fixed-lr duration contrast, separate from selected model comparisons.'),('ranking','Architecture rankings describe the tested recipes and selected finite grid.','phase37_outputs/ensemble_metrics.csv','SUPPORTED_WITH_LIMIT','Unequal budgets, only two cells and 200bp contexts.'),('transfer','External ranking and absolute-scale calibration are separate outcomes.','phase37_outputs/ensemble_metrics.csv','SUPPORTED_WITH_LIMIT','Previously examined cohort; negative R2 is retained.'),('collapse','Zero frozen undefined-Pearson flags coexist with two practical near-constant fits.','phase38_outputs/ERRATUM_collapse_count.md','VERIFIED','Post hoc descriptive screen; no altered fit inclusion or recipe selection.'),('filter','Retained/removed diagnostics use older frozen fits.','phase16_outputs/retained_removed_metrics.csv','VERIFIED','Not updated-model evidence or a causal filter retraining intervention.'),('universal','DNABERT-2 is universally superior or sequence is intrinsically insufficient.','','REMOVE','Unsupported generalization.')]
    table('final_evidence/final_claim_inventory.csv',pd.DataFrame(claims,columns=['claim_id','claim','source','status','caveat']))
    shutil.copy2(ROOT/'CONTRIBUTION_LOG.md',output/'contribution_log_at_start.md')
    protected=protected_snapshot()
    dump('final_evidence/protected_history_before.json',protected)
    inputs=[]
    paths=set(sources.authoritative_file)|{'phase32_outputs/three_seed_ensemble_oof_predictions.csv.gz','phase32_outputs/ensemble_metrics.csv','phase32_outputs/fold_metrics_independent.csv','phase32_outputs/seed_metrics_pooled_and_fold.csv','phase32_outputs/updated_classification_comparison.csv','phase32_outputs/conditional_paired_ensemble_intervals.csv','phase37_outputs/ensemble_metrics.csv','phase37_outputs/conditional_paired_intervals.csv','phase37_outputs/seed_metrics_independent.csv','phase37_outputs/seed_variability.csv','phase35_outputs/run_lock.json','phase35_outputs/training_job_summary.csv','phase36_outputs/validation_candidate_scores.csv','phase38_outputs/selected_recipe_adequacy_flags.csv','phase30_outputs/compatibility_decision.json','phase38_outputs/POSTSEAL_ADDENDUM_STATUS.json','feedback_inputs/20261015/Manuscript Preparation.docx','feedback_inputs/20261015/extracted_feedback.json','phase21_outputs/historical_test_predictions.csv','phase21_outputs/historical_test_overlap_roster.csv','final_evidence/inputs/phase3_model_robustness.csv'}
    for p in sorted(paths):
        assert (ROOT/p).is_file(),p
        inputs.append(dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)))
    for row in extracted:
        if row['path'] not in paths: inputs.append({k:row[k] for k in ['path','bytes','sha256']})
    dump('final_evidence/phase39_provenance.json',dict(status='completed',created_utc=NOW(),feedback_sha256=sha(ROOT/'feedback_inputs/20261015/Manuscript Preparation.docx'),source_sha256=sha(__file__),input_files=inputs,extracted_members=extracted,protected_file_count=len(protected),new_training=False,history_policy='Earlier artifacts read-only. Live contribution log will receive append-only entries. No old archive is resealed.',scope='Final synthesis on previously examined cohorts; optional historical experiment subject to authentication stop rule.'))
    dump('phase39_outputs/summary.json',dict(status='completed',authoritative_source_rows=len(sources),input_files_locked=len(inputs),protected_history_files=len(protected),new_training=False))
    print('Phase39 complete:',len(sources),'authorities;',len(inputs),'input locks;',len(protected),'protected files.')

def phase40():
    verify_inputs()
    compatibility=jd('phase30_outputs/compatibility_decision.json')
    p=read('phase21_outputs/historical_test_predictions.csv')
    assert set(p.model.unique())=={'gc_cpg_logreg','k3_logreg','spaced6_4_logreg'}
    provenance=json.loads(member('11A_enhancer_phase11a_overlap_impact.zip','phase11a_provenance.json'))
    for name in ['historical_test_predictions.csv','historical_test_overlap_roster.csv','historical_test_metrics.csv']:
        raw=member('11A_enhancer_phase11a_overlap_impact.zip',name)
        assert digest(raw)==provenance['output_sha256'][name]
        a=pd.read_csv(io.BytesIO(raw),float_precision='round_trip');b=read('phase21_outputs/'+name)
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-13,atol=1e-13)
    assert not compatibility['original_training_implementation_verified']
    feasibility=[dict(model=m,authenticated_historical_test_predictions=m=='spaced6_4_logreg',authenticated_original_training_implementation=m=='spaced6_4_logreg',disposition='archived descriptive rescore only' if m=='spaced6_4_logreg' else 'STOP: original historical training implementation/checkpoints and independent-test predictions unavailable',evidence='Phase11A notebook/provenance' if m=='spaced6_4_logreg' else 'Phase30 compatibility audit; Phase2 notebook is GENA correction, not complete original wrapper') for m in ['spaced6_4_logreg','finetuned_dnabert_legacy','finetuned_dnabert2']]
    table('phase40_outputs/feasibility_audit.csv',pd.DataFrame(feasibility))
    from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,matthews_corrcoef
    rows=[]
    for stage,g in p[p.model.eq('spaced6_4_logreg')&p.cohort.eq('historical_raw')].groupby('stage'):
        for condition,h in [('A_supplied_historical_test',g),('B_same_predictions_nonduplicate_test',g[~g.exact_training_match])]:
            valid=h.label.nunique()==2
            rows.append(dict(stage=stage,model='spaced6_4_logreg',condition=condition,n=len(h),positive=int(h.label.sum()),negative=int(h.label.eq(0).sum()),auroc=roc_auc_score(h.label,h.score) if valid else np.nan,auprc=average_precision_score(h.label,h.score) if valid else np.nan,balanced_accuracy=balanced_accuracy_score(h.label,h.score.ge(.5)) if valid else np.nan,mcc=matthews_corrcoef(h.label,h.score.ge(.5)) if valid else np.nan,undefined_reason='' if valid else 'One class remains after removal of all 100 positive records',source='Authenticated archived Phase11A; no new training'))
        context=read('phase21_outputs/verified_pooled_classification.csv').query('stage==@stage and model=="spaced6_4_logreg"').iloc[0]
        rows.append(dict(stage=stage,model='spaced6_4_logreg',condition='C_grouped_OOF_context_DIFFERENT_PROTOCOL',n=int(context.n),positive=int(context.positive),negative=int(context.negative),auroc=context.roc_auc,auprc=context.auprc,balanced_accuracy=context.balanced_acc_at_0_5,mcc=context.mcc_at_0_5,undefined_reason='',source='Archived Phase2 grouped OOF; not a paired population'))
    frame=pd.DataFrame(rows);table('phase40_outputs/archived_baseline_descriptive_rescore.csv',frame)
    changes=[]
    for stage,g in frame.groupby('stage'):
        a=g[g.condition.str.startswith('A_')].iloc[0];b=g[g.condition.str.startswith('B_')].iloc[0]
        for metric in ['auroc','auprc','balanced_accuracy','mcc']:
            changes.append(dict(stage=stage,model='spaced6_4_logreg',metric=metric,historical_minus_clean=a[metric]-b[metric],interpretation='changed test composition; not causal leakage effect; undefined for Stage2'))
    table('phase40_outputs/archived_baseline_subset_differences.csv',pd.DataFrame(changes))
    text('phase40_outputs/STOP_RULE_AND_LIMITS.md','''# Historical leakage experiment: authentication stop rule applied

The requested new three-model historical experiment was not run. The original historical independent-test training implementation and checkpoints for fine-tuned DNABERT and DNABERT-2 cannot be authenticated from the supplied artifacts. Their grouped-OOF predictions and newer Phase31 checkpoints do not authenticate the original supplied-training protocol. Reusing a new implementation would be a new study, not an exact historical reconstruction.

The existing Phase11A CPU spaced-mer analysis has an executable notebook, input hashes and prediction provenance. Its predictions are rescored here as an explicitly separate archived descriptive audit, without training. A and B use the same historical_raw predictions, and B removes exact-training test matches. C is the archived grouped-OOF estimate with different training and evaluation populations. Changes cannot be interpreted as a causal effect of leakage. No three-model leakage-impact figure or claim of completed historical neural replication is produced.

Stage1 loses 100 positive records (400 to 300; positive:negative 200:200 to 100:200). Stage2 loses all 100 positive records (200 to 100; 100:100 to 0:100). AUROC, AUPRC, balanced accuracy and MCC are deliberately reported as undefined for the one-class cleaned Stage2 test, rather than substituting zero or a misleading accuracy.

The duplicated-record finding remains valid. The historical tests were previously examined. This stop does not block final statistics, figures, or manuscript construction. No replacement records, new hyperparameters, new fits, or post hoc outcome-based exclusions were introduced.''')
    dump('phase40_outputs/summary.json',dict(status='completed_feasibility_audit_experiment_stopped',new_three_model_experiment_run=False,new_training=False,stop_rule_applied=True,archived_spaced_mer_rescore_rows=len(frame),original_implementation_verified=False,stage2_cleaned_discrimination='undefined: no positives',source_sha256=sha(__file__)))
    print('Phase40: stop rule applied; archived baseline rescore retained separately. No neural historical experiment claimed.')

def classification_metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, matthews_corrcoef
    y=np.asarray(y,int);p=np.asarray(p,float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(n=len(y),negative=int((y==0).sum()),positive=int(y.sum()),auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),threshold=.5,sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),mcc=float(matthews_corrcoef(y,p>=.5)),balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def regression_metrics(y,p):
    from scipy.stats import pearsonr,spearmanr
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y
    assert np.isfinite(y).all() and np.isfinite(p).all()
    return dict(n=len(y),pearson=float(pearsonr(y,p).statistic) if np.std(p)>0 and np.std(y)>0 else np.nan,spearman=float(spearmanr(y,p).statistic) if np.std(p)>0 and np.std(y)>0 else np.nan,rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),r2=float(1-np.dot(e,e)/np.sum((y-y.mean())**2)),mean_signed_error=float(e.mean()),prediction_sd=float(np.std(p)))

def current_classification():
    repeated=read('phase32_outputs/three_seed_ensemble_oof_predictions.csv.gz')
    old=read('phase21_outputs/authenticated_classification_predictions.csv.gz')
    baseline=old[~old.model.isin(repeated.model.unique())]
    columns=['record_id','stage','model','similarity_group_id','fold_5','label','score']
    result=pd.concat([repeated[columns].assign(fit_scope='current_three_seed_ensemble'),baseline[columns].assign(fit_scope='archived_single_run')],ignore_index=True)
    assert not result.duplicated(['stage','model','record_id']).any()
    assert result.groupby(['stage','record_id']).label.nunique().max()==1
    assert result.groupby(['stage','record_id']).similarity_group_id.nunique().max()==1
    return result

def weighted_corr(x,y,w):
    total=w.sum()
    if total<=1:return np.nan
    xx=x-np.dot(w,x)/total; yy=y-np.dot(w,y)/total
    den=np.sqrt(np.dot(w,xx*xx)*np.dot(w,yy*yy))
    return float(np.dot(w,xx*yy)/den) if den>0 else np.nan

def weighted_ranks(x,w):
    # Literal mean ranks of duplicated observations, including original ties.
    order=np.argsort(x,kind='stable');v=x[order];mass=w[order]
    starts=np.r_[0, np.flatnonzero(v[1:]!=v[:-1])+1]
    counts=np.add.reduceat(mass,starts)
    ranks=np.cumsum(counts)-counts+(counts+1)/2
    out=np.empty(len(x),float)
    out[order]=np.repeat(ranks,np.diff(np.r_[starts,len(x)]))
    return out

def weighted_auc(y,p,w):
    order=np.argsort(p,kind='stable');v=p[order];yy=y[order];mass=w[order]
    starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
    positive=np.add.reduceat(mass*yy,starts);negative=np.add.reduceat(mass*(1-yy),starts)
    den=positive.sum()*negative.sum()
    return float(np.dot(positive,np.cumsum(negative)-negative+.5*negative)/den) if den else np.nan

def weighted_metric(y,p,w,metric):
    if metric=='auroc':return weighted_auc(y,p,w)
    if metric=='pearson':return weighted_corr(y,p,w)
    if metric=='spearman':return weighted_corr(weighted_ranks(y,w),weighted_ranks(p,w),w)
    raise ValueError(metric)

def stats_selfcheck():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr,pearsonr
    rng=np.random.default_rng(3091); checks=[]
    for i in range(25):
        y=rng.integers(0,2,12);p=np.round(rng.random(12),1);w=rng.integers(0,5,12)
        ix=np.repeat(np.arange(12),w);yy=y[ix];pp=p[ix]
        if len(np.unique(yy))<2:continue
        for metric,expected in [('auroc',roc_auc_score(yy,pp)),('pearson',pearsonr(yy,pp).statistic),('spearman',spearmanr(yy,pp).statistic)]:
            got=weighted_metric(y,p,w,metric);error=abs(got-expected)
            assert error<1e-12,(metric,error)
            checks.append(dict(probe=i,metric=metric,absolute_error=error,meaning='weighted evaluation versus literal duplicated rows with ties and zero weights'))
    return pd.DataFrame(checks)

def synthesis_comparisons():
    rows=[]
    for stage in (1,2):
        for b in ['finetuned_dnabert_legacy','spaced6_4_logreg']:
            rows.append(dict(comparison_id=f'C_S{stage}_DB2_minus_{b}',task='classification',stage=stage,cell='',cohort='grouped_OOF',arm_a='finetuned_dnabert2',arm_b=b,metric='auroc',resampling_unit='locked_similarity_graph_component'))
    for cell in ['HepG2','K562']:
        for cohort in ['internal','DHS_natural']:
            for a,b in [('legnet_selected','dnabert2_selected'),('dnabert2_selected','mprann_selected')]:
                rows.append(dict(comparison_id=f'M_{cell}_{cohort}_{a}_minus_{b}',task='MPRA',stage=None,cell=cell,cohort=cohort,arm_a=a,arm_b=b,metric='pearson' if cohort=='internal' else 'spearman',resampling_unit='locus_group' if cohort=='internal' else 'canonical_DNA'))
        rows.append(dict(comparison_id=f'D_{cell}_internal_full_minus_own_early2',task='MPRA',stage=None,cell=cell,cohort='internal',arm_a='dnabert2_fixedlr_full',arm_b='dnabert2_fixedlr_early2',metric='pearson',resampling_unit='locus_group'))
    assert len(rows)==14
    return rows

def paired_arrays(row,classification,mpra):
    if row['task']=='classification':
        d=classification[classification.stage.eq(int(row['stage']))]
        score='score';arm='model';truth='label';group='similarity_group_id'
    else:
        d=mpra[mpra.cell.eq(row['cell'])&mpra.cohort.eq(row['cohort'])]
        score='prediction';arm='arm';truth='activity';group='group_id'
    a=d[d[arm].eq(row['arm_a'])].set_index('record_id').sort_index()
    b=d[d[arm].eq(row['arm_b'])].set_index('record_id').sort_index()
    assert a.index.equals(b.index) and np.array_equal(a[truth],b[truth]) and np.array_equal(a[group].astype(str),b[group].astype(str))
    groups,gi=np.unique(a[group].astype(str),return_inverse=True)
    return a[truth].to_numpy(),a[score].to_numpy(),b[score].to_numpy(),gi,len(groups)

def phase41():
    verify_inputs()
    rows=synthesis_comparisons()
    methods=dict(version='final_synthesis_stats_v1',family_size=14,comparisons=rows,selection_timing='prespecified for final synthesis after prior outcomes were seen; not prospective confirmation',primary_metrics={'classification':'pooled ensemble AUROC','internal_MPRA':'Pearson','DHS_natural':'Spearman'},bootstrap_draws=2000,permutation_draws=9999,random_seed=20261015,confidence_interval='percentile 95%, whole groups resampled with replacement; Spearman ranks recomputed on duplicated observations',test='two-sided whole-group paired prediction-swap permutation; sharp conditional exchangeability null, not a test of algorithm equivalence',p_value='(1 + count(abs(permuted difference)>=abs(observed difference))) / (R+1)',multiple_testing='Holm across all 14 headline contrasts, including both internal fixed-LR duration controls',alpha=.05,threshold=.5,undefined_policy='undefined draws visible and counted; no fit or cohort exclusions; no revised selection',scope='fixed fitted models and previously examined cohorts; no selection, hyperparameter-search, training-algorithm, pretraining-membership or fresh-test uncertainty',dependence='OOF folds share training data; 15 seed x fold points are not independent replicates; graph grouping does not rule out all biological relatedness',baselines='Seven archived single-run baselines kept; no fabricated seed SD. Spaced-mer MPRA baseline outside neural 54-fit grid.',secondary_policy='Other old interval/diagnostic results are separately labelled exploratory and unadjusted; no conversion of an old CI into a p-value',source_sha256=sha(__file__))
    planpath=ROOT/'final_statistics/statistical_methods.json'
    if planpath.exists():
        old=jd('final_statistics/statistical_methods.json')
        assert {k:v for k,v in old.items() if k!='source_sha256'}=={k:v for k,v in methods.items() if k!='source_sha256'},'Do not alter frozen statistical policy after testing.'
    else:
        dump('final_statistics/statistical_methods.json',methods)
        table('final_statistics/headline_comparisons.csv',pd.DataFrame(rows))
        dump('final_statistics/plan_freeze.json',dict(frozen_utc=NOW(),methods_sha256=sha(planpath),source_sha256=sha(__file__),prior_results_previously_seen=True,new_tests_started=False))
        source_dir=ROOT/'final_statistics/source_at_plan';source_dir.mkdir(exist_ok=True)
        shutil.copy2(__file__,source_dir/'final_synthesis.py')
    table('final_statistics/weighted_metric_validation.csv',stats_selfcheck())
    c=current_classification();m=read('phase37_outputs/verified_ensemble_predictions.csv.gz')
    table('final_statistics/classification_predictions.csv.gz',c)
    class_rows=[]
    seedmetrics=read('phase32_outputs/seed_metrics_pooled_and_fold.csv')
    folds=read('phase21_outputs/verified_fold_classification.csv')
    variation=read('phase32_outputs/seed_vs_fold_variability.csv')
    for (stage,model),g in c.groupby(['stage','model']):
        row=dict(stage=stage,model=model,fit_scope=g.fit_scope.iloc[0],**classification_metrics(g.label,g.score))
        v=variation[variation.stage.eq(stage)&variation.model.eq(model)]
        if len(v):
            v=v.iloc[0];s=seedmetrics[seedmetrics.stage.eq(stage)&seedmetrics.model.eq(model)]
            row.update(mean_fold_auroc=v.mean_fold_auroc_across_seeds,fold_sd_of_seed_mean_auroc=v.fold_sd_of_seed_mean_auroc,seed_sd_of_mean_fold_auroc=v.seed_sd_of_mean_fold_auroc,mean_seed_pooled_auroc=s.auroc.mean(),seed_sd_pooled_auroc=s.auroc.std(ddof=1),n_seeds=3)
        else:
            f=folds[folds.stage.eq(stage)&folds.model.eq(model)]
            assert len(f)==5
            row.update(mean_fold_auroc=f.roc_auc.mean(),fold_sd_of_seed_mean_auroc=f.roc_auc.std(ddof=1),seed_sd_of_mean_fold_auroc=np.nan,mean_seed_pooled_auroc=np.nan,seed_sd_pooled_auroc=np.nan,n_seeds=1)
        class_rows.append(row)
    summary=pd.DataFrame(class_rows)
    expected=read('phase32_outputs/updated_classification_comparison.csv').set_index(['stage','model'])
    for row in class_rows:
        old=expected.loc[(row['stage'],row['model'])]
        for key in ['auroc','auprc','brier','mcc','balanced_accuracy','sensitivity','specificity']:
            assert abs(row[key]-old[key])<1e-12,(row['model'],key)
    table('final_statistics/classification_summary.csv',summary)
    mr=[]
    for (cell,cohort,arm),g in m.groupby(['cell','cohort','arm']):
        mr.append(dict(cell=cell,cohort=cohort,arm=arm,fit_scope='current_three_seed_ensemble',**regression_metrics(g.activity,g.prediction)))
    ms=pd.DataFrame(mr);oldm=read('phase37_outputs/ensemble_metrics.csv').set_index(['cell','cohort','arm'])
    for r in mr:
        old=oldm.loc[(r['cell'],r['cohort'],r['arm'])]
        for key in ['pearson','spearman','rmse','mae','r2']:
            assert abs(r[key]-old[key])<1e-12,(r['arm'],key)
    # The spaced-mer baseline remains a separate original frozen recipe.
    import analysis_tools.readiness_pipeline as original
    archived=original.all_internal()
    spaced=archived[archived.protocol.eq('9b')&archived.model.eq('spaced6_4_ridge')]
    if len(spaced):
        # The raw archive schema is checked rather than guessed.
        table('final_statistics/archived_spaced_mer_input_columns.csv',pd.DataFrame({'column':spaced.columns}))
    table('final_statistics/mpra_internal_summary.csv',ms[ms.cohort.eq('internal')])
    table('final_statistics/external_transfer_summary.csv',ms[ms.cohort.eq('DHS_natural')])
    variation['uncertainty_type']='classification_seed_and_fold_SD_separate'
    mv=read('phase37_outputs/seed_variability.csv');mv['uncertainty_type']='MPRA_three_seed_sample_SD'
    table('final_statistics/variability_summary.csv',pd.concat([variation,mv],ignore_index=True))
    results=[]
    for number,row in enumerate(rows):
        result_path=ROOT/f'final_statistics/contrast_cache/{row["comparison_id"]}.json'
        if result_path.exists():
            cached=json.loads(result_path.read_text(encoding='utf-8'))
            assert cached['methods_sha256']==sha(planpath)
            results.append(cached['result']);continue
        y,a,b,gi,ng=paired_arrays(row,c,m);n=len(y);w0=np.ones(n)
        obs=weighted_metric(y,a,w0,row['metric'])-weighted_metric(y,b,w0,row['metric'])
        seed=int.from_bytes(hashlib.sha256(row['comparison_id'].encode()).digest()[:4],'little') ^ methods['random_seed']
        rng=np.random.default_rng(seed);boot=[]
        for i in range(methods['bootstrap_draws']):
            weights=np.bincount(rng.integers(0,ng,ng),minlength=ng)[gi].astype(float)
            boot.append(weighted_metric(y,a,weights,row['metric'])-weighted_metric(y,b,weights,row['metric']))
        boot=np.array(boot);defined=boot[np.isfinite(boot)]
        assert len(defined)>0
        lo,hi=np.quantile(defined,[.025,.975]);extreme=0;undefined=0
        for i in range(methods['permutation_draws']):
            flip=rng.integers(0,2,ng,dtype=np.int8)[gi].astype(bool)
            pa=np.where(flip,b,a);pb=np.where(flip,a,b)
            value=weighted_metric(y,pa,w0,row['metric'])-weighted_metric(y,pb,w0,row['metric'])
            if not np.isfinite(value):undefined+=1
            elif abs(value)>=abs(obs)-1e-14:extreme+=1
        assert undefined==0,'Do not silently change permutation denominator'
        raw=(extreme+1)/(methods['permutation_draws']+1)
        result=dict(**row,estimate_a_minus_b=float(obs),lower_95=float(lo),upper_95=float(hi),p_raw=raw,p_monte_carlo_se=float(np.sqrt(raw*(1-raw)/(methods['permutation_draws']+1))),extreme_permutations=extreme,permutation_draws=methods['permutation_draws'],undefined_permutations=undefined,bootstrap_draws=len(boot),defined_bootstrap_draws=len(defined),n_records=n,n_groups=ng,scope=methods['scope'],conditional_null=methods['test'])
        dump(result_path,dict(methods_sha256=sha(planpath),result=result))
        results.append(result)
        dump('phase41_outputs/live_progress.json',dict(completed_contrasts=len(results),total_contrasts=len(rows),last=row['comparison_id'],updated_utc=NOW()))
        print('Contrast',len(results),'/',len(rows),row['comparison_id'],'difference',round(obs,6),'raw p',raw,flush=True)
    frame=pd.DataFrame(results);order=np.argsort(frame.p_raw.to_numpy(),kind='stable');ranked=frame.p_raw.to_numpy()[order]
    adjusted=np.minimum(1,np.maximum.accumulate(ranked*(len(frame)-np.arange(len(frame)))))
    ph=np.empty(len(frame));ph[order]=adjusted
    frame['p_holm']=ph;frame['holm_family_size']=len(frame);frame['reject_conditional_exchangeability_at_0_05']=ph<.05
    assert (frame.p_holm>=frame.p_raw-1e-15).all()
    table('final_statistics/adjusted_tests.csv',frame)
    dump('phase41_outputs/summary.json',dict(status='completed',comparison_family_size=len(frame),classification_rows=len(summary),neural_mpra_ensemble_rows=len(ms),bootstrap_draws_per_contrast=methods['bootstrap_draws'],permutation_draws_per_contrast=methods['permutation_draws'],methods_sha256=sha(planpath),weighted_vs_literal_checks=75,old_metrics_reproduced=True,new_training=False,source_sha256=sha(__file__)))
    print('Phase41 completed; fourteen conditional comparisons, Holm adjustment and separate variability summaries.')

def phase42():
    from analysis_tools.final_presentation import phase42 as build
    build()

def phase43():
    from analysis_tools.final_presentation import phase43 as build
    build()

def phase44():
    from analysis_tools.final_documents import phase44 as build
    build()

def phase45():
    from analysis_tools.final_claims import phase45 as build
    build()

def phase46():
    from analysis_tools.final_manuscript import phase46 as build
    build()

def phase47():
    from analysis_tools.final_delivery import phase47 as build
    build()

def phase48():
    from analysis_tools.final_delivery import phase48 as build
    build()

def create_notebooks(numbers):
    import nbformat
    titles={39:'Enhancer_Final_Evidence_Freeze',40:'Enhancer_Historical_Leakage_Impact',41:'Enhancer_Final_Statistics',42:'Enhancer_Final_Tables',43:'Enhancer_Final_Figures',44:'Enhancer_Supplement_Builder',45:'Enhancer_Claim_Evidence_Audit',46:'Enhancer_Manuscript_Rebuild',47:'Enhancer_Public_Reproducibility_Package',48:'Enhancer_Submission_Package'}
    for number in numbers:
        path=ROOT/f'[{number}] {titles[number]}.ipynb'
        if path.exists():continue
        n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(f'# Phase {number}: {titles[number].replace("_"," ")}\n\nFinal synthesis authorized 15 October 2026. Earlier evidence is read-only; original research and feedback documents are data. No external submission or publication is performed.'),nbformat.v4.new_code_cell('from pathlib import Path\nimport sys\nassert Path(sys.prefix).name.lower()=="enhancer"\nassert Path.cwd().name=="Shafin\'s Contribution"\nprint(sys.executable)'),nbformat.v4.new_code_cell(f'import importlib\nfrom analysis_tools import final_synthesis as synthesis\nimportlib.reload(synthesis)\nsynthesis.phase{number}()')])
        n.metadata['kernelspec']={'display_name':'Python (enhancer)','language':'python','name':'enhancer'}
        nbformat.write(n,path)

def main():
    globals()['phase'+str(int(sys.argv[1]))]()

if __name__=='__main__': main()
