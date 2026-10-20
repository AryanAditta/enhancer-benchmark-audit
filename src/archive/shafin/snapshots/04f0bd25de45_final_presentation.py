"""Tables and scientific figures derived from the locked final-synthesis inputs."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
from analysis_tools import final_synthesis as s
ROOT=s.ROOT

LABELS={'gc_logreg':'GC/CpG logistic regression','k3_logreg':'3-mer logistic regression','spaced6_4_logreg':'Spaced 6/4-mer logistic regression','small_cnn':'Small CNN','frozen_dnabert_legacy':'Frozen DNABERT','frozen_dnabert2':'Frozen DNABERT-2','frozen_gena_lm':'Frozen GENA-LM','finetuned_dnabert_legacy':'Fine-tuned DNABERT','finetuned_dnabert2':'Fine-tuned DNABERT-2','dnabert2_selected':'DNABERT-2','legnet_selected':'LegNet','mprann_selected':'MPRAnn','spaced6_4_ridge':'Spaced-mer ridge','dnabert2_fixedlr_full':'DNABERT-2 longer','dnabert2_fixedlr_early2':'DNABERT-2 own two-epoch prefix'}
ORDER=['gc_logreg','k3_logreg','spaced6_4_logreg','small_cnn','frozen_dnabert_legacy','frozen_dnabert2','frozen_gena_lm','finetuned_dnabert_legacy','finetuned_dnabert2']

def classify_gc(predictions):
    manifests=pd.concat(s.manifests(),ignore_index=True)
    seq=manifests[['stage','record_id','sequence']].copy()
    seq['gc_count']=seq.sequence.map(lambda x:x.count('G')+x.count('C'))
    p=predictions.merge(seq[['stage','record_id','gc_count']],on=['stage','record_id'],validate='many_to_one')
    rows=[]
    for (stage,model,fold),g in p.groupby(['stage','model','fold_5']):
        numerator=0.;denominator=0;represented=0
        for _,b in g.groupby('gc_count'):
            positive=b[b.label.eq(1)].score.to_numpy();negative=b[b.label.eq(0)].score.to_numpy()
            if len(positive)==0 or len(negative)==0:continue
            d=positive[:,None]-negative[None,:]
            numerator+=np.sum(d>0)+.5*np.sum(d==0)
            denominator+=len(positive)*len(negative);represented+=len(b)
        assert denominator>0
        from sklearn.metrics import roc_auc_score
        rows.append(dict(stage=stage,model=model,fold_5=fold,raw_fold_auroc=roc_auc_score(g.label,g.score),same_gc_fold_auroc=numerator/denominator,matched_positive_negative_pairs=denominator,represented_records=represented,validation_records=len(g),fit_scope=g.fit_scope.iloc[0]))
    frame=pd.DataFrame(rows)
    assert len(frame)==90
    old=s.read('final_evidence/inputs/phase3_fold_gc_matched.csv')
    baseline=frame[frame.fit_scope.eq('archived_single_run')].merge(old,on=['stage','model','fold_5'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(baseline)==70
    assert abs(baseline.same_gc_fold_auroc_new-baseline.same_gc_fold_auroc_old).max()<1e-10
    return frame

def table_md(df):
    def value(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):
            if x==int(x):return f'{int(x):,}'
            return f'{x:.4f}' if abs(x)>=.0001 else f'{x:.2g}'
        return str(x).replace('|',' / ').replace('\n',' ')
    header='| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
    return header+'\n'.join('| '+' | '.join(value(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def phase42():
    s.verify_inputs();assert s.jd('phase41_outputs/summary.json')['status']=='completed'
    out=ROOT/'final_tables';out.mkdir(exist_ok=True)
    # Seal derived statistical outputs as a distinct, versioned input generation.
    stats=[p for p in (ROOT/'final_statistics').glob('*') if p.is_file() and p.name!='exact_gc_fold_metrics.csv']
    lock=[dict(path=p.relative_to(ROOT).as_posix(),sha256=s.sha(p),bytes=p.stat().st_size) for p in sorted(stats)]
    lp=ROOT/'final_tables/statistical_input_lock.json'
    if lp.exists():assert s.jd('final_tables/statistical_input_lock.json')==lock
    else:s.dump(lp,lock)
    index=[]
    def emit(name,title,data,source,notes,placement='supplement'):
        path=f'final_tables/{placement}/{name}.csv';s.table(path,data)
        index.append(dict(table=name,title=title,placement=placement,rows=len(data),columns=len(data.columns),file=path,sha256=s.sha(ROOT/path),source=source,notes=notes))
        if placement=='main':s.text(f'final_tables/main/{name}.md',f'# {title}\n\n'+table_md(data)+'\n'+notes)
    counts=s.read('final_evidence/final_dataset_counts.csv');audit=s.read('final_evidence/benchmark_audit.csv')
    rows=[]
    for r in audit.itertuples():
        c=counts[counts.task.eq(f'classification_stage{r.stage}')].set_index('population')
        rows.append(dict(task=f'Legacy Stage {r.stage}',task_definition='enhancer vs non-enhancer' if r.stage==1 else 'strong vs weak enhancer',initial_training=r.supplied_train_n,historical_test=r.supplied_test_n,complete_assay_candidates=np.nan,retained=r.candidate_n,removed_or_excluded=r.supplied_train_n-r.candidate_n,groups=r.candidate_groups,folds=5,positive=int(c.loc['grouped_candidates','positive']),negative=int(c.loc['grouped_candidates','negative']),train=np.nan,validation=np.nan,internal_test=np.nan,external_DHS_natural=np.nan,evaluation='fixed similarity-grouped OOF; historical test secondary'))
    attr=s.read('phase24_outputs/dataset_attrition_counts.csv')
    partition=s.read('phase34_outputs/partition_counts.csv')
    for cell in ['HepG2','K562']:
        a=attr[attr.cell_line.eq(cell)].set_index('stage').n
        retained=int(partition[partition.cell_line.eq(cell)].n.sum());complete=int(a['complete_paired_candidates'])
        p=partition[partition.cell_line.eq(cell)].set_index('split_9b').n
        en=int(counts[counts.task.eq(f'MPRA_{cell}')&counts.population.eq('external_DHS_natural')].n.iloc[0])
        rows.append(dict(task=cell,task_definition='continuous 200bp reporter activity',initial_training=np.nan,historical_test=np.nan,complete_assay_candidates=complete,retained=retained,removed_or_excluded=complete-retained,groups=np.nan,folds=np.nan,positive=np.nan,negative=np.nan,train=int(p['train']),validation=int(p['validation']),internal_test=int(p['test']),external_DHS_natural=en,evaluation='primary9B chromosome holdout; frozen previously examined DHS-natural transfer'))
    emit('T1_Datasets','Datasets and evaluation design',pd.DataFrame(rows),'Phase39 manifests/counts; Phase34 partitions; Phase24 raw-row reconstruction','Rows and unique elements are distinct. Excluded classification rows include 100 exact test duplicates plus additional test-related group exclusions. No untouched confirmatory test.','main')
    emit('T2_Benchmark_audit','Supplied benchmark overlap and corrected grouping',audit.drop(columns='group_rule'),'Phase39 independently reconstructed benchmark_audit.csv','Exact matches and reverse-complement matches are counted separately and may overlap in general. LCS>=160 is a connected-component graph rule, not 80% local identity. Mixed-label counts here are among retained candidates; seven mixed groups existed in the full Stage 1 graph. Optional three-model historical experiment stopped under authentication rule.','main')
    c=s.read('final_statistics/classification_summary.csv')
    gc=classify_gc(s.read('final_statistics/classification_predictions.csv.gz'))
    s.table('final_statistics/exact_gc_fold_metrics.csv',gc)
    gcm=gc.groupby(['stage','model']).same_gc_fold_auroc.mean().rename('exact_GC_mean_fold_AUROC').reset_index()
    c=c.merge(gcm,on=['stage','model'],validate='one_to_one')
    c['model_label']=c.model.map(LABELS)
    emit('T3_Classification','Leakage-controlled grouped classification',c[['stage','model','model_label','fit_scope','n_seeds','n','mean_fold_auroc','fold_sd_of_seed_mean_auroc','seed_sd_of_mean_fold_auroc','mean_seed_pooled_auroc','seed_sd_pooled_auroc','auroc','auprc','mcc','exact_GC_mean_fold_AUROC']],'Phase41 current classification; Phase42 fixed Phase3 exact-GC pair rule applied to same current OOF predictions','Mean-fold AUROC, pooled AUROC, fold SD and seed SD are different summaries. Current fine-tuned rows use three-seed ensembles; seven baselines are archived single runs. GC matching is evaluation-only and does not control CpG or all confounding.','main')
    internal=s.read('final_statistics/mpra_internal_summary.csv')
    neural=internal[internal.arm.isin(['dnabert2_selected','legnet_selected','mprann_selected'])].copy()
    import analysis_tools.readiness_pipeline as original
    archive=original.csv('9B_enhancer_phase9b_regression_results.zip','phase9b_predictions_validation_test.csv')
    baseline=archive[archive.model.eq('spaced6_4_ridge')&archive.split.eq('test')]
    assert len(baseline)>0
    b=[]
    for cell,g in baseline.groupby('cell_line'):
        assert len(g)==int(neural[neural.cell.eq(cell)].n.iloc[0]) and g.record_id.is_unique
        b.append(dict(cell=cell,cohort='internal',arm='spaced6_4_ridge',fit_scope='archived_single_recipe_outside_54_fit_grid',**s.regression_metrics(g.target,g.prediction)))
    t4=pd.concat([neural,pd.DataFrame(b)],ignore_index=True).sort_values(['cell','arm'])
    t4['model_label']=t4.arm.map(LABELS)
    emit('T4_MPRA_internal','Primary chromosome-held-out MPRA activity prediction',t4[['cell','arm','model_label','fit_scope','n','pearson','spearman','rmse','mae','r2']],'Phase41 neural ensembles; authenticated original9B spaced-mer predictions','Neural recipes selected using three-seed mean validation Pearson only, before inference. DNABERT-2 uses longer validation-selected trajectories. Spaced-mer is one archived recipe, not an equal-search neural competitor.','main')
    external=s.read('final_statistics/external_transfer_summary.csv');ext=external[external.arm.isin(['dnabert2_selected','legnet_selected','mprann_selected'])].copy();ext['model_label']=ext.arm.map(LABELS)
    emit('T5_External_transfer','DHS-natural external transfer: ranking and calibration',ext[['cell','arm','model_label','n','spearman','pearson','mae','rmse','r2','mean_signed_error']],'Phase41 recalculated current ensembles','Spearman is primary. Negative R2 and signed errors are retained. Frozen previously examined cohort; no external score selected recipes.','main')
    tests=s.read('final_statistics/adjusted_tests.csv').copy()
    tests['contrast_label']=tests.arm_a.map(LABELS)+' minus '+tests.arm_b.map(LABELS)
    tests['interpretation']=tests.apply(lambda r:'Conditional interval includes zero; not equivalence' if r.lower_95<=0<=r.upper_95 else ('Positive difference under tested recipes' if r.lower_95>0 else 'Negative difference under tested recipes'),axis=1)
    emit('T6_Headline_contrasts','Paired headline contrasts and final-synthesis Holm adjustment',tests[['comparison_id','task','stage','cell','cohort','contrast_label','metric','estimate_a_minus_b','lower_95','upper_95','p_raw','p_holm','n_records','n_groups','resampling_unit','interpretation']],'Phase41 final 14-contrast family','Percentile group-bootstrap intervals are conditional and pointwise. Holm-adjusted Monte Carlo p-values test whole-group prediction-swap exchangeability; they do not quantify algorithm/search uncertainty. Family fixed for final synthesis after earlier results were seen.','main')
    emit('S1_Provenance','Source provenance and hashes',pd.DataFrame(s.jd('final_evidence/phase39_provenance.json')['input_files']),'Phase39 immutable input manifest','Original sources preserved; source generations explicit.')
    f=s.read('phase32_outputs/fold_metrics_independent.csv').assign(aggregation='current_seed_fold',model_generation='Phase31')
    old=s.read('phase21_outputs/verified_fold_classification.csv');old=old[~old.model.isin(f.model.unique())].assign(aggregation='archived_single_run_fold',model_generation='Phase2')
    seeds=s.read('phase32_outputs/seed_metrics_pooled_and_fold.csv').assign(aggregation='current_seed_pooled_and_meanfold',model_generation='Phase31')
    emit('S2_Classification_fold_seed','Classification fold and seed metrics',pd.concat([f,old,seeds],ignore_index=True),'Phase21/32 fixed predictions','Overlapping training folds are dependent; fifteen seed-fold cells are not independent replicates.')
    old=s.read('phase21_outputs/mixed_label_exclusion_sensitivity.csv');old=old[~old.model.isin(f.model.unique())].assign(model_generation='archived_single_run')
    current=s.read('phase32_outputs/mixed_label_sensitivity.csv').assign(model_generation='current_three_seed_ensemble')
    emit('S3_Mixed_label_sensitivity','Mixed-label evaluation sensitivity',pd.concat([old,current],ignore_index=True),'Phase21/32 predefined six-record Stage 1 sensitivity','Evaluation only; no retraining; primary retains these records.')
    emit('S4_Exact_GC','Exact-GC within-fold opposite-label pair evaluation',gc,'Locked Phase1 manifests; Phase41 OOF; original Phase3 matching rule','Current fine-tuned ensembles are recomputed; original archived seven rows independently match Phase3.')
    emit('S5_Classification_calibration','Classification calibration and fixed-threshold confusion counts',c[['stage','model','fit_scope','n','positive','negative','threshold','tn','fp','fn','tp','auroc','auprc','brier','sensitivity','specificity','mcc','balanced_accuracy']],'Phase41 metrics independently reproduced against Phase32','Fixed0.5 threshold; no OOF-optimized cutoff.')
    emit('S6_MPRA_seed_metrics','All current MPRA seed metrics',s.read('phase37_outputs/seed_metrics_independent.csv'),'Phase37 independent checkpoint inference','Selected arms plus fixed-lr full and own-prefix controls; all seeds remain visible.')
    jobs=s.read('phase35_outputs/training_job_summary.csv');candidates=s.read('phase36_outputs/validation_candidate_scores.csv').rename(columns={'cell':'cell_line'})
    grid=jobs.merge(candidates[['cell_line','model','grid_index','mean_validation_pearson','seed_sd_validation_pearson','eligible']],on=['cell_line','model','grid_index'],validate='many_to_one')
    selected=set(s.read('phase38_outputs/selected_recipe_adequacy_flags.csv').job);grid['selected_recipe']=grid.job.isin(selected)
    emit('S7_MPRA_grid_epochs','All candidate fits, validation selection and epochs',grid,'Phase35 jobs; Phase36 validation-only selection','Undefined-seed candidate exclusion is frozen; practical collapse screen does not change finite-Pearson eligibility.')
    duration=s.read('phase37_outputs/conditional_paired_intervals.csv');duration=duration[duration.contrast.eq('duration_fixed_lr')]
    emit('S8_Duration_control','DNABERT-2 own-prefix duration contrasts',duration,'Phase37 fixedlr_full minus exact own early2','Original unadjusted secondary intervals, including external errors. Internal primary contrasts use the new Phase41 family in Table6.')
    origin=s.read('phase24_outputs/external_origin_counts.csv').assign(row_kind='origin_count',model_generation='dataset_construction')
    oldext=s.read('phase23_outputs/external_metrics_authoritative.csv').assign(row_kind='archived_origin_metrics',model_generation='historical_Phase9B_9C_12_not_current_neural')
    emit('S9_External_origins','External origin inventory and explicitly archived origin metrics',pd.concat([origin,oldext],ignore_index=True),'Phase24 origin counts; Phase23 older all-origin diagnostics','Current neural training-adequacy study evaluated DHS-natural only. Other origins are historical context, not current selected-model rankings.')
    emit('S10_Filter_attrition','Raw-row and assay-element attrition',attr,'Phase14 raw reconstruction, Phase24 audit','Row exclusions and element exclusions are not interchangeable.')
    emit('S11_Filter_performance','Retained and removed frozen-model performance',s.read('phase23_outputs/filtering_metrics.csv'),'Phase22 inference; Phase23 independent metric recomputation','Older fitted models; not new Phase35 selected models and not causal retraining effects.')
    subgroup=pd.concat([s.read('phase24_outputs/retained_test_subgroup_metrics_exact.csv').assign(diagnostic_scope='retained_test'),s.read('phase24_outputs/complete_population_subgroup_metrics_exact.csv').assign(diagnostic_scope='complete_population')],ignore_index=True)
    emit('S12_Subgroups','Training-threshold subgroup diagnostics',subgroup,'Phase24 exact thresholds and frozen older models','Suppressed small groups remain visible; historical model versions explicitly retained.')
    cov=pd.concat([s.read('phase17_outputs/interval_audit_coverage.csv').assign(table_component='overall'),s.read('phase17_outputs/interval_subgroup_coverage.csv').assign(table_component='subgroup')],ignore_index=True)
    emit('S13_Interval_coverage','Prediction interval audit coverage',cov,'Phase17 disjoint marginal calibration/audit protocol','Marginal coverage does not establish conditional or individual reliability; older frozen fits.')
    emit('S14_High_activity','Measured high-activity tail failure diagnostics',s.read('phase18_outputs/observed_activity_tail_diagnostics.csv'),'Phase18 frozen measured-activity diagnostic specification','Diagnostic using measured target; not a prospective target-free selection rule. Older model versions.')
    checkpoints=pd.concat([s.read('phase32_outputs/checkpoint_reproduction_checks.csv').assign(study='current_classification'),s.read('phase37_outputs/checkpoint_reproduction_checks.csv').assign(study='current_MPRA')],ignore_index=True)
    emit('S15_Checkpoint_reproduction','Independent checkpoint and prediction reproduction',checkpoints,'Phase32/37 independent exact-inference audits','Selected scaling, IDs, models and predictions authenticated against source locks.')
    collapse=jobs.copy();collapse['frozen_undefined_Pearson_flag']=collapse.collapsed
    collapse['posthoc_practical_nearconstant_SD_lt_1e_minus4']=collapse.validation_prediction_sd<1e-4
    assert collapse['posthoc_practical_nearconstant_SD_lt_1e_minus4'].sum()==2 and not collapse.collapsed.any()
    collapse['study']='current54_grid';oldcollapse=s.read('phase23_outputs/collapsed_seed_sensitivity.csv').assign(study='historical9C_sensitivity_separate')
    emit('S16_Collapse_diagnostics','Frozen collapse flags and practical near-constant diagnostics',pd.concat([collapse,oldcollapse],ignore_index=True),'Phase35 validation metadata plus Phase38 erratum; historical9C sensitivity separate','Two finite-Pearson practical collapses; zero frozen undefined-Pearson flags. Post hoc screen changes neither eligibility nor selection. Historical9C cause unresolved.')
    cost=jobs.groupby(['cell_line','model']).agg(fits=('job','size'),elapsed_seconds=('elapsed_seconds','sum'),optimizer_updates=('optimizer_updates','sum'),training_examples_seen=('training_examples_seen','sum'),max_peak_cuda_allocated_bytes=('peak_cuda_allocated_bytes','max')).reset_index()
    cost['elapsed_hours']=cost.elapsed_seconds/3600;cost['scope']='recorded fit runtime, not pretraining cost or standardized FLOPs'
    emit('S17_Compute','Recorded training cost across nine candidate and seed fits per model and cell',cost,'All54 Phase35 completed fit metadata','Three learning-rate recipes with three seeds each; equal candidate counts do not equal compute, pretraining, updates or exhaustive tuning.')
    emit('S18_Full_statistics','Full final headline statistical comparisons',tests,'Phase41 frozen final-synthesis family','Unadjusted secondary old results are not silently promoted into this family.')
    s.table('final_tables/table_index.csv',pd.DataFrame(index))
    s.text('final_tables/TABLE_NOTES.md','# Authoritative tables\n\n'+ '\n\n'.join('**'+r['table']+' — '+r['title']+'**\n\n'+r['notes']+'\n\nSource: '+r['source'] for r in index))
    s.dump('phase42_outputs/summary.json',dict(status='completed',main_tables=6,supplementary_tables=18,current_GC_fold_rows=len(gc),archived_GC_rows_reproduced=70,new_training=False,source_sha256=s.sha(__file__)))
    print('Phase42 completed: 6 main tables, 18 curated supplementary tables; 90 current exact-GC fold rows.')

def phase43():
    s.verify_inputs();assert s.jd('phase42_outputs/summary.json')['status']=='completed'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from PIL import Image,ImageOps,ImageDraw
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    colors={'dnabert2_selected':'#275D8C','legnet_selected':'#C65A28','mprann_selected':'#35806A','spaced6_4_ridge':'#77716A'}
    root=ROOT/'final_figures';root.mkdir(exist_ok=True);index=[]
    def save(fig,name,title,caption,sources,placement='main'):
        folder=root/placement;folder.mkdir(exist_ok=True)
        for extension in ['png','svg','pdf']:
            fig.savefig(folder/f'{name}.{extension}',dpi=200,bbox_inches='tight')
        plt.close(fig)
        index.append(dict(figure=name,title=title,placement=placement,caption=caption,source_files=sources,png=f'final_figures/{placement}/{name}.png',svg=f'final_figures/{placement}/{name}.svg',pdf=f'final_figures/{placement}/{name}.pdf'))
    datasets=s.read('final_tables/main/T1_Datasets.csv');audit=s.read('final_tables/main/T2_Benchmark_audit.csv')
    tests=s.read('final_statistics/adjusted_tests.csv');gc=s.read('final_statistics/exact_gc_fold_metrics.csv')
    # F1: six ordered scientific questions and populations.
    fig,axes=plt.subplots(2,3,figsize=(14,7));fig.subplots_adjust(wspace=.42,hspace=.70)
    first=audit.iloc[0];second=audit.iloc[1]
    nodes=[('A  Audit the supplied benchmark',f'Stage 1: {int(first.supplied_train_n):,} train / {int(first.supplied_test_n):,} test\nStage 2: {int(second.supplied_train_n):,} train / {int(second.supplied_test_n):,} test\n100 strong training-identical test records','Can the original test be treated as independent?'),('B  Fix similarity-grouped folds',f'{int(first.candidate_n):,} / {int(second.candidate_n):,} retained candidates\nFive locked folds; exact/RC plus LCS160 groups\nSeven archived baselines + two three-seed LMs','Compare fixed-model OOF performance.'),('C  Control composition at evaluation','Opposite-label pairs within each validation fold\nExactly identical 200bp GC count\nAll eligible pairs; no retraining','Does discrimination persist within GC count?'),('D  Construct quantitative activity tasks','Paired-orientation reporter candidates\nFrozen filtering and chromosome partitions\nHepG2 and K562; 200bp sequence input','Keep legacy labels separate from measured activity.'),('E  Compare training recipes','DNABERT-2, LegNet and MPRAnn\n3 learning rates × 3 seeds × 2 cells = 54 fits\nValidation-only selection; own-prefix duration control','Equal candidate counts; unequal compute.'),('F  Evaluate transfer and failure modes','Frozen DHS-natural external cohort\nRanking, errors and absolute calibration\nOlder filtering/reliability diagnostics labelled separately','Previously examined cohorts; conditional inference.')]
    import textwrap
    for ax,(title,body,foot) in zip(axes.flat,nodes):
        title=textwrap.fill(title,width=31)
        body='\n'.join(textwrap.fill(line,width=44) for line in body.splitlines())
        foot=textwrap.fill(foot,width=43)
        ax.axis('off');ax.set_title(title,loc='left',fontweight='bold',pad=18,fontsize=11)
        ax.text(0,.83,body,ha='left',va='top',transform=ax.transAxes,fontsize=9.5,linespacing=1.6)
        ax.text(0,-.10,foot,ha='left',va='bottom',transform=ax.transAxes,fontsize=9,color='#485566')
    fig.suptitle('From benchmark validity to reporter-activity generalization',fontsize=17,y=1.04)
    save(fig,'F1_Study_workflow','Study design and scientific questions','The study separates benchmark validity, grouped classification, GC composition sensitivity, quantitative reporter activity and assay transfer. All numbers derive from locked manifests or declared fit counts; no new model family or long-context experiment was added.','final_tables/main/T1_Datasets.csv; final_evidence/benchmark_audit.csv; phase34_outputs/training_protocol.json')
    # F2: actual counts, not an illustrative overlap graph.
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.4))
    roster=s.read('phase21_outputs/historical_test_overlap_roster.csv')
    for i,stage in enumerate([1,2]):
        g=roster[roster.stage.eq(stage)];over=int(g.exact_training_match.sum());other=len(g)-over
        axes[0].barh(i,over,color='#B84945',label='Training-identical strong test records' if i==0 else None)
        axes[0].barh(i,other,left=over,color='#B8C9D8',label='Other test records' if i==0 else None)
        axes[0].text(over/2,i,str(over),ha='center',va='center',color='white',fontweight='bold')
        axes[0].text(over+other/2,i,str(other),ha='center',va='center')
    axes[0].set(yticks=[0,1],yticklabels=['Stage 1','Stage 2'],xlabel='Supplied test records',title='A  Training–test identity');axes[0].legend(loc='upper left',bbox_to_anchor=(-.03,-.20),fontsize=8)
    for i,(stage,m) in enumerate(zip([1,2],s.manifests())):
        candidate=m[m.role.eq('train_candidate')];sizes=candidate.groupby('similarity_group_id').size()
        n=np.array([(sizes==1).sum(),(sizes==2).sum(),(sizes>=3).sum()])
        axes[1].bar(np.arange(3)+(i-.5)*.32,n,width=.32,label=f'Stage {stage}')
        for x,value in zip(np.arange(3)+(i-.5)*.32,n):axes[1].text(x,value+(.1 if value else .1),str(value),ha='center',va='bottom',fontsize=8)
    axes[1].set(xticks=range(3),xticklabels=['1 record','2 records','≥3 records'],ylabel='Retained graph components',title='B  Fixed similarity groups',yscale='symlog');axes[1].legend(fontsize=9)
    ax=axes[2];ax.axis('off');ax.set_title('C  Duplicate-removal limit',loc='left')
    ax.text(0,.92,'Same historical predictions;\nremove exact-training test matches',va='top',linespacing=1.6)
    ax.text(0,.63,'Stage 1: 200+ / 200− → 100+ / 200−\nStage 2: 100+ / 100− → 0+ / 100−',va='top',linespacing=1.9)
    ax.text(0,.27,'Stage 2 cleaned discrimination: undefined\nThree-model historical reconstruction stopped:\noriginal neural training provenance unavailable.',va='top',fontsize=9,linespacing=1.6)
    fig.tight_layout();fig.subplots_adjust(bottom=.23,wspace=.35)
    save(fig,'F2_Benchmark_validity','Historical overlap, grouping and decontamination limit','A: supplied test exact matches to supplied training, all in strong class. Reverse-complement-only matches were absent. B: actual retained candidate graph-component sizes, using the original LCS160 and exact/RC grouping protocol; logarithmic count scale near zero. C: duplicate-cleaned historical Stage 2 test has no positives. The optional three-model historical experiment stopped under the authentication rule; these counts do not quantify a causal leakage effect.','final_evidence/benchmark_audit.csv; final_evidence/inputs/stage1_candidate_manifest.csv; final_evidence/inputs/stage2_candidate_manifest.csv; phase40_outputs/summary.json')
    # F3: all rows use the same aggregation of the actual fixed OOF scores.
    fig,axes=plt.subplots(1,2,figsize=(12.5,6.5),sharex=True,sharey=True)
    for stage,ax in zip([1,2],axes):
        data=gc[gc.stage.eq(stage)].groupby('model').agg(raw=('raw_fold_auroc','mean'),fold_sd=('raw_fold_auroc','std'),matched=('same_gc_fold_auroc','mean')).loc[ORDER]
        y=np.arange(len(data))
        ax.errorbar(data.raw,y,xerr=data.fold_sd,fmt='o',color='#275D8C',capsize=3,label='Fixed OOF mean-fold AUROC ± fold SD')
        ax.scatter(data.matched,y,marker='D',s=40,color='#C65A28',label='Exact-GC within-fold pair AUROC')
        for j,r in enumerate(data.itertuples()):ax.plot([r.matched,r.raw],[j,j],color='#B7BEC4',lw=1,zorder=0)
        ax.axvline(.5,color='#777777',lw=1,ls=':');ax.set(yticks=y,yticklabels=[LABELS[x] for x in data.index],xlabel='AUROC',title=f'Stage {stage}',xlim=(.47,.94));ax.grid(axis='x',alpha=.18)
    axes[0].invert_yaxis();fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.55,1.01),fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig,'F3_Classification_GC','Grouped classification and exact-GC evaluation','Points show equal averages of five within-fold AUROCs calculated from fixed predictions. Fine-tuned DNABERT rows use current three-seed ensemble scores; other rows are archived single runs. Error bars are descriptive fold SD, not independent-replicate confidence intervals or seed SD. Diamonds use all opposite-label pairs with identical GC count inside each fold; matching changes eligible pairs and controls only GC count. Its lower performance cannot be interpreted as causal removal of all composition effects.','final_tables/supplement/S4_Exact_GC.csv')
    # F4: seeded mean-fold estimates and distinct pooled ensemble contrasts.
    seed=s.read('phase32_outputs/seed_metrics_pooled_and_fold.csv');fig,axes=plt.subplots(1,3,figsize=(12.5,4.4),gridspec_kw={'width_ratios':[1,1,1.25]})
    models=['finetuned_dnabert_legacy','finetuned_dnabert2']
    for stage,ax in zip([1,2],axes[:2]):
        for i,model in enumerate(models):
            g=seed[seed.stage.eq(stage)&seed.model.eq(model)].sort_values('base_seed')
            ax.scatter(np.full(len(g),i)+np.linspace(-.07,.07,len(g)),g.mean_fold_auroc,s=45,color=['#77716A','#275D8C'][i])
            ax.plot([i-.16,i+.16],[g.mean_fold_auroc.mean()]*2,color='black',lw=2)
        ax.set(xticks=[0,1],xticklabels=['DNABERT','DNABERT-2'],ylabel='Mean-fold AUROC per seed',title=f'Stage {stage}: three seeds');ax.grid(axis='y',alpha=.2)
    ax=axes[2]
    for i,stage in enumerate([1,2]):
        r=tests[tests.comparison_id.eq(f'C_S{stage}_DB2_minus_finetuned_dnabert_legacy')].iloc[0]
        ax.errorbar(r.estimate_a_minus_b,i,xerr=[[r.estimate_a_minus_b-r.lower_95],[r.upper_95-r.estimate_a_minus_b]],fmt='o',color='#275D8C',capsize=4)
    ax.axvline(0,color='#777777',ls=':');ax.set(yticks=[0,1],yticklabels=['Stage 1','Stage 2'],xlabel='Pooled ensemble AUROC difference\nDNABERT-2 minus DNABERT',title='Conditional 95% group-bootstrap intervals');ax.grid(axis='x',alpha=.18)
    fig.tight_layout(w_pad=3)
    save(fig,'F4_Repeated_DNABERT','Repeated training and conditional ensemble contrasts','Left and middle: per-seed mean-fold AUROC for three current consistent fits per fold, with horizontal mean marks. Right: paired differences of pooled three-seed ensembles, with Phase41 similarity-group bootstrap intervals; both contain zero. Seed spread, fold variation and conditional cohort uncertainty are different quantities. Fifteen seed×fold entries are not independent replicates; no equivalence conclusion follows.','phase32_outputs/seed_metrics_pooled_and_fold.csv; final_statistics/adjusted_tests.csv')
    # F5: direct activity prediction, explicit controls and model comparisons.
    internal=s.read('final_tables/main/T4_MPRA_internal.csv');seedmpra=s.read('phase37_outputs/seed_metrics_independent.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8));arms=['spaced6_4_ridge','dnabert2_selected','legnet_selected','mprann_selected']
    for cell,ax in zip(['HepG2','K562'],axes[0]):
        for i,arm in enumerate(arms):
            r=internal[internal.cell.eq(cell)&internal.arm.eq(arm)].iloc[0]
            ax.scatter(i,r.pearson,color=colors[arm],s=90,marker='D',zorder=3)
            g=seedmpra[seedmpra.cell.eq(cell)&seedmpra.cohort.eq('internal')&seedmpra.arm.eq(arm)]
            if len(g):ax.scatter(i+np.linspace(-.12,.12,len(g)),g.pearson,color=colors[arm],s=25,alpha=.6)
        ax.set(xticks=range(4),xticklabels=['Spaced-mer\narchived','DNABERT-2','LegNet','MPRAnn'],ylabel='Internal test Pearson',title=f'{cell}: chromosome-held-out activity');ax.grid(axis='y',alpha=.18)
    for ax,kind,title in [(axes[1,0],'selected','LegNet minus selected DNABERT-2'),(axes[1,1],'duration','DNABERT-2 longer minus own two-epoch prefix')]:
        for i,cell in enumerate(['HepG2','K562']):
            cid=f'M_{cell}_internal_legnet_selected_minus_dnabert2_selected' if kind=='selected' else f'D_{cell}_internal_full_minus_own_early2'
            r=tests[tests.comparison_id.eq(cid)].iloc[0]
            ax.errorbar(r.estimate_a_minus_b,i,xerr=[[r.estimate_a_minus_b-r.lower_95],[r.upper_95-r.estimate_a_minus_b]],fmt='o',capsize=4,color='#275D8C')
        ax.axvline(0,color='#777777',ls=':');ax.set(yticks=[0,1],yticklabels=['HepG2','K562'],xlabel='Pearson difference; conditional 95% interval',title=title);ax.grid(axis='x',alpha=.18)
    fig.tight_layout(h_pad=3,w_pad=3)
    save(fig,'F5_MPRA_prediction_duration','Quantitative activity prediction and duration control','Top: three-seed ensemble diamonds and individual-seed circles for selected neural recipes, plus one explicitly archived spaced-mer baseline. Bottom: Phase41 conditional locus-group bootstrap intervals. The fixed2e-5 duration comparison uses each new fit’s own exact first-two-epoch prefix. The model comparison follows validation-only grid selection. Equal numbers of candidate recipes do not establish equal compute or exhaustive tuning; grid-boundary and budget-cap caveats remain.','final_tables/main/T4_MPRA_internal.csv; phase37_outputs/seed_metrics_independent.csv; final_statistics/adjusted_tests.csv')
    # F6: latest transfer versus explicitly older frozen-model failure diagnostics.
    ext=s.read('final_tables/main/T5_External_transfer.csv');tail=s.read('phase18_outputs/observed_activity_tail_diagnostics.csv');filtering=s.read('phase23_outputs/filtering_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8));names=['DNABERT-2','LegNet','MPRAnn'];ars=['dnabert2_selected','legnet_selected','mprann_selected']
    for ci,cell in enumerate(['HepG2','K562']):
        g=ext[ext.cell.eq(cell)].set_index('arm').loc[ars]
        axes[0,0].scatter(np.arange(3)+(ci-.5)*.10,g.spearman,marker=['o','s'][ci],label=cell)
        axes[0,1].scatter(np.arange(3)+(ci-.5)*.10,g.r2,marker=['o','s'][ci],label=cell)
    for ax,title,ylabel in [(axes[0,0],'A  Current selected models: external ranking','DHS-natural Spearman'),(axes[0,1],'B  Current selected models: absolute calibration','DHS-natural R²')]:
        ax.set(xticks=range(3),xticklabels=names,title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
    axes[0,1].axhline(0,color='#777777',ls=':')
    bands=['low','middle','high']
    for ci,cell in enumerate(['HepG2','K562']):
        for mi,model in enumerate(['legnet','mprann']):
            g=tail[tail.cell_line.eq(cell)&tail.model.eq(model)&tail.population.eq('retained')].set_index('band').loc[bands]
            axes[1,0].plot(range(3),g.signed_error,marker=['o','s'][mi],ls=['-','--'][ci],label=f'{cell} {model}')
    axes[1,0].axhline(0,color='#777777',ls=':');axes[1,0].set(xticks=range(3),xticklabels=['Low activity','Middle','High activity'],ylabel='Mean prediction minus measured activity',title='C  Older frozen CNNs: retained activity bands');axes[1,0].legend(fontsize=8,ncol=2);axes[1,0].grid(axis='y',alpha=.18)
    fg=filtering[filtering.analysis.eq('primary')]
    for ci,cell in enumerate(['HepG2','K562']):
        delta=[]
        for model in ['dnabert2','legnet','mprann']:
            g=fg[fg.cell_line.eq(cell)&fg.model.eq(model)].set_index('population')
            delta.append(g.loc['removed','mae']-g.loc['retained','mae'])
        axes[1,1].scatter(np.arange(3)+(ci-.5)*.10,delta,marker=['o','s'][ci],label=cell)
    axes[1,1].axhline(0,color='#777777',ls=':');axes[1,1].set(xticks=range(3),xticklabels=names,ylabel='Removed minus retained MAE',title='D  Older frozen models: population transport');axes[1,1].legend(fontsize=9);axes[1,1].grid(axis='y',alpha=.18)
    fig.tight_layout(h_pad=3,w_pad=3)
    save(fig,'F6_Transfer_failure_modes','Assay transfer, calibration and population failure modes','A–B use current three-seed selected neural models on frozen DHS-natural sequences. All external ensemble R2 values are negative, despite positive rank correlations. C–D are explicitly different, older fitted-model diagnostics from Phases16–24; they are not predictions of the new selected models. C uses fixed measured-activity bands and is a target-dependent diagnostic. D compares errors in different retained/removed populations, not causal retraining effects; paired exploratory intervals and subgroup counts are in the supplement.','final_tables/main/T5_External_transfer.csv; phase18_outputs/observed_activity_tail_diagnostics.csv; phase23_outputs/filtering_metrics.csv')
    # Curated supplementary plots, regenerated where current OOF is available.
    pred=s.read('final_statistics/classification_predictions.csv.gz')
    from sklearn.metrics import roc_curve,precision_recall_curve
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for stage,axs in zip([1,2],axes):
        for model in ORDER:
            d=pred[pred.stage.eq(stage)&pred.model.eq(model)];fpr,tpr,_=roc_curve(d.label,d.score);pr,re,_=precision_recall_curve(d.label,d.score)
            axs[0].plot(fpr,tpr,label=LABELS[model],lw=1.4);axs[1].plot(re,pr,lw=1.4)
        axs[0].plot([0,1],[0,1],color='#AAAAAA',ls=':');axs[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title=f'Stage {stage} ROC');axs[1].set(xlabel='Recall',ylabel='Precision',title=f'Stage {stage} precision–recall')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.07));fig.tight_layout(rect=(0,.08,1,1))
    save(fig,'SF1_Classification_ROC_PR','Fixed OOF classification curves','Current three-seed fine-tuned ensembles and seven archived single-run baselines. Curves pool fixed OOF predictions; not an untouched independent test. No threshold was chosen from these curves.','final_statistics/classification_predictions.csv.gz','supplement')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for stage,ax in zip([1,2],axes):
        for model in ['gc_logreg','spaced6_4_logreg','finetuned_dnabert_legacy','finetuned_dnabert2']:
            d=pred[pred.stage.eq(stage)&pred.model.eq(model)].copy();d['bin']=np.minimum((d.score*10).astype(int),9)
            b=d.groupby('bin').agg(n=('label','size'),mean_probability=('score','mean'),positive_fraction=('label','mean'));b=b[b.n>=20]
            ax.plot(b.mean_probability,b.positive_fraction,marker='o',label=LABELS[model],markersize=4)
        ax.plot([0,1],[0,1],ls=':',color='#777777');ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title=f'Stage {stage}: bins with ≥20 records',xlim=(0,1),ylim=(0,1))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=8,bbox_to_anchor=(.5,-.08));fig.tight_layout(rect=(0,.08,1,1))
    save(fig,'SF2_Classification_reliability','Probability reliability on fixed OOF cohorts','Ten fixed probability bins, with bins below20 records suppressed from the plot; Brier and all confusion counts remain in TableS5. Current ensembles versus archived baselines are explicit. This is a descriptive OOF diagnostic, not recalibration.','final_statistics/classification_predictions.csv.gz; final_tables/supplement/S5_Classification_calibration.csv','supplement')
    jobs=s.read('phase35_outputs/training_job_summary.csv');fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=False)
    for (cell,model),ax in zip([(c,m) for c in ['HepG2','K562'] for m in ['dnabert2','legnet','mprann']],axes.flat):
        for r in jobs[jobs.cell_line.eq(cell)&jobs.model.eq(model)].itertuples():
            log=s.read(f'phase35_outputs/training_logs/{r.job}.csv')
            ax.plot(log.epoch,log.validation_pearson,lw=1,alpha=.65,color=['#275D8C','#C65A28','#35806A'][int(r.grid_index)])
        ax.set(title=f'{cell} · {model}',xlabel='Completed epoch',ylabel='Validation Pearson');ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig,'SF3_Validation_trajectories','All 54 validation trajectories','Each line is one declared candidate/seed fit. Colors denote the three ordered learning-rate grid positions. Curves retain the two practical near-constant high-LR DNABERT-2 runs. Stopping policies differ by model; no claim of exhaustive convergence.','phase35_outputs/training_logs/*.csv; final_tables/supplement/S7_MPRA_grid_epochs.csv','supplement')
    candidates=s.read('phase36_outputs/validation_candidate_scores.csv');flags=s.read('phase38_outputs/selected_recipe_adequacy_flags.csv');fig,axes=plt.subplots(2,3,figsize=(12,7))
    for (cell,model),ax in zip([(c,m) for c in ['HepG2','K562'] for m in ['dnabert2','legnet','mprann']],axes.flat):
        d=candidates[candidates.cell.eq(cell)&candidates.model.eq(model)].sort_values('grid_index')
        ax.errorbar(d.learning_rate,d.mean_validation_pearson,yerr=d.seed_sd_validation_pearson,fmt='o-',capsize=3,color='#275D8C')
        lr=flags[flags.cell_line.eq(cell)&flags.model.eq(model)].learning_rate.iloc[0];ax.axvline(lr,color='#C65A28',ls=':')
        ax.set(xscale='log',xlabel='Declared learning rate / OneCycle maximum',ylabel='Mean validation Pearson ± seed SD',title=f'{cell} · {model}')
        ax.set_xticks(d.learning_rate,[f'{v:.0e}' for v in d.learning_rate]);ax.minorticks_off();ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig,'SF4_Grid_and_selection','Candidate grids and validation-only choices','Means and sample SD across exactly three seeds. Dashed vertical lines mark locked choices. Large DNABERT-2 high-LR spread includes practical collapses; no fit was dropped. Upper/lower grid-boundary selections remain visible.','phase36_outputs/validation_candidate_scores.csv; phase38_outputs/selected_recipe_adequacy_flags.csv','supplement')
    cost=s.read('final_tables/supplement/S17_Compute.csv');fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for cell,ax in zip(['HepG2','K562'],axes):
        d=cost[cost.cell_line.eq(cell)].set_index('model').loc[['dnabert2','legnet','mprann']]
        ax.barh(['DNABERT-2','LegNet','MPRAnn'],d.elapsed_hours,color=['#275D8C','#C65A28','#35806A'])
        for i,r in enumerate(d.itertuples()):ax.text(r.elapsed_hours,i,f' {r.elapsed_hours:.3f} h',va='center',fontsize=9)
        ax.set(xscale='log',xlabel='Sum of recorded fit hours (log scale)',title=cell,xlim=(.03,15))
    fig.tight_layout()
    save(fig,'SF5_Recorded_compute','Recorded fitting time under unequal recipes','Sum across nine completed candidate/seed fits per model/cell. Runtime is hardware/implementation dependent and excludes pretrained-model production cost. Same candidate count does not imply equal GPU time, steps or FLOPs.','final_tables/supplement/S17_Compute.csv','supplement')
    cov=s.read('phase17_outputs/interval_audit_coverage.csv');fig,axes=plt.subplots(2,2,figsize=(14,8.5))
    for (cell,model),ax in zip([(c,m) for c in ['HepG2','K562'] for m in ['legnet','mprann']],axes.flat):
        d=cov[cov.cell_line.eq(cell)&cov.model.eq(model)].sort_values(['calibration_source','method','audit_population']).reset_index(drop=True)
        labels=[f'Cal {r.calibration_source} / '+('constant' if r.method=='constant_width' else 'seed scaled')+f' / audit {r.audit_population}' for r in d.itertuples()]
        ax.scatter(d.empirical_coverage,range(len(d)),color='#275D8C');ax.axvline(.9,color='#C65A28',ls=':',label='Nominal 90%')
        ax.set(yticks=range(len(d)),yticklabels=labels,xlabel='Empirical interval coverage',xlim=(.82,.96),title=f'{cell} · {model} · older frozen fits');ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.15)
    fig.tight_layout(h_pad=3,w_pad=2)
    save(fig,'SF6_Interval_coverage','Older fitted-model interval coverage','Marginal nominal 90% intervals under the original disjoint calibration/audit scheme. These are older fitted models and populations, not a coverage guarantee for Phase35 models. Subgroup coverage and n remain in TableS13.','phase17_outputs/interval_audit_coverage.csv','supplement')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for cell,ax in zip(['HepG2','K562'],axes):
        g=jobs[jobs.cell_line.eq(cell)].reset_index(drop=True);bad=g.validation_prediction_sd<1e-4
        ax.scatter(np.arange(len(g))[~bad],g.validation_prediction_sd[~bad],color='#275D8C',s=24,label='Other fitted candidates')
        ax.scatter(np.arange(len(g))[bad],g.validation_prediction_sd[bad],color='#B84945',marker='x',s=65,label='Practical near-constant fit')
        ax.axhline(1e-4,color='#777777',ls=':');ax.set(yscale='log',xlabel='Candidate/seed fit index (all 27 shown)',ylabel='Selected validation prediction SD',title=f'{cell}: 0 undefined flags, 1 practical near-constant fit');ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig,'SF7_Practical_collapse','Practical collapse alongside the frozen flag','The frozen undefined-Pearson rule flags zero fits. A post hoc descriptive SD<1e-4 screen identifies two practical near-constant fits, one per cell at DNABERT-2 LR5e-5/seed20260932. All54 fits and the locked three-seed selection remain unchanged. No training root cause is inferred.','phase35_outputs/training_job_summary.csv; phase38_outputs/ERRATUM_collapse_count.md','supplement')
    s.table('final_figures/figure_index.csv',pd.DataFrame(index))
    s.text('final_figures/FIGURE_CAPTIONS.md','# Final figure captions\n\n'+'\n\n'.join('**'+r['figure']+' — '+r['title']+'**\n\n'+r['caption']+'\n\nSources: '+r['source_files'] for r in index))
    main=[r for r in index if r['placement']=='main'];sheet=Image.new('RGB',(1800,1800),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(main):
        im=Image.open(ROOT/r['png']).convert('RGB');im.thumbnail((880,550));x=(i%2)*900+(900-im.width)//2;y=(i//2)*600+35
        sheet.paste(im,(x,y));draw.text(((i%2)*900+15,(i//2)*600+10),r['figure'],fill='black')
    sheet.save(root/'figure_contact_sheet.png')
    s.dump('final_figures/visual_QA.json',dict(passed=False,status='pending human-visible inspection by assistant',main_figures_inspected=0,main_figures=6,supplementary_figures=7))
    s.dump('phase43_outputs/summary.json',dict(status='generated_pending_visual_QA',main_figures=6,supplementary_figures=7,formats=['PNG','SVG','PDF'],source_sha256=s.sha(__file__),new_training=False))
    print('Phase43 generated 6 main and 7 supplementary figures in PNG/SVG/PDF; visual inspection is pending.')

def revision_figures():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,Rectangle,Ellipse,Circle,FancyArrowPatch
    root=ROOT/'final_figures'
    def save(fig,name):
        for extension in ['png','svg','pdf']:fig.savefig(root/'main'/f'{name}.{extension}',dpi=200,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    # Workflow uses data-store symbols, a similarity network, fixed-fold and
    # chromosome partitions, a candidate grid and explicit branch connections.
    fig,ax=plt.subplots(figsize=(14,7.5));ax.set(xlim=(0,14),ylim=(0,7.5));ax.axis('off')
    blue='#275D8C';green='#35806A';orange='#C65A28';grey='#5C6873'
    def arrow(a,b,color=grey,style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=color,lw=1.6,linestyle=style,connectionstyle='arc3'))
    def store(x,y,label,color):
        ax.add_patch(Rectangle((x-.85,y-.45),1.7,.9,facecolor='#F4F6F8',edgecolor=color,lw=1.3))
        for yy in [y-.45,y+.45]:ax.add_patch(Ellipse((x,yy),1.7,.20,facecolor='#F4F6F8',edgecolor=color,lw=1.3))
        ax.text(x,y,label,ha='center',va='center',fontsize=9 if 'DHS-natural' in label else 10)
    ax.text(7,7.18,'Separate benchmark validity from activity prediction and transfer',ha='center',fontsize=16,fontweight='bold')
    ax.text(.1,6.53,'A  Legacy classification',fontsize=12,fontweight='bold',color=blue)
    store(1.05,5.55,'Supplied\n200 bp sequences',blue)
    nodes=[(3.0,5.7),(3.5,5.95),(3.65,5.35),(3.05,5.12),(4.0,5.65)]
    for a,b in [(0,1),(0,3),(1,2),(2,3),(1,4)]:ax.plot([nodes[a][0],nodes[b][0]],[nodes[a][1],nodes[b][1]],color=blue,lw=1.5)
    for xy in nodes:ax.add_patch(Circle(xy,.09,color=blue))
    ax.text(3.5,4.62,'Audit identity\nand group related DNA',ha='center',fontsize=10)
    for i in range(5):
        for j in range(3):ax.add_patch(Rectangle((5.25+i*.25,5.14+j*.25),.20,.20,facecolor=blue if i==j else '#DBE5EF',edgecolor='white'))
    ax.text(5.85,4.62,'Five fixed folds\nGroup separation',ha='center',fontsize=10)
    ax.add_patch(FancyBboxPatch((7.2,5.05),2.3,1.0,boxstyle='round,pad=.06',facecolor='#F4F6F8',edgecolor=blue))
    ax.text(8.35,5.55,'Seven archived baselines\nDNABERT and DNABERT-2\nThree training seeds',ha='center',va='center',fontsize=9.5)
    ax.text(11.4,5.58,'Classification metrics\nRaw and exact-GC AUROC\nSingle seeds and ensembles',ha='center',va='center',fontsize=10)
    for a,b in [((1.96,5.55),(2.72,5.55)),((4.13,5.55),(5.05,5.55)),((6.6,5.55),(7.12,5.55)),((9.6,5.55),(10.15,5.55))]:arrow(a,b,blue)
    ax.text(.1,3.77,'B  Quantitative reporter activity',fontsize=12,fontweight='bold',color=green)
    store(1.05,2.85,'ENCODE\nlentiMPRA',green)
    ax.text(1.05,2.12,'HepG2 and K562\nFrozen 200 bp elements',ha='center',va='top',fontsize=9.5)
    for i,color in enumerate([green,'#8CA99F',orange]):
        ax.add_patch(FancyBboxPatch((2.9+i*.38,2.3),.2,1.05-i*.13,boxstyle='round,pad=.02',facecolor=color,edgecolor=color))
    ax.text(3.4,1.85,'Chromosome split\nTrain / validation / test',ha='center',fontsize=9.5)
    for i in range(3):
        for j in range(3):ax.add_patch(Rectangle((5.1+i*.26,2.42+j*.25),.21,.20,facecolor=['#BED7CD',green,'#DCE8E2'][i],edgecolor='white'))
    ax.text(5.5,1.70,'54 fits\n3 models × 3 rates\n× 3 seeds × 2 cells',ha='center',fontsize=9.5)
    ax.add_patch(FancyBboxPatch((7.1,2.3),2.05,1.05,boxstyle='round,pad=.06',facecolor='#F4F6F8',edgecolor=green))
    ax.text(8.12,2.82,'Validation-only selection\nLock recipes and weights',ha='center',va='center',fontsize=9.5)
    ax.text(11.35,2.82,'Internal test metrics\nOwn-prefix duration control',ha='center',va='center',fontsize=10)
    for a,b in [((1.96,2.85),(2.7,2.85)),((4.15,2.85),(4.9,2.85)),((6.1,2.85),(7.0,2.85)),((9.25,2.85),(10.1,2.85))]:arrow(a,b,green)
    store(8.1,.83,'Gosai et al.\nDHS-natural subset',orange)
    ax.text(11.35,.83,'External transfer\nRank correlation and calibration\n8,399 sequences per cell',ha='center',va='center',fontsize=9.5)
    arrow((9.12,2.24),(10.18,1.18),green);arrow((9.0,.83),(10.0,.83),orange)
    ax.text(1.8,.65,'All cohorts were previously examined.\nEarlier filtering and reliability diagnostics\nretain their original fitted models.',fontsize=9.5,ha='left',color=grey)
    save(fig,'F1_Study_workflow')
    audit=s.read('final_tables/main/T2_Benchmark_audit.csv');roster=s.read('phase21_outputs/historical_test_overlap_roster.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1,1.5]})
    ax=axes[0];x=np.arange(2);exact=audit.exact_train_test_overlap.to_numpy();remaining=audit.supplied_test_n.to_numpy()-exact
    ax.bar(x,exact,color=orange,label='Training-identical');ax.bar(x,remaining,bottom=exact,color='#DBE5EF',label='Other supplied test')
    for i,(a,b) in enumerate(zip(exact,remaining)):
        ax.text(i,a/2,str(a),ha='center',va='center',color='white');ax.text(i,a+b/2,str(b),ha='center',va='center')
    ax.set(xticks=x,xticklabels=['Stage 1','Stage 2'],ylabel='Supplied test records',title='A  Exact overlap');ax.legend(fontsize=8,loc='upper right');ax.set_ylim(0,475)
    ax=axes[1]
    for stage,color in [(1,blue),(2,green)]:
        m=s.read(f'final_evidence/inputs/stage{stage}_candidate_manifest.csv');m=m[m.role.eq('train_candidate')];counts=m.groupby('similarity_group_id').size().value_counts().sort_index()
        ax.bar(counts.index+(stage-1.5)*.27,counts.values,width=.25,color=color,label=f'Stage {stage}')
    ax.set(yscale='log',xlabel='Records per retained group',ylabel='Number of groups',title='B  Similarity groups');ax.set_xticks([1,2,3,4]);ax.legend(fontsize=8)
    ax=axes[2];labels=['Stage 1 supplied','Stage 1 duplicates removed','Stage 2 supplied','Stage 2 duplicates removed'];positive=[200,100,100,0];negative=[200,200,100,100];y=np.arange(4)
    ax.barh(y,positive,color=green,label='Positive');ax.barh(y,negative,left=positive,color='#CBD3DA',label='Negative')
    for i,(a,b) in enumerate(zip(positive,negative)):
        if a:ax.text(a/2,i,str(a),ha='center',va='center',color='white')
        ax.text(a+b/2,i,str(b),ha='center',va='center')
    ax.set(yticks=y,yticklabels=labels,xlabel='Historical test records',title='C  Test composition after duplicate removal',xlim=(0,435));ax.invert_yaxis();ax.legend(fontsize=8,loc='lower right')
    ax.text(.02,-.27,'Cleaned Stage 2 AUROC is undefined: no positives.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout(w_pad=2);fig.subplots_adjust(bottom=.26)
    save(fig,'F2_Benchmark_validity')
    # Show pooled seed and ensemble scores on the same axes, with no mixed
    # mean-fold versus pooled aggregation in this single-seed comparison.
    seed=s.read('phase32_outputs/seed_metrics_pooled_and_fold.csv');summary=s.read('final_tables/main/T3_Classification.csv');tests=s.read('final_statistics/adjusted_tests.csv')
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.7),gridspec_kw={'width_ratios':[1,1,1.2]});models=['finetuned_dnabert_legacy','finetuned_dnabert2']
    for stage,ax in zip([1,2],axes[:2]):
        for i,model in enumerate(models):
            g=seed[seed.stage.eq(stage)&seed.model.eq(model)].sort_values('base_seed');r=summary[summary.stage.eq(stage)&summary.model.eq(model)].iloc[0]
            ax.scatter(i+np.linspace(-.08,.08,3),g.auroc,s=32,color=blue,label='Individual seed' if i==0 else None)
            ax.scatter(i+.23,r.auroc,marker='D',s=55,color=orange,label='Three-seed ensemble' if i==0 else None)
        b=summary[summary.stage.eq(stage)&summary.model.eq('spaced6_4_logreg')].auroc.iloc[0];ax.axhline(b,color='#777777',ls=':',label='Spaced-mer single run')
        ax.set(xticks=[0,1],xticklabels=['DNABERT','DNABERT-2'],ylabel='Pooled out-of-fold AUROC',title=f'Stage {stage}');ax.grid(axis='y',alpha=.15)
    axes[0].legend(fontsize=8,loc='lower left',bbox_to_anchor=(0,.15))
    for i,stage in enumerate([1,2]):
        r=tests[tests.comparison_id.eq(f'C_S{stage}_DB2_minus_finetuned_dnabert_legacy')].iloc[0]
        axes[2].errorbar(r.estimate_a_minus_b,i,xerr=[[r.estimate_a_minus_b-r.lower_95],[r.upper_95-r.estimate_a_minus_b]],fmt='o',color=blue,capsize=4)
    axes[2].axvline(0,color='#777777',ls=':');axes[2].set(yticks=[0,1],yticklabels=['Stage 1','Stage 2'],xlabel='DNABERT-2 minus DNABERT\nEnsemble pooled AUROC difference',title='Fixed-model 95% intervals');fig.tight_layout(w_pad=2.6)
    save(fig,'F4_Repeated_DNABERT')
    index=s.read('final_figures/figure_index.csv')
    captions={
      'F1_Study_workflow':'The two analysis branches start from supplied legacy sequences and ENCODE lentiMPRA measurements. Identity/similarity auditing precedes fixed grouped classification; chromosome partitions precede a 54-fit validation-only activity search. Locked activity models are evaluated internally and on the processed DHS-natural subset from Gosai et al. The symbols are schematic and arrows show data flow. External and internal cohorts were previously examined.',
      'F2_Benchmark_validity':'A: supplied test records with exact training identity; reverse-complement-only matches add no records. B: sizes of retained similarity components, on a logarithmic group-count axis. C: positive and negative class counts in the supplied and exact-duplicate-removed historical test sets. Stage 2 retains no positive records after removal, so discrimination metrics are undefined. These findings apply to audited supplied copies; published benchmark-lineage citations do not establish byte identity of every earlier study. Historical neural artifacts were unavailable, so no causal inflation estimate is made.',
      'F4_Repeated_DNABERT':'Left and middle: pooled out-of-fold AUROC for each of three training seeds (circles) and the corresponding three-seed ensemble (diamonds). The dotted line is the archived single-run spaced-mer baseline on the same records. These panels use pooled AUROC throughout, rather than mean-fold AUROC. Right: unchanged ensemble DNABERT-2 minus DNABERT differences with 95% group-bootstrap intervals for the fixed fitted models. Seed variation is separate from interval uncertainty; folds share training data.',
      'F6_Transfer_failure_modes':'A and B: unconnected model-specific points for selected three-seed ensembles on the DHS-natural external cohort. C and D: older fitted-model activity-tail and retained/removed diagnostics. Models are categorical comparisons; connecting lines are not used to imply an ordered model trend. All selected external R² values are negative despite positive rank correlations. Older diagnostics retain their original populations, fitted models and exploratory uncertainty.'}
    for name,caption in captions.items():index.loc[index.figure.eq(name),'caption']=caption
    index.loc[index.figure.eq('F2_Benchmark_validity'),'title']='Benchmark overlap and effects of duplicate removal'
    index.loc[index.figure.eq('F4_Repeated_DNABERT'),'title']='Individual training seeds and ensemble comparisons'
    for a,b in [('Stage1','Stage 1'),('Stage2','Stage 2'),('nominal90%','nominal 90%'),('All54','All 54'),('All3','All 3'),('TableS','Table S'),('LR5e-5','LR 5e-5'),('seed20260932','seed 20260932'),('Phase35','the current study'),('Phase41','the final synthesis')]:index['caption']=index.caption.str.replace(a,b,regex=False)
    s.table('final_figures/figure_index.csv',index)
    s.text('final_figures/FIGURE_CAPTIONS.md','# Figure captions\n\n'+'\n\n'.join('**'+r.figure+' '+r.title+'**\n\n'+r.caption for r in index.itertuples()))
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(1800,1800),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(index[index.placement.eq('main')].itertuples()):
        im=Image.open(ROOT/r.png).convert('RGB');im.thumbnail((880,550));x=(i%2)*900+(900-im.width)//2;y=(i//2)*600+35;sheet.paste(im,(x,y));draw.text(((i%2)*900+15,(i//2)*600+10),r.figure,fill='black')
    sheet.save(root/'figure_contact_sheet.png')

_phase43_original=phase43
def phase43():
    _phase43_original()
    revision_figures()
