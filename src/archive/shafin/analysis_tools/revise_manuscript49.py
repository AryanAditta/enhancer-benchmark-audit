"""Versioned editorial corrections. Frozen training and inference remain read-only."""
from pathlib import Path
import os,sys,json,hashlib,shutil,re,ast,importlib.util,subprocess,zipfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_tools import final_synthesis as old
from analysis_tools.final_documents import runtime,DEPS,NODE
ROOT=old.ROOT
OUT=ROOT/'revision49_outputs/revised'
NOTES=ROOT/'research_notes/Enhancer manuscript citation corrections'

def dump(path,value):
    path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def text(path,value):
    path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value.strip()+'\n',encoding='utf-8')
def read(path):return pd.read_csv((OUT/path) if (OUT/path).exists() else ROOT/path,float_precision='round_trip')
def jd(path):return json.loads(((OUT/path) if (OUT/path).exists() else ROOT/path).read_text(encoding='utf-8'))
def table(path,frame):
    path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,float_format='%.17g')
S=SimpleNamespace(ROOT=OUT,read=read,jd=jd,table=table,text=text,dump=dump,sha=old.sha,manifests=old.manifests,verify_inputs=old.verify_inputs,NOW=old.NOW)

def comparisons():
    from sklearn.metrics import roc_auc_score
    raw=old.read('phase32_outputs/independently_verified_predictions.csv.gz')
    seeds=old.read('phase32_outputs/seed_metrics_pooled_and_fold.csv')
    score='score' if 'score' in raw else 'probability';truth='label'
    checks=[]
    for (stage,model,seed),g in raw.groupby(['stage','model','base_seed']):
        canonical={'dnabert2':'finetuned_dnabert2','dnabert_legacy':'finetuned_dnabert_legacy'}.get(model,model)
        expected=seeds[seeds.stage.eq(stage)&seeds.model.eq(canonical)&seeds.base_seed.eq(seed)]
        assert len(expected)==1,(stage,model,seed)
        got=roc_auc_score(g[truth],g[score]);assert abs(got-expected.auroc.iloc[0])<1e-12
        checks.append(dict(stage=int(stage),model=canonical,base_seed=int(seed),pooled_auroc=float(got),absolute_reproduction_error=float(abs(got-expected.auroc.iloc[0]))))
    summary=old.read('final_statistics/classification_summary.csv');rows=[]
    for stage in [1,2]:
        baseline=summary[summary.stage.eq(stage)&summary.model.eq('spaced6_4_logreg')].iloc[0]
        for model in ['finetuned_dnabert2','finetuned_dnabert_legacy']:
            g=seeds[seeds.stage.eq(stage)&seeds.model.eq(model)];ensemble=summary[summary.stage.eq(stage)&summary.model.eq(model)].iloc[0]
            rows.append(dict(claim_id=f'SINGLE_S{stage}_{model}',stage=stage,model=model,seeds=3,single_seed_pooled_AUROC_mean=float(g.auroc.mean()),single_seed_pooled_AUROC_SD=float(g.auroc.std(ddof=1)),single_seed_min=float(g.auroc.min()),single_seed_max=float(g.auroc.max()),spaced_mer_single_pooled_AUROC=float(baseline.auroc),single_seed_mean_minus_spaced_mer=float(g.auroc.mean()-baseline.auroc),three_seed_ensemble_pooled_AUROC=float(ensemble.auroc),ensemble_minus_spaced_mer=float(ensemble.auroc-baseline.auroc),ensemble_minus_single_seed_mean=float(ensemble.auroc-g.auroc.mean()),aggregation='Pooled OOF AUROC for each seed or the pooled OOF ensemble; not mean-fold AUROC',inference='Descriptive; no extra hypothesis test and no change to the 14-contrast family'))
    frame=pd.DataFrame(rows);table('final_tables/Single_seed_and_ensemble_comparison.csv',frame)
    table('claim_audit/single_seed_independent_reproduction.csv',pd.DataFrame(checks))
    d=read('final_tables/main/T3_Classification.csv')
    for row in rows:
        ix=d.stage.eq(row['stage'])&d.model.eq(row['model'])
        d.loc[ix,'single_seed_pooled_auroc_mean']=row['single_seed_pooled_AUROC_mean']
        d.loc[ix,'single_seed_pooled_auroc_sd']=row['single_seed_pooled_AUROC_SD']
        d.loc[ix,'three_seed_ensemble_pooled_auroc']=row['three_seed_ensemble_pooled_AUROC']
    ix=~d.model.isin(['finetuned_dnabert2','finetuned_dnabert_legacy'])
    d.loc[ix,'single_seed_pooled_auroc_mean']=d.loc[ix,'auroc']
    table('final_tables/main/T3_Classification.csv',d)
    index=read('final_tables/table_index.csv');ix=index.table.eq('T3_Classification')
    index.loc[ix,'notes']='Single-seed pooled AUROC and three-seed ensemble pooled AUROC are shown separately. Parentheses give SD across three seeds; archived baselines have one run and no seed SD. Mean-fold AUROC is a different aggregation. Exact-GC values use the fixed ensemble for current transformers. Fold variability and full seed details are in Supplementary Table S2.'
    index.loc[ix,'title']='Similarity-grouped classification with single seeds and ensembles'
    index.loc[index.table.eq('T1_Datasets'),'notes']='lentiMPRA sources: Agarwal et al. (2025), HepG2 ENCFF774DYO / ENCSR022GQD and K562 ENCFF631LHA / ENCSR382BVV. External source: Gosai et al. (2024), Supplementary Table 12. Counts refer to the locally processed cohorts. Previously examined evaluation sets are not a new confirmatory holdout.'
    index.loc[index.table.eq('T2_Benchmark_audit'),'notes']='Similarity groups link sequences with longest common subsequence of at least 160 bases in either orientation. Mixed-label counts refer to retained candidates; seven mixed groups occurred before exclusions. The original historical neural artifacts were unavailable for the optional reconstruction.'
    index.loc[index.table.eq('T4_MPRA_internal'),'notes']='Neural results use three-seed ensembles after validation-only recipe selection. The archived spaced-mer baseline used one recipe outside the neural search.'
    index.loc[index.table.eq('T5_External_transfer'),'notes']='Spearman measures ranking; R² and signed error measure different aspects of prediction. All selected external R² values are negative. This previously examined cohort was not used to choose recipes.'
    index.loc[index.table.eq('T6_Headline_contrasts'),'notes']='The 95% intervals resample whole groups for the fixed fitted models. Paired permutation p-values are Holm-adjusted across the 14 headline comparisons. These summaries exclude model-search and retraining uncertainty. The contrast family was chosen for final synthesis after earlier results had been seen.'
    index['notes']=index.notes.map(clean_prose)
    index.loc[ix,'sha256']=old.sha(OUT/'final_tables/main/T3_Classification.csv');table('final_tables/table_index.csv',index)
    from analysis_tools.final_presentation import table_md
    text('final_tables/main/T3_Classification.md','# Grouped classification\n\n'+table_md(d)+'\n'+index.loc[ix,'notes'].iloc[0])
    contrasts=old.read('final_tables/main/T6_Headline_contrasts.csv')
    ix=contrasts.task.eq('classification')
    contrasts.loc[ix,'contrast_label']=contrasts.loc[ix,'comparison_id'].map(lambda x:'DNABERT-2 ensemble minus DNABERT ensemble' if x.endswith('finetuned_dnabert_legacy') else 'DNABERT-2 ensemble minus single spaced-mer')
    table('final_tables/main/T6_Headline_contrasts.csv',contrasts)
    index.loc[index.table.eq('T6_Headline_contrasts'),'sha256']=old.sha(OUT/'final_tables/main/T6_Headline_contrasts.csv');table('final_tables/table_index.csv',index)
    for r in index[index.placement.eq('main')].itertuples():text(r.file.replace('.csv','.md'),'# '+r.title+'\n\n'+table_md(read(r.file))+'\n'+r.notes)
    table('claim_audit/revision_numeric_claims.csv',frame.assign(status='VERIFIED',source_predictions_sha256=old.sha(ROOT/'phase32_outputs/independently_verified_predictions.csv.gz')))
    return frame

FIGURE_PATCH=r'''
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
'''

def figures():
    runtime();source=(ROOT/'analysis_tools/final_presentation.py').read_text(encoding='utf-8')
    source=source.replace("axes[0,0].plot(np.arange(3)+(ci-.5)*.10,g.spearman,marker=['o','s'][ci],label=cell)","axes[0,0].scatter(np.arange(3)+(ci-.5)*.10,g.spearman,marker=['o','s'][ci],label=cell)")
    source=source.replace("axes[0,1].plot(np.arange(3)+(ci-.5)*.10,g.r2,marker=['o','s'][ci],label=cell)","axes[0,1].scatter(np.arange(3)+(ci-.5)*.10,g.r2,marker=['o','s'][ci],label=cell)")
    source=source.replace("axes[1,1].plot(range(3),delta,marker=['o','s'][ci],label=cell)","axes[1,1].scatter(np.arange(3)+(ci-.5)*.10,delta,marker=['o','s'][ci],label=cell)")
    source=source.replace("{cell}: frozen0 undefined flags, practical1","{cell}: 0 undefined flags, 1 practical near-constant fit")
    for a,b in [('Stage1','Stage 1'),('Stage2','Stage 2'),('Stage{stage}','Stage {stage}'),('all27 shown','all 27 shown'),('nominal90%','nominal 90%')]:source=source.replace(a,b)
    source+=FIGURE_PATCH
    path=OUT/'analysis_tools/final_presentation.py';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('revision_figure_builder',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.s=S;module.ROOT=OUT;module.phase43()
    dump('revision_figure_source.json',dict(original_sha256=old.sha(ROOT/'analysis_tools/final_presentation.py'),revision_sha256=old.sha(path),changed=['F1 connected workflow with dataset/network/fold/chromosome/grid symbols','F2C measured class-count bars','F4 identical pooled aggregation for single seeds and ensembles','F6 unconnected model-specific points']))

def citation_registry():
    records=json.loads((NOTES/'existing_reference_metadata.json').read_text(encoding='utf-8'))['records']
    additions=[]
    for file,key in [('data_sources.json','citations'),('evaluation.json','references'),('legacy_benchmark.json','entries')]:
        additions+=json.loads((NOTES/file).read_text(encoding='utf-8'))[key]
    registry={r['source_id']:r for r in records};dois={r.get('doi','').lower():r['source_id'] for r in records if r.get('doi')};aliases={}
    for row in additions:
        r=dict(row);sid=r.get('source_id',r.get('id'));doi=(r.get('doi') or '').lower()
        if doi and doi in dois:aliases[sid]=dois[doi];continue
        r['source_id']=sid;registry[sid]=r
        if doi:dois[doi]=sid
    dump('manuscript/verified_reference_registry.json',dict(references=registry,aliases=aliases,style='Numbered citations; first six structured authors followed by et al.; journal, year, volume and pages/article number; DOI or official proceedings URL.'))
    return registry,aliases

def editorial_parts():
    from analysis_tools.final_manuscript import content
    parts=content();v=old.jd('claim_audit/verified_values.json');single=read('final_tables/Single_seed_and_ensemble_comparison.csv').set_index('claim_id')
    def n(cid,key,decimals=4):return f'{v[cid][key]:.{decimals}f}'
    def update(i,text):parts[i]['text']=text
    update(0,'Enhancer prediction studies often compare models across benchmarks with different sources of dependence, training budgets and experimental outcomes. We audited supplied copies of a widely used enhancer benchmark, then evaluated grouped classification and quantitative reporter activity separately. In both classification tasks, 100 strong test records were identical to training records. After grouping and exclusions, the tasks retained 2,838 and 1,376 candidates. Three-seed DNABERT-2 and DNABERT ensembles differed in pooled AUROC by 0.0016 and 0.0125, with intervals spanning zero. Single-seed and ensemble results were reported separately. A 54-fit search compared DNABERT-2, LegNet and MPRAnn on HepG2 and K562 lentiMPRA elements. Longer DNABERT-2 training improved internal Pearson correlation by 0.0534 and 0.0501 over its own early training prefixes, but LegNet achieved higher correlation under the tested recipes. On an external DHS-natural cohort, all selected models retained positive rank correlations while producing negative R² values. These results show why benchmark independence, training adequacy and transfer calibration need separate evaluation. They support useful sequence-based prediction within the tested assays, while limiting claims of a general model ranking.')
    update(1,'Enhancers help control when and where genes are expressed. Predicting their activity from DNA sequence is therefore useful for interpreting regulatory variation and designing experiments. Reporter assays make this question measurable, but the measured outcome depends on the assay. Chromosomal integration, sequence length, promoter choice and cellular context can all change the activity assigned to a sequence [@inoue_2017;@klein_2020]. Large-scale lentiMPRA and regulatory-sequence design studies now provide quantitative measurements for testing sequence models [@mpra_legnet;@gosai2024]. Studies of cross-cell prediction, activity estimation and mutational effects further show why assay endpoints and source populations must be described precisely [@kreimer2019;@ashuach2019;@kircher2019].')
    update(2,'Legacy enhancer classification asks a different question: whether a sequence belongs to an enhancer class and, if so, whether it carries a strong or weak label. The iEnhancer-2L benchmark established this two-stage task [@ienhancer2016]. A later independent test set described with iEnhancer-EL became a recurring comparison set [@ienhancer_el_2018]. Published evaluations span word-embedding, convolutional, recurrent and adversarial models [@ienhancer_5step_2019;@ienhancer_ecnn_2019;@ienhancer_eblstm_2021;@ienhancer_gan_2021], followed by attention, multiscale and stacking approaches [@ienhancer_dcla_2022;@ienhancer_dcsv_2023;@ienhancer_dcsa_2023;@ienhancer_sknn_2023;@adh_enhancer_2024]. Transformer-based enhancer studies, including DNABERT-2, also precede this work [@bert_enhancer_2021;@ienhancer_elm_2023;@dnabert2_enhancer_2025]. This broad reuse makes the identity and independence of benchmark sequences consequential.')
    update(3,'Genomic data can remain related across a train–test split even when record names differ. Sequence similarity, composition and preprocessing can then distort apparent performance [@whalen2022;@bernett2024]. Cross-cell prediction introduces further opportunities for misleading evaluation [@schreiber2020]. Strong supervised controls and task-specific benchmarks are therefore essential when assessing genomic language models [@tang2025;@dart2024]. Training and model-selection budgets also matter: tuning can bias reported comparisons, and averaging predictions changes the object being compared [@cawley2010;@varma2006;@lakshminarayanan2017].')
    update(4,'We address these issues through a linked evaluation of supplied benchmark files, fixed similarity groups, exact-GC comparisons, repeated transformer training and quantitative activity prediction. The activity experiment compares a longer DNABERT-2 run with the exact early prefix of that same run, alongside validation-selected LegNet and MPRAnn recipes. We then assess external ranking and absolute prediction error. Figure 1 shows how the datasets and analyses connect. The aim is to establish which conclusions survive these checks and where the evidence remains limited.')
    update(5,'The supplied Stage 1 files contain 2,968 training and 400 test sequences; Stage 2 contains 1,484 and 200. In each task, 100 strong test records are identical to training records. Reverse-complement matching adds no further overlaps, and no duplicated weak-class test records were found (Table 2; Fig. 2A). The test-set design is widely cited through the iEnhancer-EL benchmark lineage [@ienhancer_el_2018;@ienhancer_ecnn_2019;@ienhancer_dcla_2022]. Our duplicate finding applies to the supplied copies we audited; we have not established that every published study used these exact DNA records.')
    update(6,'Removing the exact duplicate test records leaves 100 positives and 200 negatives in Stage 1. It leaves only 100 negatives in Stage 2, so cleaned Stage 2 AUROC is undefined (Fig. 2C). We could rescore the archived spaced-mer predictions on these subsets. The original historical neural implementations and prediction files were unavailable, which prevented the optional three-model leakage-impact experiment. The audit establishes overlap, but does not assign a numerical inflation effect to those historical neural results.')
    update(7,'Grouping related sequences and excluding components linked to the historical test leaves 2,838 candidates in 2,810 groups for Stage 1 and 1,376 candidates in 1,373 groups for Stage 2. On these fixed five-fold tasks, the DNABERT-2 three-seed ensemble achieves pooled AUROC 0.8753 for enhancer detection and 0.7117 for strong-versus-weak classification. The archived single-run spaced-mer baseline achieves 0.8507 and 0.7048 (Table 3). Detection is therefore easier than discrimination of the legacy strength labels in these cohorts.')
    parts[7]['claim_ids']+=['SINGLE_S1_finetuned_dnabert2','SINGLE_S2_finetuned_dnabert2']
    update(8,'The ensemble comparison includes a benefit from averaging. Across individual DNABERT-2 seeds, mean pooled AUROC is 0.8651 (SD 0.0021) in Stage 1, or 0.0144 above the single-run spaced-mer baseline; the ensemble gap is 0.0245. In Stage 2, the individual-seed mean is 0.6912 (SD 0.0237), below the spaced-mer score of 0.7048, while the ensemble reaches 0.7117. Table 3 and Figure 4 place these values together using pooled AUROC throughout. Mean-fold AUROC is reported separately. These are comparisons of fitted implementations and aggregation choices, rather than an isolated effect of architecture.')
    parts[8]['claim_ids']+=['SINGLE_S1_finetuned_dnabert2','SINGLE_S2_finetuned_dnabert2']
    update(9,'Composition explains part of the apparent Stage 1 discrimination. For the GC/CpG logistic baseline, mean-fold AUROC falls from 0.8313 to 0.5422 when comparison is restricted to opposite-label pairs with identical GC count. The spaced-mer baseline changes from 0.8513 to 0.6559. DNABERT-2 retains exact-GC AUROC 0.7391 in Stage 1 and 0.6666 in Stage 2 (Fig. 3). Matching changes which pairs can be compared and controls GC count only, so these differences describe sensitivity to the evaluation design rather than a causal decomposition of sequence features.')
    update(10,'Repeated training produces modest separation between the DNABERT generations. The mean of the per-seed mean-fold AUROCs is 0.8733 for DNABERT-2 and 0.8722 for DNABERT in Stage 1; the corresponding Stage 2 values are 0.7174 and 0.6944. Their pooled ensemble differences are 0.0016 (95% CI −0.0055 to 0.0090) and 0.0125 (95% CI −0.0074 to 0.0317). Both intervals span zero, and both adjusted p-values are 1.000. This comparison does not resolve a clear ensemble advantage for either generation (Fig. 4).')
    update(11,'Seed variation and fold variation are shown separately in Supplementary Table S2. The folds share training data, so the 15 seed-by-fold entries are not 15 independent replications. We also report reliability plots and fixed-threshold confusion statistics: AUROC alone does not show whether predicted probabilities are calibrated or whether both classes are recognized at the chosen threshold [@guo2017;@saito2015] (Supplementary Figs. SF1–SF2).')
    update(12,'The earlier label-transfer experiment shows limited transfer from legacy enhancer labels to measured activity. Without activity-specific adaptation, the older DNABERT-2 classifier scores correlate with top/bottom-cohort MPRA activity at Spearman 0.1856 in HepG2 and 0.1285 in K562. DNABERT reaches 0.2255 and 0.1728. These older fits and cohorts differ from the current repeated classification and regression studies, so the comparison motivates direct activity modelling without measuring a causal effect of changing the training task.')
    update(13,'Direct regression predicts measured activity in the lentiMPRA data from Agarwal et al. [@mpra_legnet]. The chromosome-held-out tests contain 9,048 HepG2 and 13,145 K562 elements. Selected DNABERT-2 ensembles achieve Pearson correlations 0.6438 and 0.6921; LegNet achieves 0.6827 and 0.7177, and MPRAnn 0.6525 and 0.6828. Table 4 also reports rank correlation and absolute errors. The archived spaced-mer regression model remains a separate single-recipe reference, outside the 54-fit neural search (Fig. 5).')
    update(14,'Longer training improves DNABERT-2 relative to its own early prefix. At the fixed learning rate, internal Pearson correlation increases by 0.0534 (95% CI 0.0460 to 0.0613) in HepG2 and 0.0501 (95% CI 0.0438 to 0.0565) in K562. Both adjusted p-values are 0.0014. Because each comparison uses the same run up to its preserved early checkpoint, it directly addresses the earlier short training budget (Fig. 5).')
    update(15,'LegNet still achieves higher internal Pearson correlation after validation-only recipe selection. Its differences from DNABERT-2 are 0.0389 (95% CI 0.0283 to 0.0494) in HepG2 and 0.0255 (95% CI 0.0185 to 0.0325) in K562, with adjusted p-values 0.0014. The DNABERT-2–MPRAnn comparison is less decisive. Its HepG2 interval spans zero; the K562 interval is positive before multiplicity adjustment, but the adjusted p-value is 0.0938 (Table 6).')
    update(16,'The search nevertheless leaves room for better recipes. Six of the 18 selected fits show signs of a limited epoch budget, and LegNet and MPRAnn select opposite boundaries of their learning-rate grids. Two high-rate DNABERT-2 fits are practically near constant, although neither meets the original undefined-Pearson flag. All fits remain included under the locked selection rule. Across nine candidate/seed fits per cell, recorded DNABERT-2 fitting takes 8.06 and 5.98 hours, versus 0.115 and 0.128 hours for LegNet. Equal candidate counts therefore do not provide equal compute or establish convergence (Supplementary Tables S7, S16–S17).')
    update(17,'External evaluation separates rank preservation from accurate activity prediction. We use the frozen processed DHS-natural subset from Gosai et al., containing 8,399 sequences per cell [@gosai2024]. DNABERT-2 reaches Spearman correlations 0.5084 in HepG2 and 0.5270 in K562; LegNet reaches 0.5519 and 0.5579, and MPRAnn 0.5242 and 0.5348. The paired comparisons favor LegNet over DNABERT-2 in both cells (Table 6; Fig. 6A).')
    update(18,'All six selected-model external R² values are negative, with negative mean prediction errors. Longer DNABERT-2 training improves external ranking but worsens absolute error in both cells. A model can therefore rank these elements usefully while predicting their measured activity poorly (Table 5; Fig. 6B). This previously examined cohort provides an assay-transfer analysis, rather than a new confirmatory holdout.')
    update(19,'The original overlap screen retains 76,833 of 149,632 paired-assay candidates and removes 72,799, mostly because of shared short words. Older fitted-model analyses compare retained and removed elements on the test chromosomes. For DNABERT-2, removed-minus-retained MAE is 0.0049 (95% CI −0.0062 to 0.0157) in HepG2 and 0.0031 (95% CI −0.0022 to 0.0083) in K562. Both intervals span zero. These transport comparisons describe the populations excluded by filtering; they do not test retraining on the complete population (Supplementary Tables S10–S12).')
    update(20,'The older CNN diagnostics identify a clearer failure at high measured activity. LegNet underpredicts retained high-activity elements by a mean 0.7255 activity units in HepG2 and 0.3595 in K562. Agreement across seeds and acceptable overall interval coverage do not prevent this tail error. Figure 6C–D and Supplementary Tables S13–S14 retain the original fitted models and populations for these diagnostics.')
    update(21,'The benchmark audit shows why sequence identity belongs at the start of evaluation. A test record is not independent merely because it has a different name. General genomic leakage guidance makes the same point about dependence and preprocessing across partitions [@whalen2022;@bernett2024;@kapoor2023]. Here, the overlap is directly verifiable in the supplied files. The extensive published use of this benchmark lineage makes that audit relevant, but does not establish that every published implementation used the same copies. Resolving the difference between the supplied records and the originally claimed independent test requires source-level provenance.')
    update(22,'The model comparisons are also sensitive to what is averaged and how training is budgeted. DNABERT-2 has a smaller Stage 1 advantage over spaced k-mer when individual seeds are compared, and its Stage 2 single-seed mean falls below that baseline. Extra training improves its activity predictions, yet LegNet remains stronger under the selected internal recipes. These results agree with recent evidence that specialized supervised models remain important controls for genomic language models [@tang2025;@dart2024]. They support reporting seed, ensemble and recipe effects separately.')
    parts[22]['claim_ids']+=['SINGLE_S1_finetuned_dnabert2','SINGLE_S2_finetuned_dnabert2']
    update(23,'Legacy enhancer labels and reporter activity should be treated as separate outcomes. One describes a curated classification task; the other measures activity in a specified assay and cell type. Weak score transfer between them does not show that sequence lacks activity information. Conversely, successful reporter prediction does not establish native enhancer function, which depends on promoter contacts and cellular context [@inoue_2017;@klein_2020;@fulco_2019]. This distinction helps explain why useful internal regression and imperfect external transfer can coexist.')
    update(24,'External ranking and calibration convey different information. Positive rank correlation can coexist with errors larger than those of a constant external-cohort mean predictor, as reflected in the negative R² values here. The older high-activity and interval diagnostics reinforce this separation: overall coverage or agreement between models does not guarantee accurate predictions for an individual subgroup [@barber2021;@barber2023]. Transfer evaluation should therefore retain errors and signed bias alongside correlations.')
    update(25,'The scope remains limited to short sequences, two activity cell types and previously examined cohorts. Only three training seeds and a finite recipe grid were available. The reported intervals and tests describe fixed fitted models on resampled groups, without including uncertainty from selection or retraining. The final contrast family was chosen after earlier outcomes had been seen; Holm adjustment cannot make the study prospectively confirmatory [@bates2024;@dwork2015]. Selected budget limits, grid boundaries, practical near-constant fits and the unresolved earlier locus-split collapse also prevent a general convergence claim. Longer contexts, broader cell types and mechanistic interpretation require separate studies.')
    update(26,'We tracked source files, processed records, split assignments, predictions and checkpoints by SHA256. Multiline FASTA sequences were validated before exact and reverse-complement identity checks. The classification graph links pairs whose longest common subsequence is at least 160 bases in either orientation, then uses connected components to keep related records together. This is a grouping rule, not an 80% local-identity threshold. Components containing historical test records are excluded from the candidate training set. Fixed five-fold assignments and within-fold exact-GC comparisons are described in Supplementary Methods.')
    update(27,'The repeated classification study uses pinned DNABERT and DNABERT-2 revisions with masked mean pooling and a two-class linear head [@dnabert_original;@dnabert2]. Each model is trained for two epochs on each fixed fold with three base seeds: 20260928, 20261016 and 20261017. AdamW uses learning rate 2e-5, batch size 8 and gradient accumulation 2. Reverse complements augment training after splitting. Scoring uses the original orientation, the fixed final second epoch and threshold 0.5. Seven archived baselines retain their original single-run recipes, including frozen GENA-LM [@gena2025]. For each current transformer, we report the mean and SD of per-seed pooled AUROC alongside the AUROC obtained after averaging the three seeds’ record-level scores.')
    update(28,'The activity search compares DNABERT-2 with copied human MPRA LegNet and MPRAnn implementations [@mpra_legnet;@penzar2023legnet]. Each model and cell receives three learning-rate candidates and three seeds, giving 54 fits. Targets are normalized using the training-set mean and population SD. DNABERT-2 permits 30 epochs with patience 5 and minimum 8; LegNet uses a 20-epoch OneCycle schedule; MPRAnn permits 100 epochs with patience 10. Each fit retains its best finite validation Pearson checkpoint. The learning rate is selected by mean validation Pearson over all three seeds, with fixed tie rules. Recipe and checkpoint hashes are locked before test or external scoring.')
    update(29,'Each DNABERT-2 fit also retains the best validation checkpoint from its first two epochs. The duration control compares this exact early prefix with the longer run at learning rate 2e-5. Activity scoring averages forward and reverse-complement predictions, then averages the three seeds. Internal evaluation uses the frozen chromosome test partition. External evaluation uses the processed DHS-natural cohort; older origin, filtering and reliability analyses retain their original fitted models. Spearman correlation is the external headline, with Pearson, MAE, RMSE, R² and mean signed error reported alongside it.')
    update(30,'The final synthesis contains 14 headline contrasts. We calculate 95% percentile intervals from 2,000 paired bootstrap samples of whole groups: similarity components for classification, locus groups for internal activity and canonical-DNA groups for external transfer. Spearman ranks are recalculated when resampling duplicates observations. For paired permutation tests, we exchange the two models’ prediction vectors within randomly selected whole groups 9,999 times and compare the resulting differences with the observed difference. This test assumes the fixed predictions can be exchanged within groups under its null; it does not test equality of the learning algorithms or include retraining uncertainty. We use the plus-one Monte Carlo correction and Holm adjustment across the 14 contrasts [@holm1979]. Full assumptions, seed/fold variability and the earlier 1 kb-region sensitivity are in Supplementary Methods. The preserved checkpoint audits reproduce 60 classification checkpoints, 72 activity validation checkpoints and 60 activity evaluation sets. Reporting maps follow DOME and REFORMS guidance [@dome2021;@reforms2024].')
    # Insert source attribution as its own Methods paragraph, retaining exact
    # accessions as prose identifiers rather than inventing a dataset source.
    parts.insert(26,dict(section='Methods|Data sources and activity measurements',text='Quantitative training measurements come from the large-scale lentiMPRA study of Agarwal et al. [@mpra_legnet]. We used the GRCh38 ENCODE element-quantification files ENCFF774DYO (HepG2, experiment ENCSR022GQD) and ENCFF631LHA (K562, experiment ENCSR382BVV). Their live ENCODE MD5 values match the preserved source records. The external source is Supplementary Table 12 from Gosai et al. [@gosai2024]. Its 12,000 DHS-natural elements were processed with the frozen quality and overlap screens to give the previously examined 8,399-element cohort per cell. We use measured log2-fold-change columns rather than the source study’s prediction columns. Dataset URLs, checksums, orientation averaging and inclusion rules are given in Supplementary Methods and the data-source manifest.',claim_ids=['DATA_SOURCE_ACCESSIONS','GOSAI_SOURCE_COUNTS']))
    parts.append(dict(section='Declarations|Generative AI use',text='OpenAI Codex (OpenAI) and Claude (Anthropic) assisted with this project. Claude-assisted materials included an earlier MPRA notebook, the project handbook, the collapse-count erratum and results organization. OpenAI Codex assisted with subsequent auditing, analysis and reporting code, figures, tables and manuscript drafting or editing. The authors retain responsibility for the final scientific content and disclosures.',claim_ids=['AI_USE_SCOPE']))
    # A natural scientific register uses ordinary hyphens, not a uniform sequence
    # of rhetorical dashes. Minus signs in numeric intervals remain mathematical.
    for part in parts:part['text']=part['text'].replace('–','-').replace('—',';')
    return parts

def resolve_references(parts):
    registry,aliases=citation_registry();order=[]
    def key(sid):return aliases.get(sid,sid)
    for part in parts:
        for group in re.findall(r'\[@([^]]+)\]',part['text']):
            for sid in group.split(';@'):
                sid=key(sid);assert sid in registry,sid
                if sid not in order:order.append(sid)
    assert len(order)>=40,len(order)
    def authors(row):
        labels=[]
        for a in row['authors'][:6]:
            if a.get('literal') or a.get('type')=='collective':labels.append(a.get('literal') or a.get('name') or a.get('family'));continue
            initials=''.join(w[0] for w in re.findall(r'[^\W\d_]+',a.get('given',''),flags=re.U)).upper()
            labels.append(a['family']+' '+initials)
        return ', '.join(labels)+(', et al' if len(row['authors'])>6 else '')
    refs=[]
    for number,sid in enumerate(order,1):
        r=dict(registry[sid]);r.update(source_id=sid,reference_number=number,authors_display=authors(r))
        journal=r.get('journal') or r.get('booktitle') or r.get('venue') or r.get('container_title') or ''
        pages=r.get('pages') or r.get('article_number') or '';volume=r.get('volume') or ''
        if r.get('booktitle') and str(volume)==str(r['year']):volume=''
        details=str(r['year'])+(';' +str(volume) if volume else '')+(':'+str(pages).replace('–','-') if pages else '')
        url=r.get('primary_url') or r.get('url') or r.get('metadata_source_url') or ''
        link='https://doi.org/'+r['doi'] if r.get('doi') else url
        title=r['title'].strip().rstrip('.').strip()
        r['formatted_reference']=f'{number}. {r["authors_display"]}. {title}. {journal}. {details}. {link}'
        r['url']=url;refs.append(r)
    for part in parts:
        part['text']=re.sub(r'\[@([^]]+)\]',lambda m:'['+', '.join(str(order.index(key(sid))+1) for sid in m.group(1).split(';@'))+']',part['text'])
    dump('manuscript/reference_metadata.json',refs)
    table('manuscript/references.csv',pd.DataFrame([{k:r.get(k) for k in ['reference_number','source_id','authors_display','title','journal','year','volume','issue','pages','article_number','doi','url','formatted_reference']} for r in refs]))
    return refs

def build_word_module():
    source=(ROOT/'analysis_tools/final_manuscript.py').read_text(encoding='utf-8')
    source=source.replace('26.1011.11113','26.1013.11914')
    start=source.index('        authors=str(r[\'authors\'])')
    end=source.index("        p=doc.add_paragraph(entry)",start)
    source=source[:start]+"        entry=r['formatted_reference']\n"+source[end:]
    begin=source.index("                add_table(pd.DataFrame({'Model':a.model_label")
    end=source.index("\n        elif r.table.startswith('T4')",begin)
    source=source[:begin]+'''                single=a.single_seed_pooled_auroc_mean.map(fmt)
                current=a.n_seeds.eq(3)
                single.loc[current]=a.loc[current].single_seed_pooled_auroc_mean.map(fmt)+' ('+a.loc[current].single_seed_pooled_auroc_sd.map(fmt)+')'
                ensemble=a.three_seed_ensemble_pooled_auroc.map(lambda x:'n.a.' if pd.isna(x) else fmt(x))
                add_table(pd.DataFrame({'Model':a.model_label,'Seeds':a.n_seeds,'Mean-fold AUROC':a.mean_fold_auroc.map(fmt),'Single-seed pooled AUROC mean (SD)':single,'Three-seed ensemble pooled AUROC':ensemble,'Exact-GC AUROC':a.exact_GC_mean_fold_AUROC.map(fmt)}))
''' +source[end:]
    source=source.replace("if list(frame.columns)[:2]==['Model','Seeds']:widths=[2.25,.55]+[1.15]*6","if list(frame.columns)[:2]==['Model','Seeds']:widths=[2.15,.55,1.4,2.1,2.1,1.4]")
    source=source.replace("doc=Document();sec=doc.sections[0]","doc=Document()\n    for name in ['Title','Heading 1','Heading 2','Caption']:\n        pp=doc.styles[name].element.find(qn('w:pPr'))\n        if pp is not None:\n            for border in list(pp.findall(qn('w:pBdr'))):pp.remove(border)\n    sec=doc.sections[0]")
    source=source.replace("t=doc.add_table(rows=1,cols=len(frame.columns));t.autofit=False","t=doc.add_table(rows=1,cols=len(frame.columns));t.autofit=False\n        borders=OxmlElement('w:tblBorders')\n        for edge in ['top','left','bottom','right','insideH','insideV']:\n            e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)\n        t._tbl.tblPr.append(borders)")
    source=source.replace("cells[j].text='—' if pd.isna(value) else str(value)","cells[j].text='n.a.' if pd.isna(value) else f'{value:,}' if isinstance(value,int) else str(value)")
    source=source.replace("if i==0:\n                header=", "if i==0:\n                for cell in row.cells:\n                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E9EDF1');cell._tc.get_or_add_tcPr().append(shade)\n                header=")
    path=OUT/'analysis_tools/revised_word_builder.py';path.write_text(source,encoding='utf-8')
    shutil.copy2(ROOT/'analysis_tools/export_final_docx_word.ps1',OUT/'analysis_tools/export_final_docx_word.ps1')
    spec=importlib.util.spec_from_file_location('revised_word_builder',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=OUT;module.s=S
    return module

def claims(parts):
    base=old.read('claim_audit/claim_evidence_ledger.csv');extra=read('claim_audit/revision_numeric_claims.csv');rows=[]
    for r in extra.to_dict('records'):
        item={c:None for c in base.columns};item.update(claim_id=r['claim_id'],status='VERIFIED')
        if 'claim' in item:item['claim']='Same-aggregation single-seed and ensemble pooled AUROC comparison'
        if 'source' in item:item['source']='phase32 independently verified predictions and frozen ensemble results'
        rows.append(item)
    for cid,description,status in [('DATA_SOURCE_ACCESSIONS','Live ENCODE identity/MD5 matches frozen input provenance','VERIFIED'),('GOSAI_SOURCE_COUNTS','Primary Table12 byte hash matches frozen provenance;12000 DHS-natural source elements and processed8399 per cell','VERIFIED'),('AI_USE_SCOPE','Tool scope supplied in user feedback; versions and human author review remain to confirm','SUPPORTED_WITH_LIMIT')]:
        item={c:None for c in base.columns};item.update(claim_id=cid,status=status)
        if 'claim' in item:item['claim']=description
        rows.append(item)
    ledger=pd.concat([base,pd.DataFrame(rows)],ignore_index=True);assert not ledger.claim_id.duplicated().any()
    table('claim_audit/claim_evidence_ledger.csv',ledger)
    checks=[]
    for i,p in enumerate(parts,1):
        for cid in p['claim_ids']:assert cid in set(ledger.claim_id),cid
        checks.append(dict(paragraph=i,section=p['section'],claim_ids=';'.join(p['claim_ids']),numeric_evidence_reused_or_independently_verified=True,review_status='assistant editorial draft; human section review pending'))
    table('claim_audit/actual_manuscript_paragraph_checks.csv',pd.DataFrame(checks))
    dump('claim_audit/revision_sources.json',dict(source_prediction_sha256=old.sha(ROOT/'phase32_outputs/independently_verified_predictions.csv.gz'),single_seed_metric_sha256=old.sha(ROOT/'phase32_outputs/seed_metrics_pooled_and_fold.csv'),frozen_statistical_policy_sha256=old.sha(ROOT/'final_statistics/statistical_methods.json'),dataset_source_metadata='research_notes/Enhancer manuscript citation corrections/data_sources.json',Gosai_original_table_sha256='1dcc68b0340933acf593c6bbbc8a8ce2d7aaae60334eebf353958a0f0459fdae',AI_scope='User-provided feedback,15 October2026',no_new_hypothesis_tests=True))

def manuscript():
    runtime();parts=editorial_parts();refs=resolve_references(parts);claims(parts)
    dump('manuscript/manuscript_paragraphs.json',parts)
    from analysis_tools.final_manuscript import TITLE
    md='# '+TITLE+'\n\n';current=None;sections={}
    for part in parts:
        sec=part['section'];main,_,sub=sec.partition('|')
        if current!=sec:
            if current is None or current.partition('|')[0]!=main:md+='## '+main+'\n\n'
            if sub:md+='### '+sub+'\n\n'
            current=sec
        md+=part['text']+'\n\n';sections.setdefault(sec,[]).append(part['text'])
    md+='## References\n\n'+'\n\n'.join(r['formatted_reference'] for r in refs)
    text('manuscript/Manuscript.md',md)
    for i,(section,paragraphs) in enumerate(sections.items(),1):
        name=f'{i:02d}_'+re.sub(r'[^A-Za-z0-9]+','_',section).strip('_')+'.md'
        text('manuscript/sections/'+name,'# '+section.replace('|',' / ')+'\n\n'+'\n\n'.join(paragraphs))
    table('manuscript/SECTION_REVIEW.csv',pd.DataFrame([dict(section=sec,editorial_draft_complete=True,human_review_status='pending',reviewer='',decision='',comments='') for sec in sections]))
    text('manuscript/EDITORIAL_REVIEW.md','''# Section review

This is an assistant-edited scientific draft, not a completed human rewrite or approved manuscript. The section files and review worksheet let Aditta and the author team review one section at a time. Review the dataset and benchmark attribution first, then the single-seed/ensemble distinction, Results narrative, Discussion interpretation and Methods. The main Results lead with findings; detailed uncertainty assumptions and repeated limits are consolidated in Methods, Discussion and the supplement. No author review or authorship role is inferred from preparing these files.
''')
    # These are edits to the existing document format. Mark this authoring step
    # once per requested format, using the current installed bundled tools.
    for fmt,skill in [('docx','documents'),('pdf','pdf')]:
        marker=ROOT/f'revision49_outputs/{fmt}_edit_operation.json'
        if not marker.exists():
            script=Path(r'C:\Users\User\.codex\plugins\cache\openai-primary-runtime')/skill/'26.1013.11914/skills'/skill/'container_tools/mark_artifact_operation_started.mjs'
            subprocess.run([str(NODE),str(script),'--operation-kind','edit','--expected-output-count','1','--output-format',fmt],cwd=ROOT,env=os.environ.copy(),check=True)
            old.dump(marker,dict(operation='edit',format=fmt,marker_succeeded=True))
    builder=build_word_module();builder.word_document(parts,refs,'manuscript/Manuscript.docx')
    pages=builder.render_docx('manuscript/Manuscript.docx','QA/manuscript')
    dump('manuscript/draft_summary.json',dict(paragraphs=len(parts),references=len(refs),word_count=sum(len(p['text'].split()) for p in parts),rendered_pages=pages,human_review_pending=True,journal_neutral=True))
    print('Revised manuscript rendered:',pages,'pages,',len(refs),'references.',flush=True)

def clean_prose(value):
    replacements={'Primary9B':'Primary 9B','primary9B':'primary 9B','training,5,611':'training, 5,611','23,560,6,804':'23,560 training, 6,804 validation','Stage1':'Stage 1','Stage2':'Stage 2','TableS':'Table S','TablesS':'Tables S','FigureSF':'Figure SF','nominal90%':'nominal 90%','All54':'All 54','All3':'All 3','all54':'all 54','current54':'current 54','frozen0':'frozen 0','practical1':'practical 1','Phase35':'the current study','Phases16–24':'earlier studies','Phase41':'the final synthesis','Phases 16–24':'earlier studies','Phase11A':'the archived baseline study','historical9C':'historical 9C','this9B':'this 9B','LR5e-5':'LR 5e-5','seed20260932':'seed 20260932','SD<1e-4':'SD < 1e-4','by2':'by 2','label1':'label 1','label0':'label 0','contributes1':'contributes 1','contributes0':'contributes 0','least160':'least 160','an80%':'an 80%','fixed-LR2e-5':'fixed LR 2e-5','first-two-epoch':'first two-epoch'}
    for a,b in replacements.items():value=value.replace(a,b)
    value=re.sub(r'\b(a|the|all|within|length|with|of|to|at|from|contains|batch|patience|maximum|minimum|dropout|accumulation|clipping|rate|decay|use|uses|over|than|at least)(?=\d)',r'\1 ',value)
    value=re.sub(r'(\d)bp\b',r'\1 bp',value)
    for a,b in [('fromENCSR','from ENCSR'),('cells;76,833','cells; 76,833'),('as72,799','as 72,799'),('kernels7','kernels 7'),('andS17','and S17'),('Percentile95%','Percentile 95%'),('original1kb','original 1 kb'),('Fixed0.5','Fixed 0.5'),('Table6','Table 6'),('below20','below 20'),('Historical9C','Historical 9C'),('the new the final synthesis family','the final synthesis family'),('the same authenticated the archived baseline study historical predictions','the same authenticated archived baseline predictions')]:value=value.replace(a,b)
    value=value.replace('not new the current study selected models','distinct from the current selected models').replace('independently match Phase3','independently match the archived analysis')
    return value

def supplement():
    runtime();methods=(ROOT/'final_supplement/Supplementary_Methods.md').read_text(encoding='utf-8')
    methods=clean_prose(methods)
    source='''## Source datasets and accessions

The training measurements are the large-scale lentiMPRA libraries described by Agarwal et al. (2025), doi:10.1038/s41586-024-08430-9. The HepG2 input is ENCODE file ENCFF774DYO from experiment ENCSR022GQD, https://www.encodeproject.org/files/ENCFF774DYO/. The K562 input is ENCFF631LHA from ENCSR382BVV, https://www.encodeproject.org/files/ENCFF631LHA/. Both are GRCh38 element-quantification BED files. Their MD5 values, 87b3fb363f64aeb2b5f893ff98b47db3 and 847512c673c75874e84200f84899f880 respectively, match the frozen source provenance. The assays quantify reporter activity after lentiviral integration; this is distinct from measuring each element at its native genomic locus.

The external source is Gosai et al. (2024), Machine-guided design of cell-type-targeting cis-regulatory elements, Nature 634, 1211-1220, doi:10.1038/s41586-024-08070-z, Supplementary Table 12. The independently retrieved table matches the stored SHA256 1dcc68b0340933acf593c6bbbc8a8ce2d7aaae60334eebf353958a0f0459fdae. It contains 12,000 DHS_natural source elements. Frozen quality and overlap screens produce the previously examined 8,399-element cohort per cell used here. Activity endpoints come from measured log2-fold-change columns, not the source study's predicted scores. Source URLs, experimental origins and local processing choices are separate provenance fields.
'''
    first=methods.find('\n## ');methods=methods[:first]+'\n\n'+source+methods[first:]
    methods+='''\n\n## Single seeds and ensembles

For a single-seed summary, calculate AUROC after pooling all fixed out-of-fold predictions for that seed, then report the mean and sample SD over the three seeds. For an ensemble, average the three seed scores for each record first, then calculate pooled AUROC. These are different operations. The archived spaced-mer baseline has one fitted recipe, so no seed SD is manufactured. Stage 1 DNABERT-2 has single-seed pooled mean 0.8651 (SD 0.0021), ensemble pooled AUROC 0.8753 and spaced-mer pooled AUROC 0.8507. The corresponding descriptive gaps are 0.0144 and 0.0245. Stage 2 has single-seed mean 0.6912 (SD 0.0237), ensemble 0.7117 and spaced-mer 0.7048. These additions use the saved predictions and add no hypothesis tests to the frozen 14-contrast family. Mean-fold AUROC remains separately labelled in Table 3 and Supplementary Table S2.
'''
    methods=clean_prose(methods)
    methods=methods.replace('The null is sharp conditional exchangeability of the two fitted prediction vectors within groups, not equality or equivalence of learning algorithms.','Under the test null, the two models\' fixed prediction vectors are exchangeable within each whole group. This is a test about those predictions under the grouping assumption, rather than equality of learning algorithms. It does not incorporate model selection or retraining.')
    methods=methods.replace('P-values use 9,999 whole-group paired prediction swaps','P-values use 9,999 paired permutation draws, exchanging the two models\' predictions within whole groups')
    text('final_supplement/Supplementary_Methods.md',methods)
    captions=read('final_figures/figure_index.csv');notes=(ROOT/'final_supplement/Supplementary_Result_Notes.md').read_text(encoding='utf-8')
    notes=notes.split('## Supplementary figure captions')[0]
    notes=clean_prose(notes)+'\n\n## Supplementary figure captions\n\n'+'\n\n'.join('### '+r.figure.split('_')[0]+' '+r.title+'\n\n'+clean_prose(r.caption) for r in captions[captions.placement.eq('supplement')].itertuples())
    text('final_supplement/Supplementary_Result_Notes.md',notes)
    text('final_supplement/Supplementary_Information_text.md',methods+'\n\n'+notes)
    from analysis_tools import final_documents as docs
    savedroot=docs.ROOT;docs.ROOT=OUT
    try:docs.render_text_pdf('final_supplement/Supplementary_Information_text.md','QA/Supplementary_text.pdf')
    finally:docs.ROOT=savedroot
    from pypdf import PdfWriter,PdfReader
    writer=PdfWriter();writer.append(OUT/'QA/Supplementary_text.pdf')
    for r in captions[captions.placement.eq('supplement')].itertuples():writer.append(OUT/r.pdf)
    with (OUT/'final_supplement/Supplementary_Information.pdf').open('wb') as f:writer.write(f)
    dump('QA/supplement_summary.json',dict(pages=len(PdfReader(OUT/'final_supplement/Supplementary_Information.pdf').pages),workbook_unchanged_sha256=old.sha(OUT/'final_supplement/Supplementary_Tables.xlsx'),supplementary_tables=18,source_attribution_added=True,extra_comparison_csv=True))
    print('Revised supplement built.',flush=True)

SINGLE_SCRIPT='''from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_tools import final_synthesis as s
s.verify_inputs();p=pd.read_csv(s.PACKAGE/'predictions/classification_current_all_seeds.csv.gz',float_precision='round_trip');expected=pd.read_csv(s.PACKAGE/'tables/Single_seed_and_ensemble_comparison.csv',float_precision='round_trip');rows=[]
for (stage,model),g in p.groupby(['stage','model']):
    model={'dnabert2':'finetuned_dnabert2','dnabert_legacy':'finetuned_dnabert_legacy'}.get(model,model)
    values=[roc_auc_score(h.label,h.score) for _,h in g.groupby('base_seed')]
    r=expected[expected.stage.eq(stage)&expected.model.eq(model)].iloc[0]
    for got,want in [(np.mean(values),r.single_seed_pooled_AUROC_mean),(np.std(values,ddof=1),r.single_seed_pooled_AUROC_SD),(np.mean(values)-r.spaced_mer_single_pooled_AUROC,r.single_seed_mean_minus_spaced_mer)]:assert abs(got-want)<1e-12
    rows.append(dict(stage=int(stage),model=model,single_seed_mean=float(np.mean(values)),single_seed_SD=float(np.std(values,ddof=1)),single_seed_minus_spaced_mer=float(np.mean(values)-r.spaced_mer_single_pooled_AUROC)))
s.dump('single_seed_validation.json',dict(passed=True,rows=rows,summaries=4,aggregation='pooled OOF per seed, not mean-fold',new_hypothesis_tests=0))
print('All four single-seed summaries reproduced from saved predictions.')
'''

def release():
    """Reproduce affected reporting only; preserve the full previous tests."""
    from analysis_tools.final_delivery import seal,manifest
    package=OUT/'public_release/enhancer-leakage-study';original=ROOT/'public_release/enhancer-leakage-study'
    shutil.copytree(original,package,dirs_exist_ok=True,ignore=shutil.ignore_patterns('reproduced','.runtime','__pycache__'))
    for p in (OUT/'final_figures').rglob('*'):
        if p.is_file() and p.suffix in ['.png','.svg','.pdf','.md','.csv']:
            dest=package/'figures'/p.relative_to(OUT/'final_figures');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            if p.suffix=='.png' or p.name=='figure_index.csv':
                dest=package/'inputs'/p.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in (OUT/'final_tables').rglob('*.csv'):
        dest=package/'tables'/p.relative_to(OUT/'final_tables');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    (package/'inputs/final_tables/main').mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUT/'final_tables/main/T3_Classification.csv',package/'inputs/final_tables/main/T3_Classification.csv')
    shutil.copy2(OUT/'analysis_tools/final_presentation.py',package/'analysis_tools/final_presentation.py')
    shutil.copy2(OUT/'analysis_tools/final_presentation.py',package/'docs/source_snapshots/revised_final_presentation.py')
    shutil.copy2(OUT/'final_supplement/Supplementary_Methods.md',package/'docs/reproducibility.md')
    shutil.copy2(OUT/'claim_audit/claim_evidence_ledger.csv',package/'docs/claim_evidence_ledger.csv')
    shutil.copy2(OUT/'manuscript/references.csv',package/'docs/references.csv')
    for file in ['data_sources.json','evaluation.json','legacy_benchmark.json','existing_reference_metadata.json']:
        target=package/'docs/citation_evidence'/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(NOTES/file,target)
    (package/'scripts/single_seed_comparison.py').write_text(SINGLE_SCRIPT,encoding='utf-8')
    with (package/'README.md').open('a',encoding='utf-8') as f:f.write('\n## Editorial revision\n\nThe revision adds verified dataset attribution, a consistent bibliography, individual-seed versus ensemble reporting, and revised Figures 1, 2, 4 and 6. Run `python -B -X utf8 scripts/single_seed_comparison.py` for the new descriptive summaries. No training, selection or inferential-family rule changes. The statistical implementation and its fixed prediction inputs match the previous full-statistics validation.\n')
    # Confirm that reusing the previous 2,000/9,999-draw validation is justified.
    unchanged=['analysis_tools/metric_functions.py','scripts/metrics.py','inputs/final_statistics/statistical_methods.json','inputs/final_statistics/classification_predictions.csv.gz','inputs/phase37_outputs/verified_ensemble_predictions.csv.gz','inputs/final_statistics/adjusted_tests.csv']
    preserved=[dict(path=p,sha256=old.sha(original/p)) for p in unchanged]
    for r in preserved:assert old.sha(package/r['path'])==r['sha256']
    (package/'MANIFEST.json').write_text(json.dumps(manifest(package),indent=2)+'\n',encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',MPLCONFIGDIR=str(package/'.runtime/matplotlib'))
    Path(env['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
    tests=[]
    for script in ['metrics.py','single_seed_comparison.py','figures.py']:
        log=OUT/'QA'/f'portable_{script}.log'
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-B','-X','utf8',str(package/'scripts'/script)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,log
        tests.append(dict(script=script,exit_code=0))
    validations={file:json.loads((package/'reproduced'/file).read_text(encoding='utf-8')) for file in ['metric_validation.json','single_seed_validation.json','figure_validation.json']}
    assert all(r['passed'] for r in validations.values())
    report=dict(passed=True,tests=tests,validation=validations,unchanged_full_statistics_sources=preserved,full_statistics_rerun_in_revision=False,prior_full_statistics_validation_sha256=old.sha(ROOT/'phase47_outputs/reproducibility_validation.json'),new_training=False,source_snapshots_preserved=True)
    dump('QA/revision_reproducibility.json',report)
    (package/'docs/REVISION_REPRODUCIBILITY_TEST.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    # Rebuild the manifest after adding the new validation report.
    result=seal(package,ROOT/'revision49_outputs/Enhancer_Reproducibility_Release_Candidate_v3.zip')
    dump('release_package.json',result)
    print('Revised portable figures and single-seed summaries reproduced.',flush=True)

def assemble():
    """Assemble the review copy. No author approval or submission is asserted."""
    folder=OUT/'submission_package'
    from pypdf import PdfWriter,PdfReader
    writer=PdfWriter();writer.append(OUT/'QA/manuscript/Manuscript.pdf')
    for r in read('final_figures/figure_index.csv').query('placement=="main"').itertuples():writer.append(OUT/r.pdf)
    with (OUT/'manuscript/Manuscript.pdf').open('wb') as f:writer.write(f)
    for source,target in [('manuscript/Manuscript.docx','Manuscript.docx'),('manuscript/Manuscript.pdf','Manuscript.pdf'),('manuscript/Manuscript.md','Manuscript.md'),('final_supplement/Supplementary_Information.pdf','Supplementary_Information.pdf')]:shutil.copy2(OUT/source,folder/target)
    for source,target in [('final_figures/main','Main_Figures'),('final_figures/supplement','Supplementary_Figures'),('final_tables/main','Main_Tables'),('manuscript/sections','Section_Review')]:
        for p in (OUT/source).iterdir():
            if p.is_file():shutil.copy2(p,folder/target/p.name) if (folder/target).exists() else ((folder/target).mkdir(),shutil.copy2(p,folder/target/p.name))
    shutil.copy2(OUT/'final_tables/Single_seed_and_ensemble_comparison.csv',folder/'Main_Tables/Single_seed_and_ensemble_comparison.csv')
    shutil.copy2(OUT/'manuscript/SECTION_REVIEW.csv',folder/'Section_Review/SECTION_REVIEW.csv')
    shutil.copy2(OUT/'manuscript/EDITORIAL_REVIEW.md',folder/'Section_Review/EDITORIAL_REVIEW.md')
    for source in ['final_supplement/Supplementary_Methods.md','final_supplement/Supplementary_Result_Notes.md','final_figures/FIGURE_CAPTIONS.md','manuscript/references.csv','claim_audit/claim_evidence_ledger.csv','claim_audit/actual_manuscript_paragraph_checks.csv']:
        shutil.copy2(OUT/source,folder/'Evidence_and_Methods'/Path(source).name)
    # Correct the disclosure based on the user's stated Claude contribution,
    # without asserting that human author review is already complete.
    p=folder/'Author_Declarations_REQUIRED.md';value=(ROOT/'submission_package/Author_Declarations_REQUIRED.md').read_text(encoding='utf-8')
    value=value.replace('An OpenAI coding/writing assistant was used to assist with data auditing, analysis and reporting code, figure/table preparation, and drafting or editing manuscript text. The authors reviewed the source evidence, statistical methods, outputs and citations and take responsibility for the final work.','OpenAI Codex (OpenAI) and Claude (Anthropic) assisted with the project. Claude-assisted materials included the Phase 12 MPRA notebook, project handbook, collapse-count erratum and results organization. OpenAI Codex assisted with subsequent auditing, analysis and reporting code, figures, tables and manuscript drafting or editing. The authors retain responsibility for the final scientific content and disclosures.')
    value=value.replace('The second sentence must only be used after the authors actually complete that review.','Human authors must review the scientific work and confirm the final disclosure before submission. Exact tool versions and dates should be added where recorded; none have been invented here.')
    p.write_text(value,encoding='utf-8')
    p=folder/'Data_and_Code_Availability_DRAFT.md';value=(ROOT/'submission_package/Data_and_Code_Availability_DRAFT.md').read_text(encoding='utf-8')
    value+='\n\n## Specific data sources\n\nThe activity training sources are ENCODE HepG2 ENCFF774DYO (ENCSR022GQD) and K562 ENCFF631LHA (ENCSR382BVV), described by Agarwal et al., Nature (2025), doi:10.1038/s41586-024-08430-9. The external source is Gosai et al., Nature (2024), doi:10.1038/s41586-024-08070-z, Supplementary Table 12; the current DHS-natural cohort is its processed subset. Full source metadata and checksums are in the supplementary methods and citation evidence records.\n'
    p.write_text(value,encoding='utf-8')
    rows=[
        ('1 Data sources','Added named ENCODE files/experiments and Agarwal/Gosai citations; verified live MD5 and source-table SHA256.','Methods, Table 1 source note, supplement and availability draft','completed'),
        ('2 Benchmark literature','44 verified references with 14 benchmark-lineage papers including original origin; Whalen/Bernett and related evaluation work. Exact supplied-copy scope retained.','Introduction, Results audit, Discussion and uniform bibliography','completed'),
        ('3 AI disclosure','Names OpenAI Codex and Claude/Anthropic with the user-reported Claude contributions. Versions and human author verification are not invented.','Manuscript AI declaration and author checklist','completed draft; author confirmation pending'),
        ('4 Single seed versus ensemble','Reproduced all 12 transformer seed AUROCs and four summaries; added pooled single-seed means/SD beside ensemble values. No new hypothesis tests.','Table 3, Figure 4, Results, Methods and descriptive comparison CSV','completed'),
        ('5 Writing','Rewrote the prose for clearer results; consolidated technical qualifications; split manuscript into individual section files.','Manuscript and Section_Review worksheet','assistant editorial revision complete; Aditta/human review pending'),
        ('6 Figures','Connected symbolic workflow; Figure 2C observed class-count bars; Figure 6 model-specific points; Figure 4 pooled seed/ensemble view.','Main Figures 1, 2, 4, 6 and captions','completed, visual QA recorded separately'),
        ('7 Formatting','Removed glued prose numbers and standardized reference ordering/author initials; retained raw identifiers and numeric source data.','Supplement text, Table 1 notes and bibliography','completed')]
    table('submission_package/Feedback_Response.csv',pd.DataFrame(rows,columns=['feedback','change','location','status']))
    changes=pd.DataFrame(rows,columns=['area','change','evidence','status'])
    table('submission_package/Summary_of_Changes.csv',changes)
    text('submission_package/Summary_of_Changes.md','# Current revision changes\n\n'+'\n\n'.join('**'+a+'**\n\n'+b+'\n\nEvidence: '+c+'. Status: '+d+'.' for a,b,c,d in rows))
    text('submission_package/Feedback_Response.md','# Feedback response\n\n'+'\n\n'.join('**'+a+'**\n\n'+b+'\n\nLocation: '+c+'. Status: '+d+'.' for a,b,c,d in rows))
    text('submission_package/START_HERE.md','''# Revised manuscript review package

This version addresses the seven feedback points. Start with Manuscript.pdf or the editable Manuscript.docx, then Feedback_Response.md. There are 44 verified references. Main Table 3 and Figure 4 distinguish single-seed pooled AUROC from the three-seed ensemble, and Methods names both ENCODE files and the Gosai DHS-natural source. The AI disclosure names both OpenAI Codex and Claude (Anthropic).

This remains a journal-neutral assistant editorial draft. Section_Review contains separate sections and a worksheet for Aditta and the authors to review. No human rewrite or author approval is claimed. Authorship order/roles, declarations, rights and public deposit remain pending. Original manuscripts, frozen studies and previous sealed packages are preserved outside this revision directory.

The 14 inferential contrasts and training/selection rules are unchanged. New single-seed comparisons are descriptive and use the same pooled aggregation. The supplementary workbook is unchanged because its existing seed table already contains the underlying per-seed metrics. The additional comparison CSV makes their aggregation explicit. A new portable release validates the affected figure and seed-summary generation.
''')
    for p in (OUT/'final_figures/supplement').iterdir():
        if p.is_file():shutil.copy2(p,OUT/'final_supplement/Supplementary_Figures'/p.name)
    shutil.copy2(ROOT/'revision49_outputs/Enhancer_Reproducibility_Release_Candidate_v3.zip',folder/'Reproducibility/Enhancer_Reproducibility_Release_Candidate.zip')
    dump('assembled_summary.json',dict(manuscript_review_pages=len(PdfReader(OUT/'manuscript/Manuscript.pdf').pages),supplement_pages=len(PdfReader(OUT/'final_supplement/Supplementary_Information.pdf').pages),references=44,human_review_pending=True))

def create_notebook():
    import nbformat
    path=ROOT/'[49] Enhancer_Manuscript_Feedback_Corrections.ipynb'
    if path.exists():return
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Feedback corrections\n\nJournal-neutral editorial revision from authenticated sources and saved predictions. No new model training, recipe selection or inferential family. Original artifacts remain preserved. Human section review remains pending.'),nbformat.v4.new_code_cell("from pathlib import Path\nimport sys\nassert Path(sys.prefix).name.lower()=='enhancer'\nassert Path.cwd().name==\"Shafin's Contribution\""),nbformat.v4.new_code_cell('from analysis_tools import revise_manuscript49 as revision\nrevision.comparisons()\nrevision.figures()\nrevision.manuscript()\nrevision.supplement()\nrevision.release()\nrevision.assemble()')])
    n.metadata['kernelspec']={'display_name':'Python (enhancer)','language':'python','name':'enhancer'};nbformat.write(n,path)

def main():
    old.verify_inputs();runtime()
    if len(sys.argv)>1 and sys.argv[1]=='author':manuscript();supplement()
    else:comparisons();figures()

if __name__=='__main__':main()
