from figure_common import *
import shutil

def build():
    status('Generating separate result graphs and consistent figure exports')
    # Approved Figure 1 is copied byte-for-byte; no redesign of the approved draft.
    old=ROOT/'revision52_outputs/figure1_redesign_proposal_20261018_v1'
    mapping={'.pdf':'.pdf','.svg':'.svg','.png':'_600dpi.png','.tiff':'_600dpi.tiff','_preview.png':'_preview.png'}
    for dest,src in mapping.items():shutil.copy2(old/('Figure1_Study_workflow_proposed'+src),OUT/('Figure1_Study_workflow'+dest))
    MANIFEST.append(dict(stem='Figure1_Study_workflow',figure=1,panel=None,width_mm=180,height_mm=137,sources=['Approved Figure1 proposal'],approved_draft_unchanged=True))
    audit=read('final_evidence/benchmark_audit.csv')
    sources=['final_evidence/benchmark_audit.csv']
    fig,ax=plotfig(73);plabel(fig,'a','Exact identity between supplied training and test records')
    x=np.arange(2);a=audit.exact_train_test_overlap.to_numpy();b=audit.supplied_test_n.to_numpy()-a
    ax.bar(x,a,width=.48,color=ORANGE,label='Training-identical');ax.bar(x,b,bottom=a,width=.48,color=BL,edgecolor=EDGE,lw=.4,label='Other supplied test records')
    for j in range(2):
        ax.text(j,a[j]/2,str(a[j]),ha='center',va='center',color='white',fontsize=6.2)
        ax.text(j,a[j]+b[j]/2,str(b[j]),ha='center',va='center',fontsize=6.2)
        POINTS.append(dict(figure='4a',stage=j+1,training_identical=int(a[j]),other_test=int(b[j])))
    ax.set(xticks=x,xticklabels=['Stage 1','Stage 2'],ylabel='Supplied test records',ylim=(0,450),xlim=(-.7,1.7));ax.legend(frameon=False,loc='upper right')
    save(fig,'Figure4a_Training_test_identity',4,'a',sources)
    fig,ax=plotfig(74);plabel(fig,'b','Sizes of retained similarity components')
    for i,stage in enumerate([1,2]):
        path=f'final_evidence/inputs/stage{stage}_candidate_manifest.csv';m=read(path);g=m[m.role.eq('train_candidate')].groupby('similarity_group_id').size();counts=g.value_counts().reindex([1,2,3,4],fill_value=0)
        for x,val in enumerate(counts):
            if val:ax.bar(x+(i-.5)*.28,val,width=.26,color=[BLUE,TEAL][i],label=f'Stage {stage}' if x==0 else None);ax.text(x+(i-.5)*.28,val*1.18,str(val),ha='center',va='bottom',fontsize=6)
            POINTS.append(dict(figure='4b',stage=stage,group_size=x+1,n_groups=int(val)))
    ax.set(xticks=range(4),xticklabels=['1','2','3','4'],xlabel='Records per retained group',ylabel='Number of groups (log scale)',yscale='log',ylim=(.7,7000));ax.set_yticks([1,10,100,1000],labels=['1','10','100','1,000']);ax.minorticks_off();ax.legend(frameon=False)
    save(fig,'Figure4b_Similarity_groups',4,'b',[f'final_evidence/inputs/stage{s}_candidate_manifest.csv' for s in [1,2]])
    fig,ax=plotfig(82,left=.31,bottom=.22);plabel(fig,'c','Historical test composition before and after duplicate removal')
    roster=read('phase21_outputs/historical_test_overlap_roster.csv')
    populations=[roster[roster.stage.eq(s)&(~roster.exact_training_match if cleaned else True)] for s,cleaned in [(1,False),(1,True),(2,False),(2,True)]]
    pos=[int(g.label.eq(1).sum()) for g in populations];neg=[int(g.label.eq(0).sum()) for g in populations];y=np.arange(4)
    ax.barh(y,pos,color=TEAL,label='Positive');ax.barh(y,neg,left=pos,color='#CFD5DB',label='Negative')
    for j in range(4):
        if pos[j]:ax.text(pos[j]/2,j,str(pos[j]),ha='center',va='center',color='white',fontsize=6.2)
        ax.text(pos[j]+neg[j]/2,j,str(neg[j]),ha='center',va='center',fontsize=6.2)
        POINTS.append(dict(figure='4c',population=j,positive=pos[j],negative=neg[j]))
    ax.set(yticks=y,yticklabels=['Stage 1 supplied','Stage 1 duplicates removed','Stage 2 supplied','Stage 2 duplicates removed'],xlabel='Historical test records',xlim=(0,450));ax.invert_yaxis();ax.legend(frameon=False,loc='lower right')
    fig.text(.31,.04,'Stage 2 after removal: no positives; discrimination metrics are undefined.',fontsize=6)
    save(fig,'Figure4c_Duplicate_removal',4,'c',['phase21_outputs/historical_test_overlap_roster.csv'])

    seeds=read('phase32_outputs/seed_metrics_pooled_and_fold.csv');cs=read('final_statistics/classification_summary.csv');tests=read('final_statistics/adjusted_tests.csv')
    fig=plt.figure(figsize=(180/25.4,82/25.4));axes=[fig.add_axes(p) for p in [[.09,.23,.23,.60],[.41,.23,.23,.60],[.75,.23,.22,.60]]]
    for letter,stage,ax in zip('ab',[1,2],axes[:2]):
        ax.set_title(f'Stage {stage}',loc='left');fig.text(ax.get_position().x0-.035,.875,letter,fontsize=8,fontweight='bold')
        for i,model in enumerate(['finetuned_dnabert_legacy','finetuned_dnabert2']):
            g=seeds[seeds.stage.eq(stage)&seeds.model.eq(model)].sort_values('base_seed');col=[GREY,BLUE][i]
            ax.scatter(i+np.linspace(-.07,.07,3),g.auroc,s=11,color=col,marker='o')
            ensemble=cs[cs.stage.eq(stage)&cs.model.eq(model)].auroc.iloc[0];ax.scatter(i,ensemble,s=24,color=col,marker='D')
            for r in g.itertuples():POINTS.append(dict(figure='5',stage=stage,model=model,seed=r.base_seed,auroc=r.auroc))
            POINTS.append(dict(figure='5',stage=stage,model=model,kind='ensemble',auroc=ensemble))
        baseline=cs[cs.stage.eq(stage)&cs.model.eq('spaced6_4_logreg')].auroc.iloc[0];ax.axhline(baseline,color=ORANGE,ls=':',lw=.7)
        ax.set(xticks=[0,1],xticklabels=['DNABERT','DNABERT-2'],ylabel='Pooled out-of-fold AUROC',xlim=(-.45,1.45))
    ax=axes[2];ax.set_title('Ensemble contrast',loc='left');fig.text(.715,.875,'c',fontsize=8,fontweight='bold')
    for i,stage in enumerate([1,2]):
        r=tests[tests.comparison_id.eq(f'C_S{stage}_DB2_minus_finetuned_dnabert_legacy')].iloc[0]
        ax.errorbar(r.estimate_a_minus_b,i,xerr=[[r.estimate_a_minus_b-r.lower_95],[r.upper_95-r.estimate_a_minus_b]],fmt='o',markersize=3,color=BLUE,capsize=2,lw=.7)
        POINTS.append(dict(figure='5c',stage=stage,estimate=r.estimate_a_minus_b,lower=r.lower_95,upper=r.upper_95))
    ax.axvline(0,color=GREY,ls=':',lw=.6);ax.set(yticks=[0,1],yticklabels=['Stage 1','Stage 2'],ylim=(-.5,1.5),xlim=(-.012,.037),xticks=[-.01,0,.01,.02,.03],xlabel='DNABERT-2 minus DNABERT\nPooled AUROC difference')
    fig.text(.07,.06,'Circles: individual seeds    Diamonds: three-seed ensemble    Dotted horizontal line: archived spaced-mer baseline',fontsize=5.8)
    fig.text(.75,.08,'95% conditional group-bootstrap intervals',fontsize=5.3,ha='left')
    save(fig,'Figure5_Seeds_and_ensembles',5,None,['phase32_outputs/seed_metrics_pooled_and_fold.csv','final_statistics/classification_summary.csv','final_statistics/adjusted_tests.csv'])

    gc=read('final_statistics/exact_gc_fold_metrics.csv')
    for stage,letter in [(1,'a'),(2,'b')]:
        d=gc[gc.stage.eq(stage)].groupby('model').agg(raw=('raw_fold_auroc','mean'),sd=('raw_fold_auroc','std'),matched=('same_gc_fold_auroc','mean')).loc[ORDER]
        fig,ax=plotfig(85,left=.35,bottom=.16,top=.82);plabel(fig,letter,f'Stage {stage}: grouped classification and exact-GC evaluation')
        y=np.arange(9);ax.errorbar(d.raw,y,xerr=d.sd,fmt='o',color=BLUE,markersize=3.5,capsize=2,lw=.7,label='Mean-fold AUROC ± fold SD')
        ax.scatter(d.matched,y,s=17,marker='D',color=ORANGE,label='Exact-GC AUROC')
        for j,r in enumerate(d.itertuples()):
            ax.plot([r.matched,r.raw],[j,j],color='#BAC2C8',lw=.5,zorder=0)
            POINTS.append(dict(figure=f'6{letter}',stage=stage,model=r.Index,raw=r.raw,fold_sd=r.sd,exact_gc=r.matched))
        ax.axvline(.5,color=GREY,ls=':',lw=.6);ax.set(yticks=y,yticklabels=LABELS,xlabel='AUROC',xlim=(.47,.94),xticks=[.5,.6,.7,.8,.9],ylim=(8.55,-.55))
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.98,.91),frameon=False,ncol=2)
        save(fig,f'Figure6{letter}_Stage{stage}_classification_exact_GC',6,letter,['final_statistics/exact_gc_fold_metrics.csv'])

    internal=read('final_tables/main/T4_MPRA_internal.csv');sm=read('phase37_outputs/seed_metrics_independent.csv')
    arms=['spaced6_4_ridge','dnabert2_selected','legnet_selected','mprann_selected']
    for letter,cell in [('a','HepG2'),('b','K562')]:
        fig,ax=plotfig(76,bottom=.23);plabel(fig,letter,f'{cell}: chromosome-held-out reporter activity')
        for i,arm in enumerate(arms):
            r=internal[internal.cell.eq(cell)&internal.arm.eq(arm)].iloc[0];ax.scatter(i,r.pearson,marker='D',s=28,color=MODEL_COLORS[arm],zorder=4)
            POINTS.append(dict(figure=f'7{letter}',cell=cell,arm=arm,kind='ensemble' if arm!='spaced6_4_ridge' else 'archived',pearson=r.pearson))
            ss=sm[sm.cell.eq(cell)&sm.cohort.eq('internal')&sm.arm.eq(arm)].sort_values('seed')
            if len(ss):
                assert len(ss)==3;ax.scatter(i+np.linspace(-.10,.10,3),ss.pearson,marker='o',s=11,facecolors='white',edgecolors=MODEL_COLORS[arm],lw=.7,zorder=3)
                for r2 in ss.itertuples():POINTS.append(dict(figure=f'7{letter}',cell=cell,arm=arm,seed=r2.seed,kind='seed',pearson=r2.pearson))
        ax.set(xticks=range(4),xticklabels=['Spaced-mer\narchived single recipe','DNABERT-2','LegNet','MPRAnn'],ylabel='Internal test Pearson correlation',xlim=(-.45,3.45),ylim=(.51,.735))
        fig.text(.13,.045,'Diamonds: three-seed neural ensembles or archived baseline    Open circles: individual neural seeds',fontsize=5.8)
        save(fig,f'Figure7{letter}_{cell}_internal_activity',7,letter,['final_tables/main/T4_MPRA_internal.csv','phase37_outputs/seed_metrics_independent.csv'])
    for letter,kind,title in [('c','selected','Validation-selected LegNet minus DNABERT-2'),('d','duration','DNABERT-2 longer run minus its own two-epoch prefix')]:
        fig,ax=plotfig(53,left=.18,bottom=.29,top=.78);plabel(fig,letter,title)
        for i,cell in enumerate(['HepG2','K562']):
            cid=f'M_{cell}_internal_legnet_selected_minus_dnabert2_selected' if kind=='selected' else f'D_{cell}_internal_full_minus_own_early2'
            r=tests[tests.comparison_id.eq(cid)].iloc[0]
            ax.errorbar(r.estimate_a_minus_b,i,xerr=[[r.estimate_a_minus_b-r.lower_95],[r.upper_95-r.estimate_a_minus_b]],fmt='o',color=BLUE,capsize=2,markersize=3.5,lw=.75)
            POINTS.append(dict(figure=f'7{letter}',cell=cell,comparison=cid,estimate=r.estimate_a_minus_b,lower=r.lower_95,upper=r.upper_95,p_holm=r.p_holm))
        ax.axvline(0,color=GREY,ls=':',lw=.6);ax.set(yticks=[0,1],yticklabels=['HepG2','K562'],ylim=(-.7,1.7),xlim=(-.002,.065),xticks=[0,.01,.02,.03,.04,.05,.06],xlabel='Pearson difference (95% conditional locus-group bootstrap interval)')
        if kind=='duration':fig.text(.18,.045,'Fixed learning rate 2e-5; same initialisation and exact training prefix.',fontsize=5.8)
        save(fig,f'Figure7{letter}_'+('Selected_model_contrast' if kind=='selected' else 'Own_prefix_duration_control'),7,letter,['final_statistics/adjusted_tests.csv'])

    ext=read('final_tables/main/T5_External_transfer.csv')
    fig=plt.figure(figsize=(180/25.4,77/25.4));axes=[fig.add_axes([.10,.24,.36,.59]),fig.add_axes([.60,.24,.36,.59])]
    for letter,ax,metric,title in zip('ab',axes,['spearman','r2'],['Ranking','Absolute prediction']):
        fig.text(ax.get_position().x0-.035,.875,letter,fontsize=8,fontweight='bold')
        for j,cell in enumerate(['HepG2','K562']):
            vals=[]
            for i,arm in enumerate(['dnabert2_selected','legnet_selected','mprann_selected']):
                r=ext[ext.cell.eq(cell)&ext.arm.eq(arm)].iloc[0];vals.append(r[metric]);ax.scatter(i+(j-.5)*.15,r[metric],marker=['o','s'][j],s=21,color=MODEL_COLORS[arm],label=cell if i==0 else None)
                POINTS.append(dict(figure='8',metric=metric,cell=cell,arm=arm,value=r[metric]))
        ax.set(xticks=[0,1,2],xticklabels=['DNABERT-2','LegNet','MPRAnn'],title=title,xlim=(-.4,2.4),ylabel='External Spearman correlation' if metric=='spearman' else 'External R²')
        if metric=='spearman':ax.set_ylim(.48,.58)
        else:ax.axhline(0,color=GREY,ls=':',lw=.6);ax.set_ylim(-1.25,.07)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],marker='o',color=BLACK,ls='',markersize=3.5),Line2D([],[],marker='s',color=BLACK,ls='',markersize=3.5)],['HepG2','K562'],loc='lower center',bbox_to_anchor=(.5,.04),ncol=2,frameon=False)
    save(fig,'Figure8_Transfer_failure_modes',8,None,['final_tables/main/T5_External_transfer.csv'])

if __name__=='__main__':build();finish();print('Plot files generated:',len(MANIFEST))
