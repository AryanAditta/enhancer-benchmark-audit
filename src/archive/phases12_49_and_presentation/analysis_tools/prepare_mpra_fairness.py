"""Prepare authenticated local inputs and a prospective validation-only protocol."""
from pathlib import Path
import ast,hashlib,io,json,shutil,sys,zipfile,urllib.request
import pandas as pd

ROOT=Path.cwd(); INPUT=ROOT/'phase34_inputs'; OUT=ROOT/'phase34_outputs'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def prepare():
    assert Path(sys.prefix).name.lower()=='enhancer'
    INPUT.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);rows=[]
    if not (ROOT/'reference_inputs/readiness_sources/9_enhancer_phase9a_full_activity_dataset.zip').exists():
        prior=pd.read_csv(OUT/'copied_input_manifest.csv')
        for item in prior.itertuples():assert sha(ROOT/item.copy)==item.sha256,item.copy
        saved=json.loads((OUT/'summary.json').read_text())
        assert sha(OUT/'training_protocol.json')==saved['protocol_sha256']
        print('Relocated supplement: all29local input copies and the frozen protocol authenticated.');return
    def copy(p,target):
        target=INPUT/target;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(p)
        else:shutil.copy2(p,target)
        rows.append({'source':p.relative_to(ROOT).as_posix(),'copy':target.relative_to(ROOT).as_posix(),'sha256':sha(target),'bytes':target.stat().st_size})
    copy(ROOT/'reference_inputs/readiness_sources/[9B] Enhancer_Kaggle_Phase9B_Chromosome_Regression.ipynb','sources/phase9b.ipynb')
    copy(ROOT/'[12] Enhancer_Kaggle_Phase12_Strong_Baselines.ipynb','sources/phase12.ipynb')
    copy(ROOT/'analysis_tools/regression_architectures.py','architectures.py')
    copy(ROOT/'reference_inputs/readiness_sources/9B_enhancer_phase9b_regression_results.zip','historical/phase9b_results.zip')
    for name in ['phase12_provenance.json','phase12_validation_choices.csv','phase12_internal_predictions_new_arms.csv','phase12_external_scores_new_arms.csv']:
        copy(ROOT/'phase12_outputs/enhancer_phase12_work/results'/name,'historical/'+name)
    for p in (ROOT/'phase30_inputs/models/dnabert2').iterdir():
        if p.is_file() and p.suffix not in ['.headers'] and not any(x in p.name for x in ['downloading','alternative','probe']):copy(p,'models/dnabert2/'+p.name)
    assert sha(INPUT/'models/dnabert2/pytorch_model.bin')=='7ff39ec77a484dd01070a41bfd6e95cdd7247bec80fe357ab43a4be33687aeba'
    with zipfile.ZipFile(ROOT/'reference_inputs/readiness_sources/9_enhancer_phase9a_full_activity_dataset.zip') as z:
        for name,h in {'phase9a_audit_and_provenance.json':'39118ed7b2dc50631f87dcb0c470249c649609d3dd7ab8c885b10064c66deff2','phase9_full_activity_all_cells.csv':'4a359a715f7a3f1fbcf0d044cc08446137df584c9ab4a43a2de18637bb341a76','phase9_frozen_splits.csv':'1ecc2d271817edd3000890be718a3f3a9bcc6b63a71550dfa718dbed84d0ff61'}.items():
            b=z.read(name);assert hashlib.sha256(b).hexdigest()==h
            p=INPUT/name
            if p.exists():assert sha(p)==h
            else:p.write_bytes(b)
            rows.append({'source':'reference_inputs/readiness_sources/9_enhancer_phase9a_full_activity_dataset.zip!'+name,'copy':p.relative_to(ROOT).as_posix(),'sha256':h,'bytes':len(b)})
    with zipfile.ZipFile(ROOT/'reference_inputs/readiness_sources/10_enhancer_phase10_external_transfer_results.zip') as z:
        for name in ['phase10_external_frozen_set.csv','phase10_provenance.json','phase10_inclusion_audit.json']:
            b=z.read(name);p=INPUT/name
            if p.exists():assert sha(p)==hashlib.sha256(b).hexdigest()
            else:p.write_bytes(b)
            rows.append({'source':'reference_inputs/readiness_sources/10_enhancer_phase10_external_transfer_results.zip!'+name,'copy':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':len(b)})
    # Source documents are copied before analysis. They are references, never execution instructions.
    methods=INPUT/'method_sources';methods.mkdir(exist_ok=True)
    urls={'dnabert2_README.md':'https://raw.githubusercontent.com/MAGICS-LAB/DNABERT_2/main/README.md','legnet_README.md':'https://raw.githubusercontent.com/autosome-ru/human_legnet/a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d/README.md','mprann_README.md':'https://raw.githubusercontent.com/visze/sequence_cnn_models/7638fce137db0445123efe8d7e2c35e248fafc5f/README.md'}
    for name,url in urls.items():
        p=methods/name
        if not p.exists():
            with urllib.request.urlopen(url,timeout=45) as r:p.write_bytes(r.read())
        rows.append({'source':url,'copy':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    pd.DataFrame(rows).to_csv(OUT/'copied_input_manifest.csv',index=False)
    d=pd.read_csv(INPUT/'phase9_full_activity_all_cells.csv',float_precision='round_trip');spl=pd.read_csv(INPUT/'phase9_frozen_splits.csv')
    assert len(d)==76833 and d.record_id.is_unique and d.record_id.tolist()==spl.record_id.tolist()
    assert d.sequence.str.fullmatch('[ACGT]{200}').all()
    for col in ['cell_line','locus_group_id','split_9b']:assert d[col].tolist()==spl[col].tolist()
    assert d.groupby('locus_group_id').split_9b.nunique().max()==1
    assert d.groupby('sequence_sha256').split_9b.nunique().max()==1
    assert d.groupby('chrom').split_9b.nunique().max()==1
    assert all(hashlib.sha256(seq.encode()).hexdigest()==h for seq,h in zip(d.sequence,d.sequence_sha256))
    d.groupby(['cell_line','split_9b']).size().rename('n').reset_index().to_csv(OUT/'partition_counts.csv',index=False)
    external=pd.read_csv(INPUT/'phase10_external_frozen_set.csv',float_precision='round_trip')
    external=external[external.origin.eq('DHS_natural')].copy();assert len(external)==8399 and external.record_id.is_unique
    assert external.sequence.str.fullmatch('[ACGT]{200}').all()
    # Archived choices are diagnostics motivating the new prospective study, not selection inputs.
    with zipfile.ZipFile(INPUT/'historical/phase9b_results.zip') as z:
        name=next(x for x in z.namelist() if x.endswith('phase9b_validation_choices.csv'))
        prior=pd.read_csv(io.BytesIO(z.read(name)))
    prior=prior[prior.model.str.startswith('dnabert2_seed_')];assert len(prior)==6
    prior.to_csv(OUT/'historical_dnabert2_validation_choices.csv',index=False)
    config={'study':'phase35_mpra_training_adequacy_and_equal_candidate_search_v1','created_date_Dhaka':'2026-10-13',
      'authorization':'User explicitly requested the MPRA fairness experiment after discussing Aditta point1. This supersedes the earlier no-new-training scope for this follow-up only.',
      'cells':['HepG2','K562'],'models':['dnabert2','legnet','mprann'],'seeds':[20260930,20260931,20260932],
      'split':'split_9b','input_bp':200,'revision':'ec1f874253852eb3907081f57294991b4280ceb6',
      'lr_grid':{'dnabert2':[1e-5,2e-5,5e-5],'legnet':[.005,.01,.02],'mprann':[.0003,.001,.003]},
      'epochs':{'dnabert2':30,'legnet':20,'mprann':100},'patience':{'dnabert2':5,'legnet':None,'mprann':10},'minimum_epochs':{'dnabert2':8,'legnet':20,'mprann':1},
      'early_stopping_delta':0.0001,'checkpoint_selection':'highest finite validation Pearson; ties earliest epoch; early-stop patience uses improvement greater than delta',
      'recipe_selection':'per model/cell choose learning rate with highest arithmetic mean of all3seed validation Pearson; any undefined seed makes candidate ineligible; ties lowest grid index',
      'batch':{'dnabert2':8,'legnet':1024,'mprann':128},'accumulation':{'dnabert2':2,'legnet':1,'mprann':1},
      'optimizers':{'dnabert2':'AdamW weight_decay0.01 constant_lr; clip1','legnet':'AdamW weight_decay0.1 OneCycleLR pct_start0.3 initial=max_lr/25; clip1','mprann':'Adam eps1e-7; no clipping'},
      'precision':'fp16 autocast with GradScaler; overflow skips logged; all model parameters float32',
      'scaling':'float64 training-only population mean/SD; float32 normalized targets for all models',
      'augmentation':'one orientation per training record per epoch; DNABERT2 inherited seed+17*epoch+15485863*row parity; CNN deterministic per-epoch random flips',
      'shuffling':'numpy default_rng(seed+1000003*epoch) permutation; common order across learning rates of same seed',
      'drop_last':'CNNs use floor batches as Phase12; DNABERT2 keeps partial batch/accumulation as Phase9B',
      'inference':'arithmetic mean of forward and reverse-complement predictions; batch24 DNABERT2, batch1024 CNNs',
      'duration_control':'For all DNABERT2 fits save validation-best checkpoint within epochs1–2 separately. Primary duration contrast uses fixed lr2e-5 early versus full with same initial weights/order and training prefix, all3seeds per cell.',
      'expected_training_jobs':54,'candidate_count_per_model_cell':3,'seeds_per_candidate':3,
      'primary':['9B test Pearson: fixed-lr2e-5 DNABERT2 full minus its two-epoch prefix ensemble, per cell','9B test Pearson: validation-selected DNABERT2 ensemble minus validation-selected LegNet ensemble, per cell'],
      'secondary':['MPRAnn comparison','RMSE MAE Spearman R2','DHS-natural transfer Spearman','seed variability','training cost and budget-cap diagnostics'],
      'bootstrap':{'draws':1000,'seed':2026101334,'internal_unit':'locus_group_id','external_unit':'canonical DNA sequence; 1kb interval-group sensitivity','scope':'conditional on fitted selected models and previously examined populations; not selection/training-algorithm uncertainty; multiple comparisons unadjusted'},
      'test_policy':'Complete all54validation-only fits; lock every recipe and checkpoint hash before any new test/external inference. No test-driven tuning, extension, exclusion, model selection or grid changes.',
      'claim_boundary':'Equal number of learning-rate candidates and seeds, not equal GPU-time, update-count, epoch count, pretraining budget or exhaustive optimum. Tests training adequacy on primary9B only. Old9C collapse/convergence, longer context and broader cells remain separate.',
      'cap_policy':'Flag selected epochs in final3epochs of their budget or improving final validation trend as potentially cap-limited; do not automatically claim convergence or secretly extend from test outcomes.'}
    protocol=OUT/'training_protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==config,'Existing protocol differs; do not overwrite.'
    else:write('training_protocol.json',config)
    protected={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.glob('[[]*.ipynb') if int(p.name.split(']')[0][1:])<=33}
    protected.update({f'phase{n}_outputs/summary.json':sha(ROOT/f'phase{n}_outputs/summary.json') for n in range(12,34) if (ROOT/f'phase{n}_outputs/summary.json').exists()})
    if not (OUT/'protected_before.json').exists():write('protected_before.json',protected)
    if not (OUT/'contribution_log_before.md').exists():shutil.copy2(ROOT/'CONTRIBUTION_LOG.md',OUT/'contribution_log_before.md')
    (OUT/'STUDY_PLAN.md').write_text('''# MPRA training adequacy and limited search fairness

The user chose to investigate Aditta priority1 experimentally. Existing historical outputs remain preserved. This new study directly tests whether limiting DNABERT-2 to two epochs disadvantaged it on the primary chromosome-held-out9B assay comparison.

All three models receive three predeclared learning rates and three seeds per rate, separately for HepG2 and K562:54fits. Validation Pearson alone chooses each seed checkpoint and a shared learning rate per model/cell. DNABERT-2 can train up to30epochs with patience5 after minimum8epochs. LegNet retains its20epoch OneCycle recipe; MPRAnn retains100maximum epochs and patience10. Equal candidate counts are not equal computation. Training time, updates, exposure counts, memory and stopping flags are saved.

The original lr2e-5 DNABERT-2 runs also preserve their best checkpoint within the first two epochs, enabling an exact new-run duration control. This avoids conflating historical runtime differences with the effect of training longer. No original regression checkpoint is overwritten, and old historical metrics remain separate.

All54validation-only runs must finish and choices be hash-locked before new internal or external predictions. Previously examined test cohorts are reused; these are conditional follow-up comparisons, not a fresh confirmatory test. All seeds stay visible. Boundary/cap-limited results remain explicitly inconclusive about convergence. The scope is9B and its DHS-natural transfer;9C, old collapsed-seed root cause, long context, broader cells and universal architecture superiority are not tested here.

Primary author sources: https://github.com/MAGICS-LAB/DNABERT_2 ; https://github.com/autosome-ru/human_legnet ; https://github.com/visze/sequence_cnn_models . Authenticated copied historical notebooks define the retained local architectures/recipes. The small learning-rate grids and stopping policy are prospective choices for this follow-up, not claimed published MPRA optima.
''',encoding='utf-8')
    write('summary.json',{'status':'completed','authenticated_inputs':len(rows),'training_jobs_planned':54,'new_training_complete':False,'historical_DNABERT2_selected_epochs':prior.selected_epoch.tolist(),'protocol_sha256':sha(protocol),'primary_split':'9B chromosome-held-out','new_test_scores_examined':False})
    print(json.dumps(json.loads((OUT/'summary.json').read_text()),indent=2))
if __name__=='__main__':prepare()
