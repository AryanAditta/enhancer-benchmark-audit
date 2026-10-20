"""Frozen training implementation for the prospectively specified MPRA follow-up."""
from pathlib import Path
import os,sys,json,hashlib,time,math,gc,random,importlib.util,itertools,platform
import numpy as np
import pandas as pd

ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'phase34_inputs').is_dir())
INPUT=ROOT/'phase34_inputs';P34=ROOT/'phase34_outputs';P35=ROOT/'phase35_outputs'
C=json.loads((P34/'training_protocol.json').read_text(encoding='utf-8'))
COMP=str.maketrans('ACGT','TGCA')
def rc(x):return x.translate(COMP)[::-1]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def jout(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.writing')
    tmp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def csvout(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.writing')
    d.to_csv(tmp,index=False,float_format='%.17g');os.replace(tmp,p)
def jread(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def safe_r(y,p):
    if not np.isfinite(p).all() or np.std(p)<1e-8 or np.std(y)==0:return None
    return float(np.corrcoef(np.asarray(y,float),np.asarray(p,float))[0,1])
def jobs():return list(itertools.product(C['cells'],C['models'],range(3),C['seeds']))
def key(cell,model,grid,seed):return f'{cell}__{model}__lr{grid}__seed{seed}'
def partitions(cell):
    d=pd.read_csv(INPUT/'phase9_full_activity_all_cells.csv',float_precision='round_trip')
    d=d[d.cell_line.eq(cell)]
    return tuple(d[d.split_9b.eq(split)].reset_index(drop=True) for split in ['train','validation','test'])
def protected_check():
    original=(ROOT/'phase12_outputs').exists();rows=[]
    for file,h in jread(P34/'protected_before.json').items():
        if (ROOT/file).exists():assert sha(ROOT/file)==h,file;status='verified_unchanged'
        else:assert not original,file;status='not_in_standalone_supplement; original_preservation_audit_retained'
        rows.append({'file':file,'sha256':h,'status':status})
    return rows
def runtime():
    assert Path(sys.prefix).name.lower()=='enhancer'
    for name in ['TEMP','TMP','HF_HOME','HF_MODULES_CACHE','TRANSFORMERS_CACHE','CUDA_CACHE_PATH','TORCH_HOME','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME','MPLCONFIGDIR']:
        p=P35/'runtime'/name.lower();p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    vendor=ROOT/'reference_inputs/readiness_runtime/vendor'
    if str(vendor) not in sys.path:sys.path.insert(0,str(vendor))
    import torch,transformers
    assert torch.__version__=='2.11.0+cu128' and transformers.__version__=='4.44.2'
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    return torch
def arch():
    spec=importlib.util.spec_from_file_location('mpra_fairness_architectures',INPUT/'architectures.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def build(model):
    a=arch()
    if model=='dnabert2':
        from transformers import AutoTokenizer
        from transformers.models.bert.configuration_bert import BertConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import importlib
        folder=INPUT/'models/dnabert2'
        tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        cfg=BertConfig.from_pretrained(folder,local_files_only=True)
        cls=get_class_from_dynamic_module('bert_layers.BertModel',str(folder),local_files_only=True)
        module=importlib.import_module(cls.__module__);module.flash_attn_qkvpacked_func=None
        enc,info=cls.from_pretrained(folder,config=cfg,local_files_only=True,output_loading_info=True)
        assert not info.get('mismatched_keys') and not info.get('error_msgs')
        assert all('pooler' in k for k in info['missing_keys'])
        assert all(k.startswith('cls.') or k.startswith('bert.cls.') for k in info['unexpected_keys'])
        return a.Regressor(enc).cuda(),tok
    if model=='legnet':
        net=a.LegNet(in_ch=4,stem_ch=64,stem_ks=11,ef_ks=9,ef_block_sizes=[80,96,112,128],pool_sizes=[2,2,2,2],resize_factor=4)
        net.apply(a.initialize_weights)
    else:assert model=='mprann';net=a.MPRAnn()
    return net.cuda(),None
def encode(model,tok,seqs):
    import torch
    if model=='dnabert2':
        b=tok(list(seqs),padding=True,add_special_tokens=True,truncation=False,return_tensors='pt')
        assert b['input_ids'].shape[1]<=256
        return {k:v.cuda() for k,v in b.items() if k in ['input_ids','attention_mask']}
    lut=np.zeros(256,dtype=np.int64)
    for i,x in enumerate(b'ACGT'):lut[x]=i
    codes=np.stack([lut[np.frombuffer(seq.encode(),dtype=np.uint8)] for seq in seqs])
    return torch.from_numpy(np.eye(4,dtype=np.float32)[codes].transpose(0,2,1).copy()).cuda()
def predict(net,tok,model,seqs,mean,scale):
    import torch
    bs=24 if model=='dnabert2' else 1024;out=[];net.eval()
    with torch.inference_mode():
        for start in range(0,len(seqs),bs):
            batch=seqs[start:start+bs]
            with torch.autocast('cuda',dtype=torch.float16):
                p=net(encode(model,tok,batch)).float();q=net(encode(model,tok,[rc(x) for x in batch])).float()
            out.append((((p+q)/2)*scale+mean).cpu().numpy())
    result=np.concatenate(out).astype(float);assert np.isfinite(result).all();return result
def save_torch(path,obj):
    import torch
    path=Path(path);tmp=path.with_suffix(path.suffix+'.writing');torch.save(obj,tmp);os.replace(tmp,path)
def state_cpu(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def checkpoint(net,context,epoch,score,pred):
    return dict(context,epoch=epoch,validation_pearson=score,validation_predictions=list(map(float,pred)),state_dict=state_cpu(net))
def setup():
    torch=runtime();P35.mkdir(exist_ok=True)
    for name in ['jobs','checkpoints','validation','training_logs']:(P35/name).mkdir(exist_ok=True)
    manifest=pd.read_csv(P34/'copied_input_manifest.csv')
    for row in manifest.itertuples():assert sha(ROOT/row.copy)==row.sha256,row.copy
    lock={'protocol_sha256':sha(P34/'training_protocol.json'),'source_sha256':sha(Path(__file__)),'architecture_sha256':sha(INPUT/'architectures.py'),'input_manifest_sha256':sha(P34/'copied_input_manifest.csv'),'pretrained_weight_sha256':sha(INPUT/'models/dnabert2/pytorch_model.bin')}
    path=P35/'run_lock.json'
    if path.exists():assert jread(path)==lock,'Frozen implementation/input/protocol changed.'
    else:jout(path,lock)
    # A finite synthetic forward/backward probe is discarded; no optimizer or scientific test outcomes.
    probes=[]
    if not (P35/'runtime_validation.json').exists():
        for model in C['models']:
            seed_all(1234);net,tok=build(model)
            seqs=['ACGT'*50,'CGTA'*50,'AACC'*50,'TGCA'*50]
            with torch.autocast('cuda',dtype=torch.float16):y=net(encode(model,tok,seqs));loss=y.float().square().mean()
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            probes.append({'model':model,'parameters':sum(p.numel() for p in net.parameters()),'finite_forward_backward':True})
            del net,tok;gc.collect();torch.cuda.empty_cache()
        jout(P35/'runtime_validation.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'models':probes,'probe_optimizer_steps':0,'working_directory':str(ROOT)})
    protected_check();update_summary();return lock
def update_summary():
    complete=list((P35/'jobs').glob('*.json'))
    jout(P35/'summary.json',{'status':'running' if len(complete)<54 else 'all_training_done_pending_selection','completed_jobs':len(complete),'expected_jobs':54,'test_inference_started':False})
def train_job(cell,model,grid,seed):
    torch=runtime();lock=jread(P35/'run_lock.json')
    assert sha(Path(__file__))==lock['source_sha256'] and sha(P34/'training_protocol.json')==lock['protocol_sha256']
    assert sha(INPUT/'architectures.py')==lock['architecture_sha256']
    name=key(cell,model,grid,seed);done=P35/'jobs'/f'{name}.json'
    if done.exists():
        m=jread(done);assert m['source_sha256']==lock['source_sha256'] and m['protocol_sha256']==lock['protocol_sha256']
        for item in m['artifacts']:assert sha(ROOT/item['file'])==item['sha256']
        print('REUSED',name,flush=True);return m
    tr,va,_=partitions(cell);assert set(tr.locus_group_id).isdisjoint(va.locus_group_id)
    y=tr.mean_log2_activity.to_numpy(float);mean=float(y.mean());scale=float(y.std(ddof=0));assert scale>0
    yz=torch.tensor(((y-mean)/scale).astype(np.float32),device='cuda');seqs=tr.sequence.tolist()
    bs=C['batch'][model];acc=C['accumulation'][model];epochs=C['epochs'][model];lr=C['lr_grid'][model][grid]
    steps=math.ceil(len(tr)/bs) if model=='dnabert2' else len(tr)//bs
    seed_all(seed);net,tok=build(model);torch.cuda.reset_peak_memory_stats()
    if model=='dnabert2':opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.01);sched=None
    elif model=='legnet':
        opt=torch.optim.AdamW(net.parameters(),lr=lr/25,weight_decay=.1)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=epochs*steps,pct_start=.3,cycle_momentum=False)
    else:opt=torch.optim.Adam(net.parameters(),lr=lr,eps=1e-7);sched=None
    scaler=torch.amp.GradScaler('cuda');context={'job':name,'cell_line':cell,'model':model,'grid_index':grid,'learning_rate':lr,'seed':seed,'protocol_sha256':lock['protocol_sha256'],'source_sha256':lock['source_sha256'],'architecture_sha256':lock['architecture_sha256'],'revision':C['revision'] if model=='dnabert2' else 'copied_Phase12_architecture','target_mean':mean,'target_scale':scale,'train_record_ids_sha256':digest('\n'.join(tr.record_id)),'validation_record_ids_sha256':digest('\n'.join(va.record_id))}
    best=None;prefix=None;logs=[];stale=0;patience_best=None;overflow=0;updates=0;examples=0;elapsed_prior=0.;start_epoch=1;start_step=0;loss_sum=0.;loss_n=0
    resume=P35/'current_training_resume.pt';side=resume.with_suffix('.json')
    if resume.exists() and side.exists() and jread(side).get('job')==name:
        assert sha(resume)==jread(side)['sha256'],'Resume checksum mismatch; preserve before recovery.'
        r=torch.load(resume,map_location='cpu',weights_only=True);assert r['context']==context
        net.load_state_dict(r['net'],strict=True);opt.load_state_dict(r['optimizer']);scaler.load_state_dict(r['scaler'])
        if sched:sched.load_state_dict(r['scheduler'])
        best=r['best'];prefix=r['prefix'];logs=r['logs'];stale=r['stale'];patience_best=r['patience_best'];overflow=r['overflow'];updates=r['updates'];examples=r['examples'];elapsed_prior=r['elapsed_seconds'];start_epoch=r['next_epoch'];start_step=r['next_step'];loss_sum=r['loss_sum'];loss_n=r['loss_n']
        torch.set_rng_state(r['torch_rng']);torch.cuda.set_rng_state_all(r['cuda_rng']);random.setstate(r['python_rng'])
        # The authenticated single resume contains its best/prefix weights, avoiding multi-file crash races.
        if best:save_torch(P35/'checkpoints'/f'{name}__best.pt',best)
        if prefix:save_torch(P35/'checkpoints'/f'{name}__early2.pt',prefix)
        del r;print('RESUMING',name,'epoch',start_epoch,'step',start_step,flush=True)
    if logs and len(logs)>=C['minimum_epochs'][model] and C['patience'][model] is not None and stale>=C['patience'][model]:
        start_epoch=epochs+1
    epoch=logs[-1]['epoch'] if logs else 0
    t0=time.monotonic();last_save=t0;consecutive=0
    def persist(next_epoch,next_step,total_loss,count):
        obj={'context':context,'net':state_cpu(net),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'scheduler':sched.state_dict() if sched else None,'best':best,'prefix':prefix,'logs':logs,'stale':stale,'patience_best':patience_best,'overflow':overflow,'updates':updates,'examples':examples,'elapsed_seconds':elapsed_prior+time.monotonic()-t0,'next_epoch':next_epoch,'next_step':next_step,'loss_sum':total_loss,'loss_n':count,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()}
        save_torch(resume,obj);jout(side,{'job':name,'sha256':sha(resume),'next_epoch':next_epoch,'next_step':next_step,'source_sha256':lock['source_sha256']})
    for epoch in range(start_epoch,epochs+1):
        if epoch>start_epoch:loss_sum=0.;loss_n=0
        rng=np.random.default_rng(seed+1000003*epoch);order=rng.permutation(len(tr));flips=rng.random(len(tr))<.5
        net.train();opt.zero_grad(set_to_none=True)
        for step in range(start_step if epoch==start_epoch else 0,steps):
            ix=order[step*bs:(step+1)*bs]
            if model=='dnabert2':batch=[rc(seqs[i]) if (seed+17*epoch+15485863*int(i))%2 else seqs[i] for i in ix]
            else:batch=[rc(seqs[i]) if flips[i] else seqs[i] for i in ix]
            with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.mse_loss(net(encode(model,tok,batch)).float(),yz[ix])
            assert torch.isfinite(loss),(name,epoch,step,'nonfinite loss')
            scaler.scale(loss/acc).backward();loss_sum+=float(loss.detach());loss_n+=1;examples+=len(ix)
            boundary=(step+1)%acc==0 or step+1==steps
            if boundary:
                scaler.unscale_(opt)
                if model!='mprann':torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                before=scaler.get_scale();scaler.step(opt);scaler.update();skipped=scaler.get_scale()<before
                overflow+=int(skipped);consecutive=consecutive+1 if skipped else 0;assert consecutive<20,(name,'persistent AMP overflows')
                if not skipped:
                    updates+=1
                    if sched:sched.step()
                opt.zero_grad(set_to_none=True)
                if time.monotonic()-last_save>180:
                    persist(epoch,step+1,loss_sum,loss_n);last_save=time.monotonic()
            if step%100==0:
                jout(P35/'live_progress.json',{'job':name,'cell':cell,'model':model,'grid':grid,'seed':seed,'epoch':epoch,'maximum_epochs':epochs,'batches_done':step+1,'batches_total':steps,'elapsed_this_execution_seconds':time.monotonic()-t0,'updated_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
        pv=predict(net,tok,model,va.sequence.tolist(),mean,scale);r=safe_r(va.mean_log2_activity,pv)
        improved=r is not None and (best is None or r>best['validation_pearson'])
        if improved:
            best=checkpoint(net,context,epoch,r,pv);save_torch(P35/'checkpoints'/f'{name}__best.pt',best)
        if model=='dnabert2' and epoch==2:
            prefix=best if best is not None else checkpoint(net,context,epoch,None,pv)
            save_torch(P35/'checkpoints'/f'{name}__early2.pt',prefix)
        significant=r is not None and (patience_best is None or r>patience_best+C['early_stopping_delta'])
        if significant:patience_best=r;stale=0
        else:stale+=1
        logs.append({'epoch':epoch,'training_loss':loss_sum/loss_n,'validation_pearson':r,'validation_prediction_sd':float(np.std(pv)),'best_validation_epoch':best['epoch'] if best else None,'updates_total':updates,'examples_seen_total':examples,'AMP_skipped_total':overflow,'learning_rate_at_epoch_end':float(opt.param_groups[0]['lr']),'elapsed_seconds':elapsed_prior+time.monotonic()-t0})
        csvout(P35/'training_logs'/f'{name}.csv',pd.DataFrame(logs))
        print(name,'epoch',epoch,'val',r,'best',best['epoch'] if best else None,'stale',stale,flush=True)
        persist(epoch+1,0,0.,0);last_save=time.monotonic()
        if epoch>=C['minimum_epochs'][model] and C['patience'][model] is not None and stale>=C['patience'][model]:break
    if best is None:
        best=checkpoint(net,context,epoch,None,pv);save_torch(P35/'checkpoints'/f'{name}__best.pt',best)
    artifacts=[]
    for role,payload in [('best',best)]+([('early2',prefix)] if model=='dnabert2' else []):
        p=P35/'validation'/f'{name}__{role}.csv'
        csvout(p,pd.DataFrame({'record_id':va.record_id,'prediction':payload['validation_predictions']}))
        for artifact in [p,P35/'checkpoints'/f'{name}__{role}.pt']:artifacts.append({'file':artifact.relative_to(ROOT).as_posix(),'sha256':sha(artifact),'role':role})
    logpath=P35/'training_logs'/f'{name}.csv';artifacts.append({'file':logpath.relative_to(ROOT).as_posix(),'sha256':sha(logpath),'role':'training_log'})
    score=best['validation_pearson'];cap=best['epoch']>=epochs-2 or (len(logs)>=3 and logs[-1]['validation_pearson'] is not None and logs[-3]['validation_pearson'] is not None and logs[-1]['validation_pearson']-logs[-3]['validation_pearson']>.002 and len(logs)==epochs)
    meta=dict(context,selected_epoch=best['epoch'],validation_pearson=score,early2_selected_epoch=prefix['epoch'] if prefix else None,early2_validation_pearson=prefix['validation_pearson'] if prefix else None,epochs_run=len(logs),maximum_epochs=epochs,early_stopped=len(logs)<epochs,possibly_budget_limited=bool(cap),collapsed=score is None,validation_prediction_sd=float(np.std(best['validation_predictions'])),AMP_skipped_optimizer_steps=overflow,optimizer_updates=updates,training_examples_seen=examples,elapsed_seconds=elapsed_prior+time.monotonic()-t0,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),artifacts=artifacts)
    jout(done,meta);update_summary();jout(P35/'live_progress.json',{'job':name,'status':'completed','completed_jobs':len(list((P35/'jobs').glob('*.json'))),'expected_jobs':54})
    del net,tok,opt,scaler,best,prefix,yz;gc.collect();torch.cuda.empty_cache();return meta
def finish_training():
    rows=[]
    for args in jobs():
        m=jread(P35/'jobs'/f'{key(*args)}.json')
        for item in m['artifacts']:assert sha(ROOT/item['file'])==item['sha256']
        rows.append({k:v for k,v in m.items() if k!='artifacts'})
    assert len(rows)==54;csvout(P35/'training_job_summary.csv',pd.DataFrame(rows));protected_check()
    jout(P35/'summary.json',{'status':'completed','completed_jobs':54,'expected_jobs':54,'test_inference_started':False,'all_artifact_hashes_passed':True,'next_phase':'36:freeze validation choices then infer'})
    print('All54validation-only training fits complete.',flush=True)
