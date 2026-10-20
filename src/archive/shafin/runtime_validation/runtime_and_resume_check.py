"""Synthetic checks only. No assay targets or scientific model optimizer steps."""
from pathlib import Path
import sys,json,gc,random
import numpy as np
root=Path.cwd();sys.path.insert(0,str(root/'analysis_tools'))
import mpra_fairness_training as s
torch=s.runtime();s.setup()
device='cuda'
data=np.random.default_rng(43).normal(size=(48,16)).astype('float32')
target=np.random.default_rng(44).normal(size=48).astype('float32')
def build():
    net=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(32,1)).to(device)
    return net,torch.optim.AdamW(net.parameters(),lr=1e-3),torch.amp.GradScaler('cuda')
def step(net,opt,scaler,begin,end):
    net.train();opt.zero_grad(set_to_none=True)
    for i in range(begin,end):
        with torch.autocast('cuda',dtype=torch.float16):
            loss=torch.nn.functional.mse_loss(net(torch.tensor(data[i*8:(i+1)*8],device=device)).flatten().float(),torch.tensor(target[i*8:(i+1)*8],device=device))
        scaler.scale(loss/2).backward()
        if (i+1)%2==0:
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(net.parameters(),1.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
s.seed_all(1337);net,opt,scaler=build();step(net,opt,scaler,0,6);expected=s.state_cpu(net)
del net,opt,scaler;gc.collect();torch.cuda.empty_cache()
s.seed_all(1337);net,opt,scaler=build();step(net,opt,scaler,0,4)
payload={'net':s.state_cpu(net),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'best':{'state_dict':s.state_cpu(net),'validation_predictions':[.1,.2],'validation_pearson':.3},'prefix':None}
p=root/'phase34_outputs/synthetic_resume_probe.pt';s.save_torch(p,payload);h=s.sha(p)
del net,opt,scaler,payload;gc.collect();torch.cuda.empty_cache()
net,opt,scaler=build();assert s.sha(p)==h;loaded=torch.load(p,map_location='cpu',weights_only=True)
net.load_state_dict(loaded['net']);opt.load_state_dict(loaded['optimizer']);scaler.load_state_dict(loaded['scaler']);torch.set_rng_state(loaded['torch_rng']);torch.cuda.set_rng_state_all(loaded['cuda_rng']);random.setstate(loaded['python_rng'])
step(net,opt,scaler,4,6);actual=s.state_cpu(net)
delta=max(float((actual[k]-expected[k]).abs().max()) for k in expected);assert delta==0,delta
# Weighted Pearson must equal literal duplicate-row bootstrap calculations.
y=np.array([1.,2.,4.,8.,16.]);q=np.array([1.5,2.2,3.1,9.,12.]);w=np.array([2,0,3,1,2]);my=np.average(y,weights=w);mp=np.average(q,weights=w)
weighted=np.sum(w*(y-my)*(q-mp))/np.sqrt(np.sum(w*(y-my)**2)*np.sum(w*(q-mp)**2))
ix=np.repeat(np.arange(len(y)),w);difference=abs(weighted-np.corrcoef(y[ix],q[ix])[0,1]);assert difference<1e-12
result={'passed':True,'synthetic_only':True,'research_model_optimizer_steps':0,'serialized_GPU_AdamW_AMP_dropout_resume_max_parameter_difference':delta,'weighted_bootstrap_Pearson_literal_difference':difference,'resume_probe_sha256':h}
s.jout(root/'phase34_outputs/runtime_and_resume_check.json',result);print(json.dumps(result,indent=2))
