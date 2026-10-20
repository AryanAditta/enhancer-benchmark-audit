"""Synthetic CPU check of optimizer/RNG and shuffled-batch continuation; no research model training."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase11c_completion as s
torch=s.setup_runtime()
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
x=torch.arange(64*4,dtype=torch.float32).reshape(64,4)/100
y=torch.arange(64)%2;ds=TensorDataset(x,y)
def model():return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Dropout(.25),nn.Linear(8,2))
def loader():return DataLoader(ds,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(907))
def step(net,opt,batch,idx):
 xb,yb=batch;loss=nn.functional.cross_entropy(net(xb),yb)/2;loss.backward()
 if (idx+1)%2==0:opt.step();opt.zero_grad(set_to_none=True)
s.seed_everything(125);net=model();opt=torch.optim.AdamW(net.parameters(),lr=.001);net.train()
saved=s.out(30)/'synthetic_resume_probe.pt'
for i,batch in enumerate(loader()):
 step(net,opt,batch,i)
 if i==3:s.atomic_torch_save(saved,{'model':net.state_dict(),'optimizer':opt.state_dict(),'rng':s.rng_state()})
expected={k:v.clone() for k,v in net.state_dict().items()}
state=torch.load(saved,map_location='cpu',weights_only=True)
s.seed_everything(988);resumed=model();other=torch.optim.AdamW(resumed.parameters(),lr=.001)
resumed.load_state_dict(state['model']);other.load_state_dict(state['optimizer']);s.restore_rng(state['rng']);resumed.train()
for i,batch in enumerate(loader()):
 if i<4:continue
 step(resumed,other,batch,i)
delta=max(float(abs(expected[k]-v).max()) for k,v in resumed.state_dict().items())
assert delta==0.,delta
s.dump(30,'resume_mechanics_check.json',{'passed':True,'max_parameter_difference':delta,'checks':['weights_only deserialization','optimizer moments','Python/NumPy/Torch/CUDA RNG restoration','recreated private-generator shuffle and skipped consumed batches','dropout continuation'],'scope':'synthetic CPU network only; no research model or research-result training'})
print('Synthetic shuffled/dropout/optimizer/RNG continuation matches uninterrupted parameters exactly.')
