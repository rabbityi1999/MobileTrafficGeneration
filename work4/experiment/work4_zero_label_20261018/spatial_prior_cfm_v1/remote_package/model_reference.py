# -*- coding: utf-8 -*-
"""Source-only ordinary conditional flow matching baseline. No target-data path."""
import argparse,copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
class CFM(nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('freq',2**torch.arange(8,dtype=torch.float32)*math.pi)
  self.net=nn.Sequential(nn.Linear(168+130+16,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,168))
 def forward(self,y,x,t):
  phase=t[:,None]*self.freq[None];return self.net(torch.cat([y,x,phase.sin(),phase.cos()],dim=-1))
def loss(model,x,y):
 z=torch.randn_like(y);t=torch.rand(y.shape[0],device=y.device);yt=(1-t[:,None])*z+t[:,None]*y
 return torch.mean((model(yt,x,t)-(y-z))**2)
@torch.no_grad()
def generate(model,x,samples=20,steps=20,seed=530001):
 model.eval();gen=torch.Generator(device=x.device).manual_seed(seed)
 expanded=x[:,None,:].expand(-1,samples,-1).reshape(-1,x.shape[1])
 y=torch.randn((len(expanded),168),device=x.device,generator=gen)
 for k in range(steps):y=y+model(y,expanded,torch.full((len(y),),k/steps,device=y.device))/steps
 return ((y.reshape(len(x),samples,168)+1)/2).clamp(0,1).cpu().numpy()
def crps(y,s):
 s=np.asarray(s,dtype='float64');y=np.asarray(y,dtype='float64');n=s.shape[1];c=(2*np.arange(1,n+1)-n-1)[None,:,None]
 return float(np.abs(s-y[:,None,:]).mean()-(np.sort(s,axis=1)*c).sum(1).mean()/n**2)
def validate(model,x,y):
 scores=[]
 for i in range(0,len(x),16):scores.append((len(x[i:i+16]),crps(y[i:i+16],generate(model,x[i:i+16],seed=530001+i))))
 return sum(n*v for n,v in scores)/sum(n for n,v in scores)
def smoke(out):
 torch.manual_seed(1);model=CFM();optim=torch.optim.AdamW(model.parameters(),lr=3e-4);x=torch.randn(4,130);y=torch.rand(4,168)*2-1
 before={k:v.clone() for k,v in model.state_dict().items()};ls=[]
 for _ in range(3):
  optim.zero_grad();v=loss(model,x,y);assert torch.isfinite(v);v.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters());optim.step();ls.append(float(v.detach()))
 a=generate(model,x,3,2,19);b=generate(model,x,3,2,19)
 assert a.shape==(4,3,168) and np.array_equal(a,b) and np.isfinite(a).all()
 assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
 # Identical deterministic draws reduce CRPS to MAE.
 yy=np.zeros((2,168));ss=np.ones((2,3,168));assert abs(crps(yy,ss)-1)<1e-12
 write(out,{'status':'PASS','synthetic_only':True,'gpu_used':False,'training_updates':3,'finite_losses':ls,'deterministic_noise':True,'shape':list(a.shape),'crps_point_mass_identity':True})
def train(args):
 base=Path(__file__).resolve().parent
 manifest=json.loads((base/'SOURCE_PACKAGE.json').read_text(encoding='utf-8-sig'))
 assert digest(base/'source_only.npz')==manifest['sha256']
 seal=json.loads((base/'STAGE0_SEAL.json').read_text(encoding='utf-8-sig'))
 for name,h in seal['files'].items():assert digest(base/name)==h,name
 assert args.device=='cuda:0','Real source pilot uses an allocated GPU; CPU mode is synthetic-smoke only.'
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 with np.load(base/'source_only.npz',allow_pickle=False) as z:
  assert set(z.files)==set(manifest['array_allowlist']);xtr=z['x_train'].copy();xva=z['x_valid'][:128].copy();ytr=z['y_train'].copy();yva=z['y_valid'][:128].copy()
 device=torch.device(args.device);assert torch.cuda.is_available()
 xtr=torch.tensor(xtr,device=device);xva=torch.tensor(xva,device=device)
 write(out/'STARTED.json',{'pid':os.getpid(),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_package_sha256':manifest['sha256'],'seal_sha256':digest(base/'STAGE0_SEAL.json'),'labels':['tn/train','tn/validation'],'target_labels':False,'torch':torch.__version__,'started_unix':time.time()})
 receipt={}
 for c,name in enumerate(['call_in','call_out']):
  torch.manual_seed(1);torch.cuda.manual_seed_all(1);rng=np.random.RandomState(330001);model=CFM().to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
  yt=torch.tensor(ytr[:,c]*2-1,device=device);best=float('inf');beststate=None;beststep=None;started=time.monotonic()
  for step in range(1,2001):
   model.train();idx=rng.choice(len(xtr),128,replace=False);idx=torch.tensor(idx,device=device);opt.zero_grad(set_to_none=True);v=loss(model,xtr[idx],yt[idx]);assert torch.isfinite(v)
   v.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(g);opt.step()
   if step%200==0:
    row={'channel':name,'step':step,'loss':float(v.detach()),'seconds':time.monotonic()-started}
    if step in [1000,2000]:
     score=validate(model,xva,yva[:128,c]);assert np.isfinite(score);row['source_dev_sample_crps']=score
     if score<best:best=score;beststep=step;beststate={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
    with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
  path=out/(name+'.pt');torch.save(beststate,path)
  receipt[name]={'checkpoint_sha256':digest(path),'best_step':beststep,'source_dev_sample_crps':best,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.monotonic()-started}
 write(out/'TRAINING_COMPLETE.json',{'status':'COMPLETE_SOURCE_BASELINE_ONLY','stages':receipt,'target_labels_accessed':False,'target_predictions_made':False,'label_budget_target':0})
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke-output');p.add_argument('--output');p.add_argument('--device',default='cuda:0');args=p.parse_args();torch.set_num_threads(4)
 if args.smoke_output:smoke(args.smoke_output)
 else:
  assert args.output
  try:train(args)
  except Exception as e:
   out=Path(args.output)
   if out.is_dir() and not (out/'FAILED.json').exists():write(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'no_auto_retry':True})
   raise
if __name__=='__main__':main()
