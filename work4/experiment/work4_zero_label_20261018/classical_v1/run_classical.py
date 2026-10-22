# -*- coding: utf-8 -*-
"""Source-only classical probability baselines under a prewritten protocol."""
from pathlib import Path
import json,hashlib,time
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def metrics(y,samples):
 s=np.asarray(samples,dtype='float64');y=np.asarray(y,dtype='float64');m=s.shape[1]
 coef=(2*np.arange(1,m+1)-m-1)[None,:,None];ordered=np.sort(s,axis=1)
 crps=np.abs(s-y[:,None]).mean((1,2))-(ordered*coef).sum(1).mean(1)/m**2
 low,high=np.quantile(s,[.05,.95],axis=1)
 return {'sample_crps':float(crps.mean()),'sample_mean_mae':float(np.abs(s.mean(1)-y).mean()),'coverage90':float(((y>=low)&(y<=high)).mean()),'width90':float((high-low).mean()),'per_region_crps':crps.tolist()}
def main():
 cfg=json.loads((BASE/'PROTOCOL.json').read_text(encoding='utf-8'));seal=json.loads((BASE/'SEAL.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==seal['script_sha256'];assert sha(BASE/'PROTOCOL.json')==seal['protocol_sha256'];assert sha(ROOT/'source_only.npz')==cfg['source_package_sha256']
 out=BASE/'results';out.mkdir(exist_ok=False);start=time.monotonic()
 with np.load(ROOT/'source_only.npz',allow_pickle=False) as z:
  x=z['x_train'].astype('float64');xv=z['x_valid'][:128].astype('float64');y=z['y_train'].astype('float64');yv=z['y_valid'][:128].astype('float64');rows=z['train_rows'];vrows=z['valid_rows'][:128]
 perm=np.random.RandomState(cfg['split_seed']).permutation(len(x));fit=np.sort(perm[:3156]);cal=np.sort(perm[3156:]);assert len(cal)==789 and not np.intersect1d(fit,cal).size
 draws=np.random.RandomState(cfg['sample_seed']).randint(0,len(cal),size=(len(xv),20));full=np.random.RandomState(cfg['sample_seed']).randint(0,len(y),size=(len(xv),20))
 np.savez_compressed(out/'indices.npz',fit=fit,calibration=cal,calibration_draws=draws,full_draws=full,train_rows=rows,dev_rows=vrows)
 predictions={};results=[];predmeta={};coeff={}
 def add(name,channel,samples,**extra):
  key=name+'__'+channel;predictions[key]=np.asarray(samples,dtype='float32');c=cfg['channels'].index(channel);m=metrics(yv[:,c],predictions[key]);row={'method':name,'channel':channel,**extra,**m};results.append(row);predmeta[key]={'method':name,'channel':channel};return row
 for c,channel in enumerate(cfg['channels']):
  add('SourceMean',channel,np.repeat(y[:,c].mean(0)[None,None],len(xv)*20,axis=0).reshape(len(xv),20,168))
  add('SourceEmpiricalFull',channel,y[full,c])
  add('SourceEmpiricalCalibration',channel,y[cal[draws],c])
 yflat=y.reshape(len(y),-1)
 for view,cols in [('satellite',list(range(2,130))),('coordinates',[0,1]),('all',list(range(130))),('all_permuted_fit_labels',list(range(130)))]:
  xf=x[fit][:,cols];xc=x[cal][:,cols];xq=xv[:,cols]
  center=xf.mean(0);scale=np.maximum(xf.std(0),1e-3);xf=(xf-center)/scale;xc=(xc-center)/scale;xq=(xq-center)/scale
  labels=yflat[fit].copy()
  if view=='all_permuted_fit_labels':labels=labels[np.random.RandomState(cfg['permutation_seed']).permutation(len(labels))]
  ym=labels.mean(0);rhs=xf.T@(labels-ym);gram=xf.T@xf
  for alpha in cfg['alphas']:
   beta=np.linalg.solve(gram+alpha*np.eye(len(cols)),rhs);mu=ym+xq@beta;res=yflat[cal]-(ym+xc@beta)
   rr=(mu[:,None,:]+res[draws]).reshape(len(xq),20,2,168);clipped=np.clip(rr,0,1)
   name='SourceRidgeResidual_'+view+'_a'+str(alpha);coeff[name+'__beta']=beta;coeff[name+'__center']=center;coeff[name+'__scale']=scale;coeff[name+'__intercept']=ym;coeff[name+'__residuals']=res
   for c,channel in enumerate(cfg['channels']):add(name,channel,clipped[:,:,c,:],view=view,alpha=alpha,clip_fraction=float(((rr[:,:,c,:]<0)|(rr[:,:,c,:]>1)).mean()))
 np.savez_compressed(out/'predictions.npz',**predictions);np.savez_compressed(out/'ridge_models.npz',**coeff)
 selection={}
 for channel in cfg['channels']:
  selection[channel]={}
  for view in cfg['views']:
   candidates=[r for r in results if r.get('view')==view and r['channel']==channel]
   best=min(candidates,key=lambda r:(r['sample_crps'],r['alpha']));selection[channel][view]={k:v for k,v in best.items() if k!='per_region_crps'}
 report={'status':'COMPLETE_SOURCE_DEVELOPMENT','role':'source dev model selection; not a new test or cross-city result','target_labels_accessed':False,'target_predictions_made':False,'source_roles':{'regression_fit':len(fit),'residual_calibration':len(cal),'source_development_queries':len(xv)},'sample_count':20,'seed':cfg['sample_seed'],'records':results,'selection':selection,'seconds':time.monotonic()-start,'protocol_sha256':sha(BASE/'PROTOCOL.json'),'source_package_sha256':sha(ROOT/'source_only.npz'),'script_sha256':sha(Path(__file__)),'outputs':{p.name:sha(p) for p in out.glob('*.npz')}}
 write(out/'RESULTS.json',report)
 for channel in cfg['channels']:
  print(channel)
  for r in results:
   if r['channel']==channel and (not r.get('view') or any(r['method']==v['method'] for v in selection[channel].values())):print(json.dumps({k:r[k] for k in ['method','sample_crps','sample_mean_mae','coverage90','width90']},ensure_ascii=False))
 print('seconds',report['seconds'])
if __name__=='__main__':main()
