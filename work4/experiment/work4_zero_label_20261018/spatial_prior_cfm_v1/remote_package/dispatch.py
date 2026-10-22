# -*- coding: utf-8 -*-
from pathlib import Path
import json,os,sys,subprocess,time,fcntl,argparse
from gpu_guard_reference import snapshot,idle
B=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',action='store_true');args=ap.parse_args()
    lock=(B/'dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=B/'LAUNCH.json'
    if marker.exists():
        receipt=json.loads(marker.read_text());state='ALREADY_LAUNCHED'
        if (B/'OUTPUT_SEAL.json').exists():state='COMPLETE'
        elif (B/'FAILED.json').exists():state='FAILED_NO_AUTO_RETRY'
        elif receipt.get('pid') and not Path('/proc/'+str(receipt['pid'])).exists():state='EXITED_REVIEW_REQUIRED'
        print(json.dumps({'status':state,'launch':receipt}));return
    a=snapshot();time.sleep(2);b=snapshot();candidates=sorted(set(idle(a))&set(idle(b)))
    preflight={'time_unix':time.time(),'first':a,'second':b,'idle_candidates':candidates}
    if not args.start or not candidates:
        print(json.dumps({'status':'IDLE_READY' if candidates else 'BUSY_NO_LAUNCH','preflight':preflight}));return
    gpu=candidates[0];final=snapshot();preflight['final']=final
    if gpu not in idle(final):print(json.dumps({'status':'BECAME_BUSY_NO_LAUNCH','preflight':preflight}));return
    assert not (B/'run').exists()
    receipt={'status':'RESERVED','gpu':gpu,'pid':None,'time_unix':time.time(),'preflight':preflight}
    with marker.open('x') as f:json.dump(receipt,f,indent=2)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
    cmd=[sys.executable,'-u',str(B/'worker.py')]
    with (B/'training.log').open('x') as log:proc=subprocess.Popen(cmd,cwd=str(B),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(status='LAUNCHED',pid=proc.pid,command=cmd);marker.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
