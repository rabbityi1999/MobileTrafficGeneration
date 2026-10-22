# -*- coding: utf-8 -*-
"""Start stage0 once on a genuinely idle GPU, or return BUSY without queueing."""
import argparse,csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def parse(gpus,processes):
 busy={r[0].strip() for r in csv.reader(processes.splitlines()) if r}
 rows=[]
 for r in csv.reader(gpus.splitlines()):
  if not r:continue
  index,uuid,used,util=[x.strip() for x in r]
  rows.append({'index':int(index),'uuid':uuid,'memory_used_mib':int(used),'utilization_percent':int(util),'has_compute_process':uuid in busy})
 return rows

def idle(rows):return [r['index'] for r in rows if not r['has_compute_process'] and r['memory_used_mib']<1024 and r['utilization_percent']<=5]
def snapshot():
 def query(args):
  r=subprocess.run(['nvidia-smi']+args,capture_output=True,text=True,timeout=20)
  if r.returncode:raise RuntimeError('nvidia-smi unavailable; refusing launch')
  return r.stdout.strip()
 return parse(query(['--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']),query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']))
def main():
 p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
 if args.self_test:
  rows=parse('0, GPU-a, 0, 0\n1, GPU-b, 18330, 8\n2, GPU-c, 200, 0\n3, GPU-d, 0, 98','GPU-b, 25\nGPU-c, 26')
  assert idle(rows)==[0];print(json.dumps({'status':'PASS','guard_tests':['occupied_low_util_rejected','process_without_large_memory_rejected','high_util_rejected','idle_accepted']}));return
 lock=(ROOT/'dispatcher.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 run=ROOT/'runs/stage0_seed1';marker=ROOT/'LAUNCH.json'
 if marker.exists():
  receipt=json.loads(marker.read_text());pid=receipt.get('pid');state='ALREADY_LAUNCHED'
  if (run/'TRAINING_COMPLETE.json').exists():state='COMPLETE'
  elif (run/'FAILED.json').exists():state='FAILED_NO_AUTO_RETRY'
  elif pid and not Path('/proc/'+str(pid)).exists():state='EXITED_REVIEW_REQUIRED'
  print(json.dumps({'status':state,'launch':receipt}));return
 a=snapshot();time.sleep(2);b=snapshot();candidates=sorted(set(idle(a))&set(idle(b)))
 if not args.start or not candidates:
  print(json.dumps({'status':'BUSY_NO_LAUNCH' if not candidates else 'IDLE_READY','gpus':b,'idle_candidates':candidates}));return
 gpu=candidates[0]
 # Final metadata check, directly before spawning; no existing process is terminated.
 if gpu not in idle(snapshot()):print(json.dumps({'status':'BECAME_BUSY_NO_LAUNCH'}));return
 assert not run.exists();(ROOT/'runs').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
 reservation={'status':'RESERVED','gpu':gpu,'pid':None,'time_unix':time.time(),'run':str(run)}
 with marker.open('x') as f:json.dump(reservation,f,indent=2)
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
 cmd=[sys.executable,'-u',str(ROOT/'train_stage0.py'),'--output',str(run),'--device','cuda:0']
 with (ROOT/'logs/stage0_seed1.log').open('x') as log:
  proc=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 reservation.update(status='LAUNCHED',pid=proc.pid,command=cmd)
 marker.write_text(json.dumps(reservation,indent=2));print(json.dumps(reservation))
if __name__=='__main__':main()
