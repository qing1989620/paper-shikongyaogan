"""Finite, serial experiment queue with a PID lock and resumable child runs.

Run status and reports reflect actual child processes; queued does not mean run.
No test labels are consulted by selection; all warm-start pilots are development.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from research_runtime import CHECKPOINT_HEADROOM_BYTES, write_json as write

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'work/runs/research_2026'
CONFIG=ROOT/'work/configs/screening_2026/queue_v1.json'

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def alive(pid):
    r=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'],capture_output=True,
                     text=True,encoding='utf-8',errors='replace')
    return f'"{pid}"' in (r.stdout or '')

def aggregate(cfg):
    rows=[]
    for method in cfg['methods']:
        p=DIR/f'screen_{method}_seed42_v1'
        if not (p/'COMPLETE.json').exists():continue
        s=json.loads((p/'summary.json').read_text(encoding='utf-8'))
        rows.append({k:s.get(k) for k in ('method','seed','mean_patch_rmse','ssim','sam_rad','ergas','cc','best_epoch','peak_cuda_MiB')})
    base=next((r['mean_patch_rmse'] for r in rows if r['method']=='base'),None)
    if base:
        for r in rows:r['relative_improvement_pct']=100*(base-r['mean_patch_rmse'])/base
    if rows:
        path=DIR/'screening_progress.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

def main():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    DIR.mkdir(parents=True,exist_ok=True);lock=DIR/'QUEUE_LOCK.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        if alive(old['pid']):raise RuntimeError(f'Queue already active: {old["pid"]}')
    write(lock,{'pid':os.getpid(),'started_cst':now()})
    state={'pid':os.getpid(),'started_cst':now(),'state':'running','jobs':[],'protocol':'screening_2026_v1',
           'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest()}
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for method in cfg['methods']:
        run_id=f'screen_{method}_seed42_v1';out=DIR/run_id
        record={'method':method,'run_id':run_id,'state':'queued'};state['jobs'].append(record)
    write(DIR/'QUEUE_STATUS.json',state)
    for record in state['jobs']:
        method=record['method'];out=DIR/record['run_id']
        if (out/'COMPLETE.json').exists():
            record.update(state='complete_existing');continue
        if shutil.disk_usage(DIR).free<CHECKPOINT_HEADROOM_BYTES:
            record.update(state='waiting_for_disk',reason='insufficient_checkpoint_write_space')
            state.update(state='attention_required',heartbeat_cst=now())
            write(DIR/'QUEUE_STATUS.json',state)
            return
        out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-B','-u','-X','utf8',str(ROOT/'work/tools/train_research_2026.py'),
             '--method',method,'--phase','screening','--dataset','LGC','--epochs',str(cfg['epochs']),
             '--seed','42','--lr',str(cfg['learning_rate']),'--accum',str(cfg['gradient_accumulation']),
             '--warmstart',str(ROOT/cfg['shared_warmstart']),'--out',str(out)]
        if (out/'config.json').exists():cmd+=['--resume']
        record.update(state='running',started_cst=now())
        write(DIR/'QUEUE_STATUS.json',state)
        with (out/'console.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            record['child_pid']=child.pid;write(DIR/'QUEUE_STATUS.json',state)
            print(f'START {method} pid={child.pid}',flush=True)
            while child.poll() is None:
                time.sleep(15)
                state['heartbeat_cst']=now();write(DIR/'QUEUE_STATUS.json',state)
            record.update(exit_code=child.returncode,finished_cst=now(),
                          state='complete' if child.returncode==0 and (out/'COMPLETE.json').exists() else 'failed')
        aggregate(cfg);write(DIR/'QUEUE_STATUS.json',state)
        print(record['state'].upper(),method,flush=True)
        if record['state']=='failed':
            state.update(state='attention_required',heartbeat_cst=now())
            write(DIR/'QUEUE_STATUS.json',state)
            return
    state.update(state='complete' if all(r['state'].startswith('complete') for r in state['jobs']) else 'attention_required',finished_cst=now())
    rows=aggregate(cfg)
    # This proposes follow-up experiments; it never labels a winner SOTA.
    candidates=sorted([r for r in rows if r['method']!='base' and r.get('relative_improvement_pct',0)>=cfg['promotion_min_rmse_pct']],
                      key=lambda r:r['mean_patch_rmse'])[:cfg['promotion_max_candidates']]
    write(DIR/'FOLLOWUP_PROPOSAL.json',{'time_cst':now(),'selection_data':'exposed_LGC_validation_only',
          'candidates':candidates,'next':'from-scratch paired 40-epoch runs with seeds 42/123/2026, then external SOTA and CIA; no automatic paper claims'})
    write(DIR/'QUEUE_STATUS.json',state);print('QUEUE',state['state'],flush=True)

if __name__=='__main__':main()
