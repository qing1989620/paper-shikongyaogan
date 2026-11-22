"""Finite continuation of this experiment program, not a scheduled automation.

After all screening children stop, evaluate the original shared control, check
quality gates, and train every method plus base from scratch for 3 seeds.
Unresolved external-comparison/novelty gates remain explicit; no paper claims.
"""
from __future__ import annotations
import argparse
import csv
import math
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from research_runtime import CHECKPOINT_HEADROOM_BYTES, write_json as write

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'work/runs/research_2026'
CONFIG=ROOT/'work/configs/screening_2026/followup_v2.json'
PYTHON=ROOT/'work/envs/research_v1/Scripts/python.exe'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def refresh_report():
    if os.environ.get('STF_MANAGED_RESEARCH')=='1':
        return
    try:
        import update_research_report
        update_research_report.main()
    except (OSError,ValueError,KeyError) as error:
        print('Report refresh deferred:',type(error).__name__,str(error),flush=True)

def select_candidates(original,trained_base,summaries,cfg):
    ref=min((original,trained_base),key=lambda s:s['mean_patch_rmse'])
    candidates=[]
    for summary in summaries:
        gain=100*(ref['mean_patch_rmse']-summary['mean_patch_rmse'])/ref['mean_patch_rmse']
        harm=[]
        for metric in ('ssim','cc'):
            if summary[metric] is None or not math.isfinite(summary[metric]) or summary[metric]<.99*ref[metric]:harm.append(metric)
        for metric in ('sam_rad','ergas'):
            if summary[metric] is None or not math.isfinite(summary[metric]) or summary[metric]>1.01*ref[metric]:harm.append(metric)
        candidates.append({'method':summary['method'],'gain_pct':gain,'harm_metrics':harm,
                           'rmse':summary['mean_patch_rmse'],'passes_screen':gain>=cfg['minimum_gain_pct'] and not harm})
    candidates.sort(key=lambda r:r['rmse'])
    selected=[r for r in candidates if r['passes_screen']][:cfg['maximum_candidates']]
    if not selected and cfg['run_diagnostic_if_no_promotion']:
        selected=candidates[:cfg['diagnostic_candidate_count']]
        mode='diagnostic_full_budget_no_positive_screen_claim'
    else:mode='screen_passed_exploratory_replication_not_external_confirmation'
    return {'mode':mode,'reference_rmse':ref['mean_patch_rmse'],'all_candidates':candidates,
              'selected':selected,'unresolved_gates':['external_SOTA_common_protocol','CIA_in_domain','fulltext_novelty_comparison','mechanism_ablation']}

def select_replicated(summaries,cfg):
    """Select only after ALL methods finish, using paired from-scratch validation."""
    by_method={}
    for s in summaries:by_method.setdefault(s['method'],[]).append(s)
    wanted=set(cfg['seeds'])
    for method,rows in by_method.items():
        if len(rows)!=len(wanted) or {r['seed'] for r in rows}!=wanted:
            raise ValueError('Incomplete/duplicate seed block: '+method)
        for r in rows:
            if not math.isfinite(r['mean_patch_rmse']) or r['mean_patch_rmse']<=0:
                raise ValueError('Invalid replicated RMSE: '+method)
    base={r['seed']:r for r in by_method['base']}
    ref=statistics.mean(r['mean_patch_rmse'] for r in base.values())
    candidates=[];limit=cfg['quality_metric_relative_harm_limit_pct']/100
    for method,rows in by_method.items():
        if method=='base':continue
        rmse=statistics.mean(r['mean_patch_rmse'] for r in rows)
        gains={str(r['seed']):100*(base[r['seed']]['mean_patch_rmse']-r['mean_patch_rmse'])/base[r['seed']]['mean_patch_rmse'] for r in rows}
        harm=[]
        for r in rows:
            for metric in ('ssim','cc','sam_rad','ergas'):
                v=r.get(metric);b=base[r['seed']].get(metric)
                if v is None or b is None or not math.isfinite(v) or not math.isfinite(b):harm.append(f'{r["seed"]}:{metric}:undefined')
                elif (metric in ('ssim','cc') and v<(1-limit)*b) or (metric in ('sam_rad','ergas') and v>(1+limit)*b):harm.append(f'{r["seed"]}:{metric}')
        gain=100*(ref-rmse)/ref
        candidates.append({'method':method,'rmse':rmse,'rmse_training_seed_sd':statistics.stdev(r['mean_patch_rmse'] for r in rows),
                           'gain_pct':gain,'paired_seed_gains_pct':gains,'harm_metrics':harm,
                           'passes_replication':gain>=cfg['minimum_gain_pct'] and min(gains.values())>0 and not harm})
    candidates.sort(key=lambda r:r['rmse'])
    selected=[r for r in candidates if r['passes_replication']][:cfg['maximum_candidates']]
    mode='three_seed_from_scratch_validation_selection_not_external_confirmation'
    if not selected and cfg['run_diagnostic_if_no_promotion']:
        selected=candidates[:cfg['diagnostic_candidate_count']];mode='three_seed_diagnostic_no_positive_claim'
    return {'mode':mode,'reference_rmse':ref,'selection_data':'exposed_frozen_LGC_val_only_no_flood_scores',
            'trained_methods_including_base':len(by_method),'training_seeds':cfg['seeds'],
            'all_candidates':candidates,'selected':selected,
            'unresolved_gates':['matched_training_external_benchmark','CIA_in_domain','fulltext_novelty_comparison','mechanism_ablation','independent_confirmation']}

def original_control():
    out=ROOT/'work/results/research_2026/original_base_control'
    cached=out/'summary.json'
    if cached.exists():
        return json.loads(cached.read_text(encoding='utf-8'))
    sys.path.insert(0,str(ROOT/'work/tools'))
    import torch
    from torch.utils.data import DataLoader
    from train_research_2026 import ResearchModel,Patches,alphas,validate,atomic_json,sha
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=ResearchModel('base',42);model.load_shared_checkpoint(ROOT/'work/runs/base/best.pth')
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    out.mkdir(parents=True,exist_ok=True)
    s=validate(model,DataLoader(Patches('LGC','val'),batch_size=1),alphas(device),device,
               full_metrics=True,out=out/'per_patch_validation.csv')
    s.update(source_checkpoint_sha256=sha(ROOT/'work/runs/base/best.pth'),role='unmodified_shared_epoch0_control')
    atomic_json(cached,s)
    del model
    if device=='cuda':torch.cuda.empty_cache()
    return s

def spawn_run(method,seed,cfg,state):
    out=RUNS/f'replicate_{method}_seed{seed}_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return {'method':method,'seed':seed,'state':'complete_existing','run_id':out.name}
    if shutil.disk_usage(RUNS).free<CHECKPOINT_HEADROOM_BYTES:raise RuntimeError('Disk space below checkpoint write reserve (512 MiB)')
    cmd=[str(PYTHON),'-B','-u','-X','utf8',str(ROOT/'work/tools/train_research_2026.py'),
         '--method',method,'--phase','replication','--epochs',str(cfg['epochs']),
         '--seed',str(seed),'--lr',str(cfg['learning_rate']),'--accum','2','--out',str(out)]
    if (out/'config.json').exists():cmd+=['--resume']
    record={'method':method,'seed':seed,'state':'running','run_id':out.name}
    state['current']=record;write(RUNS/'FOLLOWUP_STATUS.json',state)
    with (out/'console.log').open('a',encoding='utf-8') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                               env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'))
        record['pid']=child.pid;write(RUNS/'FOLLOWUP_STATUS.json',state)
        while child.poll() is None:
            time.sleep(15)
        record.update(state='complete' if child.returncode==0 and (out/'COMPLETE.json').exists() else 'failed',exit_code=child.returncode)
    return record

def main(stop_after_seed=None):
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError('Start this controller with work/envs/research_v1/Scripts/python.exe')
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    if stop_after_seed is not None and stop_after_seed not in cfg['seeds']:
        raise ValueError(f'Seed {stop_after_seed} is absent from the replication plan')
    state={'pid':os.getpid(),'state':'waiting_for_screening','completed_jobs':[],
                 'python_executable':sys.executable,'config_sha256':__import__('hashlib').sha256(CONFIG.read_bytes()).hexdigest()}
    if stop_after_seed is not None:
        state['stop_after_seed']=stop_after_seed
    lock=RUNS/'FOLLOWUP_LOCK.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        alive=subprocess.run(['tasklist','/FI',f'PID eq {old["pid"]}','/FO','CSV','/NH'],capture_output=True,
                             text=True,encoding='utf-8',errors='replace')
        if f'"{old["pid"]}"' in (alive.stdout or ''):raise RuntimeError('Continuation is already active')
    write(lock,{'pid':os.getpid()});write(RUNS/'FOLLOWUP_STATUS.json',state)
    while True:
        refresh_report()
        q=json.loads((RUNS/'QUEUE_STATUS.json').read_text(encoding='utf-8'))
        if q['state'] in ('complete','attention_required'):break
        time.sleep(30)
    if q['state']!='complete':
        state.update(state='waiting_for_screening_repair');write(RUNS/'FOLLOWUP_STATUS.json',state);return
    state.update(state='evaluating_original_control');write(RUNS/'FOLLOWUP_STATUS.json',state)
    original=original_control()
    base=json.loads((RUNS/'screen_base_seed42_v1/summary.json').read_text(encoding='utf-8'))
    summaries=[]
    for job in q['jobs']:
        if job['method']=='base':continue
        summary=json.loads((RUNS/job['run_id']/'summary.json').read_text(encoding='utf-8'))
        summaries.append(summary)
    decision=select_candidates(original,base,summaries,cfg)
    priority=[r['method'] for r in decision['selected']]
    write(RUNS/'SCREENING_DECISION.json',decision)
    order=['base']+priority+[j['method'] for j in q['jobs'] if j['method']!='base' and j['method'] not in priority]
    state.update(state='all_methods_from_scratch_replication',decision_mode='all_14_methods_not_screening_elimination',
                 priority_methods=priority,total_replication_jobs=len(order)*len(cfg['seeds']))
    write(RUNS/'FOLLOWUP_STATUS.json',state)
    # Train base and candidates in each seed block for transparent paired comparisons.
    for seed in cfg['seeds']:
        for method in order:
            result=spawn_run(method,seed,cfg,state);state['completed_jobs'].append(result)
            write(RUNS/'FOLLOWUP_STATUS.json',state)
            refresh_report()
            if result['state']=='failed':
                state.update(state='attention_required');write(RUNS/'FOLLOWUP_STATUS.json',state);return
        if seed==stop_after_seed:
            state.update(state='seed_block_complete',current=result)
            write(RUNS/'FOLLOWUP_STATUS.json',state)
            print(f'Completed seed {seed}; stopping at the seed boundary',flush=True)
            return
    summaries=[json.loads((RUNS/f'replicate_{method}_seed{seed}_v1/summary.json').read_text(encoding='utf-8'))
               for method in order for seed in cfg['seeds']]
    decision=select_replicated(summaries,cfg)
    write(RUNS/'FOLLOWUP_DECISION.json',decision)
    state.update(decision_mode=decision['mode'],selected_methods=[r['method'] for r in decision['selected']])
    write(RUNS/'FOLLOWUP_STATUS.json',state)
    validation=ROOT/'work/results/research_2026/replication_validation.csv'
    with validation.open('w',encoding='utf-8-sig',newline='') as f:
        fields=sorted({k for r in decision['all_candidates'] for k in r})
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(decision['all_candidates'])
    if cfg.get('evaluate_common_flood_after_replication',False):
        state.update(state='evaluating_common_LGC_flood');write(RUNS/'FOLLOWUP_STATUS.json',state)
        log=ROOT/'work/logs/research_2026/common_flood.log'
        with log.open('a',encoding='utf-8') as f:
            child=subprocess.Popen([str(PYTHON),'-B','-u','-X','utf8',str(ROOT/'scripts/evaluate_common_flood.py')],
                    cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'))
            state['external_evaluation_pid']=child.pid;write(RUNS/'FOLLOWUP_STATUS.json',state)
            while child.poll() is None:time.sleep(15)
            if child.returncode:
                state.update(state='common_LGC_evaluation_attention_required',external_exit_code=child.returncode)
                write(RUNS/'FOLLOWUP_STATUS.json',state);return
        state.update(state='exploratory_LGC_complete_formal_benchmark_CIA_and_ablations_pending')
    else:state.update(state='replication_complete_external_and_CIA_pending')
    write(RUNS/'FOLLOWUP_STATUS.json',state)
    refresh_report()
    print(state['state'],flush=True)

if __name__=='__main__':
    import ctypes
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-after-seed',type=int)
    args=parser.parse_args()
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:main(args.stop_after_seed)
    except Exception as e:
        path=RUNS/'FOLLOWUP_STATUS.json'
        state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if state.get('pid')!=os.getpid():
            path=RUNS/f'FOLLOWUP_STARTUP_ERROR_{os.getpid()}.json'
            state={'pid':os.getpid()}
        state.update(state='attention_required',error_type=type(e).__name__,error=str(e))
        write(path,state)
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
