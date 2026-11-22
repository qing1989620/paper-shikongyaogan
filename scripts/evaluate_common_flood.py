"""Same-patch LGC flood comparison after model selection, never a published-score merge.

Author SA-STF training/test dates were checked in supplementary Table 3. Native
sampling is retained. Training recipes differ and are disclosed. This scene was
historically exposed; it is not described as a new blind confirmation set.
"""
from __future__ import annotations
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'work';RUNS=WORK/'runs/research_2026'
sys.path.insert(0,str(WORK/'tools'))
from train_research_2026 import Patches,unpack,alphas,sample,stable_seed,atomic_json,sha
from research_methods_2026 import ResearchModel
from benchmark_metrics import compute_metrics,from_minus_one_one

OUT=WORK/'results/research_2026/common_flood_v1'

def evaluate(spec,device):
    out=OUT/spec['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return json.loads((out/'summary.json').read_text(encoding='utf-8'))
    source_audit=json.loads((ROOT/'docs/audit/sa_stf_protocol_from_supplement.json').read_text(encoding='utf-8'))
    if source_audit['LGC_our_test_target_is_author_training_fine']:raise RuntimeError('Author date overlap')
    if spec['kind']=='SA-STF':
        from sa_stf_adapter import build,predict
        net,info=build(device)
        inference=lambda obs,seed:predict(net,obs,seed)
        checkpoint=WORK/'checkpoints/external/sa_stf_lgc.pth'
        recipe='author_documented_7_train_triples_200epoch_pretrained_83839558_parameter_values'
    else:
        net=ResearchModel(spec['method'],spec.get('seed',42))
        checkpoint=ROOT/spec['checkpoint']
        if spec.get('shared'):net.load_shared_checkpoint(checkpoint)
        else:
            ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if ck['method']!=spec['method']:raise ValueError('Checkpoint method mismatch')
            net.load_state_dict(ck['model'],strict=True)
        net.to(device).eval();alpha=alphas(device)
        inference=lambda obs,seed:sample(net,obs,alpha,seed,steps=spec['steps'],eta=0.)
        recipe='project_shared_old_training' if spec.get('shared') else 'project_frozen_10_train_triples_40epoch_from_scratch'
    ds=Patches('LGC','test');loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    metadata={'protocol_id':'common_flood_patch_v1','scope':'historically_exposed_scene_common_patch_comparison',
              'method':spec,'checkpoint_sha256':sha(checkpoint),'manifest_sha256':sha(ds.manifest),
              'author_split_source_sha256':source_audit['source_sha256'],
              'training_recipe':recipe,'sampling_steps':spec['steps'],'eta':0.,
              'test_triple':['2004-12-28','2005-01-13','2005-01-29'],
              'n_patches':len(ds),'noise_repeats':3,'roi':'nonoverlap_256_tiles_top_left_2560x3072',
              'mask':'common_all_finite_pixels_existing_patch_protocol_no_method_specific_mask',
              'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (
                  Path(__file__),WORK/'tools/train_research_2026.py',WORK/'tools/research_methods_2026.py',
                  WORK/'tools/benchmark_metrics.py',WORK/'tools/sa_stf_adapter.py')},
              'published_SA_STF_test_patch_count':361,'our_test_patch_count':120,
              'published_absolute_scores_not_directly_comparable':True,
              'metric_version':'reflectance_v1','data_range':1.,'ergas_ratio':1/16,
              'postprocess_clipping':'none_native_model_sampling_rules_retained'}
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text(encoding='utf-8'))!=metadata:
            raise RuntimeError('Evaluation resume provenance changed: use a new protocol/output version')
    else:atomic_json(protocol,metadata)
    path=out/'per_patch.csv';rows=[]
    if path.exists():
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                rows.append({k:(v if k in ('key','date') else float(v)) for k,v in row.items()})
    completed={(r['key'],int(r['repeat'])) for r in rows}
    net.eval()
    with torch.no_grad():
        for batch in loader:
            obs,gt=unpack(batch,device)
            for repeat in range(3):
                key=batch['key'][0]
                if (key,repeat) in completed:continue
                seed=stable_seed(key,repeat,'common_flood_v1')
                if device=='cuda':torch.cuda.synchronize()
                start=time.monotonic();pred=inference(obs,seed)
                if device=='cuda':torch.cuda.synchronize()
                elapsed=time.monotonic()-start
                arr=pred[0].permute(1,2,0).cpu().numpy();truth=gt[0].permute(1,2,0).cpu().numpy()
                values=compute_metrics(from_minus_one_one(arr),from_minus_one_one(truth),ergas_ratio=1/16)
                row={'key':key,'date':batch['date'][0],'repeat':repeat,'noise_seed':seed,'inference_seconds_observed':elapsed,**values}
                exists=path.exists()
                with path.open('a',encoding='utf-8-sig' if not exists else 'utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row))
                    if not exists:w.writeheader()
                    w.writerow(row)
                rows.append(row)
            if len(rows)%30==0:print(spec['id'],len(rows),'/360 predictions',flush=True)
    if len(rows)!=3*len(ds):raise AssertionError('Incomplete common evaluation')
    summary={'method':spec['id'],'sampling_steps':spec['steps'],'n_patches':len(ds),'noise_repeats':3,
             'training_recipe':recipe,'mean_patch_rmse':float(np.mean([r['rmse'] for r in rows])),
             'pooled_rmse':float(np.sqrt(np.mean([r['rmse']**2 for r in rows]))),
             'exposure':'legacy_development_scene_not_new_blind_test','protocol_sha256':sha(out/'protocol.json')}
    for k in ('ssim','sam_rad','ergas','cc','mae','psnr_db'):
        vals=np.asarray([r[k] for r in rows]);good=vals[np.isfinite(vals)]
        summary[k]=float(good.mean()) if len(good) else None
        summary[k+'_undefined']=int((~np.isfinite(vals)).sum())
    atomic_json(out/'summary.json',summary);atomic_json(out/'COMPLETE.json',{'summary_sha256':sha(out/'summary.json')})
    del net,inference
    gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return summary

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu';OUT.mkdir(parents=True,exist_ok=True)
    decision=json.loads((RUNS/'FOLLOWUP_DECISION.json').read_text(encoding='utf-8'))
    specs=[{'id':'SA-STF_author_LGC_50NFE','kind':'SA-STF','steps':50},
           {'id':'base_shared_1NFE','kind':'research','method':'base','seed':42,'steps':1,'shared':True,'checkpoint':'work/runs/base/best.pth'},
           {'id':'base_shared_10NFE_eta0','kind':'research','method':'base','seed':42,'steps':10,'shared':True,'checkpoint':'work/runs/base/best.pth'}]
    for seed in (42,123,2026):
        for method in ['base']+[r['method'] for r in decision['selected']]:
            cp=f'work/runs/research_2026/replicate_{method}_seed{seed}_v1/best.pth'
            specs.append({'id':f'{method}_seed{seed}_1NFE','kind':'research','method':method,'seed':seed,'steps':1,'checkpoint':cp})
    rows=[]
    for spec in specs:
        atomic_json(OUT/'STATUS.json',{'state':'running','current':spec['id'],'complete_models':len(rows),'total_models':len(specs)})
        rows.append(evaluate(spec,device))
        with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    atomic_json(OUT/'STATUS.json',{'state':'complete','complete_models':len(rows),'total_models':len(specs),
                'remaining_publication_gates':['matched_training_benchmark','CIA_validation','mechanism_ablations','independent_confirmation','exact_author_spatial_protocol_if_reproducing_published_scores']})

if __name__=='__main__':main()
