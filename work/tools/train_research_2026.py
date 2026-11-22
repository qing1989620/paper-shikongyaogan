"""Paired, resumable research screening. Never writes legacy runs or paper scores.

Primary selection is mean reflectance patch RMSE over three stable noise seeds.
The current LGC validation date is exposed development data, not a blind test.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.dont_write_bytecode=True
WORK=Path(__file__).resolve().parents[1]; ROOT=WORK.parent
sys.path.insert(0,str(WORK/'tools'))
from research_methods_2026 import ResearchModel, METHODS, blur
from bi_diffusion import cosine_beta_schedule
from benchmark_metrics import compute_metrics, from_minus_one_one

def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def atomic_torch(path,value):
    tmp=Path(path).with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def stable_seed(key,repeat,namespace='validation_v1'):
    return int.from_bytes(hashlib.sha256(f'{namespace}|{key}|{repeat}'.encode()).digest()[:4],'little')%(2**31)

class Patches(Dataset):
    def __init__(self,dataset='LGC',split='train',limit=0):
        self.dataset=dataset; self.split=split
        self.files=sorted((WORK/'data'/f'patches_{dataset.lower()}'/split).glob('*.npz'))
        self.manifest=WORK/'configs/screening_2026'/f'manifest_{dataset.lower()}_frozen.csv'
        if not self.manifest.exists():self.manifest=ROOT/'docs/audit/dataset_manifest.csv'
        rows=list(csv.DictReader(self.manifest.open(encoding='utf-8-sig',newline='')))
        self.dates={r['patch_id']:r for r in rows if r['dataset']==dataset and r['split']==split}
        if limit:self.files=self.files[:limit]
        if not self.files:raise ValueError(f'No data: {dataset}/{split}')
    def __len__(self):return len(self.files)
    def __getitem__(self,index):
        fp=self.files[index]; dates=self.dates[fp.stem]
        with np.load(fp,allow_pickle=False) as z:
            data={k:torch.from_numpy(np.ascontiguousarray(z[k].astype(np.float32).transpose(2,0,1))) for k in ('f0','f1','f3','c0','c1','c3')}
        days=[datetime.fromisoformat(dates[k]) for k in ('prev_date','target_date','next_date')]
        data['gaps']=torch.tensor([(days[1]-days[0]).days,(days[2]-days[1]).days],dtype=torch.float32)
        data['key']=f'{self.dataset}/{self.split}/{fp.stem}';data['date']=dates['target_date']
        return data

def unpack(batch,device):
    gt=batch['f1'].to(device)
    obs={k:batch[k].to(device) for k in ('f0','f3','c0','c1','c3','gaps')}
    return obs,gt

SCHEDULES={}
def alphas(device):
    beta=cosine_beta_schedule(100)
    precise=torch.cumprod(1-beta,0)
    SCHEDULES[str(device)]={'recip':(1/precise).sqrt().float().to(device),
                            'recipm1':(1/precise-1).sqrt().float().to(device)}
    return precise.float().to(device)

@torch.no_grad()
def sample(model,obs,alpha,seed,steps=1,eta=0.):
    cond=model.prepare(obs)
    gen=torch.Generator(device=obs['f0'].device).manual_seed(seed)
    eps=torch.randn(obs['f0'].shape,generator=gen,device=obs['f0'].device)
    noise=model.noise_geometry(cond,eps)
    prior=model.prior(cond,noise)
    x=prior+noise
    times=list(reversed(torch.linspace(-1,99,steps+1).int().tolist()))
    for t,tn in zip(times[:-1],times[1:]):
        tc=torch.full((x.shape[0],),t,device=x.device,dtype=torch.long)
        pred=model.predict(cond,x,tc).clamp(-1,1)
        if tn<0:return pred
        a,an=alpha[t],alpha[tn]
        # Handles a zero prior (ordinary DDIM) and a mean-reverting bridge.
        if str(x.device) not in SCHEDULES:alphas(x.device)
        schedule=SCHEDULES[str(x.device)]
        centered=x-(1-a.sqrt())*prior if model.method=='i02_bridge' else x
        # Preserve the author's arithmetic and float64-to-float32 buffer path.
        pn=(schedule['recip'][t]*centered-pred)/schedule['recipm1'][t]
        sigma=eta*((1-a/an)*(1-an)/(1-a)).sqrt()
        independent=torch.randn(x.shape,device=x.device,generator=gen)
        if model.method in ('i02_bridge','i07_covariance'):
            independent=model.noise_geometry(cond,independent)
        x=an.sqrt()*pred+(1-an.sqrt())*prior+(1-an-sigma*sigma).sqrt()*pn+sigma*independent
    raise RuntimeError('No sampling step')

@torch.no_grad()
def validate(model,loader,alpha,device,repeats=3,full_metrics=False,out=None,steps=1,noise_namespace='validation_v1'):
    model.eval();rows=[];tot=0.;n=0; perrepeat=[]
    for batch in loader:
        obs,gt=unpack(batch,device)
        for rep in range(repeats):
            seed=stable_seed(batch['key'][0],rep,noise_namespace)
            pred=sample(model,obs,alpha,seed,steps=steps,eta=0.)
            err=(pred-gt)*.5
            r=float(err.square().mean().sqrt());tot+=r;n+=1
            row={'key':batch['key'][0],'date':batch['date'][0],'repeat':rep,'noise_seed':seed,'rmse':r}
            if full_metrics:
                p=pred[0].permute(1,2,0).cpu().numpy();y=gt[0].permute(1,2,0).cpu().numpy()
                row.update(compute_metrics(from_minus_one_one(p),from_minus_one_one(y),ergas_ratio=1/16))
                row.update({'diagnostic_'+k:v for k,v in model.stats.items()})
            rows.append(row)
    if out:
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for rep in range(repeats):perrepeat.append(float(np.mean([r['rmse'] for r in rows if r['repeat']==rep])))
    summary={'mean_patch_rmse':tot/n,'n_predictions':n,'n_patches':len(loader.dataset),'noise_repeat_rmse':perrepeat,
             'metric_version':'reflectance_v1','exposure':'legacy_exposed_development_only',
             'sampler':{'steps':steps,'eta':0.},'time_cst':now()}
    if full_metrics:
        keys=('mae','psnr_db','ssim','sam_rad','sam_deg','ergas','cc')
        for k in keys:
            vals=np.array([r[k] for r in rows]); finite=vals[np.isfinite(vals)]
            summary[k]=float(finite.mean()) if len(finite) else None
            summary[k+'_undefined']=int((~np.isfinite(vals)).sum())
        diagnostics=sorted({k for r in rows for k in r if k.startswith('diagnostic_')})
        summary['mechanism_diagnostics']={k:float(np.mean([r[k] for r in rows if k in r])) for k in diagnostics}
    return summary

def environment():
    import scipy
    return {'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
            'numpy':np.__version__,'scipy':scipy.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'platform':platform.platform(),'cudnn_deterministic':True,
            'grid_sample_note':'CUDA backward may be nondeterministic for i06; warning is recorded'}

def train(args):
    out=Path(args.out).resolve()
    if not out.is_relative_to((WORK/'runs/research_2026').resolve()):raise ValueError('New research run root required')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():print('Already complete',out,flush=True);return
    if (out/'config.json').exists() and not args.resume:raise FileExistsError('Existing run: use explicit --resume')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    model=ResearchModel(args.method,args.seed)
    warm_meta=model.load_shared_checkpoint(args.warmstart) if args.warmstart else None
    model=model.to(device)
    alpha=alphas(device)
    tr=Patches(args.dataset,'train',args.limit); va=Patches(args.dataset,'val',args.val_limit)
    val_loader=DataLoader(va,batch_size=1,shuffle=False,num_workers=0)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.99))
    teacher=copy.deepcopy(model).eval().requires_grad_(False) if args.method=='i14_masked_ema' else None
    code=[Path(__file__),WORK/'tools/research_methods_2026.py',WORK/'tools/benchmark_metrics.py',WORK/'repos/STF/src/model/stfdiff/model6/bi_pred_resnet.py']
    config={**vars(args),'time_cst':now(),'device':device,
            'warmstart_metadata':warm_meta,'warmstart_sha256':sha(args.warmstart) if args.warmstart else None,
            'manifest_path':str(tr.manifest.relative_to(ROOT)), 'manifest_sha256':sha(tr.manifest),
            'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code},
            'train_patches':len(tr),'val_patches':len(va),'validation_repeats':3,
            'parameters':sum(p.numel() for p in model.parameters()),
            'max_updates_per_epoch':math.ceil(len(tr)/args.accum),'metric_version':'reflectance_v1'}
    start=1;best=float('inf');step=0
    if args.resume and (out/'resume.pth').exists():
        previous=json.loads((out/'config.json').read_text(encoding='utf-8'))
        for k in ('method','epochs','seed','lr','accum','warmstart_sha256','manifest_sha256','code_sha256'):
            if previous[k]!=config[k]:raise ValueError(f'Resume provenance mismatch: {k}')
        ck=torch.load(out/'resume.pth',map_location='cpu',weights_only=True)
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
        if teacher is not None:teacher.load_state_dict(ck['teacher'],strict=True)
        start=ck['epoch']+1;best=ck['best'];step=ck['step']
    else:
        atomic_json(out/'config.json',config);atomic_json(out/'environment.json',environment())
        source=out/'source';source.mkdir(exist_ok=True)
        for p in code:shutil.copy2(p,source/p.name)
    print(f'START {args.method} seed={args.seed} train={len(tr)} val={len(va)} params={config["parameters"]} warmstart={bool(args.warmstart)}',flush=True)
    status={'state':'running','method':args.method,'seed':args.seed,'epochs':args.epochs,'pid':os.getpid()}
    total_updates=math.ceil(len(tr)/args.accum)*args.epochs
    if start==1:
        initial=validate(model,val_loader,alpha,device)
        atomic_json(out/'initial_validation.json',initial)
        print(f'INITIAL val={initial["mean_patch_rmse"]:.7f}',flush=True)
    for epoch in range(start,args.epochs+1):
        model.train();t0=time.monotonic();loss_total=primary_total=aux_total=0.;seen=0
        data_gen=torch.Generator().manual_seed(args.seed*10000+epoch)
        noise_gen=torch.Generator(device=device).manual_seed(args.seed*100000+epoch)
        mask_gen=torch.Generator().manual_seed(args.seed*20000+epoch)
        loader=DataLoader(tr,batch_size=1,shuffle=True,generator=data_gen,num_workers=0)
        opt.zero_grad(set_to_none=True)
        for i,batch in enumerate(loader):
            obs,gt=unpack(batch,device);cond=model.prepare(obs)
            t=torch.randint(0,100,(1,),generator=noise_gen,device=device)
            eps=torch.randn(gt.shape,device=device,generator=noise_gen)
            a=alpha[t].view(-1,1,1,1);prior=model.prior(cond,gt)
            xt=prior+a.sqrt()*(gt-prior)+(1-a).sqrt()*model.noise_geometry(cond,eps)
            if teacher is not None:
                with torch.no_grad():full_pred=teacher.predict(teacher.prepare(obs),xt,t).clamp(-1,1)
                student_obs=dict(obs)
                if torch.rand((),generator=mask_gen)<.5:
                    student_obs['f0']=obs['f3'];student_obs['c0']=obs['c3']
                else:student_obs['f3']=obs['f0'];student_obs['c3']=obs['c0']
                cond=model.prepare(student_obs)
            pred=model.predict(cond,xt,t).clamp(-1,1)
            loss,primary,aux=model.loss(cond,pred,gt)
            if teacher is not None:loss=loss+.1*(blur(pred)-blur(full_pred)).abs().mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {epoch}/{i}')
            (loss/args.accum).backward()
            if (i+1)%args.accum==0 or i+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                warm=min(1.,(step+1)/50)
                rate=args.lr*warm*(.1+.9*.5*(1+math.cos(math.pi*step/max(total_updates,1))))
                for group in opt.param_groups:group['lr']=rate
                opt.step();opt.zero_grad(set_to_none=True);step+=1
                if teacher is not None:
                    with torch.no_grad():
                        for p,q in zip(teacher.parameters(),model.parameters()):p.mul_(.99).add_(q,alpha=.01)
            loss_total+=float(loss.detach());primary_total+=float(primary);aux_total+=float(aux);seen+=1
            if (i+1)%100==0:
                status.update(epoch=epoch,batch=i+1,total_batches=len(loader),updated_cst=now(),loss=loss_total/seen)
                atomic_json(out/'status.json',status)
                print(f'  epoch={epoch} batch={i+1}/{len(loader)} loss={loss_total/seen:.6f}',flush=True)
        vr=validate(model,val_loader,alpha,device)
        score=vr['mean_patch_rmse'];duration=time.monotonic()-t0
        log={'epoch':epoch,'loss':loss_total/seen,'primary':primary_total/seen,'aux':aux_total/seen,
             'val_rmse':score,'seconds':duration,'optimizer_step':step,'time_cst':now()}
        lp=out/'train_log.csv';exists=lp.exists()
        with lp.open('a',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(log))
            if not exists:writer.writeheader()
            writer.writerow(log)
        if score<best:
            best=score
            atomic_torch(out/'best.pth',{'model':model.state_dict(),'method':args.method,'seed':args.seed,
                           'epoch':epoch,'val_rmse':score,'config_sha256':sha(out/'config.json')})
        resume={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'step':step}
        if teacher is not None:resume['teacher']=teacher.state_dict()
        atomic_torch(out/'resume.pth',resume)
        status.update(epoch=epoch,state='running',updated_cst=now(),val_rmse=score,best_rmse=best)
        atomic_json(out/'status.json',status)
        print(f'EPOCH {epoch}/{args.epochs} loss={loss_total/seen:.6f} val={score:.7f} best={best:.7f} seconds={duration:.1f}',flush=True)
    ck=torch.load(out/'best.pth',map_location=device,weights_only=True);model.load_state_dict(ck['model'],strict=True)
    final=validate(model,val_loader,alpha,device,full_metrics=True,out=out/'per_patch_validation.csv')
    final.update(method=args.method,seed=args.seed,best_epoch=ck['epoch'],checkpoint_sha256=sha(out/'best.pth'),
                 peak_cuda_MiB=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0)
    atomic_json(out/'summary.json',final)
    atomic_json(out/'COMPLETE.json',{'finished_cst':now(),'summary_sha256':sha(out/'summary.json'),'status':args.phase+'_complete_not_SOTA'})
    status.update(state='complete',updated_cst=now());atomic_json(out/'status.json',status)
    print('COMPLETE',json.dumps(final,ensure_ascii=False),flush=True)

def parser():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--dataset',choices=['LGC','CIA'],default='LGC');p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--phase',choices=['smoke','screening','replication'],default='screening')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--accum',type=int,default=2);p.add_argument('--out',required=True)
    p.add_argument('--warmstart',default='');p.add_argument('--resume',action='store_true')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0)
    return p

if __name__=='__main__':train(parser().parse_args())
