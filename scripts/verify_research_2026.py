"""Meaningful gates: base equivalence, sampler parity, GT exclusion, gradients."""
from __future__ import annotations
import gc
import json
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/tools'));sys.dont_write_bytecode=True
from research_methods_2026 import ResearchModel,METHODS
from train_research_2026 import Patches,unpack,alphas,sample,atomic_json,now
from torch.utils.data import DataLoader
from bi_diffusion import BiGaussianDiffusion

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    loader=DataLoader(Patches('LGC','train',1),batch_size=1)
    obs,gt=unpack(next(iter(loader)),device)
    alpha=alphas(device);checkpoint=ROOT/'work/runs/base/best.pth';report=[]
    for method in METHODS:
        torch.manual_seed(42);model=ResearchModel(method,42);model.load_shared_checkpoint(checkpoint);model.to(device)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        x=torch.randn(gt.shape,device=device);t=torch.full((1,),99,device=device,dtype=torch.long)
        tic=time.monotonic();model.eval()
        with torch.no_grad():
            cond=model.prepare(obs);pred=model.predict(cond,x,t)
            assert pred.shape==gt.shape and torch.isfinite(pred).all(),method
            if method=='base':
                native=model.core(obs['c0'],obs['c1'],obs['c3'],obs['f0'],x,obs['f3'],t,None)
                diff=float((native-pred).abs().max());assert diff==0.,diff
                wrapper=BiGaussianDiffusion(model.core,image_size=256,timesteps=100,sampling_timesteps=10,
                           objective='pred_x0',loss_type='l1',beta_schedule='cosine',ddim_sampling_eta=0.).to(device)
                gen=torch.Generator(device=device).manual_seed(512)
                initial=torch.randn(gt.shape,generator=gen,device=device)
                native10=wrapper.ddim_sample(obs['c0'],obs['c1'],obs['c3'],obs['f0'],initial,obs['f3'])
                ours10=sample(model,obs,alpha,512,steps=10)
                sample_diff=float((native10-ours10).abs().max());assert sample_diff<3e-6,sample_diff
                del wrapper,native10,ours10,native
        try:model.prepare({**obs,'f1':gt})
        except ValueError:pass
        else:raise AssertionError('GT exclusion failed')
        model.train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);cond=model.prepare(obs)
            a=alpha[50];noise=torch.randn_like(gt);prior=model.prior(cond,gt)
            xt=prior+a.sqrt()*(gt-prior)+(1-a).sqrt()*model.noise_geometry(cond,noise)
            pred=model.predict(cond,xt,torch.full_like(t,50)).clamp(-1,1)
            loss,*_=model.loss(cond,pred,gt);loss.backward()
            assert torch.isfinite(loss),method
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads),method
            extra=[p.grad for n,p in model.named_parameters() if not n.startswith('core.') and p.grad is not None]
            if method not in ('base','i07_covariance'):
                assert extra and sum(float(g.abs().sum()) for g in extra)>0,method
            optimizer.step()
        if device=='cuda':torch.cuda.synchronize()
        row={'method':method,'passed':True,'parameters':sum(p.numel() for p in model.parameters()),
             'seconds_two_updates':time.monotonic()-tic,'loss':float(loss.detach()),
             'peak_MiB':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0,'mechanism_stats':model.stats}
        if method=='base':row.update(base_max_abs_diff=diff,official_ddim10_max_abs_diff=sample_diff)
        report.append(row);print(json.dumps(row),flush=True)
        del model,optimizer,cond,pred,loss,grads,extra,xt,prior
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    atomic_json(ROOT/'docs/audit/research_2026_smoke.json',{'time_cst':now(),'device':device,'all_passed':True,
                'scope':'full 256x256 real training patch, two updates per candidate; not scientific performance evaluation','methods':report})
    print('ALL 15 GATES PASSED',flush=True)

if __name__=='__main__':main()
