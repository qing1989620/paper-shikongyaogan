"""Author SA-STF network/sampler adapter. Inference only, no source mutation.

The author's constructor allocates training-only VGG perceptual loss even for
inference. We remove exactly that allocation and import in an isolated AST copy.
Every sampling-method AST is asserted unchanged. No GDAL/torchvision is needed
for already prepared arrays. This does not reproduce the author's training.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
import torch

WORK=Path(__file__).resolve().parents[1];ROOT=WORK.parent
SOURCE=WORK/'repos/SA-STF'

def build(device='cpu'):
    path=SOURCE/'Diffusion/Model.py'
    spec=importlib.util.spec_from_file_location('sa_stf_author_unet',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    net=module.UNet(in_channel=6,out_channel=6,inner_channel=64,norm_groups=32,channel_mults=[1,2,4,8,8],dropout=0)
    ck=torch.load(WORK/'checkpoints/external/sa_stf_lgc.pth',map_location='cpu',weights_only=True)
    net.load_state_dict(ck,strict=True);net.to(device).eval()
    raw=(SOURCE/'Diffusion/diffusion_res.py').read_text(encoding='utf-8')
    before=ast.parse(raw);after=ast.parse(raw)
    after.body=[node for node in after.body if not(isinstance(node,ast.ImportFrom) and node.module=='Diffusion.loss_perceptual')]
    cls=next(n for n in after.body if isinstance(n,ast.ClassDef) and n.name=='ResidualDiffusion')
    constructor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    changed=0
    for node in constructor.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='loss_pre' for t in node.targets):
            node.value=ast.Constant(value=None);changed+=1
    if changed!=1:raise RuntimeError('Author constructor changed; re-audit adapter')
    old_cls=next(n for n in before.body if isinstance(n,ast.ClassDef) and n.name=='ResidualDiffusion')
    names=('get_x0_predict','DDIM','sample','q_sample','p_sample_loop','p_sample','p_mean_variance')
    for name in names:
        a=next(n for n in old_cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if ast.dump(a)!=ast.dump(b):raise AssertionError(f'Changed inference method {name}')
    standalone=types.ModuleType('sa_stf_inference_sampler');standalone.__file__=str(SOURCE/'Diffusion/diffusion_res.py')
    sys.modules[standalone.__name__]=standalone
    exec(compile(ast.fix_missing_locations(after),standalone.__file__,'exec'),standalone.__dict__)
    sampler=standalone.ResidualDiffusion(net,total_epoch=200,timesteps=100,sampling_steps=50,ddim_sampling_eta=0.,device=device).to(device).eval()
    return sampler,{'strict_checkpoint_load':True,'sampling_method_AST_unchanged':list(names),
                    'training_only_VGG_removed':True,'original_sampler_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                    'author_weight_training_date_overlap':'unknown','main_table_eligible':False}

@torch.no_grad()
def predict(sampler,obs,seed):
    if 'f1' in obs:raise ValueError('GT cannot enter SA-STF inference')
    torch.manual_seed(seed)
    return sampler.sample(obs['c1'],obs['f0'],obs['c0'],obs['f3'],obs['c3'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--size',type=int,default=32)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    sampler,info=build(a.device)
    sys.path.insert(0,str(WORK/'tools'))
    from train_research_2026 import Patches,unpack
    from torch.utils.data import DataLoader
    obs,_=unpack(next(iter(DataLoader(Patches('LGC','val',1),batch_size=1))),a.device)
    obs={k:(v[:,:,:a.size,:a.size] if v.ndim==4 else v) for k,v in obs.items()}
    start=time.monotonic();out=predict(sampler,obs,42)
    if out.shape!=obs['f0'].shape or not torch.isfinite(out).all():raise RuntimeError('SA-STF output invalid')
    info.update(functional_smoke_passed=True,device=a.device,spatial_size=a.size,sampling_steps=50,
                seconds=time.monotonic()-start,output_shape=list(out.shape),
                output_range=[float(out.min()),float(out.max())],
                scope='small functional test, no accuracy or speed comparison claim')
    (ROOT/'docs/audit/sa_stf_adapter_smoke.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
