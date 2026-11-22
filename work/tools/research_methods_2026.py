"""Fourteen independent STF mechanism prototypes plus the unchanged base.

These are research hypotheses, not reproductions of the papers that motivated
them. No target-date fine image is accepted by prepare()/predict(). Each full
proposal, its nearest prior work and falsification controls are in docs/.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'repos/STF/src/model/stfdiff/model6'))
from bi_pred_resnet import BiPredNoiseNet

METHODS = {
    'base':'STFDiff paired control',
    'i01_sensor':'Calibrated sensor-operator unrolling',
    'i02_bridge':'Uncertainty-conditioned dual-anchor diffusion bridge',
    'i03_frequency':'DC/LF/HF conditional factorization',
    'i04_graph':'Change-aware boundary graph propagation',
    'i05_spectral':'Dynamic cross-band spectral transport',
    'i06_registration':'Sensor alignment separated from temporal appearance',
    'i07_covariance':'Conditional spectral-covariance diffusion geometry',
    'i08_experts':'Spatial change-regime mixture of experts',
    'i09_time':'Continuous acquisition-gap-conditioned dynamics',
    'i10_unmixing':'Abundance-transport conditional prior',
    'i11_transport':'Nonlocal balanced anchor correspondence',
    'i12_recurrent':'Shared-weight innovation refinement',
    'i13_disentangle':'Sensor-style/content factorization',
    'i14_masked_ema':'Masked-anchor EMA conditional learning',
}

def blur(x,k=7):
    return F.avg_pool2d(x,k,stride=1,padding=k//2,count_include_pad=False)

def convnet(cin,cout,width=32,zero=False):
    net=nn.Sequential(nn.Conv2d(cin,width,3,padding=1),nn.SiLU(),
                      nn.Conv2d(width,width,3,padding=1,groups=width),nn.SiLU(),nn.Conv2d(width,cout,1))
    if zero:
        nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    return net

def weights(obs):
    gaps=obs['gaps'].to(obs['f0'])
    fraction=(gaps[:,0]/gaps.sum(1).clamp_min(1)).view(-1,1,1,1)
    return 1-fraction,fraction

def packed(obs):
    return torch.cat([obs[k] for k in ('f0','f3','c0','c1','c3')],1)

def delta_features(obs):
    return torch.cat((obs['f0'],obs['f3'],obs['c1']-obs['c0'],obs['c1']-obs['c3']),1)

class SensorOperator(nn.Module):
    def __init__(self):
        super().__init__()
        a=torch.arange(-2,3).float(); g=torch.exp(-(a[:,None]**2+a[None,:]**2)/2)
        self.kernel=nn.Parameter(g.log().expand(6,1,5,5).clone())
        self.mixing=nn.Parameter(torch.eye(6)*8)
    def forward(self,x,adjoint=False):
        kernel=self.kernel.flatten(1).softmax(1).reshape(6,1,5,5)
        mixing=self.mixing.softmax(1).reshape(6,6,1,1)
        if adjoint:
            x=F.conv2d(x,mixing.transpose(0,1))
            return F.conv_transpose2d(x,kernel,padding=2,groups=6)
        return F.conv2d(F.conv2d(x,kernel,padding=2,groups=6),mixing)

class BoundaryGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.barrier=nn.Parameter(torch.tensor(3.))
        self.strength=nn.Parameter(torch.tensor(-3.))
    def forward(self,x,guide,change):
        guide=F.interpolate(guide,size=x.shape[-2:],mode='bilinear',align_corners=False)
        change=F.interpolate(change,size=x.shape[-2:],mode='bilinear',align_corners=False)
        total=torch.zeros_like(x); mass=torch.zeros_like(guide)
        h,w=x.shape[-2:]
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            g=torch.roll(guide,(dy,dx),(-2,-1)); neighbor=torch.roll(x,(dy,dx),(-2,-1))
            edge=torch.exp(-F.softplus(self.barrier)*(guide-g).abs())/(1+change)
            valid=torch.ones_like(guide)
            if dy>0:valid[:,:,:dy]=0
            if dy<0:valid[:,:,dy:]=0
            if dx>0:valid[:,:,:,:dx]=0
            if dx<0:valid[:,:,:,dx:]=0
            edge=edge*valid
            total=total+edge*(neighbor-x); mass=mass+edge
        return x+self.strength.sigmoid()*total/mass.clamp_min(1e-6)

def warp(x,flow):
    b,_,h,w=x.shape
    ys=(torch.arange(h,device=x.device,dtype=x.dtype)+.5)*2/h-1
    xs=(torch.arange(w,device=x.device,dtype=x.dtype)+.5)*2/w-1
    yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    grid=torch.stack((xx,yy),-1)[None].expand(b,-1,-1,-1)
    delta=flow.permute(0,2,3,1)*torch.tensor((2/w,2/h),device=x.device)
    return F.grid_sample(x,grid+delta,mode='bilinear',padding_mode='border',align_corners=False)

class ResearchModel(nn.Module):
    def __init__(self,method='base',seed=42):
        super().__init__()
        if method not in METHODS: raise ValueError(method)
        self.method=method
        self.core=BiPredNoiseNet(dim=64,channels=6,dim_mults=(1,2,4),self_condition=False,learned_sinusoidal_cond=False)
        # Creating optional modules must not perturb the paired training RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+202600)
            if method=='i01_sensor':
                self.sensor=SensorOperator(); self.step=nn.Parameter(torch.tensor(-3.))
            elif method=='i02_bridge': self.prior_net=convnet(30,7)
            elif method=='i03_frequency':
                self.low=convnet(18,64); self.high=convnet(12,64)
                self.dc=nn.Sequential(nn.Linear(30,64),nn.SiLU(),nn.Linear(64,64))
                self.freq_gains=nn.Parameter(torch.full((3,),.05))
            elif method=='i04_graph': self.graphs=nn.ModuleList([BoundaryGraph() for _ in range(3)])
            elif method=='i05_spectral':
                self.matrix=nn.Sequential(nn.Linear(24,64),nn.SiLU(),nn.Linear(64,36))
                self.mix_gain=nn.Parameter(torch.tensor(-2.))
            elif method=='i06_registration': self.flow_net=convnet(24,4,zero=True)
            elif method=='i08_experts':
                self.experts=nn.ModuleList([nn.Sequential(nn.Conv2d(64,64,k,padding=k//2,groups=64),nn.SiLU(),nn.Conv2d(64,64,1)) for k in (3,5,7)])
                self.router=convnet(24,3)
            elif method=='i09_time':
                self.time_style=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,128))
                self.dynamic=convnet(18,64)
            elif method=='i10_unmixing':
                self.abundance=convnet(6,8); self.endmembers=nn.Parameter(torch.randn(6,8)*.2)
                self.abundance_features=nn.Conv2d(8,64,1)
            elif method=='i11_transport':
                self.query=nn.Conv2d(6,16,1); self.key=nn.Conv2d(6,16,1); self.value=nn.Conv2d(12,64,1)
            elif method=='i12_recurrent':
                self.recurrent=convnet(58,16); self.increment=nn.Conv2d(16,6,1)
                nn.init.zeros_(self.increment.weight);nn.init.zeros_(self.increment.bias)
            elif method=='i13_disentangle':
                self.style=nn.Sequential(nn.Linear(24,32),nn.SiLU(),nn.Linear(32,12))
                nn.init.zeros_(self.style[-1].weight); nn.init.zeros_(self.style[-1].bias)
                self.content=convnet(6,16);self.content_fusion=nn.Conv2d(32,64,1)
            elif method=='i14_masked_ema': self.mask_features=convnet(30,64)
        self.stats={}

    def load_shared_checkpoint(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        sd=ck.get('model',ck)
        core={k.removeprefix('model.'):v for k,v in sd.items() if k.startswith('model.')}
        if not core: raise ValueError('Expected STFDiff diffusion-wrapper checkpoint')
        self.core.load_state_dict(core,strict=True)
        return {k:v for k,v in ck.items() if k!='model'}

    def prepare(self,obs):
        """Only the five available observations and acquisition gaps are accepted."""
        if 'f1' in obs: raise ValueError('GT passed to inference conditioning')
        o=dict(obs); feat=None; aux=obs['f0'].new_zeros(()); extra={}
        wp,wn=weights(o); m=self.method
        self.stats={}
        if m=='i01_sensor':
            bias=wp*(o['c0']-self.sensor(o['f0']))+wn*(o['c3']-self.sensor(o['f3']))
            extra['sensor_bias']=blur(bias)
            # Bias is low-frequency; the operator must explain anchor structure.
            aux=.05*((self.sensor(o['f0'])+blur(bias)-o['c0']).abs().mean()+
                     (self.sensor(o['f3'])+blur(bias)-o['c3']).abs().mean())
        elif m=='i02_bridge':
            q=self.prior_net(packed(o)); p=wp*o['f0']+wn*o['f3']+.2*q[:,:6].tanh()
            extra.update(prior=p,noise_scale=.05+.45*q[:,6:].sigmoid())
        elif m=='i03_frequency':
            lo=torch.cat((blur(wp*o['f0']+wn*o['f3']),blur(o['c1']-o['c0']),blur(o['c1']-o['c3'])),1)
            hi=torch.cat((o['f0']-blur(o['f0']),o['f3']-blur(o['f3'])),1)
            dc=self.dc(packed(o).mean((2,3)))[:,:,None,None]
            feat=self.freq_gains[0]*self.low(lo)+self.freq_gains[1]*self.high(hi)+self.freq_gains[2]*dc
        elif m=='i04_graph':
            extra['graph_guide']=(o['f0'].abs()+o['f3'].abs()).mean(1,keepdim=True)/2
            extra['graph_change']=(o['c3']-o['c0']).abs().mean(1,keepdim=True)
        elif m=='i05_spectral':
            moments=torch.cat([o[k].mean((2,3)) for k in ('f0','f3','c0','c3')],1)
            matrix=self.matrix(moments).reshape(-1,6,6).softmax(-1)
            a=self.mix_gain.sigmoid()
            for k in ('c0','c1','c3'):
                o[k]=(1-a)*o[k]+a*torch.einsum('bij,bjhw->bihw',matrix,o[k])
            self.stats['spectral_mix_gain']=float(a.detach())
        elif m=='i06_registration':
            flow=2*self.flow_net(torch.cat((o['f0'],o['c0'],o['f3'],o['c3']),1)).tanh()
            o['c0']=warp(o['c0'],flow[:,:2]);o['c3']=warp(o['c3'],flow[:,2:])
            o['c1']=warp(o['c1'],wp*flow[:,:2]+wn*flow[:,2:])
            aux=.02*((blur(o['c0'])-blur(o['f0'])).abs().mean()+(blur(o['c3'])-blur(o['f3'])).abs().mean())
            aux=aux+.01*(flow[:,:,1:]-flow[:,:,:-1]).abs().mean()+.01*(flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean()
            self.stats['flow_abs_px']=float(flow.detach().abs().mean())
        elif m=='i07_covariance':
            r=torch.cat(((o['f0']-o['c0'])[:,:,::8,::8].flatten(2),
                         (o['f3']-o['c3'])[:,:,::8,::8].flatten(2)),2)
            r=r-r.mean(2,keepdim=True);cov=r@r.transpose(1,2)/max(r.shape[-1]-1,1)
            scale=cov.diagonal(dim1=1,dim2=2).clamp_min(1e-6).sqrt()
            corr=cov/(scale[:,:,None]*scale[:,None,:])
            eye=torch.eye(6,device=r.device)[None]
            extra['chol']=torch.linalg.cholesky(.5*corr+.5*eye)
        elif m=='i08_experts':
            gate=self.router(delta_features(o)).softmax(1)
            extra['expert_gate']=gate
            aux=.002*((gate.mean((0,2,3))-1/3)**2).sum()
            self.stats['router_entropy']=float((-(gate.clamp_min(1e-8).log()*gate).sum(1).mean()).detach())
            self.stats['router_spatial_std']=float(gate.detach().std((2,3)).mean())
        elif m=='i09_time':
            gaps=o['gaps'].to(o['f0']); gapf=torch.cat((gaps.log1p()/5,wn.flatten(1)[:,:1]),1)
            scale,shift=self.time_style(gapf).chunk(2,1)
            velocities=torch.cat((wp*o['f0']+wn*o['f3'],
                        ((o['c1']-o['c0'])/(gaps[:,0,None,None,None]/16).clamp_min(.25)).clamp(-2,2),
                        ((o['c3']-o['c1'])/(gaps[:,1,None,None,None]/16).clamp_min(.25)).clamp(-2,2)),1)
            feat=.05*(self.dynamic(velocities)*(1+scale[:,:,None,None].tanh())+shift[:,:,None,None])
        elif m=='i10_unmixing':
            a={k:self.abundance((o[k]+1)/2).softmax(1) for k in ('f0','f3','c0','c1','c3')}
            target=(wp*a['f0']+wn*a['f3']+.3*(a['c1']-wp*a['c0']-wn*a['c3'])).clamp_min(1e-6)
            target=target/target.sum(1,keepdim=True)
            feat=.05*self.abundance_features(target)
            end=self.endmembers.sigmoid()
            for k in ('f0','f3'):
                recon=torch.einsum('ck,bkhw->bchw',end,a[k])
                aux=aux+.05*(recon-(o[k]+1)/2).abs().mean()
            self.stats['abundance_entropy']=float((-(target*target.log()).sum(1).mean()).detach())
        elif m=='i11_transport':
            side=min(16,o['f0'].shape[-1]//4)
            q=F.adaptive_avg_pool2d(self.query(o['c1']),(side,side)).flatten(2).transpose(1,2)
            k=F.adaptive_avg_pool2d(self.key(wp*o['c0']+wn*o['c3']),(side,side)).flatten(2)
            q=F.normalize(q,dim=-1);k=F.normalize(k,dim=1)
            coords=torch.stack(torch.meshgrid(torch.linspace(-1,1,side,device=q.device),torch.linspace(-1,1,side,device=q.device),indexing='ij'),-1).reshape(-1,2)
            logp=q@k*4-torch.cdist(coords,coords).square()[None]
            for _ in range(5):
                logp=logp-torch.logsumexp(logp,dim=2,keepdim=True)
                logp=logp-torch.logsumexp(logp,dim=1,keepdim=True)
            plan=logp.exp();v=F.adaptive_avg_pool2d(self.value(torch.cat((o['f0'],o['f3']),1)),(side,side)).flatten(2).transpose(1,2)
            moved=(plan@v).transpose(1,2).reshape(-1,64,side,side)
            feat=.05*F.interpolate(moved,size=o['f0'].shape[-2:],mode='bilinear',align_corners=False)
            self.stats['transport_row_error']=float((plan.detach().sum(2)-1).abs().mean())
        elif m=='i13_disentangle':
            def style(c,f):
                vec=torch.cat((c.mean((2,3)),f.mean((2,3)),c.std((2,3)),f.std((2,3))),1)
                gain,bias=self.style(vec).chunk(2,1)
                return 1+.25*gain.tanh(),.15*bias.tanh()
            g0,b0=style(o['c0'],o['f0']);g3,b3=style(o['c3'],o['f3'])
            o['c0']=o['c0']*g0[:,:,None,None]+b0[:,:,None,None]
            o['c3']=o['c3']*g3[:,:,None,None]+b3[:,:,None,None]
            o['c1']=o['c1']*(wp*g0[:,:,None,None]+wn*g3[:,:,None,None])+wp*b0[:,:,None,None]+wn*b3[:,:,None,None]
            f=self.content(wp*o['f0']+wn*o['f3']);c=self.content(o['c1'])
            feat=.05*self.content_fusion(torch.cat((f,c-f),1))
            for ck,fk in (('c0','f0'),('c3','f3')):
                cf=self.content(o[ck]);ff=self.content(o[fk])
                aux=aux+.01*(blur(cf)-blur(ff)).abs().mean()+.01*F.relu(.1-cf.std((2,3))).mean()
        elif m=='i14_masked_ema': feat=.05*self.mask_features(packed(o))
        return {'obs':o,'feat':feat,'aux':aux,**extra}

    def noise_geometry(self,cond,eps):
        if self.method=='i07_covariance': return torch.einsum('bij,bjhw->bihw',cond['chol'],eps)
        if self.method=='i02_bridge': return cond['noise_scale']*eps
        return eps

    def prior(self,cond,like):
        return cond.get('prior',torch.zeros_like(like))

    def predict(self,cond,x,time):
        n=self.core;o=cond['obs'];m=self.method
        xf=n.fine_init_conv(torch.cat((o['f0'],o['f3']),1))
        xc=n.coarse_init_conv(torch.cat((o['c0'],o['c1'],o['c3']),1))
        xn=n.noisy_init_conv(x);clean=n.clean_init_conv(torch.cat((xf,xc),1))
        if cond['feat'] is not None: clean=clean+cond['feat']
        if m=='i08_experts':
            gate=cond['expert_gate']
            clean=clean+.05*sum(gate[:,i:i+1]*expert(clean) for i,expert in enumerate(self.experts))
        residual=xn-clean; t=n.time_mlp(time); h=[]
        for i,((cr,cd),(nr,nd)) in enumerate(zip(n.clean_downs,n.noisy_downs)):
            clean=cr(clean)
            if m=='i04_graph': clean=self.graphs[i](clean,cond['graph_guide'],cond['graph_change'])
            xn=nr(xn,t); h.append(xn-clean);clean=cd(clean);xn=nd(xn)
        y=n.noisy_mid_block(xn,t)-n.clean_mid_block(clean)
        for nr,up in n.noise_ups: y=up(nr(torch.cat((y,h.pop()),1),t))
        y=n.final_conv(n.final_res_block(torch.cat((y,residual),1),t))
        if m=='i01_sensor':
            for _ in range(2):
                mismatch=self.sensor(y)+cond['sensor_bias']-o['c1']
                y=y-self.step.sigmoid()*self.sensor(mismatch,adjoint=True)
        elif m=='i12_recurrent':
            hidden=torch.zeros((y.shape[0],16,*y.shape[-2:]),device=y.device,dtype=y.dtype)
            for _ in range(3):
                hidden=self.recurrent(torch.cat((y,packed(o),o['c1']-blur(y),hidden),1))
                y=y+.05*self.increment(hidden).tanh()
        return y

    def loss(self,cond,pred,gt):
        if self.method=='i07_covariance':
            e=pred-gt; flat=e.flatten(2)
            whitened=torch.linalg.solve_triangular(cond['chol'],flat,upper=False)
            primary=.5*e.abs().mean()+.5*whitened.abs().mean()
        else: primary=(pred-gt).abs().mean()
        aux=cond['aux']
        if self.method=='i02_bridge':
            p,s=cond['prior'],cond['noise_scale']
            aux=aux+.1*(p-gt).abs().mean()+.005*((p-gt).square()/(s.square()+1e-6)+2*s.log()).mean()
        return primary+aux,primary.detach(),aux.detach()
