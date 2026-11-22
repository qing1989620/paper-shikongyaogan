"""Reproducible bounded-memory patch preparation, with explicit date leakage checks.

Default is a dry-run. --generate requires a new empty output directory.
--verify-existing compares representative patches to the current raw images.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

WORK=Path(__file__).resolve().parents[1];ROOT=WORK.parent
KEYS=('f0','f1','f3','c0','c1','c3')

def config(dataset):
    if dataset=='LGC':
        c=json.loads((WORK/'configs/datasets/legacy_lgc_v0.json').read_text(encoding='utf-8'))
        groups=[(x['group'],x['dates']) for x in c['train_triples']]
        return groups,(2720,3200), (0,0), (2560,3072), 'raw'
    meta=json.loads((WORK/'data/patches_cia/meta.json').read_text(encoding='utf-8'))
    withheld={tr[1] for split in ('val','test') for tr in meta['splits'][split]}
    keep=[tr for tr in meta['splits']['calib'] if not(set(tr)&withheld)]
    groups=[(f'train{i}',tr) for i,tr in enumerate(keep)]
    return groups,(2040,1720),(124,92),(1792,1536),'raw_cia'

def files_by_date(raw):
    result={}
    for sensor,folder in (('f','Landsat'),('c','MODIS')):
        for d in sorted((raw/folder).iterdir()):
            if not d.is_dir():continue
            parts=d.name.split('_')
            day=(datetime(int(parts[0]),1,1)+timedelta(days=int(parts[1])-1)).date().isoformat()
            fs=[p for p in d.iterdir() if p.is_file() and p.suffix.lower() in ('.bil','.int')]
            if fs:result[sensor,day]=fs[0]
    return result

def read_tile(path,dataset,sensor,shape,r,c,size=256):
    h,w=shape;expected=h*w*6*2
    if path.stat().st_size!=expected:raise ValueError(f'Invalid raw byte size: {path.name}')
    with path.open('rb') as f:
        if dataset=='CIA' and sensor=='f':
            f.seek(r*6*w*2);v=np.fromfile(f,dtype='<u2',count=size*6*w).reshape(size,6,w)[:,:,c:c+size].transpose(0,2,1)
        else:
            v=np.empty((size,size,6),dtype=np.uint16)
            for b in range(6):
                f.seek((b*h+r)*w*2);v[:,:,b]=np.fromfile(f,dtype='<u2',count=size*w).reshape(size,w)[:,c:c+size]
    return (np.clip(v.astype(np.float32),0,10000)/10000*2-1).astype(np.float16)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['LGC','CIA'],default='LGC')
    p.add_argument('--output',default='');p.add_argument('--generate',action='store_true');p.add_argument('--verify-existing',action='store_true')
    a=p.parse_args();groups,shape,offset,crop,rawfolder=config(a.dataset)
    coords=[(r,c) for r in range(0,crop[0]-255,256) for c in range(0,crop[1]-255,256)]
    target_meta=json.loads((WORK/f'data/patches_{a.dataset.lower()}/meta.json').read_text(encoding='utf-8'))
    holdouts=target_meta['triples'] if a.dataset=='LGC' else target_meta['splits']
    withheld={holdouts[split][1] for split in ('val','test')} if a.dataset=='LGC' else {t[1] for split in ('val','test') for t in holdouts[split]}
    overlap=sorted({d for _,tr in groups for d in tr}&withheld)
    if overlap:raise ValueError(f'Training fine dates intersect withheld targets: {overlap}')
    report={'dataset':a.dataset,'groups':[{'id':g,'dates':tr} for g,tr in groups],
            'patches':len(groups)*len(coords),'withheld_target_dates':sorted(withheld),
            'train_fine_overlap':overlap,'status':'dry_run','raw_offset':offset,'crop':crop,
            'note':'CIA filters out calib triples containing any heldout target fine date; not the author training split'}
    if a.dataset=='CIA':report['excluded_calib_triples']=[t for t in target_meta['splits']['calib'] if set(t)&withheld]
    if a.generate or a.verify_existing:
        raw=WORK/'data'/rawfolder/a.dataset;maps=files_by_date(raw)
        out=Path(a.output).resolve() if a.output else WORK/f'data/patches_{a.dataset.lower()}/train'
        if not out.is_relative_to((WORK/'data').resolve()):raise ValueError('Output must stay in work/data')
        if a.generate and (not a.output or (out.exists() and any(out.iterdir()))):
            raise ValueError('--generate needs an explicit new empty --output; existing data are never overwritten')
        if a.generate:out.mkdir(parents=True,exist_ok=True)
        checked=0
        for group,tr in groups:
            selected=coords if a.generate else [coords[i] for i in sorted({0,len(coords)//2,len(coords)-1})]
            for r,c in selected:
                arrays={f'{s}{i}':read_tile(maps[s,d],a.dataset,s,shape,r+offset[0],c+offset[1])
                        for s in ('f','c') for i,d in zip((0,1,3),tr)}
                fp=out/f'{group}_r{r:04d}_c{c:04d}.npz'
                if a.generate:np.savez_compressed(fp,**arrays)
                else:
                    with np.load(fp,allow_pickle=False) as z:
                        if any(not np.array_equal(z[k],arrays[k]) for k in KEYS):raise AssertionError(f'Raw mismatch {fp}')
                checked+=1
            print(f'{group}: {len(selected)} patches '+('generated' if a.generate else 'verified'),flush=True)
        report.update(status='generated' if a.generate else 'representative_verification_passed',checked_patches=checked,output=str(out))
        if a.generate:(out/'generation_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    dest=ROOT/f'docs/audit/preparation_v2_{a.dataset.lower()}.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
