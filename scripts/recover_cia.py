"""Recover CIA main images atomically; retain existing patch data and fragments."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = (ROOT / "work/data/raw_cia").resolve()
LOG = ROOT / "work/logs/recovery_cia/status.json"
BASE = "https://data.csiro.au/dap/ws/v2/collections/5846"
LOCK = threading.Lock()
STATE = {"started_cst": datetime.now(timezone(timedelta(hours=8))).isoformat(), "files": {}}

def record(name, **values):
    with LOCK:
        STATE["files"].setdefault(name, {}).update(values)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        tmp = LOG.with_suffix('.tmp')
        tmp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(LOG)

def recover(entry):
    name, expected = entry['filename'], int(entry['fileSize'])
    target = (DEST / name).resolve()
    if not target.is_relative_to(DEST) or expected != 42105600:
        raise ValueError(f'Unexpected CIA target or byte size: {name}')
    if target.exists() and target.stat().st_size == expected:
        record(name, status='already_complete', bytes=expected)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix+'.recovery.part')
    folder = '/' + Path(name).parent.as_posix().lstrip('/')
    for attempt in range(1, 4):
        try:
            if shutil.disk_usage(DEST).free < 8*2**30:
                raise RuntimeError('Disk free below 8 GiB; recovery paused')
            url = BASE+'/files?'+urllib.parse.urlencode({'folder':folder,'page':1,'size':200})
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STF-research/1.0'}), timeout=30) as r:
                options = json.load(r).get('file',[])
            item = next(f for f in options if f['filename']==name)
            done = part.stat().st_size if part.exists() else 0
            if done == expected:
                part.replace(target)
                record(name,status='complete',bytes=expected)
                return
            headers = {'User-Agent':'STF-research/1.0'}
            if done: headers['Range'] = f'bytes={done}-'
            request = urllib.request.Request(item['presignedLink']['href'],headers=headers)
            with urllib.request.urlopen(request,timeout=45) as r:
                mode = 'ab' if done and r.status==206 else 'wb'
                with part.open(mode) as out:
                    while True:
                        chunk=r.read(1024*1024)
                        if not chunk: break
                        out.write(chunk)
                        if out.tell()>expected: raise RuntimeError('Source exceeded expected size')
            if part.stat().st_size != expected: raise RuntimeError('Incomplete transfer')
            h=hashlib.sha256()
            with part.open('rb') as f:
                for b in iter(lambda:f.read(2**20),b''): h.update(b)
            part.replace(target)
            record(name,status='complete',bytes=expected,sha256=h.hexdigest(),attempt=attempt)
            print(f'OK {name}',flush=True)
            return
        except Exception as e:
            # Never log presigned URLs or server tokens.
            record(name,status='retry',attempt=attempt,error_type=type(e).__name__,
                   bytes=part.stat().st_size if part.exists() else 0)
            print(f'RETRY {attempt} {name} {type(e).__name__}',flush=True)
            time.sleep(min(attempt*2,6))
    record(name,status='failed')

if __name__=='__main__':
    entries=json.loads((DEST/'filelist_5846.json').read_text(encoding='utf-8'))
    entries=[f for f in entries if Path(f['filename']).suffix.lower() in ('.int','.bil')]
    print(f'Recovering {len(entries)} CIA main images; three bounded download workers',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(recover,entries))
    STATE['finished_cst']=datetime.now(timezone(timedelta(hours=8))).isoformat()
    STATE['complete']=all(x['status'] in ('complete','already_complete') for x in STATE['files'].values())
    record('_summary',status='complete' if STATE['complete'] else 'incomplete')
    print('Recovery completed:',STATE['complete'],flush=True)
