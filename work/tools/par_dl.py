# -*- coding: utf-8 -*-
"""Fast parallel Range downloader for CSIRO S3 (bypasses throttling proxy).

Strategy: split each file into 1 MiB chunks; run up to N curl processes in
parallel, each写作 independent .part file with HTTP Range; retry incomplete
chunks with resume until done; concatenate. Uses direct (no-proxy) curl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://data.csiro.au/dap/ws/v2/collections"
PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy")
ENV = dict(os.environ)
for k in PROXY_KEYS:
    ENV.pop(k, None)
CHUNK = 1 * 1024 * 1024


def qs(**kw):
    return urllib.parse.urlencode(kw)


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def files(cid):
    tree = api(f"{API}/{cid}/folders?{qs(folder='/',page=1,size=200)}")
    leaves, seen = [], set()
    def walk(node, prefix):
        name = node.get("name", "")
        path = "/" if name == "/" else (name if name.startswith("/") else
               ((prefix.rstrip("/") + "/" + name) if prefix != "/" else "/" + name))
        kids = node.get("folders", [])
        if kids:
            for c in kids:
                walk(c, path)
        elif node.get("fileCount", 0) > 0 and path not in seen:
            seen.add(path); leaves.append(path)
    for root in tree.get("folders", []):
        walk(root, "")
    out = []
    for fo in leaves:
        out.extend(api(f"{API}/{cid}/files?{qs(folder=fo,page=1,size=200)}").get("file", []))
    return out


def link(cid, f):
    parent = "/" + str(Path(f["filename"]).parent).replace("\\", "/").lstrip("/")
    try:
        for ff in api(f"{API}/{cid}/files?{qs(folder=parent,page=1,size=200)}").get("file", []):
            if ff["filename"] == f["filename"]:
                return ff["presignedLink"]["href"]
    except Exception:
        pass
    return f["presignedLink"]["href"]


def dl_one_file(cid, f, out_root, workers=64, rounds=8):
    rel = Path(f["filename"])
    dest = out_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(f["fileSize"])
    if dest.exists() and dest.stat().st_size == size:
        return True, "skip"
    url = link(cid, f)
    ranges = [(s, min(s + CHUNK - 1, size - 1)) for s in range(0, size, CHUNK)]
    def part(s):
        return dest.parent / f"{dest.name}.p{s:09d}"
    def start_chunk(s, e):
        p = part(s)
        have = p.stat().st_size if p.exists() else 0
        end = e
        if have:
            end = e
            subprocess.Popen(
                ["curl", "-sS", "-m", "240", "-r", f"{s+have}-{end}", "-o", str(p), "--", url],
                env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["curl", "-sS", "-m", "240", "-r", f"{s}-{end}", "-o", str(p), "--", url],
                env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        return have
    for rnd in range(rounds):
        # only launch chunks not yet complete
        pending = [(s, e) for s, e in ranges if not part(s).exists() or part(s).stat().st_size < (e-s+1)]
        if not pending:
            break
        # each round: launch at most `workers` chunks, wait for all.
        # Download remaining bytes into a temp file, then append to the part.
        jobs = []
        for s, e in pending[:workers]:
            p = part(s)
            have = p.stat().st_size if p.exists() else 0
            s2 = s + have
            tmp = p.parent / f"{p.name}.r{rnd}"
            args = ["curl", "-sS", "-m", "240", "-r", f"{s2}-{e}", "-o", str(tmp), "--", url]
            pr = subprocess.Popen(args, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            jobs.append((pr, p, tmp))
        for pr, p, tmp in jobs:
            pr.wait()
            if tmp.exists() and tmp.stat().st_size > 0:
                with open(p, "ab") as out, open(tmp, "rb") as inp:
                    out.write(inp.read())
                tmp.unlink()
            elif tmp.exists():
                tmp.unlink()
        done = sum(1 for s, e in ranges if part(s).exists() and part(s).stat().st_size >= (e-s+1))
        print(f"  round {rnd+1}: {done}/{len(ranges)} chunks", flush=True)
    # verify & assemble
    missing = [(s, e) for s, e in ranges if not part(s).exists() or part(s).stat().st_size < (e-s+1)]
    if missing:
        return False, f"missing {len(missing)} chunks"
    tmp = dest.with_suffix(dest.suffix + ".assembling")
    with open(tmp, "wb") as out:
        for s, e in ranges:
            with open(part(s), "rb") as inp:
                out.write(inp.read())
    got = tmp.stat().st_size
    if got != size:
        tmp.unlink()
        return False, f"size {got} != {size}"
    import os as _os
    if dest.exists():
        try:
            _os.chmod(dest, 0o666)
        except Exception:
            pass
        try:
            dest.unlink()
        except Exception:
            pass
    try:
        tmp.replace(dest)
    except PermissionError:
        # fallback: copy bytes then remove temp
        with open(tmp, "rb") as _i, open(dest, "wb") as _o:
            _o.write(_i.read())
        tmp.unlink()
    for s, e in ranges:
        part(s).unlink()
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    fs = files(args.cid)
    if args.only:
        fs = [f for f in fs if args.only in f["filename"]]
    total = sum(int(f["fileSize"]) for f in fs)
    print(f"[{args.cid}] {len(fs)} files {total/1e9:.2f}GB workers={args.workers}", flush=True)
    mf = out_root / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else []
    done = {m["file"] for m in manifest}
    for f in fs:
        if f["filename"] in done:
            continue
        print(f"GET {f['filename']} {int(f['fileSize'])/1e6:.0f}MB", flush=True)
        t0 = time.time()
        ok, msg = dl_one_file(args.cid, f, out_root, args.workers)
        print(f"  -> {msg} in {time.time()-t0:.0f}s", flush=True)
        if ok:
            manifest.append({"file": f["filename"], "size": int(f["fileSize"])})
            mf.write_text(json.dumps(manifest, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
