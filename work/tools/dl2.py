# -*- coding: utf-8 -*-
"""Robust parallel downloader for CSIRO S3.

Design:
  - enumerate all files once via the public v2 API
  - each file split into fixed 2 MiB chunks, each chunk = one .partNNN file
  - a chunk is DONE iff its .part file size == expected chunk size
  - workers curl processes run in parallel; each writes its own .part file with
    `-r start-end` directly to the .part (overwrite), no resume/append complexity
  - repeat rounds until all chunks done (failed ones simply re-fetched)
  - concatenate chunks in order to final path
No delete of bulk files; completed .part files are left in a .parts/ subdir.
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
CHUNK = 2 * 1024 * 1024


def qs(**kw):
    return urllib.parse.urlencode(kw)


def api(url, retries=6):
    import time as _t
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  api retry {i+1}/{retries}: {type(e).__name__}", flush=True)
            _t.sleep(2 * (i + 1))
    raise last


def list_files(cid):
    tree = api(f"{API}/{cid}/folders?{qs(folder='/', page=1, size=200)}")
    leaves, seen = [], set()

    def walk(node, prefix):
        name = node.get("name", "")
        path = "/" if name == "/" else (
            name if name.startswith("/") else
            ((prefix.rstrip("/") + "/" + name) if prefix != "/" else "/" + name))
        kids = node.get("folders", [])
        if kids:
            for c in kids:
                walk(c, path)
        elif node.get("fileCount", 0) > 0 and path not in seen:
            seen.add(path)
            leaves.append(path)

    for root in tree.get("folders", []):
        walk(root, "")
    out = []
    for fo in leaves:
        out.extend(api(f"{API}/{cid}/files?{qs(folder=fo, page=1, size=200)}").get("file", []))
    return out


def fresh_link(cid, f):
    parent = "/" + str(Path(f["filename"]).parent).replace("\\", "/").lstrip("/")
    try:
        for ff in api(f"{API}/{cid}/files?{qs(folder=parent, page=1, size=200)}").get("file", []):
            if ff["filename"] == f["filename"]:
                return ff["presignedLink"]["href"]
    except Exception:
        pass
    return f["presignedLink"]["href"]


def expected_size(s, e):
    return e - s + 1


def download_file(cid, f, out_root, workers, rounds):
    rel = Path(f["filename"])
    dest = out_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(f["fileSize"])
    if dest.exists() and dest.stat().st_size == size:
        return True, "skip"
    # clean stale assembling temp from previous broken runs (single file, not bulk)
    stale = dest.parent / (dest.name + ".assembling")
    if stale.exists():
        try:
            os.chmod(stale, 0o666)
            stale.unlink()
        except Exception:
            pass

    url = fresh_link(cid, f)
    ranges = [(s, min(s + CHUNK - 1, size - 1)) for s in range(0, size, CHUNK)]
    pdir = dest.parent / (".parts_" + dest.name)
    pdir.mkdir(parents=True, exist_ok=True)

    def pfile(i):
        return pdir / f"chunk_{i:05d}"

    def done(i, s, e):
        p = pfile(i)
        return p.exists() and p.stat().st_size == expected_size(s, e)

    for rnd in range(rounds):
        pending = [i for i, (s, e) in enumerate(ranges) if not done(i, s, e)]
        if not pending:
            break
        procs = []
        for i in pending[:workers]:
            s, e = ranges[i]
            # fresh link occasionally; reuse per round is fine (48h expiry)
            cmd = ["curl", "-sS", "-m", "200", "-r", f"{s}-{e}",
                   "-o", str(pfile(i)), "--", url]
            procs.append(subprocess.Popen(cmd, env=ENV,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        ok = sum(1 for i, (s, e) in enumerate(ranges) if done(i, s, e))
        print(f"  round {rnd+1}: {ok}/{len(ranges)} chunks", flush=True)

    missing = [i for i, (s, e) in enumerate(ranges) if not done(i, s, e)]
    if missing:
        return False, f"missing {len(missing)} chunks"

    # concatenate in order
    tmp = dest.parent / (dest.name + ".assembling")
    with open(tmp, "wb") as out:
        for i, (s, e) in enumerate(ranges):
            with open(pfile(i), "rb") as inp:
                out.write(inp.read())
    got = tmp.stat().st_size
    if got != size:
        return False, f"assembled {got} != {size}"
    if dest.exists():
        try:
            os.chmod(dest, 0o666)
            os.replace(str(tmp), str(dest))
        except Exception:
            with open(tmp, "rb") as i, open(dest, "wb") as o:
                o.write(i.read())
    else:
        os.replace(str(tmp), str(dest))
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--only", default="")
    ap.add_argument("--need", default="", help="comma-separated substrings; keep files matching any")
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"filelist_{args.cid}.json"
    if cache.exists() and not args.only:
        fs = json.loads(cache.read_text())
        print(f"[{args.cid}] cached list: {len(fs)} files "
              f"{sum(int(f['fileSize']) for f in fs)/1e9:.2f}GB", flush=True)
    else:
        fs = list_files(args.cid)
        if not args.only:
            cache.write_text(json.dumps(fs, indent=1, ensure_ascii=False))
    if args.only:
        fs = [f for f in fs if args.only in f["filename"]]
    if args.need:
        needles = [s.strip() for s in args.need.split(",") if s.strip()]
        fs = [f for f in fs if any(n in f["filename"] for n in needles)]
    print(f"[{args.cid}] {len(fs)} files {sum(int(f['fileSize']) for f in fs)/1e9:.2f}GB", flush=True)
    mf = root / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else []
    done = {m["file"] for m in manifest}
    for f in fs:
        if f["filename"] in done:
            continue
        print(f"GET {f['filename']} {int(f['fileSize'])/1e6:.0f}MB", flush=True)
        t0 = time.time()
        ok, msg = download_file(args.cid, f, root, args.workers, args.rounds)
        print(f"  -> {msg} in {time.time()-t0:.0f}s", flush=True)
        if ok:
            manifest.append({"file": f["filename"], "size": int(f["fileSize"])})
            mf.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
