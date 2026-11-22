# -*- coding: utf-8 -*-
"""Thread-based chunk downloader (no subprocess explosion, explicit timeouts)."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://data.csiro.au/dap/ws/v2/collections"
CHUNK = 2 * 1024 * 1024
for _k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def qs(**kw):
    return urllib.parse.urlencode(kw)


def api(url, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with OPENER.open(req, timeout=40) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"  api retry {i+1}: {type(e).__name__}", flush=True)
            time.sleep(2)
    raise RuntimeError("api failed")


def list_files(cid):
    tree = api(f"{API}/{cid}/folders?{qs(folder='/', page=1, size=200)}")
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

    for r in tree.get("folders", []):
        walk(r, "")
    out = []
    for fo in leaves:
        out.extend(api(f"{API}/{cid}/files?{qs(folder=fo, page=1, size=200)}").get("file", []))
    return out


def link_of(cid, f):
    parent = "/" + str(Path(f["filename"]).parent).replace("\\", "/").lstrip("/")
    try:
        for ff in api(f"{API}/{cid}/files?{qs(folder=parent, page=1, size=200)}").get("file", []):
            if ff["filename"] == f["filename"]:
                return ff["presignedLink"]["href"]
    except Exception:
        pass
    return f["presignedLink"]["href"]


def fetch_range(url, s, e, path: Path, attempts=6):
    want = e - s + 1
    for a in range(attempts):
        have = path.stat().st_size if path.exists() else 0
        if have == want:
            return True
        if have > want:
            path.unlink(); have = 0
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0", "Range": f"bytes={s+have}-{e}"})
            with OPENER.open(req, timeout=45) as r:
                with open(path, "ab" if have else "wb") as out:
                    while True:
                        b = r.read(262144)
                        if not b:
                            break
                        out.write(b)
            if path.stat().st_size == want:
                return True
        except Exception:
            pass
    return path.exists() and path.stat().st_size == want


def dl_file(cid, f, root: Path, workers=24):
    rel = Path(f["filename"]); dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(f["fileSize"])
    if dest.exists() and dest.stat().st_size == size:
        return True, "skip"
    url = link_of(cid, f)
    ranges = [(s, min(s + CHUNK - 1, size - 1)) for s in range(0, size, CHUNK)]
    pdir = dest.parent / (".parts_" + dest.name); pdir.mkdir(exist_ok=True)
    pf = lambda i: pdir / f"c{i:05d}"
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_range, url, s, e, pf(i)): i for i, (s, e) in enumerate(ranges)}
        done = sum(1 for fu in as_completed(futs) if fu.result())
    print(f"  chunks {done}/{len(ranges)} in {time.time()-t0:.0f}s", flush=True)
    if done < len(ranges):
        return False, f"{done}/{len(ranges)}"
    tmp = dest.parent / (dest.name + ".asm")
    with open(tmp, "wb") as out:
        for i, _ in enumerate(ranges):
            with open(pf(i), "rb") as inp:
                out.write(inp.read())
    if tmp.stat().st_size != size:
        return False, "size mismatch"
    if dest.exists():
        os.chmod(dest, 0o666)
    os.replace(str(tmp), str(dest))
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--need", default="")
    args = ap.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    cache = root / f"filelist_{args.cid}.json"
    fs = json.loads(cache.read_text()) if cache.exists() else list_files(args.cid)
    if not cache.exists():
        cache.write_text(json.dumps(fs, indent=1))
    if args.need:
        needles = [s.strip() for s in args.need.split(",") if s]
        fs = [f for f in fs if any(n in f["filename"] for n in needles)]
    print(f"[{args.cid}] {len(fs)} files", flush=True)
    for i, f in enumerate(fs, 1):
        print(f"({i}/{len(fs)}) GET {f['filename']}", flush=True)
        ok, msg = dl_file(args.cid, f, root, args.workers)
        print(f"  -> {msg}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
