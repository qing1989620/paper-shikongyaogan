# -*- coding: utf-8 -*-
"""Parallel Range downloader for large wheels from download.pytorch.org."""
from __future__ import annotations

import argparse
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def size_of(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with OPENER.open(req, timeout=40) as r:
        return int(r.headers["Content-Length"])


def fetch(url, s, e, path: Path, attempts=8):
    want = e - s + 1
    for _ in range(attempts):
        have = path.stat().st_size if path.exists() else 0
        if have == want:
            return True
        if have > want:
            path.unlink(); have = 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={s+have}-{e}"})
            with OPENER.open(req, timeout=60) as r, open(path, "ab" if have else "wb") as f:
                while True:
                    b = r.read(262144)
                    if not b:
                        break
                    f.write(b)
            if path.stat().st_size == want:
                return True
        except Exception:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk-mb", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    total = size_of(args.url)
    print(f"size {total/1e9:.3f} GB", flush=True)
    if out.exists() and out.stat().st_size == total:
        print("already complete"); return
    chunk = args.chunk_mb * 1024 * 1024
    ranges = [(s, min(s + chunk - 1, total - 1)) for s in range(0, total, chunk)]
    pdir = out.parent / (out.name + ".parts"); pdir.mkdir(exist_ok=True)
    pf = lambda i: pdir / f"c{i:05d}"
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, args.url, s, e, pf(i)): i for i, (s, e) in enumerate(ranges)}
        for fu in as_completed(futs):
            if fu.result():
                done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(ranges)} chunks, {time.time()-t0:.0f}s", flush=True)
    print(f"chunks {done}/{len(ranges)} in {time.time()-t0:.0f}s", flush=True)
    if done < len(ranges):
        raise SystemExit("incomplete")
    with open(out, "wb") as f:
        for i, _ in enumerate(ranges):
            with open(pf(i), "rb") as g:
                f.write(g.read())
    print("saved", out, out.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
