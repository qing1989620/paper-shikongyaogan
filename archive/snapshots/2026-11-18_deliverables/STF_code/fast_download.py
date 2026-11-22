# -*- coding: utf-8 -*-
"""Multi-connection Range downloader bypassing the flaky local HTTP proxy.

Direct connections to CSIRO S3 presigned URLs work (~27 KB/s/conn) while the
local proxy throttles/502s them. N parallel Range connections scale throughput.
Each chunk is written to its own file by curl; completed chunks are skipped on
retry; all chunks are concatenated at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_API = "https://data.csiro.au/dap/ws/v2/collections"
PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy",
              "ALL_PROXY", "all_proxy")


def _env_no_proxy() -> dict:
    env = dict(os.environ)
    for k in PROXY_KEYS:
        env.pop(k, None)
    return env


def api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def qs(**kwargs) -> str:
    return urllib.parse.urlencode(kwargs)


def leaf_folders(cid: str) -> list[str]:
    tree = api_get(f"{BASE_API}/{cid}/folders?{qs(folder='/', page=1, size=200)}")
    leaves: list[str] = []
    seen: set[str] = set()

    def walk(node: dict, prefix: str) -> None:
        name = node.get("name", "")
        path = "/" if name == "/" else (
            name if name.startswith("/") else
            ((prefix.rstrip("/") + "/" + name) if prefix != "/" else "/" + name))
        children = node.get("folders", [])
        if children:
            for ch in children:
                walk(ch, path)
        elif node.get("fileCount", 0) > 0 and path not in seen:
            seen.add(path)
            leaves.append(path)

    for root in tree.get("folders", []):
        walk(root, "")
    return leaves


def all_files(cid: str) -> list[dict]:
    files: list[dict] = []
    for fo in leaf_folders(cid):
        d = api_get(f"{BASE_API}/{cid}/files?{qs(folder=fo, page=1, size=200)}")
        files.extend(d.get("file", []))
    return files


def fresh_link(cid: str, f: dict) -> str:
    parent = "/" + str(Path(f["filename"]).parent).replace("\\", "/").lstrip("/")
    try:
        d = api_get(f"{BASE_API}/{cid}/files?{qs(folder=parent, page=1, size=200)}")
        for ff in d.get("file", []):
            if ff["filename"] == f["filename"]:
                return ff["presignedLink"]["href"]
    except Exception:  # noqa: BLE001
        pass
    return f["presignedLink"]["href"]


def curl_chunk(url: str, start: int, end: int, part: Path, retries: int = 8) -> bool:
    want = end - start + 1
    env = _env_no_proxy()
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return True
        s = start + have
        tmp = part.with_suffix(part.suffix + f".tmp{attempt}")
        try:
            r = subprocess.run(
                ["curl", "-sS", "-m", "180", "--retry", "2", "-r", f"{s}-{end}",
                 "-o", str(tmp), url],
                env=env, capture_output=True, timeout=190,
            )
            if tmp.exists() and tmp.stat().st_size > 0:
                with open(part, "ab") as out, open(tmp, "rb") as inp:
                    out.write(inp.read())
                tmp.unlink()
            have = part.stat().st_size if part.exists() else 0
            if have >= want:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    return False


def download_file(cid: str, f: dict, dest_root: Path, conns: int = 16,
                  chunk_mb: int = 2) -> Path:
    rel = Path(f["filename"])
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(f["fileSize"])
    if dest.exists() and dest.stat().st_size == size:
        print(f"SKIP {rel}", flush=True)
        return dest
    url = fresh_link(cid, f)
    chunk = chunk_mb * 1024 * 1024
    ranges = [(s, min(s + chunk - 1, size - 1)) for s in range(0, size, chunk)]
    print(f"GET  {rel} size={size} chunks={len(ranges)} conns={conns}", flush=True)
    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(max_workers=conns) as ex:
        futs = {ex.submit(curl_chunk, url, s, e,
                dest.parent / (dest.name + f".part{s:012d}")): (s, e)
                for s, e in ranges}
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
    if ok != len(ranges):
        raise RuntimeError(f"{rel}: {ok}/{len(ranges)} chunks ok")
    with open(dest, "wb") as out:
        for s, _ in ranges:
            part = dest.parent / (dest.name + f".part{s:012d}")
            out.write(part.read_bytes())
            part.unlink()
    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"{rel}: assembled {got} != {size}")
    print(f"OK   {rel} ({got}) in {time.time()-t0:.0f}s", flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conns", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=2)
    ap.add_argument("--only", default="", help="only download files whose name contains this")
    args = ap.parse_args()
    dest_root = Path(args.out)
    dest_root.mkdir(parents=True, exist_ok=True)

    files = all_files(args.cid)
    if args.only:
        files = [f for f in files if args.only in f["filename"]]
    print(f"[{args.cid}] {len(files)} files, "
          f"{sum(int(f['fileSize']) for f in files)/1e9:.2f} GB", flush=True)
    manifest_path = dest_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done_names = {m["file"] for m in manifest}
    for f in files:
        if f["filename"] in done_names:
            continue
        try:
            p = download_file(args.cid, f, dest_root, args.conns, args.chunk_mb)
            manifest.append({"file": f["filename"], "size": int(f["fileSize"]),
                             "path": str(p), "id": f["id"]})
            manifest_path.write_text(
                json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {f['filename']}: {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
