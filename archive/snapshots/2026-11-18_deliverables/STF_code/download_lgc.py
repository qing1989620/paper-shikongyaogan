# -*- coding: utf-8 -*-
"""Download the LGC (Lower Gwydir Catchment) dataset from CSIRO DAP.

Public collection csiro:5847, doi:10.4225/08/5111AD2B7FEE6.
14 Landsat-5 TM / MODIS pairs @ 25 m + inundation mask (~2.9 GB total).
The v2 /folders endpoint returns the FULL nested tree regardless of the
`folder` query; we parse that tree to obtain concrete leaf folders, then
query /files for each leaf and use its S3 presigned link.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://data.csiro.au/dap/ws/v2/collections/5847"


def api_get(path: str, params: dict | None = None) -> dict:
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def leaf_folders() -> list[str]:
    tree = api_get("/folders", {"folder": "/", "page": 1, "size": 200})
    leaves: list[str] = []
    seen: set[str] = set()

    def walk(node: dict, prefix: str) -> None:
        name = node.get("name", "")
        # root node is "/" -> do not add; build absolute POSIX path
        if name == "/":
            path = "/"
        elif name.startswith("/"):
            path = name
        else:
            path = (prefix.rstrip("/") + "/" + name) if prefix != "/" else "/" + name
        children = node.get("folders", [])
        if children:
            for ch in children:
                walk(ch, path)
        else:
            # leaf folder: query it only if it declares files
            if node.get("fileCount", 0) > 0 and path not in seen:
                seen.add(path)
                leaves.append(path)

    for root in tree.get("folders", []):
        walk(root, "")
    return leaves


def files_in(folder: str) -> list[dict]:
    data = api_get("/files", {"folder": folder, "page": 1, "size": 200})
    return data.get("file", [])


def refresh_link(f: dict) -> str:
    fresh = files_in("/" + str(Path(f["filename"]).parent).replace("\\", "/").lstrip("/"))
    for ff in fresh:
        if ff["filename"] == f["filename"]:
            return ff["presignedLink"]["href"]
    return f["presignedLink"]["href"]


def download_one(f: dict, dest_root: Path, retries: int = 4) -> Path:
    rel = Path(f["filename"])
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    expect = int(f["fileSize"])
    if dest.exists() and dest.stat().st_size == expect:
        print(f"SKIP {rel} ({expect})", flush=True)
        return dest
    for attempt in range(retries):
        url = refresh_link(f)
        done = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r, open(
                dest, "ab" if done else "wb"
            ) as out:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            got = dest.stat().st_size
            if got == expect:
                print(f"OK   {rel} ({got})", flush=True)
                return dest
            print(f"RETRY {rel}: size {got} != {expect}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"RETRY {rel}: {e}", flush=True)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to download {rel}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dest_root = Path(args.out)
    dest_root.mkdir(parents=True, exist_ok=True)

    folders = leaf_folders()
    print(f"leaf folders: {len(folders)}", flush=True)
    files: list[dict] = []
    for fo in folders:
        files.extend(files_in(fo))
    total = sum(int(f["fileSize"]) for f in files)
    print(f"found {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    manifest = []
    for f in files:
        p = download_one(f, dest_root)
        manifest.append({"file": f["filename"], "size": int(f["fileSize"]),
                         "path": str(p), "id": f["id"]})
    (dest_root / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
