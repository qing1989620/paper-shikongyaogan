"""Fetch just the official CSIRO source files needed for selected frozen splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import tempfile
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from restore_frozen_data import ROOT, SOURCES, source_path, source_rows, split_rows

DAP = "https://data.csiro.au/dap/ws/v2"
COLLECTIONS = {"LGC": "csiro:5847", "CIA": "csiro:5846"}
LICENCE = f"{DAP}/licences/1061"
USER_AGENT = "Frozen-STF-data/2026"


def open_public(url: str, accept: str):
    return urlopen(Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT}), timeout=35)


def with_retry(operation, label: str):
    for index in range(3):
        try:
            return operation()
        except (URLError, TimeoutError, ConnectionError, IncompleteRead, ssl.SSLError) as error:
            if isinstance(error, HTTPError) and 400 <= error.code < 500 and error.code != 429:
                raise RuntimeError(f"{label}: CSIRO returned HTTP {error.code}") from None
            if index == 2:
                raise RuntimeError(f"{label}: {type(error).__name__} after three attempts") from None
            time.sleep(index + 1)


def inventory(dataset: str) -> dict[str, dict]:
    with open_public(f"{DAP}/collections/{COLLECTIONS[dataset]}/data", "application/json") as stream:
        data = json.load(stream)
    if data["licenceLink"]["href"] != LICENCE:
        raise RuntimeError("The collection now lists different terms. Review its landing page before fetching.")
    return {entry["filename"]: entry for entry in data["file"]}


def transfer(url: str, destination: Path, size: int, expected_sha256: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ("s3.data.csiro.au", "data.csiro.au"):
        raise ValueError("Expected an official CSIRO HTTPS file link")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    received = 0
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".partial", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with open_public(url, "application/octet-stream") as response, temp.open("wb") as output:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                received += len(chunk)
                if received > size:
                    raise ValueError("Download exceeds the frozen byte count")
                digest.update(chunk)
                output.write(chunk)
        if received != size or digest.hexdigest() != expected_sha256:
            raise ValueError(f"Official bytes differ from the frozen manifest: {destination.name}")
        if destination.exists():
            raise FileExistsError(destination)
        temp.rename(destination)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SOURCES), required=True)
    parser.add_argument("--split", action="append", choices=("train", "val", "test"), required=True)
    parser.add_argument("--download", action="store_true", help="Fetch data for your own use under the collection terms")
    args = parser.parse_args()
    expected = source_rows(args.dataset, split_rows(args.dataset, args.split))
    catalog = with_retry(lambda: inventory(args.dataset), "official collection metadata")
    print(f"{args.dataset}: {len(expected)} official files, {sum(int(e['expected_bytes']) for e in expected)/2**20:.1f} MiB")
    print(f"Collection terms: {LICENCE}", flush=True)
    raw_root = ROOT / "work/data" / SOURCES[args.dataset][0]
    for row in expected:
        local = source_path(args.dataset, row)
        name = local.relative_to(raw_root).as_posix()
        size = int(row["expected_bytes"])
        if name not in catalog or catalog[name]["fileSize"] != size:
            raise ValueError(f"No official file matches the manifest: {name}")
        if not args.download:
            print(name)
            continue
        if local.exists():
            with local.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Existing source differs from manifest: {name}")
            print(f"Verified {name}", flush=True)
            continue
        link = catalog[name]["presignedLink"]["href"]
        with_retry(lambda: transfer(link, local, size, row["sha256"]), name)
        print(f"Fetched and verified {name}", flush=True)


if __name__ == "__main__":
    main()
