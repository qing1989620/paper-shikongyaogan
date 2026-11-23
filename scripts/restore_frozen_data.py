"""Verify frozen patches or recreate missing patches from locally obtained official imagery."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/tools"))
from prepare_dataset_v2 import np, read_tile

SOURCES = {"LGC": ("raw", (2720, 3200)), "CIA": ("raw_cia", (2040, 1720))}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def split_rows(dataset: str, splits: list[str] | None) -> list[dict]:
    return [
        row for row in read_rows(ROOT / "docs/audit/dataset_manifest.csv")
        if row["dataset"] == dataset and (splits is None or row["split"] in splits)
    ]


def source_rows(dataset: str, patches: list[dict]) -> list[dict]:
    dates = {p[key] for p in patches for key in ("prev_date", "target_date", "next_date")}
    entries = [
        entry for entry in read_rows(ROOT / "docs/audit/raw_file_integrity.csv")
        if entry["dataset"] == dataset and entry["date"] in dates
    ]
    expected = {(sensor, date) for sensor in ("f", "c") for date in dates}
    keys = {(entry["sensor"], entry["date"]) for entry in entries}
    if not expected or len(entries) != len(expected) or keys != expected:
        raise ValueError(f"Missing or duplicated source manifest records for {dataset}")
    return entries


def source_path(dataset: str, entry: dict) -> Path:
    raw = ROOT / entry["path"]
    folder, _ = SOURCES[dataset]
    source_dir = ROOT / "work/data" / folder / dataset
    if not raw.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError(f"Unexpected official image path: {entry['path']}")
    return raw


def verify_raw(dataset: str, patches: list[dict]) -> dict:
    mapping = {}
    for entry in source_rows(dataset, patches):
        raw = source_path(dataset, entry)
        with raw.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Official source file differs from the frozen input: {entry['path']}")
        mapping[entry["sensor"], entry["date"]] = raw
    return mapping


def restore_patch(row: dict, mapping: dict) -> None:
    dataset = row["dataset"]
    dest = ROOT / row["path"]
    dirname = (ROOT / "work/data" / f"patches_{dataset.lower()}").resolve()
    if not dest.resolve().is_relative_to(dirname) or dest.suffix != ".npz":
        raise ValueError(f"Unexpected patch path in the manifest: {row['path']}")
    if dest.is_file():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Existing patch has a different hash: {row['path']}")
        return
    if not mapping:
        raise FileNotFoundError(row["path"])

    _, shape = SOURCES[dataset]
    raw_row, raw_col = int(row["raw_row"]), int(row["raw_col"])
    dates = (row["prev_date"], row["target_date"], row["next_date"])
    arrays = {
        f"{sensor}{index}": read_tile(mapping[sensor, date], dataset, sensor, shape, raw_row, raw_col)
        for sensor in ("f", "c") for index, date in zip((0, 1, 3), dates)
    }
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    content = output.getvalue()
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError(f"Generated patch differs from the frozen bytes: {row['path']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SOURCES), required=True)
    parser.add_argument("--split", action="append", choices=("train", "val", "test"))
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    chosen = split_rows(args.dataset, args.split)
    mapping = verify_raw(args.dataset, chosen) if args.generate else {}
    for row in chosen:
        restore_patch(row, mapping)
    print(f"{args.dataset}: {len(chosen)} patches match their recorded SHA-256.")


if __name__ == "__main__":
    main()
