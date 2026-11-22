"""Read-only research audit; writes evidence tables without altering experiments.

Run from any directory: python scripts/audit_workspace.py
The raw-to-patch check verifies three representative patches per temporal group.
It does not assert that every pixel of every patch has been verified.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "docs" / "audit"
TZ = timezone(timedelta(hours=8))


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(name, rows, fields=None):
    fields = fields or sorted({k for row in rows for k in row})
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def git_info(repo):
    def run(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           encoding="utf-8", errors="replace")
        return r.stdout.strip() if r.returncode == 0 else r.stderr.strip()
    return {"path": rel(repo), "commit": run("rev-parse", "HEAD"),
            "remote": run("remote", "get-url", "origin"),
            "status": run("status", "--short")}


def inventory():
    rows = []
    excluded = {".git", "__pycache__"}
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(s in excluded for s in p.relative_to(ROOT).parts):
            continue
        # Generated audit tables are intentionally excluded from their own inventory.
        if OUT in p.parents:
            continue
        st = p.stat()
        rows.append({"path": rel(p), "bytes": st.st_size,
                     "modified_cst": datetime.fromtimestamp(st.st_mtime, TZ).isoformat()})
    write_csv("file_inventory.csv", rows, ["path", "bytes", "modified_cst"])
    totals = defaultdict(lambda: [0, 0])
    for r in rows:
        parts = r["path"].split("/")
        k = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
        totals[k][0] += 1
        totals[k][1] += r["bytes"]
    write_csv("directory_sizes.csv", [{"path": k, "files": n, "bytes": b,
                                      "GiB": round(b / 2**30, 4)}
                                     for k, (n, b) in sorted(totals.items())])
    return {"files_excluding_git_and_audit": len(rows),
            "bytes_excluding_git_and_audit": sum(r["bytes"] for r in rows)}


def raw_date_map(raw):
    maps = {}
    for sensor, folder in [("f", "Landsat"), ("c", "MODIS")]:
        m = {}
        for d in sorted((raw / folder).iterdir()):
            if not d.is_dir():
                continue
            parts = d.name.split("_")
            if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
                continue
            day = (datetime(int(parts[0]), 1, 1) + timedelta(days=int(parts[1]) - 1)).date().isoformat()
            fs = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in (".int", ".bil")]
            if fs:
                m[day] = fs[0]
        maps[sensor] = m
    return maps


def datasets():
    import numpy as np
    group_rows, patch_rows, overlap_rows, raw_rows = [], [], [], []
    for dataset in ["LGC", "CIA"]:
        ds = WORK / "data" / f"patches_{dataset.lower()}"
        meta = json.loads((ds / "meta.json").read_text(encoding="utf-8"))
        h, w = meta["shape"]
        size = meta["patch"]
        top, left = (0, 0) if dataset == "LGC" else ((h - meta["crop"][0]) // 2, (w - meta["crop"][1]) // 2)
        raw = WORK / "data" / ("raw" if dataset == "LGC" else "raw_cia") / dataset
        maps = raw_date_map(raw)
        expected_bytes = h*w*6*2
        for sensor, m in maps.items():
            for day, p in m.items():
                raw_rows.append({"dataset": dataset, "sensor": sensor, "date": day,
                                 "path": rel(p), "expected_bytes": expected_bytes,
                                 "actual_bytes": p.stat().st_size,
                                 "sha256": digest(p),
                                 "size_ok": p.stat().st_size == expected_bytes})
        maps = {sensor: {day: p for day, p in m.items() if p.stat().st_size == expected_bytes}
                for sensor, m in maps.items()}
        def tile(sensor, day, r, c):
            r, c = r + top, c + left
            # Read only a strip instead of mapping every full scene on Windows.
            with maps[sensor][day].open("rb") as fh:
                if dataset == "CIA" and sensor == "f":
                    fh.seek(r * 6 * w * 2)
                    a = np.fromfile(fh, dtype="<u2", count=size*6*w).reshape(size, 6, w)
                    v = a[:, :, c:c+size].transpose(0, 2, 1)
                else:
                    v = np.empty((size, size, 6), dtype=np.uint16)
                    for band in range(6):
                        fh.seek((band*h + r) * w * 2)
                        a = np.fromfile(fh, dtype="<u2", count=size*w).reshape(size, w)
                        v[:, :, band] = a[:, c:c+size]
            return (np.clip(v.astype(np.float32), 0, 10000) / 10000 * 2 - 1).astype(np.float16)

        lookup = {}
        for sensor, m in maps.items():
            for day in m:
                lookup[sensor, hashlib.sha256(tile(sensor, day, 0, 0).tobytes()).hexdigest()] = day

        dates_by_split = defaultdict(set)
        train_all = set()
        for sp in sorted(p for p in ds.iterdir() if p.is_dir()):
            grouped = defaultdict(list)
            for fp in sorted(sp.glob("*.npz")):
                grouped[fp.stem.split("_r")[0]].append(fp)
            for group, files in grouped.items():
                with np.load(files[0], allow_pickle=False) as z:
                    dates = {k: lookup.get((k[0], hashlib.sha256(z[k].tobytes()).hexdigest()))
                             for k in z.files}
                provenance = "raw tile hash match"
                if dataset == "CIA" and not all(dates.values()):
                    if sp.name == "train":
                        generated=json.loads((sp/"generation_manifest.json").read_text(encoding="utf-8"))
                        declared=next(g["dates"] for g in generated["groups"] if g["id"]==group)
                    else:
                        index = int(group.removeprefix(sp.name))
                        declared = meta["splits"][sp.name][index]
                    dates = {f"{sensor}{idx}": day for sensor in ("f", "c")
                             for idx, day in zip((0, 1, 3), declared)}
                    provenance = "declared meta.json; raw unavailable"
                chosen = sorted({0, len(files)//2, len(files)-1})
                problems = []
                raw_available = all(day in maps.get(k[0], {}) for k, day in dates.items())
                for idx in chosen:
                    fp = files[idx]
                    r, c = map(int, re.search(r"_r(\d+)_c(\d+)$", fp.stem).groups())
                    with np.load(fp, allow_pickle=False) as z:
                        for k, day in dates.items():
                            if raw_available and (day is None or not np.array_equal(z[k], tile(k[0], day, r, c))):
                                problems.append(f"{fp.name}:{k}")
                            if z[k].shape != (size, size, 6) or not np.isfinite(z[k]).all():
                                problems.append(f"invalid patch array:{fp.name}:{k}")
                group_rows.append({"dataset": dataset, "split": sp.name, "group": group,
                                   "patches": len(files), "dates": dates,
                                   "checked_patches": [files[i].name for i in chosen],
                                   "date_provenance": provenance,
                                   "raw_match": not problems if raw_available else None,
                                   "patch_array_checks_passed": not problems,
                                   "problems": problems})
                if dates.get("f1"):
                    dates_by_split[sp.name].add(dates["f1"])
                if sp.name == "train":
                    train_all.update(d for k, d in dates.items() if k.startswith("f") and d)
                for fp in files:
                    r, c = map(int, re.search(r"_r(\d+)_c(\d+)$", fp.stem).groups())
                    patch_rows.append({"dataset": dataset, "split": sp.name, "group": group,
                                       "patch_id": fp.stem, "path": rel(fp), "sha256": digest(fp),
                                       "bytes": fp.stat().st_size, "row": r, "col": c,
                                       "raw_row": r+top, "raw_col": c+left, "height": size, "width": size,
                                       "prev_date": dates.get("f0"), "target_date": dates.get("f1"),
                                       "next_date": dates.get("f3"),
                                       "provenance": provenance + "; 3 representative patches checked"})
        overlap_rows.append({"dataset": dataset,
                             "target_dates_by_split": {s: sorted(v) for s, v in dates_by_split.items()},
                             "val_target_in_train_fine": sorted(dates_by_split["val"] & train_all) if (ds/"train").exists() else None,
                             "test_target_in_train_fine": sorted(dates_by_split["test"] & train_all) if (ds/"train").exists() else None,
                             "scope": "current local patches only; not external pretrained weights",
                             "cia_train_available": (ds / "train").exists() if dataset == "CIA" else None})
    dump("dataset_groups.json", group_rows)
    preparation_config = WORK/"configs/datasets/legacy_lgc_v0.json"
    if preparation_config.exists():
        declared=[tuple(x["dates"]) for x in json.loads(preparation_config.read_text(encoding="utf-8"))["train_triples"]]
    else:
        train_ast = ast.parse((WORK/"tools"/"prepare_train.py").read_text(encoding="utf-8"))
        declared = next(ast.literal_eval(n.value) for n in train_ast.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "TRIPLES" for t in n.targets))
    observed = [g for g in group_rows if g["dataset"] == "LGC" and g["split"] == "train"]
    dump("preparation_reproducibility.json", {
        "script": "work/tools/prepare_train.py", "declared_train_triples": declared,
        "declared_patch_count": len(declared)*120, "observed_patch_count": sum(g["patches"] for g in observed),
        "observed_extra_triples": [[g["dates"][k] for k in ("f0", "f1", "f3")] for g in observed
                                   if tuple(g["dates"][k] for k in ("f0", "f1", "f3")) not in declared],
        "status": "configuration_matches_observed" if len(declared)==len(observed) else "mismatch_requires_reproducible_preparation"})
    dump("split_overlap.json", overlap_rows)
    write_csv("raw_file_integrity.csv", raw_rows)
    write_csv("dataset_manifest.csv", patch_rows)
    return {"patches": len(patch_rows), "groups": len(group_rows),
            "raw_verified_groups": sum(g["raw_match"] is True for g in group_rows),
            "raw_unavailable_groups": sum(g["raw_match"] is None for g in group_rows),
            "raw_mismatched_groups": sum(g["raw_match"] is False for g in group_rows),
            "patch_array_checks_passed": all(g["patch_array_checks_passed"] for g in group_rows)}


def checkpoints():
    import torch
    rows = []
    paths = sorted({*WORK.glob("runs/*/best.pth"), *WORK.glob("TCCR_STF/checkpoints/*.pth"),
                    *WORK.glob("runs/research_2026/*/best.pth"),
                    *ROOT.glob("archive/snapshots/**/*.pth"), *ROOT.glob("deliverables/**/*.pth")})
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=True)
        meta = {k: v for k, v in ck.items() if k not in ("model", "state_dict", "optimizer", "scaler")}
        rows.append({"path": rel(p), "sha256": digest(p), "bytes": p.stat().st_size,
                     "epoch": meta.get("epoch", ""), "variant": meta.get("variant", ""),
                     "arch": meta.get("arch", ""), "seed_recorded": meta.get("seed", ""),
                     "val_rmse_stored": meta.get("val_rmse", ""),
                     "metadata": json.dumps(meta, ensure_ascii=False, default=str)})
    write_csv("checkpoint_inventory.csv", rows)
    dup = defaultdict(list)
    for row in rows:
        dup[row["sha256"]].append(row["path"])
    dump("checkpoint_duplicates.json", [{"sha256": h, "paths": ps} for h, ps in dup.items() if len(ps) > 1])
    logs = []
    for run in sorted((WORK / "runs").iterdir()):
        lp = run / "train_log.csv"
        if not lp.exists():
            continue
        with lp.open(encoding="utf-8-sig", newline="") as f:
            rr = list(csv.DictReader(f))
        best = min(rr, key=lambda r: float(r["val_rmse"])) if rr else {}
        textlog = WORK / "logs" / "legacy" / f"log_train_{run.name}.txt"
        if not textlog.exists():
            textlog = WORK / f"log_train_{run.name}.txt"
        if not textlog.exists() and run.name.startswith("arch_"):
            textlog = WORK / "logs" / "legacy" / f"log_{run.name}.txt"
        text_epochs = re.findall(r"ep (\d+) loss", textlog.read_text(encoding="utf-8", errors="replace")) if textlog.exists() else []
        logs.append({"run": run.name, "csv_epochs": len(rr), "csv_best_epoch": best.get("epoch", ""),
                     "csv_best_val": best.get("val_rmse", ""),
                     "textlog_max_epoch": max(map(int, text_epochs), default=0)})
    write_csv("training_log_inventory.csv", logs)
    return len(rows)


def results():
    rows = []
    for p in sorted((WORK / "results").glob("*.csv")):
        with p.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            rr = list(reader)
        rows.append({"path": rel(p), "rows": len(rr), "columns": len(cols), "sha256": digest(p),
                     "evidence_status": "legacy_exploratory_requires_metric_and_provenance_audit"})
    write_csv("result_inventory.csv", rows)
    return len(rows)


def source_snapshot():
    rows = []
    paths = [*WORK.glob("tools/*.py"), *ROOT.glob("scripts/*.py"),
             *WORK.glob("repos/STF/src/model/stfdiff/model6/*.py")]
    for p in sorted(paths):
        rows.append({"path": rel(p), "sha256": digest(p)})
    write_csv("code_hashes.csv", rows)
    repos = [git_info(p) for p in (WORK / "repos").iterdir() if (p / ".git").exists()]
    dump("repositories.json", repos)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = {"python_executable": sys.executable, "python": sys.version, "packages": {}}
    for name in ["torch", "numpy", "scipy", "einops", "scikit-image", "timm", "opencv-python"]:
        try:
            env["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            env["packages"][name] = None
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter("always")
        import scipy.ndimage  # noqa: F401
    env["import_warnings"] = [str(w.message) for w in ww]
    env["disk_free_GiB"] = round(shutil.disk_usage(ROOT).free / 2**30, 3)
    dump("environment.json", env)
    summary = {"generated_at_cst": datetime.now(TZ).isoformat(), "root": str(ROOT),
               "dataset": datasets(), "checkpoints": checkpoints(), "result_csvs": results()}
    source_snapshot()
    summary.update(inventory())
    dump("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
