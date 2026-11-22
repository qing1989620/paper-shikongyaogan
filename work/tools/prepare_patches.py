# -*- coding: utf-8 -*-
"""Prepare LGC / CIA patches for inference-time experiments.

Reads raw CSIRO files with the audited byte layouts, clips DN, normalizes to
[-1,1], and writes 256x256 patches as .npz containing
    f0, f1, f3, c0, c1, c3   (float16, HWC, 6 bands)
where index 1 is the prediction date (GT), 0/3 are the reference dates.

Splits follow the official STFDiff syy_setting-9 convention:
  LGC  train triples 1-5; val = triple 6 (stable); test = flood triple
  CIA  date-level 8/3/4 split (calib/val/test)
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

BANDS = 6
LGC_H, LGC_W = 2720, 3200
CIA_H, CIA_W = 2040, 1720
MAX_DN = 10000


def read_bsq(path: Path, rows: int, cols: int) -> np.ndarray:
    a = np.fromfile(path, dtype="<u2")
    assert a.size == rows * cols * BANDS, f"{path}: {a.size}"
    return a.reshape(BANDS, rows, cols).transpose(1, 2, 0)


def read_bil(path: Path, rows: int, cols: int) -> np.ndarray:
    a = np.fromfile(path, dtype="<u2")
    assert a.size == rows * cols * BANDS, f"{path}: {a.size}"
    return a.reshape(rows, BANDS, cols).transpose(0, 2, 1)


def norm(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0, MAX_DN)
    return (x / MAX_DN) * 2.0 - 1.0


def patch_grid(h: int, w: int, size: int = 256):
    for r in range(0, h - size + 1, size):
        for c in range(0, w - size + 1, size):
            yield r, c


def build_triple(out_dir: Path, triple_id: str, dates, f_imgs, c_imgs, size=256):
    """dates = (d0, d1, d3); f_imgs/c_imgs dicts date->HWC array."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d0, d1, d3 = dates
    f0, f1, f3 = f_imgs[d0], f_imgs[d1], f_imgs[d3]
    c0, c1, c3 = c_imgs[d0], c_imgs[d1], c_imgs[d3]
    h, w = f0.shape[:2]
    n = 0
    for r, c in patch_grid(h, w, size):
        sl = (slice(r, r + size), slice(c, c + size))
        np.savez_compressed(
            out_dir / f"{triple_id}_r{r:04d}_c{c:04d}.npz",
            f0=norm(f0[sl]).astype(np.float16),
            f1=norm(f1[sl]).astype(np.float16),
            f3=norm(f3[sl]).astype(np.float16),
            c0=norm(c0[sl]).astype(np.float16),
            c1=norm(c1[sl]).astype(np.float16),
            c3=norm(c3[sl]).astype(np.float16),
        )
        n += 1
    return n


def lgc_date_map(raw: Path) -> tuple[dict, dict]:
    """date 'YYYY-MM-DD' -> Landsat path, MODIS path."""
    fmap, cmap = {}, {}
    for d in sorted((raw / "LGC" / "Landsat").iterdir()):
        if not d.is_dir():
            continue
        fs = [p for p in d.glob("*.int") if p.is_file()]
        if not fs:
            continue
        # date from folder name 2004_107_Apr16 -> 2004-04-16
        parts = d.name.split("_")
        y, doy = parts[0], parts[1]
        import datetime as dt
        day = dt.date(int(y), 1, 1) + dt.timedelta(days=int(doy) - 1)
        fmap[day.isoformat()] = fs[0]
    for d in sorted((raw / "LGC" / "MODIS").iterdir()):
        if not d.is_dir():
            continue
        fs = [p for p in d.glob("*.int") if p.is_file()]
        if not fs:
            continue
        parts = d.name.split("_")
        y, doy = parts[0], parts[1]
        import datetime as dt
        day = dt.date(int(y), 1, 1) + dt.timedelta(days=int(doy) - 1)
        cmap[day.isoformat()] = fs[0]
    return fmap, cmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["LGC", "CIA"], required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "LGC":
        fmap, cmap = lgc_date_map(raw)
        print(f"LGC dates: landsat={len(fmap)} modis={len(cmap)}")
        common = sorted(set(fmap) & set(cmap))
        print("common dates:", common)
        f_imgs = {d: read_bsq(fmap[d], LGC_H, LGC_W) for d in common}
        c_imgs = {d: read_bsq(cmap[d], LGC_H, LGC_W) for d in common}
        splits = json.loads((out / "splits.json").read_text()) if (out / "splits.json").exists() else None
        triples = {
            "val":  ("2005-02-14", "2005-03-02", "2005-04-03"),
            "test": ("2004-12-28", "2005-01-13", "2005-01-29"),
        }
        for name, tr in triples.items():
            if not all(d in common for d in tr):
                print(f"skip {name}: missing dates {[d for d in tr if d not in common]}")
                continue
            n = build_triple(out / name, name, tr, f_imgs, c_imgs, args.size)
            print(f"{name}: {n} patches")
        (out / "meta.json").write_text(json.dumps({
            "dataset": "LGC", "shape": [LGC_H, LGC_W], "layout": "BSQ",
            "triples": triples, "patch": args.size}, indent=1))

    else:  # CIA
        fmap, cmap = {}, {}
        for d in sorted((raw / "CIA" / "Landsat").iterdir()):
            if not d.is_dir():
                continue
            fs = [p for p in d.glob("*.bil") if p.is_file()]
            if fs:
                parts = d.name.split("_")
                import datetime as dt
                day = dt.date(int(parts[0]), 1, 1) + dt.timedelta(days=int(parts[1]) - 1)
                fmap[day.isoformat()] = fs[0]
        for d in sorted((raw / "CIA" / "MODIS").iterdir()):
            if not d.is_dir():
                continue
            fs = [p for p in d.glob("*.int") if p.is_file()]
            if fs:
                parts = d.name.split("_")
                import datetime as dt
                day = dt.date(int(parts[0]), 1, 1) + dt.timedelta(days=int(parts[1]) - 1)
                cmap[day.isoformat()] = fs[0]
        print(f"CIA dates: landsat={len(fmap)} modis={len(cmap)}")
        common = sorted(set(fmap) & set(cmap))
        print("common:", common)
        # center crop to 1792x1536
        top = (CIA_H - 1792) // 2
        left = (CIA_W - 1536) // 2
        f_imgs = {d: read_bil(fmap[d], CIA_H, CIA_W)[top:top+1792, left:left+1536] for d in common}
        c_imgs = {d: read_bsq(cmap[d], CIA_H, CIA_W)[top:top+1792, left:left+1536] for d in common}
        # date-level 8/3/4 split over consecutive triples
        trips = [(common[i], common[i+1], common[i+2]) for i in range(len(common)-2)]
        print("triples:", len(trips))
        calib, val, test = trips[:8], trips[8:11], trips[11:]
        for name, group in [("val", val), ("test", test)]:
            for i, tr in enumerate(group):
                n = build_triple(out / name, f"{name}{i}", tr, f_imgs, c_imgs, args.size)
                print(f"{name}{i} {tr}: {n} patches")
        (out / "meta.json").write_text(json.dumps({
            "dataset": "CIA", "shape": [CIA_H, CIA_W], "layout": "Landsat BIL / MODIS BSQ",
            "crop": [1792, 1536], "splits": {"calib": calib, "val": val, "test": test},
            "patch": args.size}, indent=1))

    print("DONE")


if __name__ == "__main__":
    main()
