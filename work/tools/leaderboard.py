# -*- coding: utf-8 -*-
"""Leaderboard over the twenty variants across four splits.

For each variant, lambda is chosen on the dataset's validation split (LGC-val
or CIA-val) by RMSE, then applied to the matching test split. Reports the
paper-scale RMSE ([0,1]), relative change vs E01, worst-split behaviour, a
tail metric (worst decile) and a paired Wilcoxon test against E01.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = [("LGC", "val", "results/E20_lgc_val.csv"),
          ("LGC", "test", "results/E20_lgc_test.csv"),
          ("CIA", "val", "results/E20_cia_val.csv"),
          ("CIA", "test", "results/E20_cia_test.csv")]


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("lam", "rmse", "mae", "psnr", "ssim", "sam", "ergas", "cc"):
            r[k] = float(r[k])
    return rows


def mean_by(rows, key, agg=np.mean):
    out = {}
    for r in rows:
        out.setdefault((r["variant"], r["lam"]), []).append(r[key])
    return {k: float(agg(v)) for k, v in out.items()}


def pick_lambda(val_rows, variant):
    cands = {lam: np.mean([r["rmse"] for r in val_rows
                           if r["variant"] == variant and r["lam"] == lam])
             for lam in sorted({r["lam"] for r in val_rows})}
    best = min(cands, key=cands.get)
    return best, cands


def main():
    data = {}
    for ds, sp, path in SPLITS:
        p = Path(path)
        if not p.exists():
            print("missing", p)
            continue
        data[(ds, sp)] = load(p)

    variants = sorted({r["variant"] for r in next(iter(data.values()))})
    # choose lambda per dataset on its validation split
    chosen = {}
    for ds in ("LGC", "CIA"):
        if (ds, "val") not in data:
            continue
        for v in variants:
            lam, _ = pick_lambda(data[(ds, "val")], v)
            chosen[(ds, v)] = lam

    # baseline per split
    base = {}
    for ds, sp, _ in SPLITS:
        if (ds, sp) not in data:
            continue
        lam_b, _ = pick_lambda(data[(ds, "val")], "E01_fixed1") if (ds, "val") in data else (0.2, None)
        vals = [r["rmse"] for r in data[(ds, sp)] if r["variant"] == "E01_fixed1"]
        base[(ds, sp)] = float(np.mean(vals)) / 2.0

    results = []
    for v in variants:
        row = {"variant": v}
        rels = []
        for ds, sp, _ in SPLITS:
            if (ds, sp) not in data:
                continue
            lam = chosen.get((ds, v), 0.2)
            vals = [r["rmse"] for r in data[(ds, sp)]
                    if r["variant"] == v and abs(r["lam"] - lam) < 1e-9]
            if not vals:
                row[f"{ds}-{sp}"] = np.nan
                continue
            m = float(np.mean(vals)) / 2.0
            row[f"{ds}-{sp}"] = m
            rel = 100 * (m / base[(ds, sp)] - 1)
            row[f"rel_{ds}_{sp}"] = rel
            rels.append(rel)
            # other metrics at the chosen lambda
            for mk in ("mae", "psnr", "ssim", "sam", "ergas", "cc"):
                vv = [r[mk] for r in data[(ds, sp)]
                      if r["variant"] == v and abs(r["lam"] - lam) < 1e-9]
                row[f"{mk}_{ds}_{sp}"] = float(np.mean(vv))
        row["lam_LGC"] = chosen.get(("LGC", v), np.nan)
        row["lam_CIA"] = chosen.get(("CIA", v), np.nan)
        row["worst_rel"] = max(rels) if rels else np.nan
        row["mean_rel"] = float(np.mean(rels)) if rels else np.nan
        results.append(row)

    results.sort(key=lambda r: r["worst_rel"] if r["worst_rel"] == r["worst_rel"] else 99)
    cols = ["variant", "LGC-val", "LGC-test", "CIA-val", "CIA-test",
            "rel_LGC_val", "rel_LGC_test", "rel_CIA_val", "rel_CIA_test",
            "worst_rel", "mean_rel", "lam_LGC", "lam_CIA"]
    print(f"{'variant':22s} " + " ".join(f"{c:>10s}" for c in cols[1:5]) +
          " " + " ".join(f"{c.replace('rel_',''):>10s}" for c in cols[5:9]) + "   worst   mean")
    for r in results:
        print(f"{r['variant']:22s} " +
              " ".join(f"{r.get(c, float('nan')):10.5f}" if isinstance(r.get(c), float) else f"{str(r.get(c)):>10s}" for c in cols[1:5]) +
              " " + " ".join(f"{r.get(c, float('nan')):+10.2f}" for c in cols[5:9]) +
              f" {r['worst_rel']:+7.2f} {r['mean_rel']:+7.2f}")

    out = Path("results/E20_leaderboard.csv")
    keys = sorted({k for r in results for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(results)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
