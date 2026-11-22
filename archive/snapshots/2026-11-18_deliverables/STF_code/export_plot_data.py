# -*- coding: utf-8 -*-
"""Export plotting-ready data from all experiment results.

Outputs work/paper/figures/data/*.json used by the figure scripts:
  rmse_dist.json  : patch-level fixed-1 RMSE per model per split (raincloud)
  gain_matrix.json: 20 inference variants x 4 splits relative RMSE change (bubble)
  tradeoff.json   : per-model val/test delta vs base (diverging bars)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["base", "lf", "hf", "bandw", "delsup", "full", "delta", "gate", "reinject"]
NICE = {"base": "Base", "lf": "+LF loss", "hf": "+HF loss", "bandw": "+Band weights",
        "delsup": "+Delta sup.", "full": "Arch full", "delta": "Arch delta",
        "gate": "Arch gate", "reinject": "Arch reinject"}


def patch_rmse(model, split):
    p = Path(f"results/{model}_lgc_{split}.csv")
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return sorted(float(r["fixed1_rmse"]) for r in rows)


dist = {}
for m in MODELS:
    for sp in ("val", "test"):
        v = patch_rmse(m, sp)
        if v is not None:
            dist[f"{m}|{sp}"] = {"name": NICE[m], "split": sp, "values": [round(x, 6) for x in v]}
(OUT / "rmse_dist.json").write_text(json.dumps(dist), encoding="utf-8")
print("rmse_dist:", len(dist), "groups")


def rel_table(val_csv, test_csv):
    def best(rows):
        cand = {}
        for r in rows:
            lam = float(r["lam"]); cand.setdefault(lam, []).append(float(r["rmse"]))
        lam = min(cand, key=lambda l: np.mean(cand[l]))
        return 100 * (np.mean(cand[lam]) / np.mean(
            [float(r["rmse"]) for r in rows if r["variant"] == "E01_fixed1"]) - 1)
    return best(load(val_csv)), best(load(test_csv))


def load(p):
    return list(csv.DictReader(open(p)))


# 20 variants on the released checkpoint across LGC val/test and CIA val/test
files = {"LGC-val": "results/E20_lgc_val.csv", "LGC-test": "results/E20_lgc_test.csv",
         "CIA-val": "results/E20_cia_val.csv", "CIA-test": "results/E20_cia_test.csv"}
d = {k: load(v) for k, v in files.items() if Path(v).exists()}
variants = sorted({r["variant"] for r in d["LGC-val"]})
VNICE = {"E02_identity": "Identity", "E03_ols_prev": "OLS", "E05_tls": "TLS",
         "E06_deming": "Deming", "E07_huber": "Huber", "E08_ridge": "Ridge",
         "E09_wls": "WLS", "E11_invvar_bi": "Inv-var", "E12_hist_match": "Hist match",
         "E13_ot_affine": "OT affine", "E14_conformal_gate": "Gate", "E15_haar": "Haar",
         "E16_guided": "Guided", "E17_cascade": "Cascade", "E19_pixel_adaptive": "Pixel-adaptive"}
gain = {}
for v in variants:
    if v not in VNICE:
        continue
    gain[VNICE[v]] = {}
    for k, rows in d.items():
        cand = {}
        for r in rows:
            if r["variant"] == v:
                cand.setdefault(float(r["lam"]), []).append(float(r["rmse"]))
        if not cand:
            continue
        lam = min(cand, key=lambda l: np.mean(cand[l]))
        base = np.mean([float(r["rmse"]) for r in rows if r["variant"] == "E01_fixed1"])
        gain[VNICE[v]][k] = round(100 * (np.mean(cand[lam]) / base - 1), 2)
(OUT / "gain_matrix.json").write_text(json.dumps(gain, indent=1), encoding="utf-8")
print("gain_matrix:", len(gain), "variants")


# per-model val/test delta vs base (diverging bars)
trade = {}
bv = np.mean([float(r["fixed1_rmse"]) for r in load("results/base_lgc_val.csv")])
bt = np.mean([float(r["fixed1_rmse"]) for r in load("results/base_lgc_test.csv")])
for m in MODELS:
    try:
        vv = np.mean([float(r["fixed1_rmse"]) for r in load(f"results/{m}_lgc_val.csv")])
        tv = np.mean([float(r["fixed1_rmse"]) for r in load(f"results/{m}_lgc_test.csv")])
        trade[NICE[m]] = {"val": round(100 * (vv / bv - 1), 2), "test": round(100 * (tv / bt - 1), 2)}
    except FileNotFoundError:
        pass
(OUT / "tradeoff.json").write_text(json.dumps(trade, indent=1), encoding="utf-8")
print("tradeoff:", len(trade), "models")
