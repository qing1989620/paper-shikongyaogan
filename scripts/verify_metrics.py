"""Numerical sanity checks plus a cached-prediction audit, without retraining."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "tools"))
from benchmark_metrics import compute_metrics, from_minus_one_one, METRIC_VERSION


def main():
    rng = np.random.default_rng(20261119)
    gt = rng.uniform(0.2, 0.7, (24, 24, 6))
    identity = compute_metrics(gt, gt, ergas_ratio=1/16)
    assert identity["rmse"] == 0 and identity["ssim"] == 1 and math.isinf(identity["psnr_db"])
    assert identity["sam_rad"] < 2e-8 and abs(identity["cc"]-1) < 1e-12
    pred = gt+0.1
    analytic = compute_metrics(pred, gt, ergas_ratio=1/16)
    assert abs(analytic["rmse"]-0.1) < 1e-12
    assert abs(analytic["mae"]-0.1) < 1e-12
    assert abs(analytic["psnr_db"]-20) < 1e-12
    assert abs(analytic["sam_deg"]-math.degrees(analytic["sam_rad"])) < 1e-12
    expected_ergas = 100/16*np.sqrt(np.mean((0.1/gt.mean(axis=(0,1)))**2))
    assert abs(analytic["ergas"]-expected_ergas) < 1e-12
    scaled = compute_metrics(gt*0.9, gt, ergas_ratio=1/16)
    assert scaled["sam_rad"] < 2e-8
    constant = np.full_like(gt, 0.5)
    known_ergas = compute_metrics(constant*0.9, constant, ergas_ratio=1/16)
    assert abs(known_ergas["ergas"]-0.625) < 1e-12
    assert known_ergas["undefined_cc_bands"] == 6
    expected_ssim = (2*0.5*0.45+0.01**2)/(0.5**2+0.45**2+0.01**2)
    assert abs(known_ergas["ssim"]-expected_ssim) < 1e-10
    mask = np.ones((24,24), dtype=bool)
    mask[:2] = False
    missing = pred.copy()
    missing[:2] = np.nan
    masked = compute_metrics(missing, gt, ergas_ratio=1/16, mask=mask)
    assert abs(masked["rmse"]-0.1) < 1e-12
    assert masked["n_valid_pixels"] == 22*24
    from eval_methods import evaluate
    from lock_numbers import conv
    records = []
    for split in ("val", "test"):
        cache = ROOT / "work" / "results" / f"cache_base_{split}"
        files = sorted(cache.glob("*.npz"))
        for fp in [files[0], files[len(files)//2], files[-1]]:
            with np.load(fp) as z:
                x = z["x1"].transpose(1,2,0).astype(np.float32)
            with np.load(ROOT/"work"/"data"/"patches_lgc"/split/fp.name) as z:
                y = z["f1"].astype(np.float32)
            old = evaluate(x, y)
            new = compute_metrics(from_minus_one_one(x), from_minus_one_one(y), ergas_ratio=1/16)
            assert abs(old["rmse"]/2-new["rmse"]) < 1e-7
            assert abs(old["psnr"]-new["psnr_db"]) < 1e-4
            assert abs(conv(old)["psnr"]-new["psnr_db"]-20*np.log10(2)) < 1e-4
            records.append({"split": split, "patch": fp.stem, "legacy_raw": old,
                            "legacy_locked_psnr": conv(old)["psnr"], "reflectance_v1": new})
    report = {"metric_version": METRIC_VERSION, "checks_passed": True,
              "scope": "analytic examples plus 6 cached LGC base predictions; not a benchmark result",
              "ergas_ratio_for_this_check": 1/16, "real_samples": records}
    out = ROOT/"docs"/"audit"/"metric_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: analytic and six cached-prediction checks. {out}")


if __name__ == "__main__":
    main()
