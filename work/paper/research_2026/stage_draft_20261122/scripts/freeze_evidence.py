"""Extract and audit the paused LGC results. This script runs no model."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


METRICS = ("rmse", "ssim", "sam_rad", "ergas", "cc", "mae", "psnr_db")
EXPECTED_METHODS = (
    "base", "i01_sensor", "i02_bridge", "i03_frequency",
    "i05_spectral", "i06_registration", "i07_covariance", "i13_disentangle",
)
PATCH_ID = re.compile(r"val_r(\d+)_c(\d+)$")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_copy(path, dest, project, evidence, sources):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    sha = digest(path)
    assert sha == digest(dest)
    sources.append({
        "source": path.relative_to(project).as_posix(),
        "snapshot": dest.relative_to(evidence).as_posix(),
        "sha256": sha,
    })


def paired_patch_metrics(samples, reference):
    by_method = {}
    for method, rows in samples.items():
        by_method[method] = {
            (r["key"], int(r["repeat"]), int(r["noise_seed"])): float(r["rmse"])
            for r in rows
        }
        assert len(by_method[method]) == len(rows) == 360, method
    base = by_method["base"]
    keys = sorted({k[0] for k in base})
    assert len(keys) == 120
    assert all(n == 3 for n in Counter(k[0] for k in base).values())
    assert {r["date"] for rows in samples.values() for r in rows} == {"2005-03-02"}
    assert all(v.keys() == base.keys() for v in by_method.values())
    idx = np.array([[i for i, k in enumerate(sorted(base)) if k[0] == key] for key in keys])
    values = {
        m: np.array([scores[k] for k in sorted(base)])[idx].mean(axis=1)
        for m, scores in by_method.items()
    }
    for method, score in values.items():
        # Serialized reflectance metrics use double precision, while the
        # training summary accumulates the float32 tensor RMSE.
        assert math.isclose(float(score.mean()), reference[method]["best_rmse"], abs_tol=5e-9)
    return keys, values


def coarse_change(project, val_rows, keys):
    """Mean absolute coarse deviation from the date-weighted anchor blend."""
    values = {}
    import datetime

    for row in val_rows:
        prev, target, following = (
            datetime.date.fromisoformat(row[k])
            for k in ("prev_date", "target_date", "next_date")
        )
        weight = (target - prev).days / (following - prev).days
        patch_path = project / row["path"]
        assert digest(patch_path) == row["sha256"], row["patch_id"]
        with np.load(patch_path, allow_pickle=False) as patch:
            previous, coarse, later = (patch[k].astype(np.float32) for k in ("c0", "c1", "c3"))
            values[f'LGC/val/{row["patch_id"]}'] = float(
                np.abs(coarse - ((1 - weight) * previous + weight * later)).mean() / 2
            )
    assert set(values) == set(keys)
    return values


def describe_pairs(keys, values, change):
    coords = [tuple(map(int, PATCH_ID.search(k).groups())) for k in keys]
    grid = sorted({r for r, _ in coords}), sorted({c for _, c in coords})
    assert len(grid[0]) == 10 and len(grid[1]) == 12
    assert all(k % 256 == 0 for pair in coords for k in pair)
    blocks = {}
    for i, (r, c) in enumerate(coords):
        blocks.setdefault((r // 512, c // 512), []).append(i)
    assert len(blocks) == 30 and all(len(v) == 4 for v in blocks.values())
    indices = np.array([blocks[k] for k in sorted(blocks)])
    draws = np.random.default_rng(20261122).integers(0, 30, size=(10000, 30))
    ranked = sorted(range(len(keys)), key=lambda i: (change[keys[i]], keys[i]))
    strata = {name: ranked[i * 40:(i + 1) * 40] for i, name in enumerate(("low", "mid", "high"))}
    gain = []
    stratified = []
    tile_rows = []
    for method, candidate in values.items():
        if method == "base":
            continue
        delta = values["base"] - candidate
        bootstrap = delta[indices].mean(axis=1)[draws].mean(axis=1)
        lo, hi = np.quantile(bootstrap, (0.025, 0.975))
        gain.append({
            "method": method, "mean_gain": float(delta.mean()), "lo95": float(lo),
            "hi95": float(hi), "patches_improved": int((delta > 0).sum()),
            "n_patches": 120, "n_blocks": 30, "bootstrap_draws": 10000,
        })
        for label, members in strata.items():
            v = delta[members]
            b = values["base"][members]
            stratified.append({
                "method": method, "stratum": label, "n_patches": len(members),
                "base_rmse": float(b.mean()), "variant_rmse": float(candidate[members].mean()),
                "mean_gain": float(v.mean()), "gain_pct": float(100 * v.mean() / b.mean()),
                "patches_improved": int((v > 0).sum()),
            })
            for i in members:
                tile_rows.append({
                    "method": method, "patch": keys[i], "stratum": label, "coarse_deviation": change[keys[i]],
                    "row": coords[i][0], "col": coords[i][1], "block_row": coords[i][0] // 512,
                    "block_col": coords[i][1] // 512, "base_rmse": float(values["base"][i]),
                    "variant_rmse": float(candidate[i]), "gain": float(delta[i]),
                })
    return gain, stratified, tile_rows


def latex_values(scores, gains, strata, source):
    i02, i01, b = (scores[k] for k in ("i02_bridge", "i01_sensor", "base"))
    gain = {r["method"]: r for r in gains}
    bridge_strata = {r["stratum"]: r for r in strata if r["method"] == "i02_bridge"}
    macros = {
        "BaseBest": f'{b["best_rmse"]:.6f}', "BaseEnd": f'{b["epoch40_rmse"]:.6f}',
        "BridgeBest": f'{i02["best_rmse"]:.6f}', "BridgeEnd": f'{i02["epoch40_rmse"]:.6f}',
        "BridgeBestGain": f'{i02["best_gain_pct"]:.2f}', "BridgeEndHarm": f'{-i02["epoch40_gain_pct"]:.2f}',
        "SensorBestGain": f'{i01["best_gain_pct"]:.2f}', "SensorEndHarm": f'{-i01["epoch40_gain_pct"]:.2f}',
        "BridgeSSIM": f'{i02["ssim"]:.5f}', "BaseSSIM": f'{b["ssim"]:.5f}',
        "BridgePatchWins": str(gain["i02_bridge"]["patches_improved"]),
        "BridgeDiff": f'{gain["i02_bridge"]["mean_gain"]*1000:.3f}',
        "BridgeLower": f'{gain["i02_bridge"]["lo95"]*1000:.3f}',
        "BridgeUpper": f'{gain["i02_bridge"]["hi95"]*1000:.3f}',
        "BridgeLow": f'{bridge_strata["low"]["gain_pct"]:+.2f}',
        "BridgeMid": f'{bridge_strata["mid"]["gain_pct"]:+.2f}',
        "BridgeHigh": f'{bridge_strata["high"]["gain_pct"]:+.2f}',
        "BridgeExtraParams": f'{i02["parameters"]-b["parameters"]:,}',
    }
    (source / "values.tex").write_text(
        "% Verified values generated from the paused snapshots.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n",
        encoding="utf-8",
    )
    rows = []
    order = ("base", "i02_bridge", "i01_sensor", "i06_registration", "i03_frequency", "i05_spectral", "i07_covariance", "i13_disentangle")
    for method in order:
        r = scores[method]
        label = "Base" if method == "base" else method.split("_", 1)[0].upper()
        gain_pc = f'{r["best_gain_pct"]:+.2f}' if method != "base" else r"\textemdash"
        rows.append(
            f'{label} & {r["epoch40_rmse"]:.6f} & {r["best_epoch"]} & {r["best_rmse"]:.6f} & {gain_pc} & '
            f'{r["ssim"]:.5f} & {r["sam_rad"]:.5f} & {r["ergas"]:.4f} & {r["cc"]:.5f} \\\\'
        )
    (source / "result_rows.tex").write_text(
        "\\newcommand{\\StageResultRows}{%\n" + "\n".join(rows) + "\n}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--project", type=Path, help="Original workspace containing dataset and checkpoints")
    origin.add_argument("--archived", action="store_true", help="Recompute from frozen metrics and coarse deviations")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    out = args.output.resolve()
    evidence, source = out / "evidence", out / "source"
    for directory in (evidence, source):
        directory.mkdir(parents=True, exist_ok=True)
    archive = evidence / "frozen_summary.json"
    manifest = evidence / "manifest_lgc.csv"
    coarse_file = evidence / "coarse_deviation.csv"
    pause_file = evidence / "pause.json"
    sources = []
    if args.archived:
        stored = read_json(archive)
        sources = stored["verified_source_files"]
        for entry in sources:
            assert digest(evidence / entry["snapshot"]) == entry["sha256"], entry["snapshot"]
        assert digest(coarse_file) == stored["coarse_deviation_sha256"]
    else:
        project = (args.project or Path(__file__).resolve().parents[5]).resolve()
        runs = project / "work/runs/research_2026"
        record_copy(runs / "PAUSE_STATUS.json", pause_file, project, evidence, sources)
        record_copy(runs / "SCREENING_DECISION.json", evidence / "screening_decision.json", project, evidence, sources)
        manifest_path = project / "work/configs/screening_2026/manifest_lgc_frozen.csv"
        record_copy(manifest_path, manifest, project, evidence, sources)

    pause = read_json(pause_file)
    assert pause["state"] == "paused" and len(pause["completed_runs"]) == 8
    manifest_rows = read_csv(manifest)
    val_rows = [r for r in manifest_rows if r["split"] == "val"]
    train_rows = [r for r in manifest_rows if r["split"] == "train"]
    assert len(val_rows) == 120 and len(train_rows) == 1200
    assert len({r["group"] for r in train_rows}) == 10
    dates = sorted({(r["prev_date"], r["target_date"], r["next_date"]) for r in val_rows})
    assert len(dates) == 1
    for row in train_rows:
        assert dates[0][1] not in (row["prev_date"], row["target_date"], row["next_date"])

    scores, curves, samples = {}, [], {}
    for name in pause["completed_runs"]:
        run = evidence / "runs" / name
        if not args.archived:
            live = runs / name
            files = ("COMPLETE.json", "summary.json", "config.json", "train_log.csv", "per_patch_validation.csv")
            for file in files:
                record_copy(live / file, run / file, project, evidence, sources)
        cfg, summary = (read_json(run / file) for file in ("config.json", "summary.json"))
        assert digest(run / "summary.json") == read_json(run / "COMPLETE.json")["summary_sha256"]
        if not args.archived:
            assert digest(live / "best.pth") == summary["checkpoint_sha256"]
            for path, sha in cfg["code_sha256"].items():
                assert digest(project / path) == sha
        assert (cfg["phase"], cfg["seed"], cfg["epochs"], cfg["warmstart"]) == ("replication", 42, 40, "")
        assert (cfg["train_patches"], cfg["val_patches"], cfg["validation_repeats"]) == (1200, 120, 3)
        assert (cfg["lr"], cfg["accum"], cfg["max_updates_per_epoch"]) == (0.0002, 2, 600)
        assert cfg["manifest_sha256"] == digest(manifest)
        assert summary["sampler"] == {"steps": 1, "eta": 0.0}
        method = summary["method"]
        assert cfg["method"] == method
        log = read_csv(run/"train_log.csv")
        assert [int(r["epoch"]) for r in log] == list(range(1, 41))
        assert [int(r["optimizer_step"]) for r in log] == [600*e for e in range(1, 41)]
        best = min(log, key=lambda r: float(r["val_rmse"]))
        assert int(best["epoch"]) == summary["best_epoch"]
        assert math.isclose(float(best["val_rmse"]), summary["mean_patch_rmse"], abs_tol=3e-6)
        samples[method] = read_csv(run/"per_patch_validation.csv")
        scores[method] = {
            "method": method, "seed": cfg["seed"], "parameters": cfg["parameters"],
            "best_epoch": summary["best_epoch"], "epoch40_rmse": float(log[-1]["val_rmse"]),
            "best_rmse": summary["mean_patch_rmse"], **{k: summary[k] for k in METRICS[1:]},
        }
        curves.extend({"method": method, "epoch": r["epoch"], "rmse": r["val_rmse"]} for r in log)
    assert set(scores) == set(EXPECTED_METHODS)
    base = scores["base"]
    for record in scores.values():
        record["best_gain_pct"] = 100 * (1 - record["best_rmse"] / base["best_rmse"])
        record["epoch40_gain_pct"] = 100 * (1 - record["epoch40_rmse"] / base["epoch40_rmse"])
    keys, values = paired_patch_metrics(samples, scores)
    if args.archived:
        change = {r["patch"]: float(r["coarse_deviation"]) for r in read_csv(coarse_file)}
        assert set(change) == set(keys)
    else:
        change = coarse_change(project, val_rows, keys)
        write_csv(coarse_file, [{"patch": key, "coarse_deviation": change[key]} for key in keys])
    gains, strata, tiles = describe_pairs(keys, values, change)
    for filename, rows in (
        ("scores.csv", list(scores.values())), ("curves.csv", curves), ("paired_gain.csv", gains),
        ("coarse_strata.csv", strata), ("patch_gain.csv", tiles),
    ):
        write_csv(evidence / filename, rows)
    latex_values(scores, gains, strata, source)
    summary = {
        "paused_cst": pause["paused_cst"], "val_triple": dates[0], "train_patches": 1200,
        "val_patches": 120, "train_seed": 42, "inference_repeats": 3,
        "scores": list(scores.values()), "paired_gain": gains,
        "coarse_strata": strata, "coarse_deviation_sha256": digest(coarse_file),
        "verified_source_files": sources,
    }
    archive.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print("Audited eight completed runs, 40 epochs each, 120 matched validation patches and their three noise realizations.")
    print(dates[0], scores["i02_bridge"], next(r for r in gains if r["method"] == "i02_bridge"))
    print([r for r in strata if r["method"] == "i02_bridge"])


if __name__ == "__main__":
    main()
