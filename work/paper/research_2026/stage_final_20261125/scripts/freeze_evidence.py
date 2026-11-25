"""Recompute manuscript tables and figure inputs from the archived records."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from archive_records import PROBE_PATH, load_runs, read_csv, read_json, require


PACKAGE = Path(__file__).resolve().parents[1]
SEEDS = (42, 123)
MAIN = "i10_unmixing"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(method):
    return "Base" if method == "base" else method[:3].upper()


def paired(values, strata_path):
    keys = sorted(set(row["key"] for row in values[42, "base"]["patches"]))
    positions = [tuple(map(int, re.search(r"_r(\d+)_c(\d+)$", key).groups())) for key in keys]
    blocks = {}
    for i, (row, col) in enumerate(positions):
        blocks.setdefault((row // 512, col // 512), []).append(i)
    require(len(blocks) == 30 and all(len(members) == 4 for members in blocks.values()), "Spatial blocks")
    indices = np.asarray([blocks[key] for key in sorted(blocks)])
    draws = np.random.default_rng(20261122).integers(0, 30, size=(10000, 30))
    change = {row["patch"]: float(row["coarse_deviation"]) for row in read_csv(strata_path)}
    require(set(keys) == set(change), "Frozen coarse strata keys")
    strata = np.array_split(sorted(range(120), key=lambda i: (change[keys[i]], keys[i])), 3)
    intervals, stratified, tile_rows, entropy = [], [], [], []
    for seed in SEEDS:
        by_method = {}
        for (run_seed, method), run in values.items():
            if run_seed != seed:
                continue
            samples = {}
            for row in run["patches"]:
                samples.setdefault(row["key"], []).append(float(row["rmse"]))
            require(set(samples) == set(keys), f"{seed}: {method}: patch alignment")
            by_method[method] = np.array([np.mean(samples[key]) for key in keys])
            if method == MAIN:
                entropy_by_tile = {}
                for row in run["patches"]:
                    entropy_by_tile.setdefault(row["key"], set()).add(float(row["diagnostic_abundance_entropy"]))
                require(all(len(v) == 1 for v in entropy_by_tile.values()), f"{seed}: entropy differs by noise")
                entropy.extend({"seed": seed, "patch": key, "entropy_nat": next(iter(entropy_by_tile[key]))}
                               for key in keys)
        base = by_method["base"]
        for method, candidate in by_method.items():
            if method == "base":
                continue
            delta = base - candidate
            bootstrap = delta[indices].mean(axis=1)[draws].mean(axis=1)
            lo, hi = np.quantile(bootstrap, (0.025, 0.975))
            intervals.append({"seed": seed, "method": method, "mean_gain": float(delta.mean()),
                              "lo95": float(lo), "hi95": float(hi), "patches_improved": int((delta > 0).sum()),
                              "n_patches": 120, "n_blocks": 30, "bootstrap_draws": 10000})
            for stratum, members in zip(("low", "mid", "high"), strata):
                stratified.append({"seed": seed, "method": method, "stratum": stratum, "n_patches": len(members),
                                   "gain_pct": float(100 * delta[members].mean() / base[members].mean()),
                                   "patches_improved": int((delta[members] > 0).sum())})
                for i in members:
                    row, col = positions[i]
                    tile_rows.append({"seed": seed, "method": method, "patch": keys[i], "row": row, "col": col,
                                      "stratum": stratum, "coarse_deviation": change[keys[i]],
                                      "base_rmse": float(base[i]), "variant_rmse": float(candidate[i]),
                                      "gain": float(delta[i])})
    return intervals, stratified, tile_rows, entropy


def tex_rows(macro, rows):
    body = "\n".join(" & ".join(map(str, row)) + r" \\%" for row in rows)
    return fr"\newcommand{{\{macro}}}{{%" + "\n" + body + "\n}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True,
                        help="Project root or the extracted experiment-records ZIP")
    args = parser.parse_args()
    root = args.records_root.resolve()
    evidence = PACKAGE / "evidence"
    source = PACKAGE / "source"
    runs = load_runs(root)
    controls = {(run["phase"], run["seed"]): run for run in runs if run["method"] == "base"}
    values = {(run["seed"], run["method"]): run for run in runs if run["phase"] == "replication"}
    orig = read_json(root / "work/results/research_2026/original_base_control/summary.json")
    screening_reference = orig["mean_patch_rmse"]
    scores, curves = [], []
    for run in runs:
        s = run["summary"]
        base = controls[(run["phase"], run["seed"])]
        best_ref = (screening_reference if run["phase"] == "screening"
                    else base["summary"]["mean_patch_rmse"])
        end = float(run["history"][-1]["val_rmse"])
        final_ref = float(base["history"][-1]["val_rmse"])
        scores.append({"phase": run["phase"], "seed": run["seed"], "method": run["method"],
                       "run_id": run["name"], "best_epoch": s["best_epoch"], "best_rmse": s["mean_patch_rmse"],
                       "best_reference": best_ref, "best_gain_pct": 100 * (1 - s["mean_patch_rmse"] / best_ref),
                       "final_epoch": run["epochs"], "final_rmse": end,
                       "final_paired_base_rmse": final_ref, "final_gain_pct": 100 * (1 - end / final_ref),
                       "ssim": s["ssim"], "sam_rad": s["sam_rad"], "ergas": s["ergas"], "cc": s["cc"],
                       "parameters": run["config"]["parameters"], "peak_cuda_MiB": s["peak_cuda_MiB"],
                       "summary_sha256": run["marker"]["summary_sha256"], "best_sha256": s["checkpoint_sha256"]})
        curves.extend({"phase": run["phase"], "seed": run["seed"], "method": run["method"], **row}
                      for row in run["history"])
    write_csv(evidence / "scores.csv", scores)
    write_csv(evidence / "curves.csv", curves)
    rep = {(row["seed"], row["method"]): row for row in scores if row["phase"] == "replication"}
    methods = sorted(method for seed, method in rep if seed == 42)
    best_both = [method for method in methods if all(rep[seed, method]["best_gain_pct"] > 0 for seed in SEEDS)]
    final_both = [method for method in methods if all(rep[seed, method]["final_gain_pct"] > 0 for seed in SEEDS)]
    both = sorted(set(best_both) & set(final_both))
    intervals, stratified, tiles, entropy = paired(values, evidence / "coarse_deviation.csv")
    for name, records in (("pair_intervals", intervals), ("coarse_strata", stratified),
                          ("patch_pairs", tiles), ("i10_entropy", entropy)):
        write_csv(evidence / (name + ".csv"), records)
    probe = read_json(root / PROBE_PATH / "probe_report.json")
    require(probe["n_patches"] == 2 and probe["checkpoint_seed"] == 42 and probe["noise_repeats"] == 1,
            "Probe record identity")
    (evidence / "i10_probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"n_screening": 15, "n_replication": 30, "screening_reference_rmse": screening_reference,
               "screening_positive": sum(row["phase"] == "screening" and row["method"] != "base"
                                         and row["best_gain_pct"] > 0 for row in scores),
               "best_both_methods": best_both, "epoch40_both_methods": final_both, "all_four_methods": both,
               "i10_pair_intervals": [row for row in intervals if row["method"] == MAIN],
               "i10_entropy_median_nat": {seed: float(np.median([row["entropy_nat"] for row in entropy
                                                               if row["seed"] == seed])) for seed in SEEDS}}
    (evidence / "stage_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    source.joinpath("primary_rows.tex").write_text(tex_rows("PrimaryRows", [
        (r"\textbf{I10}" if method == MAIN else label(method),
         *[f"{rep[seed, method][metric]:.6f}" for seed in SEEDS for metric in ("best_rmse", "final_rmse")])
        for method in methods]), encoding="utf-8")
    source.joinpath("secondary_rows.tex").write_text(tex_rows("SecondaryRows", [
        (seed, label(method), rep[seed, method]["best_epoch"],
         *[f"{rep[seed, method][metric]:.5f}" for metric in ("ssim", "sam_rad", "ergas", "cc")])
        for seed in SEEDS for method in ("base", MAIN)]), encoding="utf-8")
    probe_rows = {row["mode"]: row["rmse"] for row in probe["results"]}
    macros = {}
    for suffix, seed in zip(("A", "B"), SEEDS):
        for prefix, method in (("Base", "base"), ("Main", MAIN)):
            for output, metric in (("Best", "best_rmse"), ("Final", "final_rmse")):
                macros[prefix + output + suffix] = f"{rep[seed, method][metric]:.6f}"
        for output, metric in (("BestGain", "best_gain_pct"), ("FinalGain", "final_gain_pct")):
            macros[output + suffix] = f"{rep[seed, MAIN][metric]:.2f}"
        interval = next(row for row in summary["i10_pair_intervals"] if row["seed"] == seed)
        for output, metric in (("MeanPair", "mean_gain"), ("LowPair", "lo95"), ("HighPair", "hi95")):
            macros[output + suffix] = f"{1000 * interval[metric]:.3f}"
        macros["BetterPatches" + suffix] = str(interval["patches_improved"])
        macros["MedianEntropy" + suffix] = f"{1e4 * summary['i10_entropy_median_nat'][seed]:.3f}"
    macros["ScreenReference"] = f"{screening_reference:.6f}"
    macros["BaseParameters"] = f"{rep[42, 'base']['parameters']:,}"
    macros["ExtraParameters"] = f"{rep[42, MAIN]['parameters'] - rep[42, 'base']['parameters']:,}"
    for key, name in (("ProbeBase", "base"), ("ProbeFull", "i10_full"), ("ProbeZero", "i10_no_coarse_delta"),
                      ("ProbeShuffle", "i10_shuffle_coarse_delta"), ("ProbeNoInjection", "i10_no_injection")):
        macros[key] = f"{probe_rows[name]:.8f}"
    source.joinpath("values.tex").write_text(
        "".join(fr"\newcommand{{\{name}}}{{{value}}}" + "\n" for name, value in macros.items()), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
