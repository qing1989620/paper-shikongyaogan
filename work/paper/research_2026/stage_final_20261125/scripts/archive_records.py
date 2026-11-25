"""Verify and archive the 45 completed screening and replication runs."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path, PureWindowsPath
from zipfile import ZIP_DEFLATED, ZipFile


RUNS_PATH = Path("work/runs/research_2026")
CONFIG_PATH = Path("work/configs/screening_2026")
LOGS_PATH = Path("work/logs/research_2026")
PROBE_PATH = Path("work/deliverables/I10_collaboration_20261124/qa/probe_repacked_smoke")
RECORDS = (
    "COMPLETE.json", "config.json", "console.log", "environment.json",
    "initial_validation.json", "per_patch_validation.csv", "status.json",
    "summary.json", "train_log.csv",
)
SECRET = re.compile(rb"gh[pousr]_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{30,}|"
                    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
                    rb"(?i:authorization:\s*bearer)\s+[0-9A-Za-z._-]{20,}")


def read_json(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, label):
    if not condition:
        raise ValueError(label)


def load_runs(project):
    """Load and independently check the saved paper inputs."""
    queue = read_json(project / CONFIG_PATH / "queue_v1.json")
    methods = sorted(queue["methods"])
    require(len(methods) == 15 and len(set(methods)) == 15, "Unexpected method roster")
    stages = [
        ("screening", "screen", 42, 6),
        ("replication", "replicate", 42, 40),
        ("replication", "replicate", 123, 40),
    ]
    manifest_sha = digest(project / CONFIG_PATH / "manifest_lgc_frozen.csv")
    observations_by_stage = {}
    runs = []
    for phase, prefix, seed, epochs in stages:
        for method in methods:
            name = f"{prefix}_{method}_seed{seed}_v1"
            rel = RUNS_PATH / name
            directory = project / rel
            config = read_json(directory / "config.json")
            marker = read_json(directory / "COMPLETE.json")
            summary = read_json(directory / "summary.json")
            status = read_json(directory / "status.json")
            history = read_csv(directory / "train_log.csv")
            patches = read_csv(directory / "per_patch_validation.csv")
            require((config["method"], config["seed"], config["phase"], config["epochs"])
                    == (method, seed, phase, epochs), f"{name}: configuration")
            require((summary["method"], summary["seed"]) == (method, seed), f"{name}: summary identity")
            require(status["state"] == "complete" and status["epoch"] == epochs, f"{name}: status")
            require(marker["status"] == f"{phase}_complete_not_SOTA", f"{name}: completion marker")
            require(digest(directory / "summary.json") == marker["summary_sha256"], f"{name}: summary hash")
            require(config["manifest_sha256"] == manifest_sha, f"{name}: frozen manifest")
            require([int(row["epoch"]) for row in history] == list(range(1, epochs + 1)),
                    f"{name}: missing epochs")
            for path, expected in config["code_sha256"].items():
                snapshot = directory / "source" / PureWindowsPath(path).name
                require(digest(snapshot) == expected, f"{name}: source hash {path}")
            paired = {(row["key"], row["date"], int(row["repeat"]), int(row["noise_seed"])) for row in patches}
            require(len(paired) == len(patches) == 360, f"{name}: prediction count")
            require(len({row["key"] for row in patches}) == 120, f"{name}: patch count")
            require(paired == observations_by_stage.setdefault((phase, seed), paired), f"{name}: noise pairing")
            require(summary["n_patches"] == 120 and summary["n_predictions"] == len(patches), f"{name}: counts")
            expected_best_epoch = min(history, key=lambda row: float(row["val_rmse"]))
            require(int(expected_best_epoch["epoch"]) == summary["best_epoch"], f"{name}: best epoch")
            require(math.isclose(float(expected_best_epoch["val_rmse"]), summary["mean_patch_rmse"], abs_tol=1e-9),
                    f"{name}: best checkpoint RMSE")
            patch_mean = math.fsum(float(row["rmse"]) for row in patches) / len(patches)
            # The exported patch metric uses NumPy reflectance conversion, while the
            # selected checkpoint summary accumulates torch's patch-level RMSE.
            require(math.isclose(patch_mean, summary["mean_patch_rmse"], abs_tol=1e-9), f"{name}: patch mean")
            runs.append({
                "name": name, "relative": rel, "phase": phase, "seed": seed, "epochs": epochs,
                "method": method, "marker": marker, "config": config, "summary": summary,
                "history": history, "patches": patches,
            })
    return runs


def write_archive(project, out, paths, expected=None):
    expected = expected or {}
    temporary = out.with_suffix(".partial")
    entries = []
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for rel in sorted(set(paths)):
            path = project / rel
            require(path.is_file(), f"Missing archive input {rel}")
            if path.suffix != ".pth":
                require(not SECRET.search(path.read_bytes()), f"Credential-like text in {rel}")
            sha = digest(path)
            if rel in expected:
                require(sha == expected[rel], f"Checkpoint hash {rel}")
            archive.write(path, arcname=rel.as_posix())
            entries.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha})
        archive.writestr("ARCHIVE_MANIFEST.json",
                         json.dumps({"files": entries}, ensure_ascii=False, indent=2) + "\n")
    with ZipFile(temporary) as archive:
        require(archive.testzip() is None, f"{out.name}: ZIP CRC")
        archived = {row.filename: row.file_size for row in archive.infolist()}
    require(len(archived) == len(entries) + 1, f"{out.name}: duplicate members")
    for entry in entries:
        require(archived.get(entry["path"]) == entry["bytes"], f"{out.name}: member size")
    require(temporary.stat().st_size < 2 ** 31, f"{out.name}: GitHub Release size")
    os.replace(temporary, out)
    return {"name": out.name, "bytes": out.stat().st_size, "sha256": digest(out), "members": len(entries)}


def record_paths(project, runs):
    paths = []
    for run in runs:
        paths.extend(run["relative"] / name for name in RECORDS)
        paths.extend(path.relative_to(project) for path in (project / run["relative"] / "source").glob("*.py"))
    paths.extend(CONFIG_PATH / name for name in (
        "queue_v1.json", "followup_v2.json", "manifest_lgc_frozen.csv", "manifest_cia_frozen.csv",
    ))
    paths.extend(RUNS_PATH / name for name in (
        "QUEUE_STATUS.json", "FOLLOWUP_STATUS.json", "MONITOR_STATUS.json",
        "PAUSE_STATUS.json", "SCREENING_DECISION.json", "screening_progress.csv",
    ))
    paths.extend(path.relative_to(project) for path in (project / RUNS_PATH).glob("PAUSE_STATUS_*.json"))
    paths.extend(path.relative_to(project) for path in (project / LOGS_PATH).iterdir()
                 if path.suffix in {".log", ".err", ".jsonl"})
    paths.extend(Path("work/results/research_2026/original_base_control") / name
                 for name in ("summary.json", "per_patch_validation.csv"))
    paths.extend(PROBE_PATH / name for name in ("probe_report.json", "conditions.csv", "paired_scores.csv"))
    paths.extend(Path("docs") / name for name in (
        "DATASET_MANUAL.md", "RESEARCH_LIVE_STATUS.md", "EXPERIMENT_QUEUE.csv",
        "BENCHMARK_PROTOCOL.md", "EXPERIMENT_PLAN.md", "I10_EXPERIMENT_PLAN.md",
        "INNOVATION_PROGRAM_2026.md", "LITERATURE_2026_VERIFIED.md",
    ))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(output).free > 6_000_000_000, "Space for archives")
    runs = load_runs(project)
    pause = read_json(project / RUNS_PATH / "PAUSE_STATUS.json")
    require(pause["state"] == "paused" and pause["target_seed"] == 123 and
            pause["replication_completed"] == 30, "Completed seed 123 pause")
    report = {"runs": len(runs), "paused_cst": pause["paused_cst"], "assets": []}

    def add(name, paths, expected=None):
        print(f"Packaging {name}", flush=True)
        asset = write_archive(project, output / name, paths, expected)
        report["assets"].append(asset)
        print(json.dumps(asset), flush=True)

    add("STF_experiment_records_20261125.zip", record_paths(project, runs))
    groups = [("screening", 42, "screen42"), ("replication", 42, "seed42"), ("replication", 123, "seed123")]
    for phase, seed, label in groups:
        group = [run for run in runs if (run["phase"], run["seed"]) == (phase, seed)]
        paths = [run["relative"] / file for run in group for file in ("best.pth", "resume.pth")]
        expected = {run["relative"] / "best.pth": run["summary"]["checkpoint_sha256"] for run in group}
        if phase == "screening":
            warm = Path("work/runs/base/best.pth")
            paths.append(warm)
            expected[warm] = group[0]["config"]["warmstart_sha256"]
        add(f"STF_checkpoints_{label}_20261125.zip", paths, expected)
    (output / "experiment_archive.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
