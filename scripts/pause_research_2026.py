"""Pause this experiment at a scheduled, durable epoch boundary."""

import argparse
import csv
import ctypes
import hashlib
import json
import msvcrt
import os
import re
import subprocess
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research_runtime import write_json


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "work/runs/research_2026"
LOGS = ROOT / "work/logs/research_2026"
PAUSE = RUNS / "PAUSE_STATUS.json"
CHINA = timezone(timedelta(hours=8))
EPOCH_LINE = re.compile(r"^EPOCH (\d+)/(\d+) ", re.MULTILINE)
OUT_ARG = re.compile(r'(?<!\S)--out\s+(?:"([^"]+)"|(\S+))')

PROCESS_QUERY = r"""
$ErrorActionPreference = 'Stop'
$scripts = [ordered]@{
    'scripts/supervise_research_2026.py' = 0
    'scripts/run_research_queue.py' = 1
    'scripts/continue_research_2026.py' = 1
    'work/tools/train_research_2026.py' = 2
    'scripts/evaluate_common_flood.py' = 3
}
$rows = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='pythonw.exe'" |
    ForEach-Object {
        $candidate = $_
        foreach ($relative in $scripts.Keys) {
            $absolute = [IO.Path]::GetFullPath((Join-Path $env:STF_PAUSE_ROOT $relative))
            if ($candidate.CommandLine -match ([regex]::Escape($absolute) + '(?:"|\s|$)')) {
                [pscustomobject]@{
                    pid = $candidate.ProcessId
                    kind = $scripts[$relative]
                    started_ticks = $candidate.CreationDate.ToUniversalTime().Ticks
                    started_utc = $candidate.CreationDate.ToUniversalTime().ToString('o')
                    command = $candidate.CommandLine
                }
                break
            }
        }
    })
ConvertTo-Json -InputObject $rows -Compress
"""

STOP_PROCESSES = """
$ErrorActionPreference = 'Stop'
$expected = ConvertFrom-Json -InputObject $env:STF_PAUSE_PIDS
foreach ($row in $expected) {
    $proc = Get-Process -Id $row.pid -ErrorAction SilentlyContinue
    if (!$proc) { continue }
    try {
        $actual = $proc.StartTime.ToUniversalTime().Ticks
        if ([math]::Abs($actual - [long]$row.started_ticks) -gt 10000000) {
            throw ('Process identity changed for PID ' + $row.pid)
        }
        $proc.Kill()
        if (!$proc.WaitForExit(5000)) {
            throw ('Process did not stop for PID ' + $row.pid)
        }
    } catch {
        if (Get-Process -Id $row.pid -ErrorAction SilentlyContinue) { throw }
    }
}
exit 0
"""


def now():
    return datetime.now(CHINA)


def powershell(command, **env):
    command = "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n" + command
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        env=dict(os.environ, STF_PAUSE_ROOT=str(ROOT), **env),
        creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True,
        encoding="utf-8", errors="replace", timeout=45,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Windows process command exited {result.returncode}: {detail}")
    return result.stdout


def project_processes():
    return json.loads(powershell(PROCESS_QUERY))


def active_run(processes):
    runs = set()
    for process in processes:
        if process["kind"] != 2:
            continue
        match = OUT_ARG.search(process["command"])
        if not match:
            raise RuntimeError("Could not identify the training worker's output directory")
        run = Path(match.group(1) or match.group(2)).resolve()
        if run.parent != RUNS.resolve() or not run.name.startswith(("screen_", "replicate_")):
            raise RuntimeError(f"Unexpected training output directory: {run}")
        runs.add(run)
    if len(runs) > 1:
        raise RuntimeError("More than one distinct training run is active")
    return next(iter(runs), None)


def epoch_boundary(run):
    """The trainer prints EPOCH after replacing the checkpoint."""
    if run is None:
        return None
    try:
        lines = (run / "console.log").read_text(encoding="utf-8", errors="replace")
        epochs = EPOCH_LINE.findall(lines)
        if not epochs:
            return None
        epoch, total = map(int, epochs[-1])
        with (run / "train_log.csv").open(encoding="utf-8", newline="") as stream:
            last = list(csv.DictReader(stream))[-1]
        if int(last["epoch"]) != epoch:
            return None
        checkpoint = run / "resume.pth"
        written = datetime.fromtimestamp(checkpoint.stat().st_mtime, CHINA)
        logged_at = datetime.fromisoformat(last["time_cst"])
        if written < logged_at - timedelta(seconds=1):
            return None
        return {
            "epoch": epoch, "total_epochs": total,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_cst": written.isoformat(timespec="seconds"),
            "last_val_rmse": float(last["val_rmse"]),
        }
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        return None


def stop_processes(processes):
    # Stop launchers before workers so no new job can start during the pause.
    for _ in range(3):
        if not processes:
            return
        ordered = sorted(processes, key=lambda p: (p["kind"], -p["started_ticks"]))
        identities = [{key: p[key] for key in ("pid", "started_ticks")} for p in ordered]
        powershell(STOP_PROCESSES, STF_PAUSE_PIDS=json.dumps(identities))
        processes = project_processes()
    if processes:
        raise RuntimeError("Some verified experiment processes are still running")


def wait_for_checkpoint(target):
    deadline = time.monotonic() + 600
    while True:
        processes = project_processes()
        run = active_run(processes)
        boundary = epoch_boundary(run)
        if not run or (run / "COMPLETE.json").exists():
            return processes, run, boundary, "between_runs"
        started = [
            datetime.fromisoformat(p["started_utc"])
            for p in processes if p["kind"] == 2
        ]
        if min(started) >= target and not (run / "resume.pth").exists():
            return processes, run, None, "new_run_has_no_checkpoint"
        if boundary and datetime.fromisoformat(boundary["checkpoint_cst"]) >= target:
            if boundary["epoch"] < boundary["total_epochs"]:
                return processes, run, boundary, "completed_epoch"
            # At the final epoch, let the worker finish exporting its full metrics.
        if time.monotonic() >= deadline:
            return processes, run, boundary, "checkpoint_wait_limit"
        time.sleep(3)


def verify_finished_run(run):
    marker = read_json(run / "COMPLETE.json")
    summary = (run / "summary.json").read_bytes()
    if hashlib.sha256(summary).hexdigest() != marker["summary_sha256"]:
        raise RuntimeError(f"Completed summary has a different hash: {run.name}")


def finish_current_run(run, status):
    follow_path = RUNS / "FOLLOWUP_STATUS.json"
    # On Windows this read handle delays replacement of the controller status.
    # The controller records completion there before it can launch another job.
    with follow_path.open("rb") as launch_gate:
        current = json.load(launch_gate).get("current", {})
        if current.get("run_id") != run.name:
            raise RuntimeError("The current replication run changed before the pause was armed")
        if not current.get("pid"):
            raise RuntimeError("The active replication worker is not registered")
        if not (run / "COMPLETE.json").exists() and active_run(project_processes()) != run:
            raise RuntimeError("The selected replication worker is not active")
        status.update(state="waiting_for_run_completion", run_id=run.name)
        write_json(PAUSE, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        next_process_check = time.monotonic() + 10
        while not (run / "COMPLETE.json").exists():
            if time.monotonic() >= next_process_check:
                processes = project_processes()
                if active_run(processes) != run:
                    if (run / "COMPLETE.json").exists():
                        break
                    stop_processes(processes)
                    raise RuntimeError("The selected worker exited before it wrote a completion marker")
                next_process_check = time.monotonic() + 10
            time.sleep(0.5)
        status["state"] = "pausing"
        write_json(PAUSE, status)
        stop_processes(project_processes())
    verify_finished_run(run)
    record_pause(status, run, epoch_boundary(run), "completed_requested_run")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_pause(status, run, boundary, reason):
    follow_path = RUNS / "FOLLOWUP_STATUS.json"
    monitor_path = RUNS / "MONITOR_STATUS.json"
    timestamp = now().isoformat(timespec="seconds")
    resume_path = run / "resume.pth" if run else None
    if resume_path and resume_path.exists():
        with zipfile.ZipFile(resume_path) as archive:
            bad_entry = archive.testzip()
        if bad_entry is not None:
            raise RuntimeError(f"Checkpoint integrity check failed: {bad_entry}")
        if reason == "checkpoint_wait_limit":
            boundary = epoch_boundary(run)
        if boundary is None:
            # A valid saved file may precede a log interruption. Read its own epoch.
            import torch
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
            boundary = {
                "epoch": int(checkpoint["epoch"]),
                "checkpoint": str(resume_path.relative_to(ROOT)),
                "checkpoint_cst": datetime.fromtimestamp(resume_path.stat().st_mtime, CHINA).isoformat(timespec="seconds"),
            }

    completed = sorted(p.parent.name for p in RUNS.glob("replicate_*/COMPLETE.json"))
    status.update(
        state="paused", paused_cst=timestamp, run_id=run.name if run else None,
        boundary=boundary, reason=reason, replication_completed=len(completed),
        completed_runs=completed,
    )
    write_json(PAUSE, status)
    if follow_path.exists():
        follow = read_json(follow_path)
        follow.update(state="paused", pid=None)
        recorded = {entry.get("run_id") for entry in follow.get("completed_jobs", [])}
        if run and run.name in completed and run.name not in recorded:
            summary = read_json(run / "summary.json")
            follow.setdefault("completed_jobs", []).append({
                "method": summary["method"], "seed": summary["seed"],
                "run_id": run.name, "state": "complete",
            })
        if follow.get("current"):
            current = follow["current"]
            current.update(state="complete" if current["run_id"] in completed else "paused", pid=None)
            if boundary and run and current["run_id"] == run.name:
                current.update(saved_epoch=boundary["epoch"])
        write_json(follow_path, follow)
    if monitor_path.exists():
        monitor = read_json(monitor_path)
        monitor.update(
            state="paused", stage_state="paused", checked_cst=timestamp,
            pid=None, controller_pid=None, next_hourly_check_cst=None, next_health_check_cst=None,
            system_awake_requested=False, replication_completed=len(completed),
            run_id=run.name if run else None,
        )
        if boundary:
            monitor["checkpoint_cst"] = boundary["checkpoint_cst"]
        write_json(monitor_path, monitor)
        with (LOGS / "hourly_health.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(monitor, ensure_ascii=False) + "\n")
    from update_research_report import main as refresh_report
    refresh_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", type=datetime.fromisoformat)
    parser.add_argument("--finish-run")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        processes = project_processes()
        run = active_run(processes)
        print(json.dumps({
            "project_pids": [p["pid"] for p in processes],
            "run": run.name if run else None,
            "boundary": epoch_boundary(run),
        }, ensure_ascii=False))
        return
    if args.finish_run:
        if args.at or not re.fullmatch(r"replicate_[a-z0-9_]+_seed\d+_v1", args.finish_run):
            parser.error("--finish-run requires the current replication run ID and no --at")
        run = (RUNS / args.finish_run).resolve()
        if run.parent != RUNS.resolve() or not run.is_dir():
            parser.error("The selected run directory does not exist in the replication workspace")
    elif not args.at or args.at.utcoffset() is None:
        parser.error("--at requires a date, time and UTC offset")
    target = args.at.astimezone(CHINA) if args.at else now()
    with (RUNS / "PAUSE_LOCK").open("a+b", buffering=0) as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        status = {"state": "scheduled", "pid": os.getpid(), "scheduled_for_cst": target.isoformat(), "set_cst": now().isoformat(timespec="seconds")}
        if args.finish_run:
            status.update(mode="after_run", target_run_id=args.finish_run)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            try:
                finish_current_run(run, status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
            except Exception as error:
                status.update(state="attention_required", error=repr(error))
                write_json(PAUSE, status)
                raise
            finally:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            return
        write_json(PAUSE, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        while now() < target:
            time.sleep(min(30, max(0, (target - now()).total_seconds())))
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            status["state"] = "waiting_for_checkpoint"
            write_json(PAUSE, status)
            processes, run, boundary, reason = wait_for_checkpoint(target)
            status.update(state="pausing", run_id=run.name if run else None, boundary=boundary)
            write_json(PAUSE, status)
            stop_processes(processes)
            record_pause(status, run, boundary, reason)
            print(json.dumps(status, ensure_ascii=False), flush=True)
        except Exception as error:
            status.update(state="attention_required", error=repr(error))
            write_json(PAUSE, status)
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
