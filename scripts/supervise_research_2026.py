"""Run the resumable LGC stages serially and record periodic health checks."""

import argparse
import ctypes
import json
import msvcrt
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research_runtime import write_json


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "work/runs/research_2026"
LOGS = ROOT / "work/logs/research_2026"
STATUS = RUNS / "MONITOR_STATUS.json"
HOURLY_LOG = LOGS / "hourly_health.jsonl"
CHINA = timezone(timedelta(hours=8))
HOUR = 3600
SUCCESS = "exploratory_LGC_complete_formal_benchmark_CIA_and_ablations_pending"
TEMPORARY_IO = re.compile(r"PermissionError: \[WinError (5|32|33)\]")
NO_WINDOW = subprocess.CREATE_NO_WINDOW


def read_status(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def tail(path, size=4096):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - size))
            return stream.read(size).decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError):
        return ""


def active_research_processes():
    """Find verified project processes before launching or retrying a controller."""
    query = """
    Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='pythonw.exe'" |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine.IndexOf($env:STF_RESEARCH_ROOT, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $_.CommandLine -match '(run_research_queue|continue_research_2026|train_research_2026|evaluate_common_flood)\\.py'
        } | ForEach-Object { $_.ProcessId }
    """
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", query],
        env=dict(os.environ, STF_RESEARCH_ROOT=str(ROOT)),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=20, check=True, creationflags=NO_WINDOW,
    )
    return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]


class Monitor:
    def __init__(self, health_check_hours=1, stop_after_seed=None):
        self.next_check = time.monotonic()
        self.check_interval = health_check_hours * HOUR
        self.stop_after_seed = stop_after_seed
        self.stage = "screening"
        self.stage_start = time.time()
        self.controller = None
        self.controller_log = None
        self.last_report_key = None
        self.prevent_sleep = False

    def stage_status(self):
        filename = "QUEUE_STATUS.json" if self.stage == "screening" else "FOLLOWUP_STATUS.json"
        path = RUNS / filename
        if not path.exists() or path.stat().st_mtime < self.stage_start:
            return {}
        return read_status(path)

    def worker_log(self, stage_status):
        if self.stage == "screening":
            active = [
                job for job in stage_status.get("jobs", [])
                if job["state"] in ("running", "failed")
            ]
            current = active[-1] if active else {}
        else:
            if stage_status.get("state", "").startswith(("evaluating_common", "common_LGC")):
                return "common_flood", LOGS / "common_flood.log"
            current = stage_status.get("current", {})
        run_id = current.get("run_id")
        return (run_id, RUNS / run_id / "console.log") if run_id else (None, self.controller_log)

    def update(self, state, *, force=False, **details):
        stage_status = self.stage_status()
        current, progress_log = self.worker_log(stage_status)
        now = datetime.now(CHINA)
        elapsed = time.monotonic()
        check_due = elapsed >= self.next_check
        if check_due:
            while self.next_check <= elapsed:
                self.next_check += self.check_interval
        next_check = (now + timedelta(seconds=self.next_check - elapsed)).isoformat(timespec="seconds")
        snapshot = {
            "state": state,
            "pid": os.getpid(),
            "stage": self.stage,
            "controller_pid": self.controller.pid if self.controller else None,
            "stage_state": stage_status.get("state"),
            "run_id": current,
            "checked_cst": now.isoformat(timespec="seconds"),
            "next_hourly_check_cst": next_check if self.check_interval == HOUR else None,
            "next_health_check_cst": next_check,
            "health_check_interval_seconds": self.check_interval,
            "stop_after_seed": self.stop_after_seed,
            "screening_completed": sum(1 for _ in RUNS.glob("screen_*/COMPLETE.json")),
            "replication_completed": sum(1 for _ in RUNS.glob("replicate_*/COMPLETE.json")),
            "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
            "system_awake_requested": self.prevent_sleep,
            **details,
        }
        if progress_log and progress_log.exists():
            snapshot["progress_log"] = progress_log.relative_to(ROOT).as_posix()
            snapshot["progress_age_seconds"] = round(max(0, time.time() - progress_log.stat().st_mtime))
            lines = tail(progress_log, 2048).strip().splitlines()
            if lines:
                snapshot["last_progress"] = lines[-1][:350]
            checkpoint = progress_log.with_name("resume.pth")
            if current and checkpoint.exists():
                snapshot["checkpoint_cst"] = datetime.fromtimestamp(
                    checkpoint.stat().st_mtime, CHINA
                ).isoformat(timespec="seconds")
        if check_due or force:
            try:
                sample = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    creationflags=NO_WINDOW, capture_output=True, text=True,
                    timeout=10, check=True,
                )
                temperature, used, total, utilization = map(int, sample.stdout.strip().split(","))
                snapshot["gpu"] = {
                    "temperature_c": temperature, "memory_used_mib": used,
                    "memory_total_mib": total, "utilization_pct": utilization,
                }
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                snapshot["gpu_check_error"] = str(error)
            if snapshot.get("progress_age_seconds", 0) >= 1800 and state == "running":
                snapshot["progress_note"] = "No new log output for 30 minutes. The controller is still alive."
            with HOURLY_LOG.open("a", encoding="utf-8") as log:
                log.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
        write_json(STATUS, snapshot)
        report_key = (self.stage, current, state, snapshot["screening_completed"], snapshot["replication_completed"])
        if check_due or force or report_key != self.last_report_key:
            try:
                from update_research_report import main as refresh_report
                refresh_report()
            except (OSError, ValueError, KeyError) as error:
                print(f"Report refresh deferred: {error}", flush=True)
            self.last_report_key = report_key

    def run_stage(self, stage, script, expected, *stage_args):
        self.stage = stage
        for attempt in range(1, 4):
            self.stage_start = time.time()
            stamp = datetime.now(CHINA).strftime("%Y%m%d_%H%M%S_%f")
            self.controller_log = LOGS / f"{stage}_{stamp}.log"
            with self.controller_log.open("w", encoding="utf-8") as log:
                self.controller = subprocess.Popen(
                    [sys.executable, "-B", "-u", "-X", "utf8", str(ROOT / "scripts" / script), *stage_args],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, STF_MANAGED_RESEARCH="1"),
                    creationflags=NO_WINDOW,
                )
                while self.controller.poll() is None:
                    self.update("running", attempt=attempt)
                    try:
                        wait = max(0.1, min(60, self.next_check - time.monotonic()))
                        self.controller.wait(timeout=wait)
                    except subprocess.TimeoutExpired:
                        pass
            code = self.controller.returncode
            self.controller = None
            stage_status = self.stage_status()
            if code == 0 and stage_status.get("state") == expected:
                return True
            current, worker_log = self.worker_log(stage_status)
            diagnostic = tail(self.controller_log)
            if worker_log and worker_log.exists() and worker_log.stat().st_mtime >= self.stage_start:
                diagnostic += "\n" + tail(worker_log)
            if attempt < 3 and TEMPORARY_IO.search(diagnostic):
                self.update("retrying_file_lock", force=True, exit_code=code, attempt=attempt)
                # If the controller died first, let its current training worker finish.
                while active_research_processes():
                    self.update("waiting_for_previous_worker")
                    time.sleep(30)
                time.sleep(5)
                continue
            self.update(
                "attention_required", force=True, exit_code=code, failed_run=current,
                controller_log=self.controller_log.relative_to(ROOT).as_posix(),
                diagnostic=diagnostic.strip().splitlines()[-6:],
            )
            return False
        return False


def main(stop_after_seed=None, health_check_hours=1):
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    # A Windows byte-range lock is released even after an unexpected exit.
    with (RUNS / "MONITOR_LOCK").open("a+b", buffering=0) as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        active = active_research_processes()
        if active:
            raise RuntimeError(f"Existing research controller or worker: {active}")
        monitor = Monitor(health_check_hours, stop_after_seed)
        monitor.prevent_sleep = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        try:
            if stop_after_seed is not None:
                if read_status(RUNS / "QUEUE_STATUS.json").get("state") != "complete":
                    raise RuntimeError("A bounded replication requires completed screening")
                if not monitor.run_stage(
                    "replication_and_flood", "continue_research_2026.py", "seed_block_complete",
                    "--stop-after-seed", str(stop_after_seed),
                ):
                    return 1
                from pause_research_2026 import epoch_boundary, record_pause, verify_finished_run

                follow = monitor.stage_status()
                last_run = RUNS / follow["current"]["run_id"]
                verify_finished_run(last_run)
                boundary = epoch_boundary(last_run)
                if boundary is None or boundary["epoch"] != boundary["total_epochs"]:
                    raise RuntimeError(f"Missing final checkpoint boundary: {last_run.name}")
                record_pause({
                    "mode": "after_seed", "target_seed": stop_after_seed,
                    "scheduled_for_cst": datetime.now(CHINA).isoformat(timespec="seconds"),
                }, last_run, boundary, "completed_requested_seed")
                return 0
            if not monitor.run_stage("screening", "run_research_queue.py", "complete"):
                return 1
            if not monitor.run_stage("replication_and_flood", "continue_research_2026.py", SUCCESS):
                return 1
            monitor.update("complete", force=True)
            return 0
        except Exception as error:
            monitor.update("attention_required", force=True, error=repr(error))
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-after-seed", type=int)
    parser.add_argument("--health-check-hours", type=int, default=1)
    args = parser.parse_args()
    if args.health_check_hours < 1:
        parser.error("--health-check-hours must be at least one")
    raise SystemExit(main(args.stop_after_seed, args.health_check_hours))
