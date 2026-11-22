"""Restore the external source versions recorded in the experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

WORK = Path(__file__).resolve().parents[1] / "work"
LOCK = WORK / "configs/upstreams.json"


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
    return result.stdout.decode("utf-8").strip()


def restore(entry: dict) -> None:
    name, address, commit = (entry[key] for key in ("directory", "url", "commit"))
    target = WORK / "repos" / name
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", "--config", "core.autocrlf=true", address, str(target)],
            check=True,
        )
        git("checkout", "--detach", commit, cwd=target)
    if git("rev-parse", "HEAD", cwd=target) != commit:
        raise RuntimeError(f"{name} is already present at a different commit; use a clean restore directory.")
    for path, expected in entry.get("windows_crlf_sha256", {}).items():
        actual = hashlib.sha256((target / path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Byte provenance mismatch for {name}/{path}; check the recorded CRLF checkout.")
    overlay = WORK / "upstream_overlays" / name
    if overlay.is_dir():
        for owned in sorted(p for p in overlay.rglob("*.py") if p.is_file()):
            destination = target / owned.relative_to(overlay)
            if destination.exists():
                if destination.read_bytes() != owned.read_bytes():
                    raise RuntimeError(f"Local overlay differs from the backup: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(owned, destination)
    print(f"{name}: {commit}")


def main() -> None:
    entries = json.loads(LOCK.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[entry["directory"] for entry in entries])
    options = parser.parse_args()
    (WORK / "repos").mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if options.only is None or options.only == entry["directory"]:
            restore(entry)


if __name__ == "__main__":
    main()
