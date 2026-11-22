"""Reversible relocation of historical documents and logs. No file deletion.

Default: report the explicit move plan. Use --apply to perform it once.
Every source/destination is resolved and checked to remain within this workspace.
Each moved file is hashed before and after; the JSON ledger is written first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "audit" / "organization_moves.json"


def safe(path):
    p = (ROOT / path).resolve()
    if p == ROOT or not p.is_relative_to(ROOT):
        raise ValueError(f"Target outside workspace: {p}")
    return p


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def plan():
    moves = []
    for p in sorted(ROOT.iterdir()):
        if not p.is_file() or p.name == "README.md":
            continue
        if p.name == "HANDOVER.md":
            target = "archive/handover/HANDOVER_2026-11-18.md"
        elif p.name == "聊天记录.txt":
            target = "archive/handover/聊天记录_截至2026-11-17.txt"
        elif p.suffix == ".xlsx":
            target = f"references/literature/{p.name}"
        elif p.name in ("DeepFDP.pdf", "GSINet.pdf"):
            target = f"references/papers/{p.name}"
        elif p.suffix == ".zip":
            target = f"archive/packages/{p.name}"
        elif p.suffix.lower() in (".md", ".pdf"):
            target = f"archive/reports/2026-09/{p.name}"
        else:
            continue
        moves.append((p, safe(target)))
    for p in sorted((ROOT / "work").glob("*.md")):
        if p.name != "README.md":
            moves.append((p, safe(f"docs/history/2026-11-17_18/{p.name}")))
    for p in sorted((ROOT / "work").glob("log_*.txt")):
        moves.append((p, safe(f"work/logs/legacy/{p.name}")))
    for src, dst in [
        ("work/sastf.pdf", "references/papers/SA-STF_CVPR2026.pdf"),
        ("work/sastf_txt", "references/extracted/SA-STF_CVPR2026.txt"),
        ("deliverables", "archive/snapshots/2026-11-18_deliverables"),
        ("work/tools/cleanup.py", "archive/legacy_tools/cleanup_legacy_2026-11-18.py"),
    ]:
        p = safe(src)
        if p.exists():
            moves.append((p, safe(dst)))
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if LEDGER.exists():
        print(f"Organization ledger already exists; no repeat moves: {LEDGER}")
        return
    moves = plan()
    for src, dst in moves:
        safe(src)
        safe(dst)
        if dst.exists():
            raise FileExistsError(dst)
    if not args.apply:
        for src, dst in moves:
            print(f"{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        print(f"{len(moves)} planned moves. No files changed.")
        return
    ledger = {"created_at_cst": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "mode": "move_only_no_deletion", "moves": [], "verified": False}
    for src, dst in moves:
        files = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file()]
        ledger["moves"].append({"source": src.relative_to(ROOT).as_posix(),
                                "destination": dst.relative_to(ROOT).as_posix(), "done": False,
                                "files": [{"relative": "" if p == src else p.relative_to(src).as_posix(),
                                           "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]})
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    def save():
        LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    for (src, dst), entry in zip(moves, ledger["moves"]):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        for item in entry["files"]:
            p = dst / item["relative"] if item["relative"] else dst
            if p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]:
                raise RuntimeError(f"Post-move integrity check failed: {p}")
        entry["done"] = True
        save()
    ledger["verified"] = True
    save()
    print(f"Moved {len(moves)} entries; verified {sum(len(x['files']) for x in ledger['moves'])} files by SHA256.")
    print(LEDGER)


if __name__ == "__main__":
    main()
