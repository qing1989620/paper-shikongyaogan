"""Read-only cache inventory. The old automatic deletion script is archived.

This entry point preserves existing callers while never deleting experiment
evidence. Current caches remain required for metric repair and ablations.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    for p in sorted((ROOT / "results").glob("cache_*")):
        files = [f for f in p.rglob("*") if f.is_file()]
        print(f"{p.name}: {len(files)} files, {sum(f.stat().st_size for f in files)/2**20:.1f} MiB; retained")
    print("Read-only inventory complete. No files deleted.")
