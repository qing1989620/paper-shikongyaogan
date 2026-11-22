# -*- coding: utf-8 -*-
"""Build the LGC training patches (official syy_setting train triples)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_patches import read_bsq, build_triple, lgc_date_map, LGC_H, LGC_W  # noqa: E402

TRIPLES = [
    ("2004-04-16", "2004-05-02", "2004-07-05"),
    ("2004-05-02", "2004-07-05", "2004-08-22"),
    ("2004-07-05", "2004-08-06", "2004-08-22"),
    ("2004-08-22", "2004-10-25", "2004-11-26"),
    ("2004-10-25", "2004-11-26", "2004-12-12"),
]


def main():
    raw = Path("data/raw")
    out = Path("data/patches_lgc/train")
    out.mkdir(parents=True, exist_ok=True)
    fmap, cmap = lgc_date_map(raw)
    need = sorted({d for t in TRIPLES for d in t})
    print("loading", len(need), "dates", flush=True)
    f_imgs = {d: read_bsq(fmap[d], LGC_H, LGC_W) for d in need}
    c_imgs = {d: read_bsq(cmap[d], LGC_H, LGC_W) for d in need}
    total = 0
    for i, tr in enumerate(TRIPLES):
        n = build_triple(out, f"train{i}", tr, f_imgs, c_imgs, 256)
        total += n
        print(f"train{i} {tr}: {n} patches", flush=True)
    print("TOTAL", total)


if __name__ == "__main__":
    main()
