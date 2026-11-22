# -*- coding: utf-8 -*-
"""Project cleanup: remove regenerable intermediates and stale artefacts.

Batch 1: eval caches (regenerable from best.pth), smoke data, temp files.
Batch 2: stale paper versions, render previews, old logs, deprecated figures.
Everything final lives in deliverables/.
"""
import os
import shutil
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
freed = 0
n = 0


def rm_file(p: Path):
    global freed, n
    try:
        s = p.stat().st_size
        os.remove(p)
        freed += s
        n += 1
    except Exception as e:
        print("skip", p, e)


def rm_tree(p: Path):
    global freed, n
    if not p.exists():
        return
    for f in sorted(p.rglob("*"), reverse=True):
        if f.is_file():
            rm_file(f)
        elif f.is_dir():
            try:
                f.rmdir(); n += 1
            except Exception:
                pass
    try:
        p.rmdir(); n += 1
    except Exception as e:
        print("dir kept", p, e)


batch = sys.argv[1] if len(sys.argv) > 1 else "1"

if batch == "1":
    # regenerable eval caches
    for d in Path("results").glob("cache_*"):
        rm_tree(d)
    # smoke/temp data
    for pat in ["runs/_smoke*", "runs/*/last.pth", "data/mini", "data/smoke_real",
                "data/synth", "data/patches_lgc/*.mm.npy", "wheels", "speedtest",
                "work", "test_s3_url.txt", "testfiles.json", "templates"]:
        for p in Path(".").glob(pat):
            if p.is_dir():
                rm_tree(p)
            else:
                rm_file(p)

elif batch == "2":
    # stale paper versions & build artefacts (keep: icip_en.tex, icip_en_v2.pdf,
    # icip_zh.tex/pdf, refs.bib, spconf.sty, IEEEbib.bst, icip_template/, figures/)
    paper = Path("paper")
    for pat in ["prev*", "preview*", "preview_icip", "main.tex", "main.pdf",
                "main_v2.*", "main_icip_en.*", "main_icip_zh.*", "ICIP2027_main_cn.*",
                "main.aux", "main.bbl", "main.blg", "main.log",
                "icip_en.aux", "icip_en.bbl", "icip_en.blg", "icip_en.log",
                "icip_en.pdf", "icip_en_build.*", "icip_zh.aux", "icip_zh.bbl",
                "icip_zh.blg", "icip_zh.log", "refs_official_sample.bib"]:
        for p in paper.glob(pat):
            if p.is_dir():
                rm_tree(p)
            else:
                rm_file(p)
    # deprecated figures (superseded or unused in final paper)
    for name in ["RSP_mechanism.pptx", "method_rsp.png", "qual_lgc.png",
                 "fig1_tradeoff_diverging.png", "fig1_tradeoff_diverging.pdf",
                 "fig2_rounded_robustness.png", "fig2_rounded_robustness.pdf",
                 "fig3_ranked_rose.png", "fig3_ranked_rose.pdf",
                 "make_icip_figs.py"]:
        rm_file(paper / "figures" / name)
    # stale logs (keep log_arch_*, log_train_*)
    for pat in ["log_dl*", "log_eval*", "log_par*", "log_prepare*", "log_chain.txt",
                "logs_download_*.txt"]:
        for p in Path(".").glob(pat):
            rm_file(p)

print(f"batch {batch}: removed {n} items, freed {freed/1e9:.2f} GB")
