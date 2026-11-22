#!/usr/bin/env bash
# Sequentially train the four architecture innovations on 1200 patches.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="/c/Users/qing/AppData/Local/Programs/Python/Python312/python.exe"
for a in full delta gate reinject; do
  echo "=== ARCH $a ==="
  MSYS_NO_PATHCONV=1 "$PY" tools/train_stfdiff.py --variant "arch_$a" --arch "$a" \
      --epochs 40 --batch 2 --lr 2e-4 --out "runs/arch_$a" 2>&1 | tee "log_arch_$a.txt"
done
echo "ALL ARCH DONE"
