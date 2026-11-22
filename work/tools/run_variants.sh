#!/usr/bin/env bash
# Sequentially train the remaining variants (base is run separately).
set -u
cd "$(dirname "$0")/.." || exit 1
PY="/c/Users/qing/AppData/Local/Programs/Python/Python312/python.exe"
for v in lf hf bandw onestep; do
  echo "=== TRAIN $v ==="
  MSYS_NO_PATHCONV=1 "$PY" tools/train_stfdiff.py --variant "$v" --epochs 40 --batch 4 \
      --lr 2e-4 --out "runs/$v" 2>&1 | tee "log_train_$v.txt"
done
echo "ALL VARIANTS DONE"
