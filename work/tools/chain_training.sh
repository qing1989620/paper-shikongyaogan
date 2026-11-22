#!/usr/bin/env bash
# Wait for the base run to finish, then train the remaining variants in sequence.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="/c/Users/qing/AppData/Local/Programs/Python/Python312/python.exe"

echo "[watcher] waiting for base to finish..."
for i in $(seq 1 240); do
  if grep -q "^BEST" log_train_base.txt 2>/dev/null; then
    echo "[watcher] base finished"
    break
  fi
  sleep 60
done

for v in lf hf bandw onestep; do
  echo "=== TRAIN $v ==="
  MSYS_NO_PATHCONV=1 "$PY" tools/train_stfdiff.py --variant "$v" --epochs 40 --batch 2 \
      --lr 2e-4 --out "runs/$v" 2>&1 | tee "log_train_$v.txt"
done
echo "ALL VARIANTS DONE"
