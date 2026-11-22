"""Compatibility entry point for the corrected ten-triple preparation.

Default is read-only. Example:
  python work/tools/prepare_train.py --dataset LGC --verify-existing
  python work/tools/prepare_train.py --dataset LGC --generate --output <new-empty-directory>
"""
from prepare_dataset_v2 import main

if __name__=='__main__':main()
