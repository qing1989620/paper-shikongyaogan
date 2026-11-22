"""Windows file handling for the experiment controllers."""

import json
import os
import time


# Measured best + resume is about 115 MiB (about 144 MiB with EMA).
# Leave room for an additional atomic temporary checkpoint and other run files.
CHECKPOINT_HEADROOM_BYTES = 512 * 2**20


def replace_with_retry(source, destination):
    """A status reader can briefly prevent replacement on Windows."""
    deadline = time.monotonic() + 12
    while True:
        try:
            source.replace(destination)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


def write_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    replace_with_retry(temporary, path)
