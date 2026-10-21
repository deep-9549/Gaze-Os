"""Download the runtime models that are not built by this project.

    python -m tools.download_models            # MediaPipe Face Landmarker (Apache 2.0)

The gaze and gesture ONNX models come from training (training/README in
docs/TRAINING.md) or from training/convert_l2cs.py for the L2CS-Net baseline.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from gazeos.vision.landmarks import DEFAULT_MODEL_PATH, MODEL_URL


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def main() -> int:
    dest = DEFAULT_MODEL_PATH
    if dest.exists() and "--force" not in sys.argv:
        print(f"{dest} already present")
    else:
        download(MODEL_URL, dest)
        print(f"downloaded {dest}")
    print("sha256", hashlib.sha256(dest.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
