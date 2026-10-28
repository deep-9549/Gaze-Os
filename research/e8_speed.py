"""E8 (PRD 16): speed against accuracy of the deployable gaze models.

Times each model the way the app runs it (`benchmark_ms`: the ONNX estimator on a blank
normalized frame, after a warm-up) and keeps the best of several rounds, so a background
task cannot inflate the number. Accuracy is the held-out error from `training.eval_onnx`,
passed in. Run on an idle machine.

    python -m research.e8_speed --model "M3 fp32" models/gazenet_m3.onnx 3.647 \
        --model "M4 fp32" models/gazenet_lite.onnx 3.657 --out e8.json --figure e8.png
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from gazeos.inference.gaze import benchmark_ms


def time_models(models: list[tuple[str, str, float]], rounds: int = 3, runs: int = 50) -> dict[str, Any]:
    ms = [min(benchmark_ms(path, runs=runs) for _ in range(rounds)) for _, path, _ in models]
    import onnxruntime

    return {"names": [n for n, _, _ in models], "paths": [p for _, p, _ in models], "ms": ms,
            "error": [e for _, _, e in models],
            "note": f"best of {rounds} rounds of the median of {runs} inferences, ONNX Runtime CPU",
            "machine": {"cpu": platform.processor(), "cores": os.cpu_count(), "onnxruntime": onnxruntime.__version__,
                        "measured": time.strftime("%Y-%m-%d %H:%M")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "PATH", "ERROR_DEG"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", default=None)
    a = ap.parse_args()
    r = time_models([(n, p, float(e)) for n, p, e in a.model], a.rounds, a.runs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2))
    for n, m, e in zip(r["names"], r["ms"], r["error"], strict=True):
        print(f"{n:12s} {m:6.2f} ms  {e:.2f} deg")
    if a.figure:
        from research.figures import scatter_speed_accuracy

        scatter_speed_accuracy(r["names"], r["ms"], r["error"], a.figure)


if __name__ == "__main__":
    main()
