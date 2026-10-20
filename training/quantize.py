"""Static INT8 quantization with ONNX Runtime (PRD 10.8 step 3).

    python -m training.quantize models/gazenet_lite.onnx models/gazenet_lite_int8.onnx \
        --calib "data/shards/mpiifacegaze/*.h5" --eval "data/shards/mpiifacegaze/*.h5"

Calibrated on 500 normalized frames; accepted only if the accuracy drop is
under 0.2 degrees.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

INPUTS = ("face", "right_eye", "left_eye", "head")


class ShardReader:
    def __init__(self, shards: str | None, n: int = 500, input_names: tuple[str, ...] = INPUTS) -> None:
        self.items = list(_samples(shards, n))
        self.names = input_names
        self._i = 0

    def get_next(self):  # type: ignore[no-untyped-def]
        if self._i >= len(self.items):
            return None
        s = self.items[self._i]
        self._i += 1
        return {k: s[k] for k in self.names}

    def rewind(self) -> None:
        self._i = 0


def _samples(shards: str | None, n: int):  # type: ignore[no-untyped-def]
    if shards:
        from training.data.shards import GazeShardDataset

        ds = GazeShardDataset(sorted(glob.glob(shards)))
        for j in np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int):
            it = ds[int(j)]
            yield {k: it[k].numpy()[None] for k in INPUTS} | {"gaze_deg": it["gaze_deg"].numpy()}
    else:
        rng = np.random.default_rng(0)
        for _ in range(n):
            yield {"face": rng.standard_normal((1, 3, 224, 224), dtype=np.float32),
                   "right_eye": rng.standard_normal((1, 1, 36, 60), dtype=np.float32),
                   "left_eye": rng.standard_normal((1, 1, 36, 60), dtype=np.float32),
                   "head": (rng.standard_normal((1, 3)) * 0.2).astype(np.float32),
                   "gaze_deg": np.zeros(2, np.float32)}


def quantize(src: str | Path, dst: str | Path, calib: str | None, n: int = 500) -> Path:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class _Reader(CalibrationDataReader):  # type: ignore[misc]
        def __init__(self) -> None:
            self.r = ShardReader(calib, n)

        def get_next(self):  # type: ignore[no-untyped-def]
            return self.r.get_next()

    pre = Path(dst).with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(dst), _Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    )
    pre.unlink(missing_ok=True)
    _copy_metadata(src, dst)
    return Path(dst)


def _copy_metadata(src: str | Path, dst: str | Path) -> None:
    import onnx

    a, b = onnx.load(str(src)), onnx.load(str(dst))
    existing = {p.key for p in b.metadata_props}
    for p in a.metadata_props:
        if p.key not in existing:
            e = b.metadata_props.add()
            e.key, e.value = p.key, p.value
    onnx.save(b, str(dst))


def evaluate(path: str | Path, shards: str | None, n: int = 1000) -> dict[str, float]:
    import onnxruntime as ort

    from gazeos.inference.gaze import decode_bins
    from gazeos.vision.geometry import angular_error_deg

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    k, w, lo = int(meta.get("num_bins", 28)), float(meta.get("bin_width_deg", 3)), float(meta.get("bin_min_deg", -42))
    centers = np.arange(k) * w + lo + w / 2
    errs, times = [], []
    for s in _samples(shards, n):
        t0 = time.perf_counter()
        p, y = sess.run(["pitch_logits", "yaw_logits"], {kk: s[kk] for kk in INPUTS})
        times.append((time.perf_counter() - t0) * 1000)
        pred = np.degrees([decode_bins(p[0], centers), decode_bins(y[0], centers)])
        errs.append(float(angular_error_deg(np.radians(pred)[None], np.radians(s["gaze_deg"])[None])[0]))
    return {"mean_error_deg": float(np.mean(errs)), "ms_per_frame": float(np.median(times)), "n": len(errs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    quantize(a.src, a.dst, a.calib, a.n)
    fp32, int8 = evaluate(a.src, a.eval), evaluate(a.dst, a.eval)
    report = {"fp32": fp32, "int8": int8, "drop_deg": int8["mean_error_deg"] - fp32["mean_error_deg"]}
    print(json.dumps(report, indent=2))
    if a.eval and report["drop_deg"] >= 0.2:
        raise SystemExit("INT8 model rejected: accuracy drop >= 0.2 degrees")


if __name__ == "__main__":
    main()
