"""Export a trained gaze model to ONNX with a parity check (PRD 10.8 steps 1-2).

    python -m training.export_onnx outputs/m4/best.pt models/gazenet_lite.onnx --parity-shards "data/shards/mpiifacegaze/*.h5"

Fixed input shapes, opset 17. Parity: mean difference between PyTorch and
ONNX angles must be under 0.05 degrees on up to 1,000 test images.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from training.models.gazenet import BIN_MIN_DEG, BIN_WIDTH_DEG, ExportWrapper, GazeNetGOS, bin_centers_deg

INPUT_NAMES = ["face", "right_eye", "left_eye", "head"]


def load_checkpoint(path: str | Path) -> tuple[GazeNetGOS, dict]:  # type: ignore[type-arg]
    state = torch.load(path, map_location="cpu")
    m = state["config"]["model"]
    model = GazeNetGOS(backbone=m["backbone"], pretrained=False, use_eyes=m["use_eyes"], use_head=m["use_head"],
                       use_face=m.get("use_face", True), fusion=m["fusion"], num_bins=m["num_bins"], dropout=m["dropout"])
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def dummy_inputs(batch: int = 1) -> tuple[torch.Tensor, ...]:
    return (torch.randn(batch, 3, 224, 224), torch.randn(batch, 1, 36, 60), torch.randn(batch, 1, 36, 60),
            torch.randn(batch, 3) * 0.2)


def export(model: GazeNetGOS, out: str | Path, metadata: dict[str, str] | None = None, opset: int = 17) -> Path:
    import onnx

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model).eval()
    names = ["pitch_logits", "yaw_logits", "log_var", "features"]
    if model.encoder is not None:
        names.append("attention")
    torch.onnx.export(wrapper, dummy_inputs(), str(out), input_names=INPUT_NAMES, output_names=names,
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    proto = onnx.load(str(out))
    meta = {"num_bins": str(model.num_bins), "bin_width_deg": str(BIN_WIDTH_DEG), "bin_min_deg": str(BIN_MIN_DEG),
            **(metadata or {})}
    for k, v in meta.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))
    return out


def _batches(shards: str | None, n: int, batch: int = 50):  # type: ignore[no-untyped-def]
    if shards:
        from training.data.shards import GazeShardDataset

        ds = GazeShardDataset(sorted(glob.glob(shards)))
        idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
        for i in range(0, len(idx), batch):
            items = [ds[int(j)] for j in idx[i:i + batch]]
            yield tuple(torch.stack([it[k] for it in items]) for k in INPUT_NAMES)
    else:
        g = torch.Generator().manual_seed(0)
        for _ in range(max(n // batch, 1)):
            yield (torch.randn(batch, 3, 224, 224, generator=g), torch.randn(batch, 1, 36, 60, generator=g),
                   torch.randn(batch, 1, 36, 60, generator=g), torch.randn(batch, 3, generator=g) * 0.2)


def parity(model: GazeNetGOS, onnx_path: str | Path, shards: str | None = None, n: int = 1000) -> dict[str, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    centers = bin_centers_deg(model.num_bins)
    diffs = []
    with torch.no_grad():
        for b in _batches(shards, n):
            ref = model(*b).angles_deg(centers).numpy()
            for i in range(b[0].shape[0]):
                feeds = {name: t[i:i + 1].numpy() for name, t in zip(INPUT_NAMES, b, strict=True)}
                p, y = sess.run(["pitch_logits", "yaw_logits"], feeds)
                got = np.stack([_decode(p[0], centers.numpy()), _decode(y[0], centers.numpy())])
                diffs.append(np.abs(got - ref[i]))
    d = np.stack(diffs)
    return {"mean_abs_diff_deg": float(d.mean()), "max_abs_diff_deg": float(d.max()), "n": int(len(d))}


def _decode(logits: np.ndarray, centers: np.ndarray) -> float:
    z = logits - logits.max()
    p = np.exp(z) / np.exp(z).sum()
    return float(p @ centers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--parity-shards", default=None, help="glob of HDF5 shards for the parity check")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    model, state = load_checkpoint(a.checkpoint)
    path = export(model, a.out, {"commit": str(state.get("commit", "")), "source": str(a.checkpoint)})
    report = parity(model, path, a.parity_shards, a.n)
    print(json.dumps(report, indent=2))
    if report["mean_abs_diff_deg"] >= 0.05:
        raise SystemExit("parity check failed: mean difference >= 0.05 degrees")


if __name__ == "__main__":
    main()
