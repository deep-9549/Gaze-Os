"""Convert the official L2CS-Net weights (M1 baseline) to the GazeOS ONNX contract.

Download `L2CSNet_gaze360.pkl` or the MPIIGaze fold weights from the L2CS-Net
repository (https://github.com/Ahmednull/L2CS-Net), then:

    python -m training.convert_l2cs L2CSNet_mpiigaze.pkl models/l2cs_mpii.onnx --dataset mpiigaze
    python -m training.convert_l2cs L2CSNet_gaze360.pkl  models/l2cs_gaze360.onnx --dataset gaze360

L2CS returns (yaw, pitch) logits; the angle is sum(softmax * index) * width + offset:
MPIIGaze 28 bins of 3 degrees from -42, Gaze360 90 bins of 4 degrees from -180.
Its input is a 448x448 ImageNet-normalized face; GazeOS resizes the normalized
face to that size automatically.
"""

from __future__ import annotations

import argparse

import torch
import torchvision
from torch import nn

SPECS = {"mpiigaze": (28, 3.0, -42.0), "gaze360": (90, 4.0, -180.0)}


class L2CS(nn.Module):
    def __init__(self, num_bins: int) -> None:
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_yaw_gaze = nn.Linear(2048, num_bins)
        self.fc_pitch_gaze = nn.Linear(2048, num_bins)

    def forward(self, face: torch.Tensor):  # type: ignore[no-untyped-def]
        x = self.maxpool(self.relu(self.bn1(self.conv1(face))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc_pitch_gaze(x), self.fc_yaw_gaze(x), x


def convert(weights: str, out: str, dataset: str) -> None:
    import onnx

    bins, width, start = SPECS[dataset]
    model = L2CS(bins)
    state = torch.load(weights, map_location="cpu")
    state = state.get("model_state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if [m for m in missing if not m.startswith("fc_finetune")]:
        raise SystemExit(f"missing weights: {missing[:5]}")
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, 448, 448), out, input_names=["face"],
                      output_names=["pitch_logits", "yaw_logits", "features"], opset_version=17, dynamo=False)
    proto = onnx.load(out)
    # L2CS bin k maps to k * width + start, i.e. bin centres start at `start`.
    for k, v in {"num_bins": str(bins), "bin_width_deg": str(width), "bin_min_deg": str(start - width / 2),
                 "source": "L2CS-Net official weights", "model_id": "M1"}.items():
        e = proto.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(proto, out)
    print(f"wrote {out} (unexpected keys ignored: {len(unexpected)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--dataset", choices=sorted(SPECS), default="mpiigaze")
    a = ap.parse_args()
    convert(a.weights, a.out, a.dataset)


if __name__ == "__main__":
    main()
