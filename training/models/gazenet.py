"""GazeNet-GOS (M3), the face-only ablation (M2) and GazeNet-GOS-Lite (M4) (PRD 10.2, 10.6).

Multi-stream CNN: a face stream, a shared eye stream (the right eye is
mirrored so both eyes share weights), and a head-pose MLP; the four feature
vectors become 256-d tokens fused by a 2-layer transformer encoder. Heads
predict 28 pitch bins, 28 yaw bins (L2CS classify-then-regress) and a
log-variance per angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

NUM_BINS = 28
BIN_WIDTH_DEG = 3.0
BIN_MIN_DEG = -42.0


def bin_centers_deg(num_bins: int = NUM_BINS, width: float = BIN_WIDTH_DEG, lo: float = BIN_MIN_DEG) -> torch.Tensor:
    return torch.arange(num_bins, dtype=torch.float32) * width + lo + width / 2.0


@dataclass
class GazeOutput:
    pitch_logits: torch.Tensor  # (B, K)
    yaw_logits: torch.Tensor  # (B, K)
    log_var: torch.Tensor  # (B, 2)
    features: torch.Tensor  # (B, 256)
    attention: torch.Tensor | None = None  # (B, T, T)

    def angles_deg(self, centers: torch.Tensor) -> torch.Tensor:
        """(B, 2) pitch/yaw in degrees: softmax-weighted average of bin centres."""
        c = centers.to(self.pitch_logits.device)
        p = (self.pitch_logits.softmax(-1) * c).sum(-1)
        y = (self.yaw_logits.softmax(-1) * c).sum(-1)
        return torch.stack([p, y], dim=-1)


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)
    )


class EyeStream(nn.Module):
    """4 conv blocks, channels 32-64-128-128, global pool -> 128."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(_conv_block(1, 32), _conv_block(32, 64), _conv_block(64, 128), _conv_block(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x)).flatten(1)


def _face_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    import torchvision

    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        m.fc = nn.Identity()
        return m, 512
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=weights)
        m.fc = nn.Identity()
        return m, 2048
    if name == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.mobilenet_v3_small(weights=weights)
        m.classifier = nn.Identity()
        return m, 576
    raise ValueError(f"unknown backbone {name}")


class GazeNetGOS(nn.Module):
    """M3 (resnet18 face stream) and M4 Lite (mobilenet_v3_small face stream).

    Ablation switches for E2: `use_eyes`, `use_head`, `fusion` ("transformer" | "concat").
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        use_eyes: bool = True,
        use_head: bool = True,
        use_face: bool = True,
        fusion: str = "transformer",
        num_bins: int = NUM_BINS,
        dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not (use_face or use_eyes):
            raise ValueError("need the face or the eyes")
        self.use_face, self.use_eyes, self.use_head, self.fusion = use_face, use_eyes, use_head, fusion
        self.num_bins = num_bins
        tokens = []
        if use_face:
            self.face, fdim = _face_backbone(backbone, pretrained)
            self.face_proj = nn.Linear(fdim, dim)
            tokens.append("face")
        if use_eyes:
            self.eye = EyeStream()
            self.eye_proj = nn.Linear(128, dim)
            tokens += ["right_eye", "left_eye"]
        if use_head:
            self.head = nn.Sequential(nn.Linear(3, 32), nn.ReLU(inplace=True), nn.Linear(32, 32), nn.ReLU(inplace=True))
            self.head_proj = nn.Linear(32, dim)
            tokens.append("head")
        self.token_names = tokens
        n = len(tokens)
        self.type_embed = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.normal_(self.type_embed, std=0.02)
        if fusion == "transformer":
            layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 2, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            fused = dim
        elif fusion == "concat":
            self.encoder = None
            fused = dim * n
        else:
            raise ValueError(fusion)
        self.trunk = nn.Sequential(nn.Linear(fused, 256), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.pitch_head = nn.Linear(256, num_bins)
        self.yaw_head = nn.Linear(256, num_bins)
        self.var_head = nn.Linear(256, 2)
        self._last_attention: torch.Tensor | None = None

    def tokens(self, face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        toks = []
        if self.use_face:
            toks.append(self.face_proj(self.face(face)))
        if self.use_eyes:
            # Inputs: the left eye arrives already mirrored by normalization, so one
            # stream serves both eyes.
            toks.append(self.eye_proj(self.eye(right_eye)))
            toks.append(self.eye_proj(self.eye(left_eye)))
        if self.use_head:
            toks.append(self.head_proj(self.head(head)))
        return torch.stack(toks, dim=1) + self.type_embed

    def forward(
        self, face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor, head: torch.Tensor
    ) -> GazeOutput:
        x = self.tokens(face, right_eye, left_eye, head)
        attention = None
        if self.encoder is not None:
            attention = self._attention(x)
            x = self.encoder(x).mean(dim=1)
        else:
            x = x.flatten(1)
        feats = self.trunk(x)
        return GazeOutput(self.pitch_head(feats), self.yaw_head(feats), self.var_head(feats), feats, attention)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights of the first encoder layer (for the viva explainability visual)."""
        layer = self.encoder.layers[0]  # type: ignore[union-attr]
        q = layer.norm1(x) if layer.norm_first else x
        _, w = layer.self_attn(q, q, q, need_weights=True, average_attn_weights=True)
        return w.detach()

    def finetune_parameters(self) -> list[nn.Parameter]:
        """P4: only the fusion output and heads are unfrozen."""
        params: list[nn.Parameter] = []
        for m in (self.trunk, self.pitch_head, self.yaw_head, self.var_head):
            params += list(m.parameters())
        return params


def gazenet_m2(pretrained: bool = True) -> GazeNetGOS:
    """M2: face-only ResNet-18 with the L2CS head."""
    return GazeNetGOS("resnet18", pretrained, use_eyes=False, use_head=False, fusion="concat")


def gazenet_m3(pretrained: bool = True) -> GazeNetGOS:
    return GazeNetGOS("resnet18", pretrained)


def gazenet_m4(pretrained: bool = True) -> GazeNetGOS:
    return GazeNetGOS("mobilenet_v3_small", pretrained)


MODELS = {"m2": gazenet_m2, "m3": gazenet_m3, "m4": gazenet_m4}


class ExportWrapper(nn.Module):
    """Plain-tensor outputs with the GazeOS ONNX names."""

    def __init__(self, model: GazeNetGOS) -> None:
        super().__init__()
        self.model = model

    def forward(self, face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor, head: torch.Tensor):  # type: ignore[no-untyped-def]
        out = self.model(face, right_eye, left_eye, head)
        if out.attention is None:
            return out.pitch_logits, out.yaw_logits, out.log_var, out.features
        return out.pitch_logits, out.yaw_logits, out.log_var, out.features, out.attention
