"""Augmentations that mimic webcam artifacts (PRD 10.5).

Colour jitter, random gamma, Gaussian and motion blur, JPEG re-compression,
shifts up to 8 px, and horizontal flip with the label mirrored (yaw negated,
eyes swapped, head yaw and roll negated).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GazeAugment:
    p_color: float = 0.8
    p_gamma: float = 0.5
    p_blur: float = 0.3
    p_motion: float = 0.2
    p_jpeg: float = 0.5
    p_flip: float = 0.5
    p_glare: float = 0.1
    max_shift: int = 8
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        r = self.rng
        s = dict(s)
        face = s["face"].astype(np.float32)
        eyes = [s["right_eye"].astype(np.float32), s["left_eye"].astype(np.float32)]
        if r.random() < self.p_color:
            gain = r.uniform(0.7, 1.3, 3)
            bias = r.uniform(-20, 20)
            face = face * gain + bias
            g = float(gain.mean())
            eyes = [e * g + bias for e in eyes]
        if r.random() < self.p_gamma:
            gamma = r.uniform(0.6, 1.6)
            face = 255.0 * (np.clip(face, 0, 255) / 255.0) ** gamma
            eyes = [255.0 * (np.clip(e, 0, 255) / 255.0) ** gamma for e in eyes]
        face = np.clip(face, 0, 255).astype(np.uint8)
        eyes = [np.clip(e, 0, 255).astype(np.uint8) for e in eyes]
        if r.random() < self.p_glare:  # glasses glare on one eye
            i = int(r.integers(2))
            e = eyes[i].astype(np.float32)
            cx, cy = r.integers(10, 50), r.integers(6, 30)
            yy, xx = np.mgrid[:36, :60]
            e += 160 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r.uniform(3, 8) ** 2))
            eyes[i] = np.clip(e, 0, 255).astype(np.uint8)
        if r.random() < self.p_blur:
            k = int(r.choice([3, 5]))
            face = cv2.GaussianBlur(face, (k, k), 0)
            eyes = [cv2.GaussianBlur(e, (3, 3), 0) for e in eyes]
        if r.random() < self.p_motion:
            k = int(r.integers(3, 8))
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2] = 1.0 / k
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(r.uniform(0, 180)), 1.0)
            kernel = cv2.warpAffine(kernel, M, (k, k))
            kernel /= max(kernel.sum(), 1e-6)
            face = cv2.filter2D(face, -1, kernel)
        if r.random() < self.p_jpeg:
            q = int(r.integers(30, 90))
            ok, buf = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                face = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if self.max_shift:
            dx, dy = (int(v) for v in r.integers(-self.max_shift, self.max_shift + 1, 2))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            face = cv2.warpAffine(face, M, (224, 224), borderMode=cv2.BORDER_REPLICATE)
            ex, ey = int(round(dx * 60 / 224)), int(round(dy * 36 / 224))
            Me = np.float32([[1, 0, ex], [0, 1, ey]])
            eyes = [cv2.warpAffine(e, Me, (60, 36), borderMode=cv2.BORDER_REPLICATE) for e in eyes]
        gaze, head = s["gaze"].copy(), s["head"].copy()
        if r.random() < self.p_flip:
            face = np.ascontiguousarray(face[:, ::-1])
            # The flip turns the subject's left eye into the image-left (right) eye, seen
            # mirrored: exactly the stored left crop, which is already mirrored. So a
            # plain swap of the two crops gives the flipped sample.
            right, left = eyes
            eyes = [np.ascontiguousarray(left), np.ascontiguousarray(right)]
            gaze[1] = -gaze[1]
            head[1], head[2] = -head[1], -head[2]
        s["face"], s["right_eye"], s["left_eye"] = face, eyes[0], eyes[1]
        s["gaze"], s["head"] = gaze, head
        return s
