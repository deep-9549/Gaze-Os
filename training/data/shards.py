"""HDF5 shard format and the PyTorch dataset over it (PRD 15.4).

Every shard holds, per sample:
  face       uint8 (224, 224, 3) normalized face, RGB
  right_eye  uint8 (36, 60)      normalized right eye, grayscale
  left_eye   uint8 (36, 60)      normalized left eye, mirrored
  head       float32 (3,)        normalized head pose (pitch, yaw, roll), radians
  gaze       float32 (2,)        normalized gaze (pitch, yaw), radians
  subject    int32               subject id (for leave-one-subject-out)
  condition  int32 bitfield      lighting/glasses/distance codes where known (E9)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

FIELDS = ("face", "right_eye", "left_eye", "head", "gaze", "subject", "condition")

# Normalized face: eyes sit about 51 px either side of centre and 10 px above it
# (60 mm apart at 600 mm with a 960 px focal length). Used when a dataset gives
# pre-normalized faces without eye landmarks.
_EYE_CENTERS = ((112 - 51, 102), (112 + 51, 102))
_EYE_BOX = (60, 36)


class ShardWriter:
    def __init__(self, path: str | Path, chunk: int = 256) -> None:
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "w")
        self.n = 0
        self.chunk = chunk
        shapes = {"face": (224, 224, 3), "right_eye": (36, 60), "left_eye": (36, 60), "head": (3,), "gaze": (2,),
                  "subject": (), "condition": ()}
        dtypes = {"face": "u1", "right_eye": "u1", "left_eye": "u1", "head": "f4", "gaze": "f4",
                  "subject": "i4", "condition": "i4"}
        for k in FIELDS:
            self.f.create_dataset(k, shape=(0, *shapes[k]), maxshape=(None, *shapes[k]), dtype=dtypes[k],
                                  chunks=(min(chunk, 64), *shapes[k]),
                                  compression="lzf" if k in ("face", "right_eye", "left_eye") else None)

    def add(self, face: NDArray[np.uint8], right_eye: NDArray[np.uint8], left_eye: NDArray[np.uint8],
            head: Sequence[float], gaze: Sequence[float], subject: int, condition: int = 0) -> None:
        i = self.n
        for k in FIELDS:
            ds = self.f[k]
            if ds.shape[0] <= i:
                ds.resize(ds.shape[0] + self.chunk, axis=0)
        self.f["face"][i] = face
        self.f["right_eye"][i] = right_eye
        self.f["left_eye"][i] = left_eye
        self.f["head"][i] = np.asarray(head, np.float32)
        self.f["gaze"][i] = np.asarray(gaze, np.float32)
        self.f["subject"][i] = subject
        self.f["condition"][i] = condition
        self.n += 1

    def close(self, meta: dict[str, object] | None = None) -> None:
        for k in FIELDS:
            self.f[k].resize(self.n, axis=0)
        self.f.attrs["meta"] = json.dumps(meta or {})
        self.f.close()


def fixed_eye_crops(face_rgb: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Eye crops from a normalized 224x224 face at the canonical eye positions."""
    from gazeos.vision.enhance import apply_clahe

    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    w, h = _EYE_BOX
    crops = []
    for (cx, cy) in _EYE_CENTERS:
        x0, y0 = cx - w // 2, cy - h // 2
        crops.append(apply_clahe(np.ascontiguousarray(gray[y0:y0 + h, x0:x0 + w])))
    # Image-left eye is the subject's right eye; mirror the other.
    return crops[0], np.ascontiguousarray(cv2.flip(crops[1], 1))


def file_checksum(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(paths: Sequence[str | Path], out: str | Path, extra: dict[str, object] | None = None) -> None:
    """Checksums for every shard, tied to the commit that produced them (PRD 15.4 step 5)."""
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        commit = ""
    data = {"commit": commit, "files": {str(p): file_checksum(p) for p in paths}, **(extra or {})}
    Path(out).write_text(json.dumps(data, indent=2))


class GazeShardDataset:
    """torch Dataset over one or more shards, optionally filtered by subject."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        subjects: Sequence[int] | None = None,
        exclude_subjects: Sequence[int] | None = None,
        transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
        stride: int = 1,
    ) -> None:
        import h5py

        self.paths = [str(p) for p in paths]
        self.transform = transform
        self.index: list[tuple[int, int]] = []
        for fi, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                subj = f["subject"][:]
            keep = np.ones(len(subj), bool)
            if subjects is not None:
                keep &= np.isin(subj, subjects)
            if exclude_subjects is not None:
                keep &= ~np.isin(subj, exclude_subjects)
            idx = np.nonzero(keep)[0][::stride]  # subsampling, recorded in the run config
            self.index += [(fi, int(i)) for i in idx]
        self._files: dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, fi: int):  # type: ignore[no-untyped-def]
        import h5py

        if fi not in self._files:  # opened lazily so DataLoader workers each get their own handle
            self._files[fi] = h5py.File(self.paths[fi], "r")
        return self._files[fi]

    def raw(self, i: int) -> dict[str, np.ndarray]:
        fi, j = self.index[i]
        f = self._file(fi)
        return {k: f[k][j] for k in FIELDS}

    def __getitem__(self, i: int):  # type: ignore[no-untyped-def]
        import torch

        s = self.raw(i)
        if self.transform is not None:
            s = self.transform(s)
        face = (s["face"].astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return {
            "face": torch.from_numpy(face.transpose(2, 0, 1).astype(np.float32)),
            "right_eye": torch.from_numpy((s["right_eye"].astype(np.float32) / 127.5 - 1.0)[None]),
            "left_eye": torch.from_numpy((s["left_eye"].astype(np.float32) / 127.5 - 1.0)[None]),
            "head": torch.from_numpy(s["head"].astype(np.float32)),
            "gaze_deg": torch.from_numpy(np.degrees(s["gaze"]).astype(np.float32)),
            "subject": int(s["subject"]),
            "condition": int(s["condition"]),
        }

    def subjects(self) -> list[int]:
        import h5py

        out: set[int] = set()
        for p in self.paths:
            with h5py.File(p, "r") as f:
                out |= set(int(s) for s in np.unique(f["subject"][:]))
        return sorted(out)
