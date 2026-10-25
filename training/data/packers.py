"""Pack public datasets and GazeOS-Webcam recordings into HDF5 shards (PRD 15.1, 15.4).

    python -m training.data.packers mpiifacegaze /path/to/MPIIFaceGaze data/shards/mpiifacegaze
    python -m training.data.packers ethxgaze     /path/to/xgaze_224/train  data/shards/ethxgaze --stride 3
    python -m training.data.packers rtgene       data/raw/rtgene           data/shards/rtgene
    python -m training.data.packers gazeos       /path/to/recordings       data/shards/gazeos_webcam
    python -m training.data.packers synthetic    -                          data/shards/synthetic

Datasets are non-commercial research data: they are never redistributed and
never committed (PRD 15.5).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from gazeos.vision.geometry import rotation_to_euler, vector_to_pitchyaw
from gazeos.vision.head_pose import CameraIntrinsics
from gazeos.vision.normalization import FACE_SIZE, _crop_eye, _warp_points, warp_matrix
from training.data.shards import ShardWriter, fixed_eye_crops, write_manifest

log = logging.getLogger(__name__)


def pack_mpiifacegaze(root: str, out_dir: str) -> list[Path]:
    """MPIIFaceGaze: per subject pXX/pXX.txt with 6 landmarks, head pose, face centre and gaze target.

    Normalized with the same code as the runtime (PRD 15.4 step 2).
    """
    import scipy.io

    outs = []
    for subj_dir in sorted(glob.glob(os.path.join(root, "p[0-9][0-9]"))):
        sid = int(Path(subj_dir).name[1:])
        cam = scipy.io.loadmat(os.path.join(subj_dir, "Calibration", "Camera.mat"))
        K = CameraIntrinsics(matrix=np.asarray(cam["cameraMatrix"], float), dist=np.asarray(cam["distCoeffs"], float).ravel())
        out = Path(out_dir) / f"mpiifacegaze_p{sid:02d}.h5"
        w = ShardWriter(out)
        with open(os.path.join(subj_dir, f"p{sid:02d}.txt")) as f:
            lines = f.read().splitlines()
        for line in lines:
            parts = line.split()
            img = cv2.imread(os.path.join(subj_dir, parts[0]))
            if img is None:
                continue
            img = cv2.cvtColor(cv2.undistort(img, K.matrix, K.dist), cv2.COLOR_BGR2RGB)
            vals = np.asarray(parts[1:], float)
            lm = vals[2:14].reshape(6, 2)  # 4 eye corners, 2 mouth corners
            rvec = vals[14:17]  # head translation (vals[17:20]) is not needed: fc gives the face centre
            fc, gt = vals[20:23], vals[23:26]
            R, _ = cv2.Rodrigues(rvec)
            K0 = CameraIntrinsics(K.matrix, np.zeros(5))
            W, R_n, _ = warp_matrix(fc, R, K0)
            face = cv2.warpPerspective(img, W, (FACE_SIZE, FACE_SIZE))
            pts = _warp_points(W, lm)
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            # MPII landmark order: right eye outer, right eye inner, left eye inner, left eye outer.
            from gazeos.vision.enhance import apply_clahe

            right = apply_clahe(_crop_eye(gray, pts[0], pts[1], mirror=False))
            left = apply_clahe(_crop_eye(gray, pts[2], pts[3], mirror=True))
            gaze_n = R_n @ (gt - fc)
            head = rotation_to_euler(R_n @ R)
            w.add(face, right, left, head, vector_to_pitchyaw(gaze_n), sid)
        w.close({"dataset": "MPIIFaceGaze", "subject": sid})
        outs.append(out)
        log.info("subject %02d: %d samples", sid, w.n)
    return outs


def pack_ethxgaze(root: str, out_dir: str, stride: int = 1) -> list[Path]:
    """ETH-XGaze 224 release: pre-normalized `face_patch` (BGR), `face_gaze`, `face_head_pose`.

    `stride` subsamples frames (record it, PRD 15.4 step 4).
    """
    import h5py

    outs = []
    for i, path in enumerate(sorted(glob.glob(os.path.join(root, "subject*.h5")))):
        sid = 1000 + int("".join(c for c in Path(path).stem if c.isdigit()) or i)
        out = Path(out_dir) / f"ethxgaze_{Path(path).stem}.h5"
        w = ShardWriter(out)
        with h5py.File(path, "r") as f:
            n = f["face_patch"].shape[0]
            for j in range(0, n, stride):
                face = cv2.cvtColor(f["face_patch"][j], cv2.COLOR_BGR2RGB)
                right, left = fixed_eye_crops(face)
                hp = f["face_head_pose"][j]
                w.add(face, right, left, (float(hp[0]), float(hp[1]), 0.0), f["face_gaze"][j], sid)
        w.close({"dataset": "ETH-XGaze", "source": Path(path).name, "stride": stride})
        outs.append(out)
    return outs


RTGENE_SUBJECT_OFFSET = 2000
# (pitch, yaw) signs of the head pose relative to the shard convention; checked after
# packing (docs/PLAN.md step 16).
RTGENE_HEAD_SIGN = (1.0, 1.0)


def _rtgene_labels(text: str) -> dict[int, tuple[float, float, float, float]]:
    """`frame, [head A, head B], [gaze A, gaze B], timestamp` -> {frame: (A_h, B_h, A_g, B_g)}.

    A is yaw and B is pitch (verified on extreme-gaze frames): gaze B > 0 looks up,
    gaze A < 0 looks toward image-right, the same convention as `vector_to_pitchyaw`.
    """
    out = {}
    for line in text.splitlines():
        parts = [p.strip(" []") for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit():
            out[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    return out


def pack_rtgene(root: str, out_dir: str, stride: int = 1, max_abs_deg: float = 42.0) -> list[Path]:
    """RT-GENE: one `sNNN_glasses.tar` per subject, read in a single sequential pass.

    Uses the inpainted 224x224 face and 60x36 eye crops. The `right`/`left`
    folders are the subject's own eyes (template matching puts `right` on the
    image-left side), so `left` is mirrored like every other packer's left eye.
    Frames whose gaze falls outside the bin range (±42°) are dropped.
    """
    import tarfile

    from gazeos.vision.enhance import apply_clahe

    def decode(data: bytes, flags: int) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    limit = np.radians(max_abs_deg)
    outs = []
    for path in sorted(glob.glob(os.path.join(root, "s[0-9][0-9][0-9]_*.tar"))):
        name = Path(path).stem
        sid = RTGENE_SUBJECT_OFFSET + int(name[1:4])
        out = Path(out_dir) / f"rtgene_{name}.h5"
        dropped = 0
        with tarfile.open(path) as t:
            members = t.getmembers()
            label_member = next(m for m in members if m.name.endswith("/label_combined.txt"))
            labels = _rtgene_labels(t.extractfile(label_member).read().decode())  # type: ignore[union-attr]
            keep = set(sorted(labels)[::stride])
            # Eyes come before faces in the archives; whichever arrives first waits here.
            parts: dict[int, dict[str, bytes]] = {}
            w = ShardWriter(out)
            for m in members:
                pieces = m.name.split("/")
                if len(pieces) != 4 or pieces[1] != "inpainted" or pieces[2] not in ("face", "left", "right"):
                    continue
                fid = int(pieces[3].split("_")[1])
                if fid not in keep:
                    continue
                got = parts.setdefault(fid, {})
                got[pieces[2]] = t.extractfile(m).read()  # type: ignore[union-attr]
                if len(got) < 3:
                    continue
                del parts[fid]
                head_a, head_b, gaze_a, gaze_b = labels[fid]
                if abs(gaze_a) > limit or abs(gaze_b) > limit:
                    dropped += 1
                    continue
                face = cv2.cvtColor(decode(got["face"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                right = apply_clahe(decode(got["right"], cv2.IMREAD_GRAYSCALE))
                left = apply_clahe(np.ascontiguousarray(cv2.flip(decode(got["left"], cv2.IMREAD_GRAYSCALE), 1)))
                head = (RTGENE_HEAD_SIGN[0] * head_b, RTGENE_HEAD_SIGN[1] * head_a, 0.0)
                w.add(face, right, left, head, (gaze_b, gaze_a), sid)
        w.close({"dataset": "RT-GENE", "source": Path(path).name, "subject": sid, "stride": stride,
                 "dropped_out_of_range": dropped, "incomplete_frames": len(parts)})
        outs.append(out)
        log.info("%s: %d samples (%d out of range, %d without all three images)", name, w.n, dropped, len(parts))
    return outs


def pack_gazeos(root: str, out_dir: str) -> list[Path]:
    """GazeOS-Webcam: research recordings (normalized crops + targets) from tools/collect.py.

    The gaze label is the direction from the eye to the on-screen target,
    normalized with the stored rotation.
    """
    import json

    outs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
        data = np.load(path, allow_pickle=False)
        if "faces" not in data.files or "gaze_norm" not in data.files:
            log.warning("%s has no crops or labels; skipped", path)
            continue
        meta = json.loads(str(data["meta"]))
        sid = int(meta["participant"].lstrip("P") or 0)
        cond = int(meta.get("condition_code", 0))
        out = Path(out_dir) / f"gazeos_{Path(path).stem}.h5"
        w = ShardWriter(out)
        for k in range(len(data["faces"])):
            if not np.isfinite(data["gaze_norm"][k]).all():
                continue
            w.add(data["faces"][k], data["right_eyes"][k], data["left_eyes"][k], data["head_norm"][k],
                  data["gaze_norm"][k], sid, cond)
        w.close({"dataset": "GazeOS-Webcam", "source": Path(path).name, **meta})
        outs.append(out)
    return outs


def pack_synthetic(out_dir: str, subjects: int = 6, per_subject: int = 200, seed: int = 0) -> list[Path]:
    """Procedural faces whose eye appearance depends on gaze: for smoke tests and CI only."""
    rng = np.random.default_rng(seed)
    outs = []
    for sid in range(subjects):
        out = Path(out_dir) / f"synthetic_s{sid:02d}.h5"
        w = ShardWriter(out)
        skin = rng.uniform(90, 200, 3)
        kappa = rng.normal(0, np.radians(2.0), 2)
        for _ in range(per_subject):
            gaze = rng.uniform(-0.5, 0.5, 2)
            head = np.array([*rng.uniform(-0.3, 0.3, 2), rng.uniform(-0.1, 0.1)])
            face, right, left = synthetic_images(gaze - kappa, head, skin, rng)
            w.add(face, right, left, head, gaze, sid)
        w.close({"dataset": "synthetic", "subject": sid})
        outs.append(out)
    return outs


def synthetic_images(gaze: np.ndarray, head: np.ndarray, skin: np.ndarray, rng: np.random.Generator):  # type: ignore[no-untyped-def]
    face = np.empty((224, 224, 3), np.uint8)
    face[:] = np.clip(skin, 0, 255).astype(np.uint8)
    face = (face + rng.normal(0, 6, face.shape)).clip(0, 255).astype(np.uint8)
    eyes = []
    for side in (0, 1):
        e = np.full((36, 60), 230, np.uint8)
        dx = int(np.clip(-np.sin(gaze[1]) * 22 * (1 if side == 0 else -1), -20, 20))
        dy = int(np.clip(-np.sin(gaze[0]) * 14, -10, 10))
        cv2.circle(e, (30 + dx, 18 + dy), 9, 40, -1)
        cv2.circle(e, (30 + dx, 18 + dy), 4, 10, -1)
        e = (e + rng.normal(0, 8, e.shape)).clip(0, 255).astype(np.uint8)
        eyes.append(e)
    for (cx, cy), e in zip(((61, 102), (163, 102)), (eyes[0], cv2.flip(eyes[1], 1)), strict=True):
        face[cy - 18:cy + 18, cx - 30:cx + 30] = e[..., None]
    shift = int(head[1] * 30)
    face = np.roll(face, shift, axis=1)
    return face, eyes[0], eyes[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["mpiifacegaze", "ethxgaze", "rtgene", "gazeos", "synthetic"])
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()
    if a.dataset == "mpiifacegaze":
        outs = pack_mpiifacegaze(a.root, a.out)
    elif a.dataset == "ethxgaze":
        outs = pack_ethxgaze(a.root, a.out, a.stride)
    elif a.dataset == "rtgene":
        outs = pack_rtgene(a.root, a.out, a.stride)
    elif a.dataset == "gazeos":
        outs = pack_gazeos(a.root, a.out)
    else:
        outs = pack_synthetic(a.out)
    write_manifest(outs, Path(a.out) / "manifest.json", {"dataset": a.dataset, "stride": a.stride})


if __name__ == "__main__":
    main()
