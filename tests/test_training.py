from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from training.data.augment import GazeAugment  # noqa: E402
from training.data.packers import pack_synthetic  # noqa: E402
from training.data.shards import GazeShardDataset  # noqa: E402
from training.losses.gaze import (  # noqa: E402
    angle_to_bin,
    angular_error_deg,
    distillation_loss,
    gaussian_nll,
    l2cs_loss,
)
from training.models.gazenet import GazeNetGOS, bin_centers_deg, gazenet_m2  # noqa: E402
from training.models.gesture import GRUClassifier, TemporalCNN, count_parameters  # noqa: E402


def _inputs(b: int = 2):
    return torch.randn(b, 3, 224, 224), torch.randn(b, 1, 36, 60), torch.randn(b, 1, 36, 60), torch.randn(b, 3)


@pytest.mark.parametrize("backbone", ["resnet18", "mobilenet_v3_small"])
def test_gazenet_shapes(backbone: str) -> None:
    m = GazeNetGOS(backbone, pretrained=False).eval()
    out = m(*_inputs())
    assert out.pitch_logits.shape == (2, 28) and out.log_var.shape == (2, 2)
    assert out.features.shape == (2, 256) and out.attention is not None and out.attention.shape == (2, 4, 4)
    ang = out.angles_deg(bin_centers_deg())
    assert ang.shape == (2, 2) and bool((ang.abs() <= 42).all())


def test_face_only_ablation() -> None:
    m = gazenet_m2(pretrained=False).eval()
    assert m.token_names == ["face"] and m(*_inputs()).attention is None


def test_bins_and_losses() -> None:
    c = bin_centers_deg()
    assert float(c[0]) == -40.5 and float(c[-1]) == 40.5
    assert angle_to_bin(torch.tensor([-42.0, 0.0, 41.9, 99.0]), 28).tolist() == [0, 14, 27, 27]
    m = GazeNetGOS("mobilenet_v3_small", pretrained=False)
    out = m(*_inputs())
    target = torch.tensor([[5.0, -10.0], [0.0, 20.0]])
    loss, pred = l2cs_loss(out, target, c)
    assert loss.item() > 0 and pred.shape == (2, 2)
    loss.backward()
    assert gaussian_nll(pred.detach(), target, torch.zeros(2, 2)).item() > 0
    assert distillation_loss(out, out, torch.tensor(1.0)).item() == pytest.approx(0.5, abs=1e-5)
    e = angular_error_deg(torch.tensor([[0.0, 10.0]]), torch.tensor([[0.0, 0.0]]))
    assert e.item() == pytest.approx(10.0, abs=1e-4)


def test_gesture_models() -> None:
    x = torch.randn(4, 30, 17)
    tcn = TemporalCNN()
    assert tcn(x).shape == (4, 9)
    assert 20_000 < count_parameters(tcn) < 45_000  # "about 30,000 parameters"
    assert GRUClassifier()(x).shape == (4, 9)


def test_shards_and_augmentation(tmp_path) -> None:
    paths = pack_synthetic(str(tmp_path), subjects=2, per_subject=10)
    ds = GazeShardDataset(paths, subjects=[1])
    assert len(ds) == 10 and ds.subjects() == [0, 1]
    item = ds[0]
    assert item["face"].shape == (3, 224, 224) and item["right_eye"].shape == (1, 36, 60)
    raw = ds.raw(0)
    aug = GazeAugment(p_flip=1.0, seed=0)(raw)
    assert aug["gaze"][1] == pytest.approx(-raw["gaze"][1])
    assert aug["face"].shape == (224, 224, 3)
    # With every photometric/geometric step off, a flip is exactly an eye swap.
    plain = GazeAugment(p_color=0, p_gamma=0, p_blur=0, p_motion=0, p_jpeg=0, p_glare=0, max_shift=0, p_flip=1.0)
    flipped = plain(raw)
    assert np.array_equal(flipped["right_eye"], raw["left_eye"]) and np.array_equal(flipped["left_eye"], raw["right_eye"])
    assert np.array_equal(flipped["face"], raw["face"][:, ::-1])


def test_image_chunks_hold_one_sample_and_rechunk_keeps_data(tmp_path) -> None:
    import h5py

    from training.data.shards import rechunk

    (path,) = pack_synthetic(str(tmp_path / "s"), subjects=1, per_subject=70)
    old = tmp_path / "old.h5"
    with h5py.File(path, "r") as a, h5py.File(old, "w") as b:  # the previous 64-sample layout
        assert a["face"].chunks[0] == 1 and a["gaze"].chunks[0] == 64
        for k in a:
            b.create_dataset(k, data=a[k][:], chunks=(64, *a[k].shape[1:]) if a[k].ndim > 1 else None)
    new = rechunk(old, tmp_path / "new.h5")
    with h5py.File(old, "r") as a, h5py.File(new, "r") as b:
        assert b["face"].chunks[0] == 1 and b["left_eye"].compression == "lzf"
        assert all(np.array_equal(a[k][:], b[k][:]) for k in a)


def test_export_parity_and_runtime(tmp_path) -> None:
    """Exported model loads in the runtime estimator and matches PyTorch (PRD 10.8, 23)."""
    from gazeos.inference.gaze import OnnxGazeEstimator
    from gazeos.vision.normalization import NormalizationResult
    from training.export_onnx import export, parity

    m = GazeNetGOS("mobilenet_v3_small", pretrained=False).eval()
    path = export(m, tmp_path / "m.onnx")
    rep = parity(m, path, None, n=50)
    assert rep["mean_abs_diff_deg"] < 0.05
    est = OnnxGazeEstimator(path)
    rng = np.random.default_rng(0)
    norm = NormalizationResult(
        face=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        right_eye=rng.integers(0, 255, (36, 60), dtype=np.uint8),
        left_eye=rng.integers(0, 255, (36, 60), dtype=np.uint8),
        R_n=np.eye(3), S=np.eye(3), W=np.eye(3), head_pyr=(0.0, 0.1, 0.0),
    )
    g = est.estimate(norm)
    assert g.pitchyaw.shape == (2,) and g.features is not None and g.features.shape == (256,)
    assert g.attention is not None and np.all(g.sigma > 0)


def test_l2cs_conversion_contract(tmp_path) -> None:
    from gazeos.inference.gaze import OnnxGazeEstimator
    from training.convert_l2cs import L2CS, convert

    ckpt = tmp_path / "l2cs.pkl"
    torch.save(L2CS(28).state_dict(), ckpt)
    out = str(tmp_path / "l2cs.onnx")
    convert(str(ckpt), out, "mpiigaze")
    est = OnnxGazeEstimator(out)
    assert est.face_size == 448 and np.isclose(est.centers[0], -42.0) and len(est.centers) == 28


def _fake_rtgene_tar(path, frames: dict[int, tuple[float, float, float, float]], missing_left: set[int]) -> None:
    """A tiny RT-GENE-style archive: images first, labels last, plus files the packer must ignore."""
    import io
    import tarfile

    import cv2

    def add(t, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))

    def png(img) -> bytes:
        return cv2.imencode(".png", img)[1].tobytes()

    rng = np.random.default_rng(0)
    root = path.stem
    with tarfile.open(path, "w") as t:
        for side in ("left", "right"):
            for fid in frames:
                if side == "left" and fid in missing_left:
                    continue
                eye = rng.integers(0, 255, (36, 60, 3), dtype=np.uint8)
                eye[:, :30] //= 4  # asymmetric, so a mirror is detectable
                add(t, f"{root}/inpainted/{side}/{side}_{fid:06d}_rgb.png", png(eye))
        for fid in frames:
            add(t, f"{root}/original/face/face_{fid:06d}_rgb.png", png(np.zeros((224, 224, 3), np.uint8)))
            add(t, f"{root}/inpainted/face/face_{fid:06d}_rgb.png",
                png(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)))
        lines = [f"{fid}, [{v[0]}, {v[1]}], [{v[2]}, {v[3]}], 1502457728.46" for fid, v in frames.items()]
        add(t, f"{root}/label_combined.txt", "\n".join(lines).encode())


def test_rtgene_packer(tmp_path) -> None:
    import tarfile

    import cv2

    from gazeos.vision.enhance import apply_clahe
    from training.data.packers import pack_rtgene

    frames = {2: (0.1, -0.3, -0.2, 0.05), 3: (0.0, 0.0, 1.0, 0.0), 4: (0.0, 0.0, 0.1, 0.1), 7: (-0.2, 0.1, 0.3, -0.25)}
    raw = tmp_path / "raw"
    raw.mkdir()
    tar = raw / "s012_glasses.tar"
    _fake_rtgene_tar(tar, frames, missing_left={4})
    paths = pack_rtgene(str(raw), str(tmp_path / "shards"))
    ds = GazeShardDataset(paths)
    assert len(ds) == 2 and ds.subjects() == [2012]  # frame 3 out of range, frame 4 incomplete
    by_gaze = {round(float(ds.raw(i)["gaze"][0]), 3): ds.raw(i) for i in range(2)}
    s = by_gaze[0.05]
    # Labels are [yaw, pitch]: shards store (pitch, yaw) for gaze and (pitch, yaw, 0) for head.
    assert np.allclose(s["gaze"], [0.05, -0.2]) and np.allclose(s["head"], [-0.3, 0.1, 0.0])
    with tarfile.open(tar) as t:
        left = t.extractfile("s012_glasses/inpainted/left/left_000002_rgb.png").read()  # type: ignore[union-attr]
    expected = apply_clahe(np.ascontiguousarray(cv2.flip(cv2.imdecode(np.frombuffer(left, np.uint8), 0), 1)))
    assert np.array_equal(s["left_eye"], expected)  # the subject's left eye, mirrored


def test_mpiifacegaze_packer_reads_real_line_format(tmp_path) -> None:
    """Real pXX.txt lines have 28 columns; the last one is the word "left" or "right"."""
    import cv2
    import scipy.io

    from training.data.packers import pack_mpiifacegaze

    subj = tmp_path / "raw" / "p03"
    (subj / "Calibration").mkdir(parents=True)
    (subj / "day01").mkdir()
    K = np.array([[960.0, 0, 320], [0, 960.0, 240], [0, 0, 1]])
    scipy.io.savemat(subj / "Calibration" / "Camera.mat", {"cameraMatrix": K, "distCoeffs": np.zeros((1, 5))})
    cv2.imwrite(str(subj / "day01" / "0001.jpg"), np.full((480, 640, 3), 128, np.uint8))
    landmarks = [260, 220, 300, 220, 340, 220, 380, 220, 290, 320, 350, 320]
    face_centre, target = [0.0, 0.0, 600.0], [0.0, -100.0, 0.0]  # target above the face: looking up
    cols = ["day01/0001.jpg", 800, 400, *landmarks, 0, 0, 0, 0, 0, 600, *face_centre, *target, "right"]
    (subj / "p03.txt").write_text(" ".join(str(c) for c in cols) + "\n")
    ds = GazeShardDataset(pack_mpiifacegaze(str(tmp_path / "raw"), str(tmp_path / "shards")))
    assert len(ds) == 1 and ds.subjects() == [3]
    pitch, yaw = ds.raw(0)["gaze"]
    assert pitch > 0.1 and abs(yaw) < 1e-3


def test_face_stream_init_from_safetensors(tmp_path) -> None:
    import torchvision
    from safetensors.torch import save_file

    from training.train import load_face_weights

    eth = torchvision.models.resnet18(weights=None)
    eth.fc = torch.nn.Linear(512, 2)  # the ETH-XGaze model's pitch/yaw output, which is dropped
    path = str(tmp_path / "eth.safetensors")
    save_file({k: v.contiguous() for k, v in eth.state_dict().items()}, path)
    m = GazeNetGOS("resnet18", pretrained=False)
    load_face_weights(m, path)
    assert torch.equal(m.face.conv1.weight, eth.conv1.weight)
    with pytest.raises(ValueError):
        load_face_weights(GazeNetGOS("mobilenet_v3_small", pretrained=False), path)


def test_eval_onnx_reports_per_subject_error(tmp_path) -> None:
    from training.eval_onnx import evaluate_onnx
    from training.export_onnx import export

    pack_synthetic(str(tmp_path / "s"), subjects=2, per_subject=6)
    path = export(GazeNetGOS("mobilenet_v3_small", pretrained=False).eval(), tmp_path / "m.onnx")
    r = evaluate_onnx(str(path), str(tmp_path / "s" / "*.h5"), every=2, threads=1)
    assert r["n"] == 6 and set(r["per_subject_deg"]) == {0, 1} and r["mean_of_subjects_deg"] > 0


def test_l2cs_conversion_from_safetensors(tmp_path) -> None:
    from safetensors.torch import save_file

    from gazeos.inference.gaze import OnnxGazeEstimator
    from training.convert_l2cs import L2CS, convert

    ckpt = str(tmp_path / "l2cs.safetensors")
    save_file({k: v.contiguous() for k, v in L2CS(90).state_dict().items()}, ckpt)
    out = str(tmp_path / "l2cs.onnx")
    convert(ckpt, out, "gaze360")
    assert len(OnnxGazeEstimator(out).centers) == 90


def test_train_resumes_and_reads_absolute_shard_paths(tmp_path) -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from training.train import train

    pack_synthetic(str(tmp_path / "shards"), subjects=3, per_subject=8)
    overrides = ["experiment=smoke_synthetic", f"data.shards=[{(tmp_path / 'shards').as_posix()}/*.h5]",
                 f"output_dir={(tmp_path / 'run').as_posix()}", "data.val_fraction=0.5", "train.batch_size=4"]
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        first = train(compose("train", overrides=[*overrides, "train.epochs=1"]))
        resumed = train(compose("train", overrides=[*overrides, "train.epochs=2", "train.resume=true"]))
    assert [h["epoch"] for h in first["history"]] == [0]
    assert [h["epoch"] for h in resumed["history"]] == [0, 1]  # epoch 0 was not repeated


def test_gesture_training_smoke(tmp_path) -> None:
    from gazeos.inference.gesture import OnnxGestureClassifier
    from training.train_gesture import export, synthetic_sessions, train

    model, report, mean, std = train(synthetic_sessions(3), "tcn", epochs=2)
    path = export(model, tmp_path / "g.onnx", mean, std)
    probs = OnnxGestureClassifier(path).predict(np.zeros((1, 30, 17), np.float32))
    assert abs(sum(probs.values()) - 1.0) < 1e-5 and set(probs) >= {"none", "long_blink"}


def test_meta_train_produces_runtime_init(tmp_path, geometry) -> None:
    from gazeos.calibration.personalization import MetaLearnedAdaptation
    from training.meta_train import meta_train

    rng = np.random.default_rng(0)
    data = {s: (rng.normal(size=(40, 8)), rng.normal(0, 0.05, (40, 2)), rng.normal(0, 0.05, (40, 2))) for s in range(3)}
    init = meta_train(data, k_shot=9, episodes=20)
    np.savez(tmp_path / "meta.npz", **init)
    p = MetaLearnedAdaptation.from_file(geometry, tmp_path / "meta.npz")
    assert p._mu is not None and p._mu.shape == (8,)


def test_small_gpt_exports_dynamic_length(tmp_path) -> None:
    import onnxruntime as ort

    from training.models.gpt import SmallGPT

    m = SmallGPT(50, dim=32, layers=1, heads=2, context=16).eval()
    path = str(tmp_path / "lm.onnx")
    torch.onnx.export(m, torch.zeros(1, 8, dtype=torch.long), path, input_names=["tokens"], output_names=["logits"],
                      dynamic_axes={"tokens": {1: "T"}, "logits": {1: "T"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(path)
    for T in (3, 11):
        out = sess.run(None, {"tokens": np.zeros((1, T), np.int64)})[0]
        assert out.shape == (1, T, 50)


def test_fit_uncertainty_changes_only_the_variance_head(tmp_path) -> None:
    from training.fit_uncertainty import fit, fit_head, log_var

    rng = np.random.default_rng(0)
    feats = rng.normal(size=(2000, 8)).astype(np.float32)
    sigma = np.exp(0.5 * feats[:, :1])  # the error scale depends on the first feature
    err = rng.normal(size=(2000, 2)) * sigma
    w, b = fit_head(feats, err, steps=300)
    from scipy.stats import spearmanr

    assert spearmanr(log_var(feats, w, b)[:, 0], np.abs(err[:, 0])).correlation > 0.3

    pack_synthetic(str(tmp_path / "s"), subjects=4, per_subject=6)
    model = GazeNetGOS("mobilenet_v3_small", pretrained=False).eval()
    cfg = {"model": {"backbone": "mobilenet_v3_small", "use_eyes": True, "use_head": True, "use_face": True,
                     "fusion": "transformer", "num_bins": 28, "dropout": 0.3},
           "data": {"shards": [str(tmp_path / "s" / "*.h5")], "fold": None, "val_fraction": 0.25}, "seed": 0}
    torch.save({"model": model.state_dict(), "config": cfg}, tmp_path / "best.pt")
    report = fit(tmp_path / "best.pt", tmp_path / "best_unc.pt", workers=0)
    assert report["calibrated_on"] == "val" and report["val"]["n"] == 6
    assert abs(np.mean(report["val"]["mean_z2_after"]) - 1.0) < 0.05  # sigma calibrated on validation subjects
    new = torch.load(tmp_path / "best_unc.pt")["model"]
    old = model.state_dict()
    assert all(torch.equal(new[k], old[k]) for k in old if not k.startswith("var_head"))
    assert not torch.equal(new["var_head.weight"], old["var_head.weight"])


def test_camera_calibration_recovers_focal_length(tmp_path) -> None:
    """Views of the generated board through a known camera give back its focal length."""
    import cv2

    from tools.camera_calibration import calibrate, is_new_view, make_board

    board = cv2.imread(str(make_board(tmp_path / "b.png", 9, 6, square_px=40, margin_px=40)), cv2.IMREAD_GRAYSCALE)
    bh, bw = board.shape
    K = np.array([[520.0, 0, 320], [0, 520.0, 240], [0, 0, 1]])
    rng = np.random.default_rng(0)
    views = []
    for _ in range(12):
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(420, 600)])
        obj = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], np.float64) * 0.5 - [bw / 4, bh / 4, 0]
        img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]]), img_pts.reshape(4, 2).astype(np.float32))
        views.append(cv2.warpPerspective(board, H, (640, 480), borderValue=255))
    intr, rms = calibrate(views, (9, 6), 20.0)
    assert abs(intr.matrix[0, 0] - 520) < 15 and rms < 1.0
    corners = np.zeros((54, 1, 2), np.float32)
    assert is_new_view(corners, [], 640) and not is_new_view(corners, [corners + 5], 640)


def test_consistency_training_sees_two_views_of_each_sample(tmp_path) -> None:
    """train.consistency > 0 adds a second, slightly perturbed view (noise, JPEG, a 1-2 px
    shift) of every training sample and penalizes different answers for the two."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from training.data.augment import GazeAugment, TwoViews
    from training.data.shards import GazeShardDataset
    from training.train import train

    pack_synthetic(str(tmp_path / "shards"), subjects=3, per_subject=8)
    ds = GazeShardDataset(sorted(str(p) for p in (tmp_path / "shards").glob("*.h5")),
                          transform=TwoViews(GazeAugment(seed=0, p_noise=1.0), seed=0))
    item = ds[0]
    assert item["face2"].shape == item["face"].shape and not torch.equal(item["face2"], item["face"])
    overrides = ["experiment=smoke_synthetic", f"data.shards=[{(tmp_path / 'shards').as_posix()}/*.h5]",
                 f"output_dir={(tmp_path / 'run').as_posix()}", "data.val_fraction=0.5", "train.batch_size=4",
                 "train.epochs=1", "train.consistency=1.0", "data.sensor_noise_p=0.5"]
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        result = train(compose("train", overrides=overrides))
    assert np.isfinite(result["history"][0]["train_loss"])


def test_uncertainty_fit_survives_a_nearly_constant_feature() -> None:
    """A dead unit (near-constant feature) that varies a little on new samples must not
    push the fitted log-variance to its clamp (E12 fold 2: median sigma 56 deg)."""
    from training.fit_uncertainty import fit_head, log_var

    rng = np.random.default_rng(0)
    feats = rng.normal(size=(2000, 16)).astype(np.float32)
    feats[:, 3] = 1e-7 * rng.normal(size=2000)
    err = np.abs(rng.normal(0, 3, (2000, 2))) * (1 + np.abs(feats[:, :1]))
    w, b = fit_head(feats, err)
    test = rng.normal(size=(500, 16)).astype(np.float32)
    test[:, 3] = 0.05 * rng.normal(size=500)  # the dead unit wakes up a little
    sigma_deg = np.degrees(np.sqrt(np.exp(log_var(test, w, b))))
    assert np.median(sigma_deg) < 20
    assert sigma_deg.min() >= 0.49 and sigma_deg.max() <= 58  # the plausible range


def test_ablation_starts_from_a_checkpoint_with_other_shapes(tmp_path) -> None:
    """E2 arms (concat fusion, fewer streams) init from the full model's Stage A checkpoint:
    the reshaped layers start fresh instead of failing the load."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from training.train import train

    pack_synthetic(str(tmp_path / "shards"), subjects=3, per_subject=8)
    base = ["experiment=smoke_synthetic", f"data.shards=[{(tmp_path / 'shards').as_posix()}/*.h5]",
            "data.val_fraction=0.5", "train.batch_size=4", "train.epochs=1"]
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        train(compose("train", overrides=[*base, f"output_dir={(tmp_path / 'a').as_posix()}"]))
        init = f"init_checkpoint={(tmp_path / 'a' / 'best.pt').as_posix()}"
        for i, arm in enumerate(["model.fusion=concat", "model.use_face=false"]):
            result = train(compose("train", overrides=[*base, init, arm, f"output_dir={(tmp_path / str(i)).as_posix()}"]))
            assert np.isfinite(result["history"][0]["train_loss"])
