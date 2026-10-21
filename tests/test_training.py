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
