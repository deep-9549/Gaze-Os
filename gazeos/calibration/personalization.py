"""Per-user personalization methods P0-P5 (PRD 11.4).

Every method fits on CalibrationSamples and maps a GazeObservation to a
corrected screen point in pixels. Methods are compared head to head in
experiment E6 (research/e6_personalization.py).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.calibration.protocol import CalibrationSample, GazeObservation, true_pitchyaw
from gazeos.vision.geometry import pitchyaw_to_vector

FloatArray = NDArray[np.float64]


class Personalizer(ABC):
    method_id: ClassVar[str] = ""

    def __init__(self, geometry: ScreenGeometry) -> None:
        self.geometry = geometry
        self.fitted = False

    @abstractmethod
    def fit(self, samples: list[CalibrationSample]) -> None: ...

    @abstractmethod
    def predict_px(self, obs: GazeObservation) -> FloatArray: ...

    def predict_sigma_px(self, obs: GazeObservation) -> float | None:
        """Optional per-prediction uncertainty in pixels."""
        return None

    # Serialization: a few kilobytes of JSON per profile (PRD 11.7).
    def state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        self.fitted = True

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"method": self.method_id, "state": self.state()}))

    def _ray_px(self, obs: GazeObservation, pitchyaw: FloatArray) -> FloatArray:
        px = self.geometry.gaze_to_px(obs.origin, pitchyaw_to_vector(pitchyaw))
        if px is None:
            return np.asarray(obs.screen_px, dtype=np.float64)
        return np.asarray(px, dtype=np.float64)


def _weights(samples: list[CalibrationSample]) -> FloatArray:
    return np.asarray([max(s.weight, 0.0) for s in samples], dtype=np.float64)


class NoPersonalization(Personalizer):
    """P0: the raw network."""

    method_id = "none"

    def fit(self, samples: list[CalibrationSample]) -> None:
        self.fitted = True

    def predict_px(self, obs: GazeObservation) -> FloatArray:
        return np.asarray(obs.screen_px, dtype=np.float64)


class AngularOffset(Personalizer):
    """P1: a fixed pitch and yaw bias, i.e. the personal visual-axis angle."""

    method_id = "offset"

    def __init__(self, geometry: ScreenGeometry) -> None:
        super().__init__(geometry)
        self.bias = np.zeros(2)

    def fit(self, samples: list[CalibrationSample]) -> None:
        if not samples:
            return
        w = _weights(samples)
        diffs = np.stack([true_pitchyaw(s, self.geometry) - s.obs.pitchyaw for s in samples])
        self.bias = np.average(diffs, axis=0, weights=w if w.sum() > 0 else None)
        self.fitted = True

    def predict_px(self, obs: GazeObservation) -> FloatArray:
        return self._ray_px(obs, obs.pitchyaw + self.bias)

    def state(self) -> dict[str, Any]:
        return {"bias": self.bias.tolist()}

    def load_state(self, state: dict[str, Any]) -> None:
        self.bias = np.asarray(state["bias"], dtype=np.float64)
        self.fitted = True


def _distinct_targets(samples: list[CalibrationSample]) -> int:
    return len({(round(float(s.target_px[0])), round(float(s.target_px[1]))) for s in samples})


def _poly2(xy: FloatArray) -> FloatArray:
    x, y = xy[:, 0], xy[:, 1]
    return np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], axis=1)


def _weighted_ridge(X: FloatArray, Y: FloatArray, w: FloatArray, alpha: float, prior: FloatArray | None = None) -> FloatArray:
    """argmin_B sum_i w_i |X_i B - Y_i|^2 + alpha |B - prior|^2 (intercept not penalized)."""
    d = X.shape[1]
    reg = alpha * np.eye(d)
    reg[0, 0] = 1e-9
    P = np.zeros((d, Y.shape[1])) if prior is None else prior
    Xw = X * w[:, None]
    A = X.T @ Xw + reg
    b = Xw.T @ Y + reg @ P
    return np.asarray(np.linalg.solve(A, b), dtype=np.float64)


class ScreenPolynomial(Personalizer):
    """P2: second-order polynomial from predicted (x, y) to true (x, y), ridge regularized.

    The regularizer pulls toward the identity map, so with few points the
    method degrades gracefully toward a translation/scale correction.
    """

    method_id = "polynomial"

    def __init__(self, geometry: ScreenGeometry, alpha: float = 1e-2) -> None:
        super().__init__(geometry)
        self.alpha = alpha
        self.coef = self._identity()

    def _identity(self) -> FloatArray:
        B = np.zeros((6, 2))
        B[1, 0] = 1.0
        B[2, 1] = 1.0
        return B

    def _norm(self, px: FloatArray) -> FloatArray:
        return np.atleast_2d(px) / np.array([self.geometry.width_px, self.geometry.height_px])

    def fit(self, samples: list[CalibrationSample]) -> None:
        if not samples:
            return
        X = _poly2(self._norm(np.stack([s.obs.screen_px for s in samples])))
        Y = self._norm(np.stack([s.target_px for s in samples]))
        w = _weights(samples)
        # Few distinct targets cannot constrain scale or curvature: pull hard toward
        # the identity so the fit degrades to a translation (an offset correction).
        distinct = _distinct_targets(samples)
        if distinct >= 9:
            alpha = self.alpha
        elif distinct >= 5:
            alpha = self.alpha * 30.0
        else:
            alpha = 1e4
        self.coef = _weighted_ridge(X, Y, w, alpha, prior=self._identity())
        self.fitted = True

    def predict_px(self, obs: GazeObservation) -> FloatArray:
        out = _poly2(self._norm(np.asarray(obs.screen_px, dtype=np.float64))) @ self.coef
        return np.asarray(out[0] * np.array([self.geometry.width_px, self.geometry.height_px]), dtype=np.float64)

    def state(self) -> dict[str, Any]:
        return {"coef": self.coef.tolist(), "alpha": self.alpha}

    def load_state(self, state: dict[str, Any]) -> None:
        self.coef = np.asarray(state["coef"], dtype=np.float64)
        self.alpha = float(state.get("alpha", self.alpha))
        self.fitted = True


def _gp_kernel() -> Any:
    """Smooth residual field over (screen x, y, head pitch, yaw) in normalized units.

    Head angles are scaled so 0.35 rad (20 degrees) is one unit. The length-scale
    floor (a quarter of the screen, wider than the 9-point grid spacing allows to
    resolve) and the noise floor (std >= 1% of screen width) stop the GP from
    interpolating per-frame jitter, which would make it worse than P2.
    """
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    return ConstantKernel(1e-3, (1e-6, 1e-1)) * RBF(
        length_scale=[0.4, 0.4, 1.0, 1.0], length_scale_bounds=(0.25, 100.0)
    ) + WhiteKernel(1e-3, (1e-4, 1e-1))


class GaussianProcessCorrection(Personalizer):
    """P3: GP regression of the residual error as a smooth function of the predicted
    point and head pose, on top of an affine screen fit. Also gives uncertainty."""

    method_id = "gp"
    max_samples = 400

    def __init__(self, geometry: ScreenGeometry) -> None:
        super().__init__(geometry)
        self.affine = ScreenPolynomial(geometry, alpha=1.0)
        self._gp: Any = None
        self._X: FloatArray | None = None
        self._Y: FloatArray | None = None

    def _inputs(self, obs: GazeObservation) -> FloatArray:
        n = np.asarray(obs.screen_px, dtype=np.float64) / np.array(
            [self.geometry.width_px, self.geometry.height_px]
        )
        return np.concatenate([n, np.asarray(obs.head_pyr[:2], dtype=np.float64) / 0.35])

    def fit(self, samples: list[CalibrationSample]) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor

        if not samples:
            return
        if len(samples) > self.max_samples:  # keep fitting under a second
            idx = np.linspace(0, len(samples) - 1, self.max_samples).astype(int)
            samples = [samples[i] for i in idx]
        self.affine.fit(samples)
        # Fit on one median point per fixation (same target, close in time). With
        # raw frames the inputs are as noisy as the residuals, and the GP learns to
        # "undo" per-frame noise locally, which extrapolates badly.
        X, base, targets, w = self._aggregate(samples)
        Y = (targets - base) / self.geometry.width_px
        kernel = _gp_kernel()
        alpha = 1e-6 * w.max() / np.clip(w, 1e-3, None)
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=False,
                                            n_restarts_optimizer=0, random_state=0)
        self._gp.fit(X, Y)
        self._X, self._Y = X, Y
        self.fitted = True

    def _aggregate(
        self, samples: list[CalibrationSample]
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        groups: dict[tuple[float, float, int], list[CalibrationSample]] = {}
        for smp in samples:
            key = (round(float(smp.target_px[0]), 1), round(float(smp.target_px[1]), 1), int(smp.t // 5.0))
            groups.setdefault(key, []).append(smp)
        X, base, targets, w = [], [], [], []
        for group in groups.values():
            X.append(np.median(np.stack([self._inputs(g.obs) for g in group]), axis=0))
            base.append(np.median(np.stack([self.affine.predict_px(g.obs) for g in group]), axis=0))
            targets.append(group[0].target_px)
            w.append(float(np.mean([g.weight for g in group])) * len(group))
        return np.stack(X), np.stack(base), np.stack(targets).astype(np.float64), np.asarray(w)

    def _predict(self, obs: GazeObservation) -> tuple[FloatArray, float]:
        base = self.affine.predict_px(obs)
        if self._gp is None:
            return base, 0.0
        mean, std = self._gp.predict(self._inputs(obs)[None], return_std=True)
        corr = np.asarray(mean[0], dtype=np.float64) * self.geometry.width_px
        s = float(np.mean(np.atleast_1d(std))) * self.geometry.width_px
        return base + corr, s

    def predict_px(self, obs: GazeObservation) -> FloatArray:
        return self._predict(obs)[0]

    def predict_sigma_px(self, obs: GazeObservation) -> float | None:
        return self._predict(obs)[1]

    def state(self) -> dict[str, Any]:
        if self._X is None or self._Y is None or self._gp is None:
            return {"affine": self.affine.state()}
        return {
            "affine": self.affine.state(),
            "X": self._X.tolist(),
            "Y": self._Y.tolist(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.affine.load_state(state["affine"])
        if "X" in state:
            from sklearn.gaussian_process import GaussianProcessRegressor

            X = np.asarray(state["X"], dtype=np.float64)
            Y = np.asarray(state["Y"], dtype=np.float64)
            self._gp = GaussianProcessRegressor(kernel=_gp_kernel(), alpha=1e-6, random_state=0).fit(X, Y)
            self._X, self._Y = X, Y
        self.fitted = True


class LastLayerFineTune(Personalizer):
    """P4: fine-tune an additive linear layer on the network's penultimate
    features with 100 steps of gradient descent (numpy, no PyTorch at runtime).

    With meta-learned weights (P5) the same adaptation starts from the
    meta-trained initialization passed as `init`.
    """

    method_id = "finetune"

    def __init__(
        self,
        geometry: ScreenGeometry,
        steps: int = 100,
        lr: float = 0.05,
        l2: float = 1e-2,
        init: tuple[FloatArray, FloatArray] | None = None,
        norm: tuple[FloatArray, FloatArray] | None = None,
    ) -> None:
        super().__init__(geometry)
        self.steps, self.lr, self.l2 = steps, lr, l2
        self.W: FloatArray | None = None
        self.b = np.zeros(2)
        self._init = init
        # A meta-learned init is only valid with the feature normalization it was trained with.
        self._fixed_norm = norm
        self._mu: FloatArray | None = None if norm is None else norm[0]
        self._sd: FloatArray | None = None if norm is None else norm[1]

    def _feat(self, obs: GazeObservation) -> FloatArray:
        if obs.features is None:
            raise ValueError("LastLayerFineTune needs network features in the observation")
        f = np.asarray(obs.features, dtype=np.float64)
        if self._mu is not None and self._sd is not None:
            f = (f - self._mu) / self._sd
        return f

    def fit(self, samples: list[CalibrationSample]) -> None:
        if not samples:
            return
        raw = np.stack([np.asarray(s.obs.features, dtype=np.float64) for s in samples])
        if self._fixed_norm is None:
            self._mu = raw.mean(axis=0)
            self._sd = raw.std(axis=0) + 1e-6
        else:
            self._mu, self._sd = self._fixed_norm
        F = (raw - self._mu) / self._sd
        target = np.stack([true_pitchyaw(s, self.geometry) - s.obs.pitchyaw for s in samples])
        w = _weights(samples)
        w = w / max(w.sum(), 1e-9)
        d = F.shape[1]
        # With fewer than 5 distinct targets only the bias (an angular offset) is
        # learnable; the feature weights would just fit per-frame noise.
        bias_only = _distinct_targets(samples) < 5 and self._init is None
        W = np.zeros((d, 2)) if self._init is None else self._init[0].copy()
        b = target.T @ w if self._init is None else self._init[1].copy()
        # Adam on the weighted squared error plus L2 toward zero correction.
        # Adam steps are about `lr` in size, so scale lr to the size of the targets.
        lr = self.lr * max(float(np.abs(target).mean()), 1e-3)
        mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros(2), np.zeros(2)
        b1, b2, eps = 0.9, 0.999, 1e-12
        for step in range(1, self.steps + 1):
            err = F @ W + b - target
            gW = 2 * F.T @ (err * w[:, None]) + 2 * self.l2 * W
            gb = 2 * err.T @ w
            mW = b1 * mW + (1 - b1) * gW
            vW = b2 * vW + (1 - b2) * gW**2
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb**2
            c1, c2 = 1 - b1**step, 1 - b2**step
            if not bias_only:
                W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
            b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        self.W, self.b = W, b
        self.fitted = True

    def predict_px(self, obs: GazeObservation) -> FloatArray:
        if self.W is None:
            return np.asarray(obs.screen_px, dtype=np.float64)
        delta = self._feat(obs) @ self.W + self.b
        return self._ray_px(obs, obs.pitchyaw + delta)

    def state(self) -> dict[str, Any]:
        if self.W is None or self._mu is None or self._sd is None:
            return {}
        return {"W": self.W.tolist(), "b": self.b.tolist(), "mu": self._mu.tolist(), "sd": self._sd.tolist()}

    def load_state(self, state: dict[str, Any]) -> None:
        if state:
            self.W = np.asarray(state["W"], dtype=np.float64)
            self.b = np.asarray(state["b"], dtype=np.float64)
            self._mu = np.asarray(state["mu"], dtype=np.float64)
            self._sd = np.asarray(state["sd"], dtype=np.float64)
        self.fitted = True


class MetaLearnedAdaptation(LastLayerFineTune):
    """P5: adaptation from a MAML-style meta-trained initialization
    (training/meta_train.py exports the init as an .npz with W and b)."""

    method_id = "meta"

    @classmethod
    def from_file(cls, geometry: ScreenGeometry, path: str | Path, steps: int = 20) -> MetaLearnedAdaptation:
        data = np.load(path)
        return cls(geometry, steps=steps, init=(data["W"], data["b"]), norm=(data["mu"], data["sd"]))


METHODS: dict[str, type[Personalizer]] = {
    "none": NoPersonalization,
    "offset": AngularOffset,
    "polynomial": ScreenPolynomial,
    "gp": GaussianProcessCorrection,
    "finetune": LastLayerFineTune,
    "meta": MetaLearnedAdaptation,
}


def make_personalizer(method: str, geometry: ScreenGeometry) -> Personalizer:
    try:
        return METHODS[method](geometry)
    except KeyError as exc:
        raise ValueError(f"unknown personalization method {method!r}") from exc


def load_personalizer(path: str | Path, geometry: ScreenGeometry) -> Personalizer:
    data = json.loads(Path(path).read_text())
    p = make_personalizer(data["method"], geometry)
    p.load_state(data["state"])
    return p
