from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gazeos.calibration.geometry import ScreenGeometry
from gazeos.settings import Settings
from gazeos.sim import SimulatedUser, SyntheticFace
from gazeos.vision.head_pose import CameraIntrinsics


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def geometry(settings: Settings) -> ScreenGeometry:
    return ScreenGeometry.from_settings(settings.screen)


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.default(640, 480)


@pytest.fixture
def user(geometry: ScreenGeometry, intrinsics: CameraIntrinsics) -> SimulatedUser:
    return SimulatedUser(geometry, SyntheticFace(intrinsics, noise_px=0.0))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
