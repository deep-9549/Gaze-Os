"""User profiles: settings, calibration, dictionary and phrases per person (FR-13, FR-35, PRD 11.7).

Everything lives in the local profile folder; raw camera frames are never
stored. The privacy panel lists exactly these files and can delete them all
(PRD 21.1).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gazeos.settings import Settings

_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,40}$")


def default_root() -> Path:
    env = os.environ.get("GAZEOS_HOME")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "GazeOS"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GazeOS"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "gazeos"


@dataclass
class Profile:
    name: str
    path: Path

    @property
    def settings_file(self) -> Path:
        return self.path / "settings.json"

    @property
    def calibration_file(self) -> Path:
        return self.path / "calibration.json"

    @property
    def calibration_features_file(self) -> Path:
        return self.path / "calibration_features.npz"

    @property
    def dictionary_file(self) -> Path:
        return self.path / "dictionary.txt"

    @property
    def phrases_file(self) -> Path:
        return self.path / "phrases.json"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def heatmaps_dir(self) -> Path:
        return self.path / "heatmaps"

    @property
    def recordings_dir(self) -> Path:
        return self.path / "recordings"

    def load_settings(self) -> Settings:
        if self.settings_file.exists():
            return Settings.model_validate_json(self.settings_file.read_text(encoding="utf-8"))
        return Settings()

    def save_settings(self, settings: Settings) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.settings_file.write_text(settings.model_dump_json(indent=2), encoding="utf-8")

    def has_calibration(self) -> bool:
        return self.calibration_file.exists()

    def save_calibration_features(self, **arrays: NDArray[np.float64]) -> None:
        """Calibration feature vectors (not images) as a NumPy file."""
        self.path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.calibration_features_file, **arrays)  # type: ignore[arg-type]

    def load_calibration_features(self) -> dict[str, NDArray[np.float64]]:
        if not self.calibration_features_file.exists():
            return {}
        with np.load(self.calibration_features_file) as data:
            return {k: np.asarray(data[k], dtype=np.float64) for k in data.files}

    def stored_items(self) -> list[tuple[str, Path, int]]:
        """What is stored, for the privacy panel: (description, path, bytes)."""
        labels = {
            "settings.json": "Settings",
            "calibration.json": "Calibration parameters",
            "calibration_features.npz": "Calibration feature vectors (no images)",
            "dictionary.txt": "Personal dictionary",
            "phrases.json": "Phrase bank",
            "logs": "Experiment logs (opt-in)",
            "heatmaps": "Gaze heatmaps (opt-in)",
            "recordings": "Research recordings: normalized face and eye crops (opt-in)",
        }
        out = []
        for name, label in labels.items():
            p = self.path / name
            if p.exists():
                size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                out.append((label, p, size))
        return out

    def delete_all(self) -> None:
        """'Delete all my data' (PRD 21.1)."""
        if self.path.exists():
            shutil.rmtree(self.path)


class ProfileStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else default_root()
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.root / "state.json"

    def list(self) -> list[str]:
        return sorted(p.name for p in self.profiles_dir.iterdir() if p.is_dir())

    def get(self, name: str) -> Profile:
        if not _NAME_RE.match(name):
            raise ValueError("profile names use letters, digits, spaces, _ and - (max 40)")
        return Profile(name, self.profiles_dir / name)

    def create(self, name: str, settings: Settings | None = None) -> Profile:
        p = self.get(name)
        if p.path.exists():
            raise FileExistsError(name)
        p.path.mkdir(parents=True)
        p.save_settings(settings or Settings())
        return p

    def get_or_create(self, name: str) -> Profile:
        p = self.get(name)
        if not p.path.exists():
            return self.create(name)
        return p

    def delete(self, name: str) -> None:
        self.get(name).delete_all()

    @property
    def last_used(self) -> str | None:
        if self._state_file.exists():
            name = json.loads(self._state_file.read_text()).get("last_profile")
            return name if name in self.list() else None
        return None

    @last_used.setter
    def last_used(self, name: str) -> None:
        self._state_file.write_text(json.dumps({"last_profile": name}))
