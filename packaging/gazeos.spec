# PyInstaller spec for the Windows build (PRD 19, 22). Run from the repo root:
#   pyinstaller packaging/gazeos.spec --noconfirm
# Training code and PyTorch are excluded: the app ships with ONNX Runtime only.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH).parent  # noqa: F821 - provided by PyInstaller
models = [(str(p), "models") for p in (root / "models").glob("*") if p.suffix in {".onnx", ".task", ".txt", ".npz"}]

a = Analysis(  # noqa: F821
    [str(root / "gazeos" / "main.py")],
    pathex=[str(root)],
    binaries=collect_dynamic_libs("onnxruntime"),
    datas=collect_data_files("mediapipe") + collect_data_files("gazeos") + models,
    hiddenimports=["gazeos.os_backends.windows", "gazeos.os_backends.pynput_backend", "uiautomation", "pyttsx3.drivers.sapi5"],
    excludes=["torch", "torchvision", "training", "research", "matplotlib", "hydra", "tensorboard"],
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz, a.scripts, [], exclude_binaries=True, name="GazeOS", console=False, uac_admin=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="GazeOS")  # noqa: F821
