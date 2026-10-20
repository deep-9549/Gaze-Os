"""Start GazeOS with the operating system (FR-37)."""

from __future__ import annotations

import contextlib
import importlib
import os
import plistlib
import shlex
import sys
from pathlib import Path
from typing import Any

APP_ID = "GazeOS"


def _winreg() -> Any:
    # Imported dynamically: the module exists only on Windows.
    return importlib.import_module("winreg")


def launch_command() -> list[str]:
    if getattr(sys, "frozen", False):  # PyInstaller build
        return [sys.executable]
    return [sys.executable, "-m", "gazeos.main"]


def _linux_file() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "autostart" / "gazeos.desktop"


def _mac_file() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "org.gazeos.app.plist"


def set_enabled(enabled: bool, platform: str = sys.platform) -> None:
    cmd = launch_command()
    if platform == "win32":
        winreg = _winreg()
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                             winreg.KEY_SET_VALUE)
        try:
            if enabled:
                winreg.SetValueEx(key, APP_ID, 0, winreg.REG_SZ, " ".join(f'"{c}"' for c in cmd))
            else:
                with contextlib.suppress(FileNotFoundError):
                    winreg.DeleteValue(key, APP_ID)
        finally:
            winreg.CloseKey(key)
    elif platform == "darwin":
        f = _mac_file()
        if enabled:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(plistlib.dumps({"Label": "org.gazeos.app", "ProgramArguments": cmd, "RunAtLoad": True}))
        else:
            f.unlink(missing_ok=True)
    else:
        f = _linux_file()
        if enabled:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(
                "[Desktop Entry]\nType=Application\nName=GazeOS\n"
                f"Exec={' '.join(shlex.quote(c) for c in cmd)}\nX-GNOME-Autostart-enabled=true\n"
            )
        else:
            f.unlink(missing_ok=True)


def is_enabled(platform: str = sys.platform) -> bool:
    if platform == "win32":
        winreg = _winreg()
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run")
            winreg.QueryValueEx(key, APP_ID)
            return True
        except FileNotFoundError:
            return False
    if platform == "darwin":
        return _mac_file().exists()
    return _linux_file().exists()
