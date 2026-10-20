"""Shared key names used by every backend's hotkey()."""

from __future__ import annotations

# Windows virtual-key codes for named keys.
WINDOWS_VK: dict[str, int] = {
    "ctrl": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "win": 0x5B,
    "cmd": 0x5B,
    "enter": 0x0D,
    "esc": 0x1B,
    "tab": 0x09,
    "backspace": 0x08,
    "delete": 0x2E,
    "space": 0x20,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
}

# Standard shortcut panel actions (FR-26). Destructive ones need confirmation (PRD 21.3).
SHORTCUTS: dict[str, tuple[str, ...]] = {
    "copy": ("ctrl", "c"),
    "paste": ("ctrl", "v"),
    "cut": ("ctrl", "x"),
    "undo": ("ctrl", "z"),
    "select_all": ("ctrl", "a"),
    "switch_window": ("alt", "tab"),
    "close_window": ("alt", "f4"),
    "start_menu": ("win",),
    "new_tab": ("ctrl", "t"),
    "close_tab": ("ctrl", "w"),
}
MAC_SHORTCUTS: dict[str, tuple[str, ...]] = {
    "copy": ("cmd", "c"),
    "paste": ("cmd", "v"),
    "cut": ("cmd", "x"),
    "undo": ("cmd", "z"),
    "select_all": ("cmd", "a"),
    "switch_window": ("cmd", "tab"),
    "close_window": ("cmd", "q"),
    "start_menu": ("cmd", "space"),
    "new_tab": ("cmd", "t"),
    "close_tab": ("cmd", "w"),
}
DESTRUCTIVE_SHORTCUTS = frozenset({"close_window", "close_tab"})


def shortcut_keys(name: str, platform: str) -> tuple[str, ...]:
    table = MAC_SHORTCUTS if platform == "darwin" else SHORTCUTS
    return table[name]
