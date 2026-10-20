"""Element under a screen point on Linux through AT-SPI (PRD 20.4)."""

from __future__ import annotations

from gazeos.types import Rect, UIElement

_INVOKABLE_ROLES = {"push button", "link", "menu item", "check box", "radio button", "text", "entry", "toggle button", "page tab"}


def element_at(x: int, y: int) -> UIElement | None:
    try:
        import pyatspi  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        desktop = pyatspi.Registry.getDesktop(0)
        for app in desktop:
            if app is None:
                continue
            for win in app:
                if win is None:
                    continue
                comp = win.queryComponent()
                acc = comp.getAccessibleAtPoint(x, y, pyatspi.DESKTOP_COORDS)
                if acc is None:
                    continue
                ext = acc.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                role = acc.getRoleName()
                return UIElement(
                    rect=Rect(ext.x, ext.y, ext.width, ext.height),
                    role=role,
                    name=acc.name or "",
                    invokable=role in _INVOKABLE_ROLES,
                    is_password=role == "password text",
                )
    except Exception:
        return None
    return None
