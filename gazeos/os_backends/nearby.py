"""Finding the clickable elements near a gaze point through an accessibility tree
(Windows UI Automation; duck-typed, so it can be tested without Windows)."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from gazeos.types import UIElement


def nearby_elements(
    from_point: Callable[[int, int], Any], to_element: Callable[[Any], UIElement | None], x: int, y: int,
    radius: int, parent_levels: int = 2, max_nodes: int = 400,
) -> list[UIElement]:
    """Invokable elements within `radius` of (x, y). UIA has no region query: a few probe
    points find the controls around the gaze, then their containers are searched, pruned to
    the region. Probes alone (every radius/2) missed every button under 48 px (28 Oct)."""
    roots: list[Any] = []
    for dx, dy in [(0, 0)] + [(round(f * radius * c), round(f * radius * s)) for f in (0.5, 1.0)
                              for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1))]:
        try:
            ctrl = from_point(x + dx, y + dy)
            for _ in range(parent_levels):
                parent = ctrl.GetParentControl() if ctrl is not None else None
                if parent is None:
                    break
                ctrl = parent
        except Exception:  # UIA calls can fail for transient elements
            continue
        if ctrl is not None:
            roots.append(ctrl)
    found: dict[tuple[int, int, int, int, str], UIElement] = {}
    seen: set[tuple[int, int, int, int, str, str]] = set()
    queue, visited = list(roots), 0
    while queue and visited < max_nodes:
        ctrl = queue.pop(0)
        visited += 1
        try:
            r = ctrl.BoundingRectangle
            key = (int(r.left), int(r.top), int(r.right), int(r.bottom), str(ctrl.ControlTypeName), str(ctrl.Name or ""))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        # Distance from the gaze to the rectangle (0 inside it).
        ddx = max(key[0] - x, 0, x - key[2])
        ddy = max(key[1] - y, 0, y - key[3])
        if ddx * ddx + ddy * ddy > radius * radius:
            continue  # children lie inside their parent: skip the whole branch
        e = to_element(ctrl)
        if e is not None and e.invokable:
            found[(e.rect.left, e.rect.top, e.rect.width, e.rect.height, e.name)] = e
        with contextlib.suppress(Exception):
            queue.extend(ctrl.GetChildren())
    return list(found.values())
