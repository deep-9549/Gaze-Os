"""Camera capture into a one-slot buffer (PRD 8.1, FR-01, FR-02).

The capture thread always overwrites the slot, so processing gets the newest
frame and frames are dropped rather than queued when the CPU is busy (PRD 18.3).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image_rgb: NDArray[np.uint8]
    t_capture: float  # time.monotonic()
    index: int


class FrameSource(Protocol):
    def read(self) -> NDArray[np.uint8] | None: ...
    def release(self) -> None: ...


def _backends() -> tuple[int, ...]:
    """Capture backends to try, in order. On some Windows laptops Media Foundation
    cannot open the built-in webcam at all while DirectShow can."""
    if sys.platform == "win32":
        return (cv2.CAP_MSMF, cv2.CAP_DSHOW)
    if sys.platform == "darwin":
        return (cv2.CAP_AVFOUNDATION,)
    return (cv2.CAP_V4L2,)


def _is_dshow(cap: cv2.VideoCapture) -> bool:
    try:
        return bool(cap.getBackendName() == "DSHOW")
    except cv2.error:  # not opened
        return False


def _open_working(index: int, configure=lambda cap: None) -> cv2.VideoCapture | None:  # type: ignore[no-untyped-def]
    """First backend that opens camera `index` and returns a frame, or None."""
    for backend in _backends():
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            configure(cap)
            if cap.read()[0]:
                return cap
        cap.release()
    return None


def list_cameras(max_index: int = 6) -> list[int]:
    """Indices of cameras that open and return a frame (FR-01)."""
    found = []
    for i in range(max_index):
        cap = _open_working(i)
        if cap is not None:
            found.append(i)
            cap.release()
    return found


class OpenCVSource:
    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.cap = self._open()

    def _configure(self, cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if _is_dshow(cap):
            # DirectShow can start with auto exposure off at a fixed, dark setting.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    def _open(self) -> cv2.VideoCapture:
        cap = _open_working(self.index, self._configure)
        if cap is None:  # keep a closed capture; CaptureThread retries via reopen()
            cap = cv2.VideoCapture()
        return cap

    def lock_exposure(self) -> None:
        """Freeze auto exposure at its current value (FR-02)."""
        if _is_dshow(self.cap):
            # DirectShow reports the last manual value, not the one auto exposure chose,
            # so freezing would jump back to a dark image. Keep auto exposure instead.
            log.info("exposure lock not supported with DirectShow; keeping auto exposure")
            return
        exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        # 0.25 means "manual" on V4L2 / DirectShow; 1 on some MSMF drivers.
        for manual in (0.25, 1.0):
            if self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual):
                break
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def read(self) -> NDArray[np.uint8] | None:
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    def reopen(self) -> bool:
        self.cap.release()
        self.cap = self._open()
        return bool(self.cap.isOpened())

    def release(self) -> None:
        self.cap.release()


class CaptureThread(threading.Thread):
    """Reads frames continuously into a one-slot buffer and reconnects on failure."""

    def __init__(self, source: FrameSource, reconnect_delay_s: float = 1.0) -> None:
        super().__init__(name="gazeos-capture", daemon=True)
        self.source = source
        self.reconnect_delay_s = reconnect_delay_s
        self._slot: Frame | None = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._index = 0
        self.dropped = 0
        self.connected = True

    def run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            image = self.source.read()
            if image is None:
                failures += 1
                if failures >= 5:
                    self.connected = False
                    log.warning("camera read failed; reconnecting")
                    time.sleep(self.reconnect_delay_s)
                    reopen = getattr(self.source, "reopen", None)
                    if reopen is not None and reopen():
                        failures = 0
                else:
                    time.sleep(0.005)
                continue
            failures = 0
            self.connected = True
            frame = Frame(image_rgb=image, t_capture=time.monotonic(), index=self._index)
            self._index += 1
            with self._cond:
                if self._slot is not None:
                    self.dropped += 1
                self._slot = frame
                self._cond.notify()

    def latest(self, timeout: float = 0.5) -> Frame | None:
        """Take the newest frame, waiting up to `timeout` for one to arrive."""
        with self._cond:
            if self._slot is None:
                self._cond.wait(timeout)
            frame, self._slot = self._slot, None
            return frame

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self.source.release()
