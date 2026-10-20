"""Offline text-to-speech (FR-30, PRD 14.5): pyttsx3 by default, Piper for quality."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


class SilentSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class Pyttsx3Speaker:
    def __init__(self, rate: int = 170) -> None:
        import pyttsx3

        self._lock = threading.Lock()
        self.rate = rate
        self._pyttsx3 = pyttsx3

    def speak(self, text: str) -> None:
        def run() -> None:
            with self._lock:
                engine = self._pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.say(text)
                engine.runAndWait()
                engine.stop()

        threading.Thread(target=run, daemon=True).start()


class PiperSpeaker:
    """Calls the `piper` binary with a downloaded voice model (.onnx)."""

    def __init__(self, voice_model: str | Path, piper_bin: str = "piper") -> None:
        self.voice = str(voice_model)
        self.bin = shutil.which(piper_bin) or piper_bin

    def speak(self, text: str) -> None:
        def run() -> None:
            with tempfile.TemporaryDirectory() as d:
                wav = Path(d) / "out.wav"
                subprocess.run([self.bin, "--model", self.voice, "--output_file", str(wav)],
                               input=text.encode(), check=False, capture_output=True)
                _play_wav(wav)

        threading.Thread(target=run, daemon=True).start()


def _play_wav(path: Path) -> None:
    import sys

    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    elif sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        subprocess.run(["aplay", "-q", str(path)], check=False)


def make_speaker(engine: str, piper_voice: str | None = None) -> Speaker:
    try:
        if engine == "piper" and piper_voice:
            return PiperSpeaker(piper_voice)
        if engine == "pyttsx3":
            return Pyttsx3Speaker()
    except Exception as exc:
        log.warning("text-to-speech unavailable (%s)", exc)
    return SilentSpeaker()
