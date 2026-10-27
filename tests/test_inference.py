from __future__ import annotations

import gazeos.inference.gaze as gaze


def test_auto_gaze_model_prefers_m3_when_fast_enough(tmp_path, monkeypatch) -> None:
    """"auto" takes the first model that exists and runs within the budget here, else the
    last one that exists (M4); an explicit path is used as given (ADR 0008)."""
    m3, m4 = tmp_path / "gazenet_m3.onnx", tmp_path / "gazenet_lite.onnx"
    m4.write_bytes(b"")
    speed = {m3: 12.0}
    monkeypatch.setattr(gaze, "benchmark_ms", lambda p: speed[p])
    assert gaze.resolve_gaze_model("auto", candidates=(m3, m4)) == m4  # no M3 installed
    m3.write_bytes(b"")
    assert gaze.resolve_gaze_model("auto", 15.0, (m3, m4)) == m3
    speed[m3] = 22.0  # a slower CPU
    assert gaze.resolve_gaze_model("auto", 15.0, (m3, m4)) == m4
    assert gaze.resolve_gaze_model(str(m4)) == m4
    assert gaze.resolve_gaze_model(str(tmp_path / "missing.onnx")) is None
