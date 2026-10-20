from __future__ import annotations

from gazeos.gestures.blink import BlinkDetector, BlinkEvent, EyeState, eye_state, tune_long_blink_threshold
from gazeos.gestures.events import FeatureWindow, GestureEventDetector, scale_by_gesture_size
from gazeos.settings import BlinkSettings


def _closure(det, start, ms, left=True, right=True):
    t = start
    events = []
    while t < start + ms / 1000:
        events.append(det.update(t, EyeState(left, right)))
        t += 1 / 30
    events.append(det.update(t, EyeState(False, False)))
    return [e for e in events if e is not None], t


def test_normal_blink_never_clicks() -> None:
    ev, _ = _closure(BlinkDetector(), 0.0, 200)
    assert ev == [BlinkEvent.NORMAL_BLINK]


def test_long_blink_and_pause() -> None:
    assert _closure(BlinkDetector(), 0.0, 600)[0] == [BlinkEvent.LONG_BLINK]
    assert _closure(BlinkDetector(), 0.0, 2300)[0] == [BlinkEvent.PAUSE_TOGGLE]
    assert _closure(BlinkDetector(), 0.0, 1500)[0] == []  # between long blink and pause


def test_winks_only_when_enabled() -> None:
    assert _closure(BlinkDetector(), 0.0, 500, right=False)[0] == []
    det = BlinkDetector(BlinkSettings(winks_enabled=True))
    assert _closure(det, 0.0, 500, right=False)[0] == [BlinkEvent.LEFT_WINK]
    assert _closure(det, 2.0, 500, left=False)[0] == [BlinkEvent.RIGHT_WINK]


def test_eye_state_prefers_blendshapes() -> None:
    s = eye_state({"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}, 0.3, 0.3)
    assert s.left_closed and not s.right_closed
    s2 = eye_state({}, 0.1, 0.3)
    assert s2.left_closed and not s2.right_closed


def test_tune_threshold() -> None:
    assert tune_long_blink_threshold([150, 180, 200, 220, 160], [500, 600, 550, 650, 700]) == 360
    assert tune_long_blink_threshold([], []) == 400
    assert 250 <= tune_long_blink_threshold([100], [120]) <= 800


def _probs(top: str, p: float = 0.9) -> dict[str, float]:
    base = {"none": 0.0, "eyebrow_raise": 0.0, "mouth_open": 0.0, "talking_or_yawning": 0.0}
    base[top] = p
    rest = (1 - p) / (len(base) - 1)
    return {k: (v if k == top else rest) for k, v in base.items()}


def test_gesture_event_fires_on_return_to_neutral() -> None:
    det = GestureEventDetector()
    t, fired = 0.0, []
    for _ in range(12):  # 400 ms of eyebrow raise
        t += 1 / 30
        fired.append(det.update(t, _probs("eyebrow_raise")))
    assert all(f is None for f in fired)
    t += 1 / 30
    assert det.update(t, _probs("none")) == "eyebrow_raise"
    # Refractory period.
    assert det.update(t + 0.1, _probs("eyebrow_raise")) is None


def test_short_or_talking_gestures_do_not_fire() -> None:
    det = GestureEventDetector()
    det.update(0.0, _probs("mouth_open"))
    assert det.update(0.1, _probs("none")) is None  # too short
    for i in range(20):
        assert det.update(1 + i / 30, _probs("talking_or_yawning")) is None


def test_feature_window_and_gesture_size() -> None:
    w = FeatureWindow(30, 17)
    assert w.array().shape == (1, 30, 17) and not w.full
    for _ in range(30):
        w.push([1.0] * 17)  # type: ignore[arg-type]
    assert w.full
    scaled = scale_by_gesture_size(_probs("eyebrow_raise", 0.6), 2.0)
    assert scaled["eyebrow_raise"] < 0.6
    assert abs(sum(scaled.values()) - 1.0) < 1e-9
