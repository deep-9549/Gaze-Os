# Build status against the PRD

Legend: **Done** = implemented and covered by automated tests here.
**Done, verify** = implemented, but it can only be fully proven on real
hardware (a webcam, Windows, macOS) or with real data. **Needs you** = cannot
be done without a person, data access, GPUs or approvals (see HANDOFF.md).
**Not started** = P2 stretch items left for later.

## Functional requirements (PRD 6)

| ID | Requirement | Priority | Status | Where |
| --- | --- | --- | --- | --- |
| FR-01 | List webcams, pick one | P0 | Done, verify | `capture/camera.py`, `gazeos --list-cameras` |
| FR-02 | 640x480 / 1280x720 at 30 FPS, exposure lock | P0 | Done, verify | `capture/camera.py` |
| FR-03 | Low light / no face / several faces warnings | P0 | Done | `vision/quality.py`, overlay chip |
| FR-04 | 478 landmarks incl. iris every frame | P0 | Done, verify | `vision/landmarks.py` |
| FR-05 | 3D head pose every frame | P0 | Done | `vision/head_pose.py` |
| FR-06 | Gaze pitch/yaw with confidence | P0 | Done (M0 now; CNN once trained) | `inference/gaze.py` |
| FR-07 | Gaze to screen point with calibration | P0 | Done | `calibration/geometry.py`, personalization |
| FR-08 | Multiple monitors | P2 | Not started | Windows moves use the virtual desktop already |
| FR-09 | 9-point calibration in <= 30 s | P0 | Done | `ui/calibration.py` |
| FR-10 | Validation with per-point error and score | P0 | Done | `calibration/protocol.py` |
| FR-11 | One-point drift correction | P1 | Done | menu "Quick fix" |
| FR-12 | Implicit recalibration from clicks | P1 | Done | `calibration/implicit.py` |
| FR-13 | Save/load calibration per profile | P0 | Done | `profiles/store.py` |
| FR-14 | Adaptive smoothing | P0 | Done | `control/filters.py` |
| FR-15 | Fixation hold | P0 | Done | `control/fixation.py` |
| FR-16 | Snap to clickable elements | P1 | Done, verify (needs UI Automation on Windows) | `control/snapping.py`, backends |
| FR-17 | Zoom-click | P0 | Done | `control/zoom.py`, overlay |
| FR-18 | Head-assisted fine adjustment | P2 | Done (basic) | engine, mode "head_fine_tune" |
| FR-19 | Dwell click with progress ring | P0 | Done | `control/dwell.py`, overlay |
| FR-20 | Long-blink click | P0 | Done | `gestures/blink.py` |
| FR-21 | Winks to left/right click | P1 | Done | `gestures/blink.py` (enable in settings) |
| FR-22 | Action selector incl. drag | P0 | Done | action bar, engine |
| FR-23 | Map facial gestures to actions | P1 | Done (rules now; learned model once trained) | `gestures/events.py`, settings mapping |
| FR-24 | Gaze scroll zones | P0 | Done | engine (Scroll action) |
| FR-25 | Gesture scrolling | P1 | Done | gesture mapping `scroll_up` / `scroll_down` |
| FR-26 | Shortcut panel | P1 | Done | `ui/panels.py`, `os_backends/keys.py` |
| FR-27 | On-screen keyboard (3 layouts) | P0 | Done | `ui/keyboard.py`, `text_entry/layouts.py` |
| FR-28 | 3-5 word predictions | P0 | Done | `text_entry/predictor.py` |
| FR-29 | Phrase bank | P1 | Done | `text_entry/phrases.py` |
| FR-30 | Offline text-to-speech | P1 | Done, verify | `text_entry/tts.py` |
| FR-31 | Hindi and Gujarati | P2 | Not started | Unicode typing already works on Windows |
| FR-32 | Dwell-free swipe typing | P2 | Not started | |
| FR-33 | Pause by gesture, corner or hotkey | P0 | Done | engine, `ui/app.py` hotkeys |
| FR-34 | Overlay with gaze point and status | P0 | Done | `ui/overlay.py` |
| FR-35 | User profiles | P0 | Done | `profiles/store.py`, `--profile` |
| FR-36 | Settings screen for every value | P0 | Done | `ui/settings_dialog.py` (generated from the schema) |
| FR-37 | Tray icon, start with OS | P1 | Done, verify | `ui/app.py`, `autostart.py` |
| FR-38 | Opt-in logging and metric export | P0 | Done | `profiles/logs.py`, `recording.py` |
| FR-39 | Debug/demo view | P0 | Done | `ui/debug_view.py` |
| FR-40 | Auto-pause on face loss; emergency stop | P0 | Done | engine, Ctrl+Alt+G |
| FR-41 | Gaze heatmaps | P1 | Done | `profiles/logs.py` |

## ML and research components

| Item | Status |
| --- | --- |
| M0 geometric baseline | Done (runs in the app today) |
| M1 L2CS-Net baseline | Converter done (`training/convert_l2cs.py`); **needs you** to download the weights |
| M2 / M3 GazeNet-GOS, M4 Lite | Architecture, losses, training, export, INT8 done and smoke-tested; **needs you** for datasets + GPU training |
| Distillation (Stage C), meta-learning P5 (Stage D) | Code done and tested on synthetic data; needs trained models |
| Personalization P0-P4 | Done and tested; P5 runtime done |
| Gesture models B0 / B1 / B2 | B0 runs today; B1/B2 training and export done; **needs you** for the gesture dataset |
| Language models L0 / L1 / L2 / L3 | L0 + L1 run today (bundled word list and seed corpus); L2 training done; L3 local LLM client done (opt-in) |
| Experiments E1-E12 | Runners for E6 and E7 work today on simulated users; training scripts cover E1-E5, E8, E12; the rest need data |
| User study tools | Fitts task, questionnaires, statistics, figures done; **needs you** for ethics approval and participants |

## Engineering (PRD 18, 22, 23)

| Item | Status |
| --- | --- |
| Repository layout, typed settings schema, ADRs | Done |
| Ruff, strict mypy on the app | Done (clean) |
| Unit, property, replay, backend, UI, model tests | Done (113 tests; app coverage 79%) |
| CI (Linux + Windows), training smoke run, tagged Windows build | Done; runs once the repo has GitHub Actions enabled |
| Latency budget on the reference laptop, 2-hour soak | Profiler done; **needs you** to run it on the reference machine |
