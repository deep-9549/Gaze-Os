# GazeOS

Control a computer with only your eyes and face, using an ordinary webcam.
GazeOS points with your gaze, clicks by dwelling or blinking, scrolls with
gaze zones or gestures, and types on a gaze keyboard with word prediction. It
runs offline on a laptop CPU; no video ever leaves the computer.

The full product and technical design is in [docs/PRD.md](docs/PRD.md).
What is built, and what still needs a person (data, GPUs, a webcam, a study),
is in [docs/STATUS.md](docs/STATUS.md) and [docs/HANDOFF.md](docs/HANDOFF.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[app]"
python -m tools.download_models                         # MediaPipe Face Landmarker
gazeos                                                  # first run opens onboarding
```

- `gazeos --simulate` runs the whole pipeline without a camera: a synthetic face
  looks wherever your mouse is, and no real input is injected.
- `gazeos --dry-run` tracks your eyes and shows the overlay but never clicks or types.
- **Ctrl+Alt+G** is the emergency stop; **Ctrl+Alt+P** pauses; closing your eyes
  for 2 seconds also pauses or resumes.

Without a trained gaze network, GazeOS uses the geometric M0 baseline (iris
position plus head pose), so it works end to end from day one. Accuracy then comes from
calibration. Drop a trained model at `models/gazenet_lite_int8.onnx`, or convert
the L2CS-Net weights (see [docs/TRAINING.md](docs/TRAINING.md)), and GazeOS uses
it automatically.

## How it works

```
camera ─► MediaPipe landmarks ─► head pose + normalization ─► gaze CNN (ONNX) ─► personalization
                                         │                                              │
                                         └► blink / gesture features ─► event logic     ▼
                                                                         │      One Euro filter ─► fixation hold
                                                                         ▼              │
                                                       dwell · blink · gesture clicks ◄─┘ ─► snapping / zoom-click ─► OS input
```

| Package | What it holds |
| --- | --- |
| `gazeos/capture` | Camera listing, one-slot capture thread, reconnects |
| `gazeos/vision` | Landmarks, head pose (solvePnP), Zhang normalization, quality gates, EAR |
| `gazeos/inference` | ONNX Runtime wrappers: gaze (M1/M3/M4), gestures, word LM; geometric M0 |
| `gazeos/calibration` | 9/16-point protocol, P0-P5 personalization, implicit recalibration, drift fix |
| `gazeos/control` | One Euro and Kalman filters, I-VT/I-DT fixations, dwell, snapping, zoom, engine |
| `gazeos/gestures` | Long blink, winks, pause; learned-gesture event logic |
| `gazeos/text_entry` | Layouts, prefix trie, Kneser-Ney 3-gram, noisy-channel decoder, phrases, TTS |
| `gazeos/os_backends` | Windows SendInput + UI Automation; macOS Quartz; Linux X11 and uinput |
| `gazeos/ui` | PySide6 overlay, action bar, keyboard, calibration, onboarding, settings, debug |
| `gazeos/profiles` | Per-user profiles, privacy panel, opt-in logs and heatmaps |
| `training/` | GazeNet-GOS models, losses, data packing, training, distillation, export (never shipped) |
| `research/` | Metrics, statistics, experiment runners (E6, E7) and report figures |
| `tools/` | Data collection, gesture labelling, Fitts task, latency profiler, camera calibration |

## Development

```bash
pip install -e ".[app,train,research,dev]"
pytest                      # unit, property, replay, backend, UI and training tests
ruff check . && mypy gazeos
```

CI runs lint, strict type checks of the app, the test suite on Linux and Windows
(coverage gate 70%), and an end-to-end training and export smoke run.
Tagged releases (`v*`) build the Windows app with PyInstaller and publish a checksum.

## Privacy and safety

No network calls by default; camera frames are processed in memory and never
saved; research recording is opt-in, shows a banner, and stores only
normalized face and eye crops. GazeOS is an assistive tool, not a certified
medical device, and must not be the only way a person can call for help.

## Licence

The project licence has not been chosen yet (see docs/HANDOFF.md). Bundled word
frequencies come from wordfreq, CC-BY-SA 4.0 (see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)).
Public gaze datasets are for non-commercial research and are never redistributed.
