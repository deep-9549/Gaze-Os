# GazeOS — Product Requirements Document

Oct 20, 2026 · @Deep

## 1. Document overview

GazeOS lets a person control a computer with only their eyes and face, using an ordinary webcam and deep learning instead of dedicated eye-tracking hardware. The user points with gaze, clicks by dwelling or blinking, scrolls with head or eye gestures, and types on a gaze keyboard with AI word prediction.

| Field | Value |
| --- | --- |
| Product | GazeOS |
| Document | Product requirements + technical design (end to end) |
| Version | 1.0, first full draft |
| Project type | 7th-semester minor project, software only |
| Primary platform | Windows 10/11 desktop app; macOS and Linux as secondary targets |
| Hardware needed | Any built-in or USB webcam, 720p or better; a laptop CPU, no GPU at runtime |
| Core techniques | Face landmark tracking, appearance-based gaze estimation (CNN), few-shot per-user calibration, temporal filtering, blink and gesture classification, on-device language model |
| Budget | Rs 0 baseline; optional Rs 2,000 to 4,000 (Section 26) |

**Elevator pitch:** dedicated eye trackers used by people with ALS or paralysis cost lakhs of rupees and need special hardware. GazeOS turns the webcam already on every laptop into a usable eye-control system, privately and offline.

**Headline targets** (justified in Section 3):

- Gaze accuracy: 2.0 degrees or less after a 9-point calibration, which is about 2 cm on screen at a 60 cm viewing distance
- Speed: 30 frames per second or more on a mid-range laptop CPU, with under 100 ms from eye movement to cursor movement
- Usability: a new user can open a website, search, and type a short sentence hands-free within 10 minutes of first launch
- Typing: 8 or more words per minute with prediction on, against a dwell-only baseline of about 4 to 5

**What makes it a deep learning project, not just an app:** a gaze network trained and benchmarked on public datasets, a personalization method that adapts it to each user from a few seconds of data, learned blink and gesture classifiers, and a language model for prediction. Each has a baseline, a metric, and an ablation (Sections 16 and 17).

## 2. Problem statement and motivation

People who cannot use their hands still have working eyes, but the hardware that lets them control a computer with their eyes is expensive, imported, and hard to get in India. Webcam-only alternatives exist but are too inaccurate or too limited to replace it.

### 2.1 Who is affected

- India's Census 2011 counted 2.68 crore persons with disabilities, 2.21% of the population ([Census of India via UN Stats](https://unstats.un.org/unsd/demographic-social/meetings/2017/new-york--disability-egm/Session%204/India.pdf)).
- Movement disability was the largest single category, at 20.3% of them ([same source](https://unstats.un.org/unsd/demographic-social/meetings/2017/new-york--disability-egm/Session%204/India.pdf)).
- The most severe cases need eye control specifically: ALS and motor neurone disease, high spinal cord injury (quadriplegia), cerebral palsy, locked-in syndrome after stroke, and muscular dystrophy. Many of these users keep reliable eye movement long after hand and voice control are lost.
- A wider group benefits from hands-free control: people with repetitive strain injury, arthritis, tremor, temporary injuries, and surgeons or lab workers whose hands are busy or sterile.

### 2.2 Why existing solutions fall short

| Option | How it works | Limitation |
| --- | --- | --- |
| [Tobii Dynavox PCEye 5](https://montech.ruralinstitute.umt.edu/mtdb/item_view.asp?ID=8366) | Dedicated infrared eye tracker clipped under the screen | Estimated cost US$1,249 for the tracker alone (about Rs 1 lakh before import duty); Windows only; screens up to 27 inches |
| Full eye-gaze communication devices | Tablet with built-in tracker and speech software | A [Tobii I-15+ cost upwards of US$13,000 in 2014](https://montech.ruralinstitute.umt.edu/eye-gaze-finally-affordable) |
| [WebGazer](https://www.ijcai.org/Abstract/16/540) | Browser JavaScript, pupil features plus regression | Accuracy around 4 degrees or worse, which [drops further when the head moves](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017); works only inside a web page, not the whole OS |
| [Google Project Gameface](https://github.com/google/project-gameface) | Webcam head movement and facial gestures move the cursor | Uses the head, not the eyes, so it needs neck control that many target users lack; built for gaming |
| Windows Eye Control (built in) | OS eye-control feature | Needs a supported hardware eye tracker; does nothing with a webcam |

### 2.3 The gap GazeOS fills

No free tool today combines all four of: (1) true eye-gaze pointing from a normal webcam, (2) accuracy good enough for real desktop targets, (3) control of the whole operating system rather than one web page, and (4) fast text entry. Recent appearance-based deep learning makes this feasible. [L2CS-Net](https://arxiv.org/pdf/2203.03339) reports 3.92 degrees mean error on MPIIGaze without any user calibration. MIT's [iTracker](https://gazecapture.csail.mit.edu/index.php) shows that per-user calibration cuts webcam-class error substantially (from 1.71 cm to 1.34 cm on phones). GazeOS combines a modern gaze network, fast personalization, and careful interaction design so that the remaining error stops mattering to the user.

### 2.4 Problem statement (one line)

Design and build a real-time, privacy-preserving, webcam-only system that estimates where a user is looking with 2 degrees or less of error after brief calibration, and turns gaze, blinks, and facial gestures into complete hands-free control of a desktop computer, including text entry.

## 3. Goals, non-goals, and success metrics

GazeOS succeeds if a person who cannot use their hands can do everyday computer tasks with it, and if every ML component beats a clear baseline on a public benchmark.

### 3.1 Product goals

1. Full hands-free control of the desktop: point, click, double-click, right-click, drag, scroll, type, and switch apps.
2. Works with the webcam the user already has, with no extra hardware.
3. Usable in under 10 minutes from install, including a calibration of 30 seconds or less.
4. Fully offline and private: no video leaves the computer.
5. Comfortable for long sessions: no accidental clicks while the user is just reading (the "Midas touch" problem).

### 3.2 Technical and research goals

1. Train and benchmark an appearance-based gaze network that matches published results within about 0.5 degrees on a standard benchmark.
2. Show that few-shot personalization reduces error by at least 40% compared with the uncalibrated network.
3. Show that each pipeline stage (normalization, filtering, personalization, snapping) improves the end result, through ablation studies.
4. Run the whole pipeline in real time on a laptop CPU.

### 3.3 Success metrics

| Area | Metric | Target | Stretch | How measured |
| --- | --- | --- | --- | --- |
| Gaze model | Mean angular error, no calibration, MPIIFaceGaze leave-one-subject-out | 4.5 degrees or less | 4.0 or less | Offline benchmark (Section 17) |
| Gaze model | Mean angular error after 9-point calibration, own test set | 2.0 degrees or less | 1.5 or less | User study, 15+ participants |
| Pointing | On-screen error at 60 cm viewing distance | 2.1 cm or less | 1.6 cm or less | Derived: distance x tan(error) |
| Pointing | Precision (RMS jitter during a steady fixation, after filtering) | 0.5 degrees or less | 0.3 or less | Fixation task |
| Pointing | Selection success on standard Windows UI targets with zoom-click | 90% or more | 95% or more | Target-acquisition task |
| Performance | Processing frame rate on a mid-range laptop CPU | 30 FPS or more | 60 FPS | Profiler logs |
| Performance | End-to-end latency, camera frame to cursor move | 100 ms or less (p95 150 ms) | 60 ms | Timestamp tracing |
| Performance | Total CPU use while running | 30% or less | 20% or less | Task Manager / psutil |
| Clicking | Intentional-blink detection F1 score | 0.95 or more | 0.98 | Labelled blink set |
| Clicking | Accidental clicks while reading | 1 or fewer per 5 minutes | 1 per 15 minutes | Reading task |
| Typing | Words per minute with prediction | 8 or more | 12 or more | Transcription task |
| Typing | Keystrokes saved by prediction | 40% or more | 55% or more | Keystroke log |
| Usability | System Usability Scale (SUS) | 70 or more ("good") | 80 or more | Post-study questionnaire |
| Usability | Time from launch to first successful hands-free web search | 10 minutes or less | 5 minutes or less | Observed onboarding |

For reference, 1 degree of error at 60 cm is about 1.05 cm on screen. On a 15.6-inch 1080p laptop (about 141 pixels per inch) that is about 58 pixels, which is why GazeOS needs snapping and zoom-click for small targets (Section 12).

### 3.4 Non-goals (out of scope for this version)

- Mobile phone or tablet apps
- Medical-device certification or clinical claims
- Infrared or any additional hardware
- VR and AR headsets
- Multiple people controlling one computer at once
- Precision high enough for pixel-level work such as photo editing or competitive gaming
- Cloud processing of any kind
- Voice control (listed as future scope, Section 29)

## 4. Target users and personas

The primary user has reliable eye movement and blinking but little or no hand control. Every design choice is checked against the most limited persona (Ramesh), since a system that works for him works for everyone else.

| Persona | Situation | What they can move | What they need from GazeOS | Design implications |
| --- | --- | --- | --- | --- |
| **Ramesh, 52** (primary) | Retired teacher with ALS; speech is fading | Eyes and blinks; very little head movement | Message family on WhatsApp Web, read news, watch YouTube, speak typed text aloud | Pure eye control with no head gestures; dwell and blink clicks; large-key keyboard; text-to-speech output; long-session comfort |
| **Arjun, 24** (primary) | Engineering graduate with a high spinal cord injury | Eyes, blinks, good head and face control | Return to work: email, documents, browsing, light coding | Fast mode with face gestures for click and scroll; strong word prediction; app switching |
| **Meera, 31** (primary) | Cerebral palsy with involuntary head movements | Eyes and blinks; head moves unpredictably | Online classes, video calls, shopping | Head-pose compensation; heavier smoothing; larger dwell tolerance; recalibration in one tap |
| **Sunita, 45** (secondary) | Caregiver or occupational therapist | Full ability | Set up the system, adjust settings, check that it works | Simple settings screen; profiles per user; clear status of camera and tracking |
| **Karan, 29** (secondary) | Software developer with repetitive strain injury | Full eye, head, and face control; hands hurt | Reduce mouse use during long work days | Hybrid mode: gaze points, keyboard or gesture confirms; low-latency cursor |

**Environment assumptions:** a laptop or desktop with a webcam at the top of the screen, the user 40 to 80 cm from the screen, indoor lighting of at least ordinary room brightness, and possibly glasses. Users may be in a wheelchair or reclined in bed, so the face can be tilted relative to the camera.

## 5. User stories and core use cases

Each story has an acceptance test used in the user study (Section 17), so "done" means a participant completed it, not that the code exists.

| ID | As a... | I want to... | So that... | Accepted when |
| --- | --- | --- | --- | --- |
| US-01 | New user | Calibrate by looking at a few dots | The cursor goes where I look | 9-point calibration finishes in 30 s or less and validation error is shown |
| US-02 | User | Move the cursor by looking | I can point without my hands | Cursor follows gaze at 30+ FPS with no visible jitter during a fixation |
| US-03 | User | Click by staring at something briefly (dwell) | I can click without moving anything but my eyes | A dwell of the set time (default 800 ms) on a target clicks it; a progress ring shows the countdown |
| US-04 | User | Click with a deliberate long blink | Clicking is faster than dwell | A blink held 400 ms or longer clicks; normal blinks never do |
| US-05 | User | Right-click, double-click, and drag | I can use every normal mouse action | A radial action menu selects the action for the next click |
| US-06 | User | Hit small buttons reliably | I am not frustrated by tiny UI elements | Zoom-click magnifies the area and a second dwell selects inside it |
| US-07 | User | Scroll pages | I can read long pages | Looking at the top or bottom scroll zone, or a gesture, scrolls smoothly |
| US-08 | User | Type text with an on-screen keyboard | I can write messages | Gaze keyboard types into any focused app |
| US-09 | User | See word and sentence predictions | I type much faster | Top 3 to 5 predictions appear and update after every key; selecting one inserts it |
| US-10 | User who cannot speak | Have my typed text read aloud | I can talk to people in the room | A "Speak" key reads the current text with offline text-to-speech |
| US-11 | User | Pause tracking instantly | I can watch a video or rest without accidental clicks | A pause gesture or looking at a corner toggles tracking; state is clearly shown |
| US-12 | User whose position shifts | Fix drift without full recalibration | I keep working after moving in bed or chair | One-point quick recalibration in 3 s or less |
| US-13 | Caregiver | Set up a profile for each user | Settings and calibration are saved per person | Profiles store calibration, dwell time, sensitivity, and gestures |
| US-14 | User | Use quick shortcuts (copy, paste, switch window, close) | Common tasks take one action | A shortcut panel triggers OS shortcuts |
| US-15 | Any user | Know my camera images are private | I trust the system | No network calls by default; a privacy panel says what is stored |

## 6. Functional requirements

There are 41 requirements: 26 are P0 (must work in the final demo), 11 are P1 (should have), and 4 are P2 (stretch). P0 alone gives a complete, demonstrable system.

| ID | Module | Requirement | Priority |
| --- | --- | --- | --- |
| FR-01 | Camera | List available webcams and let the user pick one | P0 |
| FR-02 | Camera | Capture at 640x480 or 1280x720 at 30 FPS, with automatic exposure lock after setup | P0 |
| FR-03 | Camera | Warn when lighting is too low, no face is found, or more than one face is visible | P0 |
| FR-04 | Tracking | Detect the face and 478 face landmarks (including iris) on every frame | P0 |
| FR-05 | Tracking | Estimate 3D head pose (rotation and translation) on every frame | P0 |
| FR-06 | Tracking | Estimate gaze direction (pitch, yaw) on every frame, with a confidence score | P0 |
| FR-07 | Tracking | Convert gaze direction to a screen point using the user's calibration | P0 |
| FR-08 | Tracking | Support more than one monitor | P2 |
| FR-09 | Calibration | 9-point calibration with animated shrinking targets, finished in 30 s or less | P0 |
| FR-10 | Calibration | Validation step that shows error at each point and an overall score | P0 |
| FR-11 | Calibration | One-point quick drift correction | P1 |
| FR-12 | Calibration | Implicit recalibration: confirmed clicks become new calibration samples | P1 |
| FR-13 | Calibration | Save and load calibration per user profile | P0 |
| FR-14 | Cursor | Smooth cursor movement with an adaptive filter (still when fixating, fast during saccades) | P0 |
| FR-15 | Cursor | Fixation detection that holds the cursor steady while the user looks at one spot | P0 |
| FR-16 | Cursor | Snap to nearby clickable elements using the OS accessibility tree | P1 |
| FR-17 | Cursor | Zoom-click: magnify the region around the gaze, then select inside it | P0 |
| FR-18 | Cursor | Head-assisted fine adjustment for users with head control | P2 |
| FR-19 | Clicking | Dwell click, 300 to 2,000 ms (default 800 ms), with a progress ring | P0 |
| FR-20 | Clicking | Long-blink click (eyes closed 400 ms or more, adjustable) | P0 |
| FR-21 | Clicking | Left wink and right wink mapped to left and right click | P1 |
| FR-22 | Clicking | Action selector for left, right, double click, and drag start and end | P0 |
| FR-23 | Clicking | Map facial gestures (eyebrow raise, mouth open, smile) to any action | P1 |
| FR-24 | Navigation | Gaze scroll zones at the top and bottom of the screen | P0 |
| FR-25 | Navigation | Gesture-based scrolling | P1 |
| FR-26 | Navigation | Shortcut panel: copy, paste, undo, switch window, close, Start menu | P1 |
| FR-27 | Typing | On-screen keyboard (QWERTY, alphabetical, and large-key layouts) that types into the focused app | P0 |
| FR-28 | Typing | Word completion and next-word prediction showing 3 to 5 suggestions | P0 |
| FR-29 | Typing | Phrase bank of saved sentences ("I need water", "Please call") | P1 |
| FR-30 | Typing | Offline text-to-speech of typed text | P1 |
| FR-31 | Typing | Hindi and Gujarati keyboard layouts and prediction | P2 |
| FR-32 | Typing | Dwell-free typing by sweeping the gaze across letters | P2 |
| FR-33 | System | Pause and resume by gesture, by looking at a screen corner, or by hotkey | P0 |
| FR-34 | System | Always-on-top overlay showing the gaze point and tracking status, which can be hidden | P0 |
| FR-35 | System | User profiles | P0 |
| FR-36 | System | Settings screen for every adjustable value | P0 |
| FR-37 | System | System tray icon and start with the OS | P1 |
| FR-38 | System | Opt-in logging and metric export for experiments | P0 |
| FR-39 | System | Debug and demo view: landmarks, eye crops, gaze vector, live error | P0 |
| FR-40 | Safety | Auto-pause when the face is lost for more than 1 s; emergency stop hotkey | P0 |
| FR-41 | Research | Record gaze heatmaps of a session | P1 |

## 7. Interaction design and UX

The core UX problem is the "Midas touch": the eyes are always looking at something, so the system must never treat looking as clicking unless the user clearly means it. GazeOS solves this with explicit confirmation (dwell with a visible countdown, deliberate blinks, or gestures) and an easy pause.

### 7.1 Control modes

| Mode | Pointing | Clicking | Best for |
| --- | --- | --- | --- |
| Eyes only (default) | Gaze | Dwell or long blink | Ramesh: no head or hand control |
| Eyes + face | Gaze | Facial gestures (eyebrow raise, mouth open) | Arjun: good face control, wants speed |
| Hybrid | Gaze | A physical key or switch confirms | Karan: RSI users; also adaptive switches |
| Head fine-tune | Gaze for coarse, head for the last few centimetres | Dwell or gesture | Users with head control who need precision |

### 7.2 Default input mapping

| User action | Detection rule (default, adjustable) | Result |
| --- | --- | --- |
| Look at a point | Gaze estimate, filtered | Cursor moves there |
| Look steadily at one spot | Fixation for the dwell time (800 ms) | Click with the current action |
| Long blink | Both eyes closed 400 to 1,000 ms | Left click |
| Left wink | Left eye closed 300 ms or more, right eye open | Left click (P1) |
| Right wink | Right eye closed 300 ms or more, left eye open | Right click (P1) |
| Eyes closed 2 s or more | Both eyes closed | Pause or resume tracking |
| Look at the top-right corner for 1.5 s | Corner dwell | Open the GazeOS menu |
| Raise eyebrows | Gesture classifier | Configurable, default double click (P1) |
| Open mouth | Gesture classifier | Configurable, default drag toggle (P1) |

Normal blinks last about 100 to 300 ms, so the 400 ms threshold separates them from intentional blinks. The exact value is tuned per user during onboarding.

### 7.3 Clicking small targets: zoom-click

1. The user dwells on an area; a ring fills around the cursor.
2. If a clickable element larger than the error radius sits under the gaze, it is clicked directly (snapping, FR-16).
3. Otherwise a 3x magnifier window opens around that area.
4. The user dwells again inside the magnified view, where every target is now three times larger than the gaze error.
5. The click is sent to the exact original screen location.

### 7.4 Screen overlays

- **Gaze cursor:** a soft translucent circle whose size shows current confidence (larger when less certain).
- **Dwell ring:** fills clockwise; looking away cancels it.
- **Action bar:** a slim dockable bar with Left, Right, Double, Drag, Scroll, Keyboard, Shortcuts, and Pause. The current action is highlighted.
- **Status chip:** shows Tracking, Paused, No face, or Low light, with a colour and a text label (never colour alone).

### 7.5 Gaze keyboard

- Three layouts: QWERTY (familiar), alphabetical (for users new to keyboards), and large 6-key grouped (for low accuracy). Keys are at least 2.5 times the user's measured error radius.
- A prediction bar above the keys shows 3 to 5 words, plus a sentence completion when confident.
- Special keys: Space, Backspace, Speak, Phrases, Enter, Shift, Numbers, Close.
- Typed characters go into whichever app has focus, through OS keyboard injection (Section 20).

### 7.6 Onboarding flow

1. Welcome screen: camera permission and a live preview with a face-position guide ("move a little closer").
2. Lighting and position check, with fixes suggested on screen.
3. Blink tuning: the user does five normal and five long blinks, which sets the personal threshold.
4. 9-point calibration, then validation with a score out of 100.
5. Guided tutorial: open a website, scroll it, click a link, type "hello".
6. Done; the profile is saved.

### 7.7 Accessibility and comfort

- High-contrast and large-text themes; every element meets WCAG 2.1 AA contrast.
- No time-critical interactions except dwell, whose time is adjustable.
- A rest reminder every 20 minutes, since gaze control tires the eyes.
- The settings screen itself is fully usable by gaze.

## 8. System architecture

Each webcam frame passes through five stages in about 60 ms, and the neural models sit in the middle. The interface, typing module, and profile store hang off the control engine rather than the frame path, so a slow screen redraw never delays the cursor.

&#91;embedded content: GazeOS architecture · 5 frame stages, 3 side modules\]

Read top to bottom for one frame; the right column shows the parts that feed or listen to the control engine.

### 8.1 Threads

| Thread | Runs | Why it is separate |
| --- | --- | --- |
| Capture | Camera reads into a one-slot buffer | The camera blocks while waiting for frames; a one-slot buffer means processing always gets the newest frame |
| Processing | Stages 2 to 4 in sequence | MediaPipe and ONNX Runtime release Python's global interpreter lock, so this thread runs at full speed alongside the others |
| UI (main) | Qt event loop; overlays redrawn at 60 Hz | Qt requires its widgets on the main thread; it receives cursor state through Qt signals |
| Background worker | Calibration refits, transformer and LLM predictions, logging | Slow jobs must never stall the frame path |

If profiling shows thread contention, stages 2 and 3 move into a separate process connected through shared memory. This is a fallback, not the starting design.

### 8.2 Per-frame data contract

Every stage reads and adds to one typed record, which also makes recording and replay testing easy (Section 23).

```python
@dataclass
class FrameResult:
    t_capture: float            # monotonic timestamp, seconds
    face_found: bool
    landmarks: np.ndarray       # (478, 3)
    blendshapes: dict[str, float]
    head_R: np.ndarray          # (3, 3) rotation
    head_t: np.ndarray          # (3,) translation, mm
    gaze_pitch_yaw: tuple[float, float]   # radians, camera frame
    gaze_sigma: tuple[float, float]       # predicted uncertainty
    gesture_probs: dict[str, float]
    screen_xy: tuple[int, int] | None     # after calibration and filtering
    state: str                  # idle, dwelling, fired, cooldown, paused
    timings_ms: dict[str, float]          # per-stage latency
```

### 8.3 Key design decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Runtime engine | ONNX Runtime, not PyTorch | Several times faster on CPU; the app ships without PyTorch, so the installer is far smaller |
| Language | Python first | One language for the team; hot paths are already C++ inside MediaPipe, OpenCV, and ONNX Runtime |
| Concurrency | Threads first, processes as a fallback | Simpler; the heavy libraries release the interpreter lock |
| Input method | OS-level event injection | Works with every app with no plugins or app-specific code |
| Personalization placement | After the network, in the control engine | Calibration updates take milliseconds and never require re-exporting a model |

## 9. Computer vision front-end

The front-end turns each raw webcam frame into clean, normalized face and eye images plus head pose, in under 10 ms. Normalization is the single most important step: it removes head rotation and distance from the image so the network only has to learn eye appearance.

### 9.1 Face and landmark detection

- **Model:** MediaPipe Face Landmarker (Tasks API) in VIDEO mode, which runs BlazeFace detection once and then tracks between frames.
- **Outputs per frame:** 478 3D landmarks, with 10 iris points at indices 468 to 477 ([MediaPipe docs via Comfy](https://docs.comfy.org/tutorials/utility/face-detection/mediapipe)); 52 blendshape scores such as eyeBlinkLeft, browInnerUp, and jawOpen ([Google Developers Blog](https://developers.googleblog.com/en/project-gameface-makes-gaming-accessible-to-everyone/)); and a 4x4 facial transformation matrix.
- **Settings:** one face; minimum detection and tracking confidence 0.5; blendshapes on; transformation matrix on.
- **License:** Apache 2.0, so it can be used freely.

### 9.2 Camera model

- Default intrinsics with no setup: focal length fx = fy = image width in pixels, principal point at the image centre, no lens distortion.
- Optional accurate mode: a one-time OpenCV checkerboard calibration (print an A4 checkerboard, take 15 photos) gives true intrinsics and distortion. The ablation in Section 16 measures how much this helps.

### 9.3 Head pose estimation

- Solve a Perspective-n-Point problem (OpenCV `solvePnP`, iterative method) between 6 stable landmarks (eye corners, nose tip, mouth corners, chin) and the matching points on MediaPipe's canonical 3D face model.
- Output: rotation R (3x3) and translation t (in mm) of the head relative to the camera.
- Cross-check against MediaPipe's own transformation matrix; if they disagree by more than 10 degrees, the frame is marked low-confidence.
- Head pose is smoothed with the same One Euro filter used for the cursor (Section 12).

### 9.4 Data normalization

Based on Zhang, Sugano and Bulling's normalization method, used by MPIIGaze and ETH-XGaze. A virtual camera is rotated to look straight at the face centre from a fixed distance, which cancels head roll and scale.

1. Face centre c = mean of the eye corners and nose landmarks in camera coordinates (from R and t).
2. Build the rotation R\_n whose z-axis points from the camera to c, and whose x-axis is parallel to the head's x-axis (this removes roll).
3. Scale matrix S = diag(1, 1, d\_n / ‖c‖), with normalized distance d\_n = 600 mm.
4. Warp the image with the perspective transform below, giving a 224x224 face image (normalized focal length 960 px, matching ETH-XGaze).
5. Crop 36x60 eye images (MPIIGaze convention) from the normalized face using the landmark positions.
6. Rotate the ground-truth gaze vector by R\_n during training; at runtime, rotate the predicted gaze back with R\_n transposed.

```latex
W = C_n \, S \, R_n \, C_r^{-1}
```

Here C\_r is the real camera matrix and C\_n the virtual camera matrix.

### 9.5 Image enhancement

- Convert eye crops to grayscale and apply CLAHE (contrast-limited adaptive histogram equalization) to reduce the effect of uneven lighting.
- Face crops stay in RGB with per-channel normalization using ImageNet mean and standard deviation.

### 9.6 Quality gates (frames that skip the gaze update)

| Check | Threshold | Action |
| --- | --- | --- |
| Face too small | Inter-eye distance under 60 px | Status: "Move closer"; hold cursor |
| Head turned too far | Yaw or pitch over 35 degrees | Hold cursor; lower confidence |
| Eye closed | Blink score over 0.5 | Skip gaze update; pass frame to blink logic |
| Too dark | Mean face brightness under 50 of 255 | Status: "Low light" |
| Tracking lost | No face for 1 s | Auto-pause (FR-40) |

## 10. Gaze estimation model

The core model, GazeNet-GOS, is a multi-stream CNN that looks at the whole face and both eyes together, predicts gaze pitch and yaw with an L2CS-style classify-then-regress head, and also predicts its own uncertainty. A small MobileNet version distilled from it runs in real time on CPU.

### 10.1 Inputs and output

- **Inputs:** normalized 224x224 RGB face, two normalized 36x60 grayscale eye crops (the right eye is mirrored so both share weights), and the normalized head pose (pitch, yaw, roll).
- **Output:** gaze pitch and yaw in the normalized camera frame, plus a log-variance for each, which becomes the confidence score.

### 10.2 Architecture

| Block | Layers | Output size |
| --- | --- | --- |
| Face stream | ResNet-18 pretrained on ImageNet, global average pool | 512 |
| Eye stream (shared by both eyes) | 4 conv blocks (3x3 conv, BatchNorm, ReLU, max-pool), channels 32-64-128-128, global pool | 128 per eye |
| Head-pose stream | MLP 3 to 32 to 32, ReLU | 32 |
| Fusion | The four feature vectors projected to 256-d tokens, then a 2-layer transformer encoder (4 heads), mean-pooled | 256 |
| Shared trunk | FC 256 to 256, ReLU, dropout 0.3 | 256 |
| Pitch head | FC to 28 bins (3-degree bins covering -42 to +42 degrees) | 28 |
| Yaw head | FC to 28 bins (same range) | 28 |
| Uncertainty head | FC to 2 (log-variance of pitch and yaw) | 2 |

The fusion transformer lets the network learn when to trust the eyes (sharp, frontal) and when to lean on the face (blurry eyes, glasses glare). The attention weights are logged, which gives a nice explainability visual for the viva.

### 10.3 Classify-then-regress output (from L2CS-Net)

Each angle is first classified into a bin with softmax, then the continuous angle is the softmax-weighted average of bin centres. [L2CS-Net](https://arxiv.org/pdf/2203.03339) showed that handling each angle separately with this combined loss improves accuracy.

```latex
\hat{\theta} = \sum_{k=1}^{K} \mathrm{softmax}(z)_k \cdot b_k
```

Here z are the bin logits, b\_k the bin centres, and K = 28.

### 10.4 Loss function

```latex
\mathcal{L} = \sum_{\theta \in \{\mathrm{pitch},\, \mathrm{yaw}\}} \Big[ \mathrm{CE}(z_\theta, k_\theta) + \beta \, (\hat{\theta} - \theta)^2 \Big] + \lambda \, \mathcal{L}_{\mathrm{NLL}}
```

- CE is cross-entropy against the true bin; β = 1 (L2CS reported both β = 1 and 2; tune on validation).
- The Gaussian negative log-likelihood term trains the uncertainty head, with λ = 0.1, switched on after 5 warm-up epochs so it does not destabilize early training:

```latex
\mathcal{L}_{\mathrm{NLL}} = \sum_{\theta} \left[ \frac{(\hat{\theta} - \theta)^2}{2\sigma_\theta^2} + \tfrac{1}{2}\log \sigma_\theta^2 \right]
```

### 10.5 Training setup

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | 1e-4 for the pretrained backbone, 1e-3 for new layers; 1 epoch linear warm-up, then cosine decay |
| Batch size | 128 (64 if GPU memory runs short) |
| Epochs | 25 on ETH-XGaze, then 10 fine-tuning epochs on MPIIFaceGaze |
| Precision | Mixed precision (fp16) on a Kaggle T4 or P100 GPU |
| Augmentation | Colour jitter, random gamma, Gaussian and motion blur, JPEG re-compression (mimics webcam artifacts), random shifts up to 8 px, horizontal flip with mirrored label (yaw negated, eyes swapped) |
| Early stopping | On validation angular error, patience 5 epochs |
| Seeds | 3 seeds per experiment; report mean and standard deviation |

### 10.6 Model variants

| ID | Model | Role |
| --- | --- | --- |
| M0 | Geometric pupil-centre method (iris landmark offset from eye-corner midpoint, plus polynomial regression) | Classical baseline, no deep learning |
| M1 | L2CS-Net, ResNet-50, official released weights | Published deep-learning baseline |
| M2 | Face-only ResNet-18 with L2CS head | Ablation: are eyes and head pose needed? |
| M3 | Full GazeNet-GOS (Section 10.2) | Main model, accuracy-first |
| M4 | GazeNet-GOS-Lite: MobileNetV3-Small face stream, distilled from M3 | Deployed real-time model |

### 10.7 Distillation for real time (M3 to M4)

- The student M4 learns from both the true labels and the teacher M3's soft bin probabilities (temperature T = 4).
- Distillation loss = 0.5 x task loss + 0.5 x KL divergence between teacher and student bin distributions, scaled by T squared.
- Goal: M4 within 0.3 degrees of M3 while running at least 3 times faster on CPU.

### 10.8 Export and inference

1. Export PyTorch to ONNX (opset 17) with fixed input shapes.
2. Check parity: mean difference between PyTorch and ONNX outputs under 0.05 degrees on 1,000 test images.
3. Apply ONNX Runtime static INT8 quantization, calibrated on 500 normalized frames; accept only if accuracy drops by under 0.2 degrees.
4. Run with ONNX Runtime CPU provider, 4 intra-op threads. Target per-frame inference: 8 ms or less for M4.

## 11. Calibration and personalization

Personalization is where GazeOS gets most of its accuracy: a generic network is typically off by around 4 degrees because every eye is shaped differently, and 30 seconds of calibration data should bring that close to 2. This section is also the project's main research contribution, because it compares five personalization methods head to head.

### 11.1 Why per-user calibration is needed

- Each person's visual axis (where they actually see) differs from the optical axis of the eye that the camera observes by a personal angle, often several degrees. No generic model can know it.
- Eye shape, eyelids, glasses, and the exact camera and screen geometry also vary per person and setup.
- [iTracker](https://gazecapture.csail.mit.edu/index.php) cut its error from 1.71 cm to 1.34 cm on phones just by adding calibration.

### 11.2 Calibration protocol

1. Nine targets at 10%, 50%, and 90% of screen width and height, shown in random order.
2. Each target appears as a large circle that shrinks over 1.5 s, which draws the eye to its centre.
3. Frames from 0.5 to 1.5 s after the target appears are kept (skipping the eye's travel time). At 30 FPS that is about 30 frames per point, about 270 in total.
4. Blink frames and outliers (more than 3 median absolute deviations from the point's median) are dropped.
5. Validation: 4 new target positions; the mean error in degrees and centimetres is shown as a score.
6. Extended option: 16 points for users who want maximum accuracy.

### 11.3 From gaze direction to a screen point

The gaze ray starts at the 3D eye centre o (from head pose) and points along the de-normalized gaze direction g. It is intersected with the screen plane. By default the camera is assumed to sit at the top centre of the screen, in the same plane, with the physical screen size read from the OS.

```latex
p = o + t\,g, \qquad t = -\frac{o_z}{g_z}
```

### 11.4 Personalization methods compared

| ID | Method | What it learns | Cost |
| --- | --- | --- | --- |
| P0 | None | Nothing; raw network | Zero |
| P1 | Angular offset | A fixed pitch and yaw bias (the personal angle) | Under 1 ms to fit |
| P2 | Screen-space polynomial | Second-order polynomial from predicted (x, y) to true (x, y), ridge regularized | Under 10 ms |
| P3 | Gaussian process regression | Corrects residual error as a smooth function of predicted point and head pose; also gives uncertainty | Under 1 s |
| P4 | Last-layer fine-tuning | Unfreeze only the fusion output and heads; 100 steps of SGD on the calibration samples | Under 5 s on CPU |
| P5 | Meta-learned adaptation | Train the network with MAML-style episodes (inspired by FAZE, Park et al., ICCV 2019) so it adapts from 9 points by design | Training is heavier; adaptation under 5 s |

The expected finding is that P3 or P4 gives the best accuracy-to-effort ratio and P5 the best accuracy with very few points. Section 16 reports error against the number of calibration points (1, 3, 5, 9, 16) for every method, which makes a strong results chart.

### 11.5 Implicit (continuous) recalibration

- Every confirmed click on a UI element whose bounding box is known (from the accessibility tree) becomes a free calibration sample: the gaze samples from the 300 ms before the click, labelled with the element's centre.
- A sample is accepted only if its residual is under twice the current error, so a wrong click does not corrupt the model.
- Samples carry a weight that decays over 30 minutes, so the model tracks slow changes in posture and lighting.
- The personalization model is refit after every 20 new samples, in a background thread.

### 11.6 Drift correction

- One-point correction: the user looks at a single centre dot for 1 s; the measured offset is applied as an additive screen-space correction.
- Automatic drift detection: if the running residual on implicit samples grows by more than 50%, GazeOS suggests a quick correction.

### 11.7 What is stored

- Per profile: calibration parameters (a few kilobytes as JSON) and the calibration feature vectors (not images) as a NumPy file.
- Raw camera frames are never saved unless the user turns on research recording (Section 21).

## 12. Signal processing and cursor control

Raw per-frame gaze is noisy, so this stage decides what the eye is doing (fixating or jumping), smooths only when that helps, and turns the result into a cursor that feels steady yet responsive. It also uses the model's own uncertainty, which ties the deep learning and signal-processing parts together.

### 12.1 What the eyes do

| Movement | Typical duration or speed | What the cursor should do |
| --- | --- | --- |
| Fixation | About 200 to 400 ms on one spot | Stay perfectly still |
| Saccade | Very fast jump, tens of milliseconds | Jump to the new spot at once |
| Smooth pursuit | Following a moving object | Follow smoothly |
| Micro-movements and model noise | Small jitter within about 1 degree | Ignore |

### 12.2 Adaptive smoothing: One Euro filter

The One Euro filter (Casiez, Roussel and Vogel, CHI 2012) is a low-pass filter whose cutoff frequency rises with speed. Slow movement is smoothed heavily (no jitter) and fast movement passes almost unfiltered (no lag).

```latex
f_c = f_{c,\min} + \beta \, |\dot{\hat{x}}|, \qquad \tau = \frac{1}{2\pi f_c}, \qquad \alpha = \frac{1}{1 + \tau / T_e}
```

- T\_e is the time between frames; the filtered value is x̂\_i = α x\_i + (1 − α) x̂\_(i−1).
- Starting values: minimum cutoff 1.0 Hz, β = 0.007, derivative cutoff 1.0 Hz, tuned in the user study.
- Applied separately to screen x and y.

### 12.3 Alternative: uncertainty-aware Kalman filter

- A constant-velocity Kalman filter with state (x, y, vx, vy).
- The measurement noise for each frame is set from the gaze network's predicted variance (Section 10.4), so frames the model is unsure about move the cursor less.
- The ablation compares no filter, moving average, One Euro, standard Kalman, and uncertainty-aware Kalman on jitter, lag, and selection time.

### 12.4 Fixation and saccade detection

- **Velocity rule (I-VT):** angular velocity above 30 degrees per second means a saccade.
- **Dispersion rule (I-DT):** samples staying within 1 degree for at least 100 ms form a fixation.
- During a fixation the cursor sits at the running median of the fixation's samples, which removes jitter almost completely.
- When a saccade is detected the filter state is reset so the cursor jumps without trailing.

### 12.5 Mapping to pixels

- Screen point in millimetres (Section 11.3) is converted to pixels using the monitor's physical size and resolution.
- Windows display scaling (125%, 150%) is handled by making the process DPI-aware, so the cursor lands on the right pixel.
- The cursor is clamped to the screen, with magnetic edges: within 1 degree of an edge, it snaps onto the edge, which makes the taskbar and window close buttons easy to hit.

### 12.6 Target snapping (Bayesian target selection)

1. Read clickable elements near the gaze point from the OS accessibility tree (buttons, links, menu items, text fields), within a radius of 2 standard deviations of the current error.
2. Score each candidate by how likely the user is looking at it, given the gaze point and its uncertainty, weighted by a prior favouring larger and recently used elements:

```latex
P(T_j \mid g) \propto \exp\!\left( -\frac{\lVert g - c_j \rVert^2}{2\sigma^2} \right) \cdot \pi_j
```

3. If the best candidate's probability is above 0.7, the cursor snaps to its centre; otherwise zoom-click is used (Section 7.3).

### 12.7 Dwell click state machine

1. **Idle:** the cursor is moving.
2. **Dwelling:** the cursor has stayed within a dwell radius (default 1.5 degrees) of an anchor point; the ring starts filling.
3. **Fired:** the dwell time is reached; the selected action is sent to the OS.
4. **Cooldown:** for 500 ms no new dwell can start at the same spot, which prevents repeated clicks.
5. Leaving the radius at any time returns to Idle and cancels the ring.

## 13. Blink and facial gesture recognition

A small temporal neural network classifies intentional blinks, winks, and facial gestures from a one-second window of face features. It replaces fixed thresholds, which misfire when the user talks, yawns, or wears glasses.

### 13.1 Input features (per frame, 30 FPS)

- **Eye Aspect Ratio (EAR)** for each eye, from six eye-contour landmarks (Soukupová and Čech, 2016):

```latex
\mathrm{EAR} = \frac{\lVert p_2 - p_6 \rVert + \lVert p_3 - p_5 \rVert}{2 \, \lVert p_1 - p_4 \rVert}
```

- **MediaPipe blendshapes** (12 selected of 52): eyeBlinkLeft, eyeBlinkRight, eyeSquintLeft, eyeSquintRight, browInnerUp, browOuterUpLeft, browOuterUpRight, jawOpen, mouthSmileLeft, mouthSmileRight, mouthPucker, cheekPuff.
- **Head-pose change** since the previous frame (3 values), so a nod is not read as a gesture.
- Total: 17 features per frame; a window of 30 frames gives a 30 x 17 input.

### 13.2 Models

| ID | Model | Notes |
| --- | --- | --- |
| B0 | Rule-based: EAR below a threshold for a set duration | Classical baseline |
| B1 | 1D temporal CNN: 3 conv layers (kernel 5, channels 32-64-64), global pool, FC to 9 classes | About 30,000 parameters; under 1 ms per window |
| B2 | GRU (2 layers, 64 hidden units) on the same input | Compared with B1 for accuracy and latency |

**Classes (9):** none, normal blink, long blink, left wink, right wink, eyebrow raise, mouth open, smile, and "talking or yawning" (a deliberate negative class so speech does not trigger clicks).

### 13.3 Data

- **Own dataset (main contribution):** 20 or more participants; each records 20 repetitions of every gesture plus 5 minutes of natural behaviour (reading, talking, watching a video) as negatives. Labelled with a simple keypress tool while recording.
- **Public data for blinks:** the Eyeblink8 and Talking Face video datasets, commonly used in blink-detection research, for pre-training and testing.
- **Augmentation:** time warping (0.8 to 1.2x speed), Gaussian noise on features, random feature dropout to imitate landmark failures.

### 13.4 Personal adaptation

- During onboarding (Section 7.6), the user performs 5 examples of each enabled gesture.
- Only the final layer is fine-tuned on these, in under 2 seconds.
- The user's gesture-size setting scales how strong a gesture must be, the same idea Google used in [Project Gameface](https://blog.google/technology/ai/google-project-gameface/).

### 13.5 Event logic

- An event fires when one class stays above probability 0.8 for its minimum duration and the face then returns to neutral.
- Refractory period: 300 ms after any event.
- During a blink, the gaze cursor is frozen at its pre-blink position, so the blink itself does not move the click point.

### 13.6 Metrics

- Event-level precision, recall, and F1, counting a detection as correct within 150 ms of the labelled event.
- False activations per 10 minutes of natural use (target: 1 or fewer every 5 minutes).
- Latency from the end of the gesture to the event.

## 14. Predictive text and language model

Gaze typing is slow, so every saved keystroke matters. GazeOS combines a language model with a model of gaze error: it asks which word the user most likely meant, given both what people usually say and which keys were near where they looked. That makes it tolerant of near-miss key selections.

### 14.1 Prediction stack

| Layer | Model | Job | Latency target |
| --- | --- | --- | --- |
| L0 | Prefix trie with word frequencies | Instant completion of the current word | Under 1 ms |
| L1 | Word 3-gram with Kneser-Ney smoothing (KenLM) | Fast next-word prediction | Under 5 ms |
| L2 | Small transformer LM trained by the team (4 layers, 256 dimensions, about 10 million parameters), exported to ONNX | Better context-aware next word | Under 30 ms |
| L3 (P2) | Local LLM through Ollama, for example Qwen 2.5 1.5B | Whole-sentence completion and suggested replies, shown when confident | Under 500 ms, runs asynchronously |

Suggestions from all layers are merged and re-ranked by combined score; the top 3 to 5 are shown.

### 14.2 Noisy-channel decoding of gaze keystrokes

Each gaze key selection is treated as noisy evidence. The probability of a word w given the sequence of gaze points g\_1 to g\_n combines the language model with a Gaussian model of gaze error around each key centre, with spread σ taken from the user's measured calibration error:

```latex
P(w \mid g_{1:n}) \propto P_{\mathrm{LM}}(w \mid \mathrm{context}) \prod_{i=1}^{n} \exp\!\left( -\frac{\lVert g_i - k(w_i) \rVert^2}{2\sigma^2} \right)
```

Here k(w\_i) is the centre of the key for the i-th letter of w. This is the same idea smartphone keyboards use to fix fat-finger typos, adapted to gaze.

### 14.3 Training data

- Conversational English: DailyDialog and OpenSubtitles (from the OPUS collection), since AAC users mostly type everyday conversation.
- AAC-style messages: the crowdsourced AAC corpus from Vertanen and Kristensson (EMNLP 2011), built for exactly this use.
- Hindi and Gujarati (P2): AI4Bharat's IndicCorp.
- Preprocessing: lowercase, keep a 30,000-word vocabulary, byte-pair encoding for the transformer.

### 14.4 Personalization

- A local user dictionary learns new words and boosts frequent ones.
- A phrase bank stores whole sentences the user saves or types often.
- Personal data stays in the local profile and is never uploaded.

### 14.5 Text-to-speech

- Default: Windows' built-in speech engine through pyttsx3 (offline, zero setup).
- Better quality: Piper, an offline neural text-to-speech engine.

### 14.6 Metrics

- **Keystroke savings rate:** KSR = 1 − (keystrokes with prediction ÷ characters in the text). Target 40% or more.
- **Top-3 next-word accuracy** on a held-out conversational test set.
- **Perplexity** of L1 and L2 on the same test set.
- **Words per minute** in the user study, with and without prediction.

## 15. Datasets and data pipeline

Training uses ETH-XGaze for scale and head-pose variety, MPIIFaceGaze as the main laptop-webcam benchmark, and a new GazeOS-Webcam dataset collected by the team, which is the closest match to real use and a contribution in its own right.

### 15.1 Gaze datasets

| Dataset | Size | What it captures | Used for | Access |
| --- | --- | --- | --- | --- |
| [ETH-XGaze](https://arxiv.org/pdf/2007.15837) | 110 participants, 1,083,492 images, 18 cameras | Extreme head poses and gaze angles, controlled lighting, high resolution | Pre-training | Request through the project site; academic use |
| MPIIGaze / MPIIFaceGaze | 15 participants, [213,659 images](https://models.luxonis.com/luxonis/l2cs-net/7051c9d2-78a4-420b-91a8-2d40ecf958dd?backTo=%2F) | Laptop webcams in daily life over several months | Fine-tuning and the main benchmark (leave-one-subject-out) | Free download for research |
| Gaze360 | [238 subjects](https://models.luxonis.com/luxonis/l2cs-net/7051c9d2-78a4-420b-91a8-2d40ecf958dd?backTo=%2F), indoor and outdoor | Very wide gaze range, varied people | Cross-dataset robustness test | Registration form |
| [GazeCapture](https://gazecapture.csail.mit.edu/index.php) | 1,474 participants, about 2.5 million frames | Phones and tablets, crowdsourced, varied lighting | Optional extra pre-training | Registration form |
| **GazeOS-Webcam** (new, own) | Target: 15 to 20 participants, 3 sessions each | Real laptop webcams, the target use case | Personalization experiments, final test set | Collected by the team with consent |

### 15.2 GazeOS-Webcam collection protocol

- **Setup:** each participant's own laptop where possible (camera variety is valuable), seated at 50, 65, and 80 cm.
- **Conditions:** bright room and dim room; with and without glasses where relevant.
- **Tasks per session:** 16-point static targets (calibration-style), a moving target following a smooth path (pursuit), and free reading for gesture negatives.
- **Labels:** target position in screen pixels and millimetres, timestamp, screen size, camera resolution, and distance.
- **Volume:** about 2,000 labelled frames per session; about 100,000 frames in total.
- **Stored as:** normalized face and eye crops (not full frames), plus landmarks and head pose. Full frames are deleted after processing.
- **Consent:** written consent form; participants may withdraw and have data deleted (Section 21).

### 15.3 Splits

- **ETH-XGaze:** the official training split; a held-out 10% of training subjects for validation (official test labels are withheld for the leaderboard).
- **MPIIFaceGaze:** standard 15-fold leave-one-subject-out, for comparison with published numbers.
- **GazeOS-Webcam:** subject-independent split (no person appears in both training and test); sessions 1 and 2 for adaptation, session 3 for testing.

### 15.4 Pipeline

1. **Download and verify** each dataset; store checksums in a manifest file.
2. **Normalize** with the same code used at runtime (Section 9.4), so training and deployment see identical inputs. ETH-XGaze's pre-normalized 224x224 release can be used directly.
3. **Pack into HDF5 shards** (images, gaze labels, head pose, subject ID) for fast loading on Kaggle and Colab.
4. **Subsample** ETH-XGaze if disk or time limits bite (for example every third frame per camera), and record the exact subset.
5. **Version the data** with DVC or a manifest tied to each Git commit, so every result can be traced to its data.

### 15.5 Licensing

All public gaze datasets are for non-commercial research use. The project cites each dataset's paper, does not redistribute images, and keeps GazeOS-Webcam private to the team.

## 16. Training strategy, baselines, and ablations

Twelve planned experiments each answer one question with one table or chart in the final report. Together they prove that every component earns its place, which is what separates a research-grade project from a demo.

### 16.1 Training stages

1. **Stage A, pre-training:** GazeNet-GOS (M3) on ETH-XGaze for 25 epochs.
2. **Stage B, fine-tuning:** on MPIIFaceGaze for 10 epochs. For the benchmark this runs inside each leave-one-subject-out fold; for deployment it uses all 15 subjects.
3. **Stage C, distillation:** train the Lite model (M4) from M3 (Section 10.7).
4. **Stage D, meta-training (optional):** episodic training for personalization method P5.
5. **Gesture model:** train B1 and B2 on the gesture data (Section 13).
6. **Language model:** train the n-gram and the small transformer (Section 14).

### 16.2 Experiment plan

| ID | Question | Compared | Main metric | Data |
| --- | --- | --- | --- | --- |
| E1 | How good is our model against baselines? | M0 geometric, M1 L2CS-Net, M3 ours | Mean angular error | MPIIFaceGaze, leave-one-subject-out |
| E2 | Which input streams matter? | Face only, eyes only, face + eyes, + head pose, concat fusion vs transformer fusion | Angular error | MPIIFaceGaze |
| E3 | Which output and loss work best? | Pure L2 regression, classification only, combined loss with β = 1 and 2; 3-degree vs 4-degree bins | Angular error | MPIIFaceGaze |
| E4 | Does normalization matter? | No normalization, crop only, full normalization | Angular error | MPIIFaceGaze and GazeOS-Webcam |
| E5 | Does large-scale pre-training help? | ImageNet only vs ImageNet + ETH-XGaze | Angular error, cross-dataset error on Gaze360 | All |
| E6 | Which personalization works, with how few points? | P0 to P5, each with 1, 3, 5, 9, and 16 calibration points | Angular error after calibration | GazeOS-Webcam |
| E7 | Which filter gives the best cursor? | None, moving average, One Euro, Kalman, uncertainty-aware Kalman | RMS jitter, lag in ms, selection time | GazeOS-Webcam + user study |
| E8 | How much speed for how much accuracy? | M3, M4 fp32, M4 INT8 | Error vs milliseconds per frame | MPIIFaceGaze + laptop benchmark |
| E9 | Where does it fail? | Breakdown by lighting, glasses, distance, head yaw | Angular error per condition | GazeOS-Webcam |
| E10 | Do learned gestures beat thresholds? | B0 rules, B1 CNN, B2 GRU | Event F1, false activations per 10 min | Gesture dataset |
| E11 | How much does prediction speed typing? | No prediction, L0, L0+L1, L0+L1+L2, with and without noisy-channel decoding | Keystroke savings, words per minute | Text test set + user study |
| E12 | Can the model tell when it is wrong? | Predicted uncertainty vs actual error | Spearman correlation, reliability diagram | MPIIFaceGaze |

### 16.3 Experiment hygiene

- **Configs:** every run is defined by a Hydra YAML config committed to Git.
- **Tracking:** Weights & Biases free tier (or TensorBoard) logs losses, metrics, configs, and the Git commit of every run.
- **Seeds:** 3 seeds for headline results; report mean ± standard deviation.
- **Significance:** paired t-test or Wilcoxon signed-rank test across subjects when claiming one method beats another (p < 0.05).
- **No test-set tuning:** all hyperparameters are chosen on validation subjects only.

### 16.4 Compute budget (estimate)

| Job | GPU hours (approximate) |
| --- | --- |
| Stage A pre-training on an ETH-XGaze subset | 15 to 20 |
| Stage B, 15 leave-one-subject-out folds | 10 to 15 |
| Ablations E2 to E5 | 15 to 20 |
| Distillation, meta-training, gesture model, language model | 8 to 12 |
| **Total** | **About 50 to 65** |

Kaggle's roughly 30 free GPU hours per week covers this over 2 to 3 weeks of training. Colab is the backup.

## 17. Evaluation plan

GazeOS is evaluated twice: offline against public benchmarks (is the ML good?) and in a user study with real tasks (is the product usable?). Both use standard metrics so the results can be compared with published work.

### 17.1 Offline metrics

- **Angular error:** the angle between predicted and true 3D gaze vectors, averaged over test samples.

```latex
\varepsilon = \arccos\!\left( \frac{\hat{g} \cdot g}{\lVert \hat{g} \rVert \, \lVert g \rVert} \right)
```

- **On-screen error:** Euclidean distance between predicted and true screen points, in cm and in pixels.
- **Precision:** root mean square of sample-to-sample distances during a fixation (lower means steadier).
- **Per-condition breakdown:** error by lighting, glasses, distance, and head angle (experiment E9).
- **Speed:** per-stage timing on a reference laptop (Section 18).

### 17.2 User study design

- **Participants:** 15 to 20 able-bodied adults (needed for statistics), plus 2 to 3 users with motor impairments as case studies if a local rehabilitation centre or NGO can partner, with caregiver involvement.
- **Design:** within-subject; each person uses every condition, in an order balanced with a Latin square.
- **Conditions:** (1) GazeOS eyes-only, (2) GazeOS eyes + face, (3) a WebGazer-style baseline (classical features + linear regression), (4) a normal mouse as the upper bound.
- **Session length:** about 60 minutes per participant, with breaks.

### 17.3 Tasks

| Task | What the participant does | Measures |
| --- | --- | --- |
| T1 Calibration | 9-point calibration, then validation | Calibration time, validation error |
| T2 Target acquisition | ISO 9241-411 multi-directional task: 13 circular targets in a ring, widths 32, 64, and 128 px | Selection time, error rate, Fitts' throughput |
| T3 Web task | Open a news site, search a phrase, open the third result, scroll to the end | Completion time, success, errors |
| T4 Typing | Transcribe 5 short phrases from the MacKenzie-Soukoreff phrase set | Words per minute, error rate, keystroke savings |
| T5 Reading | Read an article for 5 minutes with tracking on | Accidental clicks |
| T6 Robustness | Lean back or shift position, then repeat a short T2 | Error increase; whether quick recalibration fixes it |

### 17.4 Fitts' law throughput

Throughput summarizes speed and accuracy in bits per second, using the effective target width so it accounts for where clicks actually landed:

```latex
TP = \frac{ID_e}{MT}, \qquad ID_e = \log_2\!\left( \frac{D}{W_e} + 1 \right), \qquad W_e = 4.133 \times SD_x
```

D is the distance to the target, MT the movement time, and SD\_x the standard deviation of click positions along the movement axis.

### 17.5 Questionnaires

- **System Usability Scale (SUS):** 10 questions, scored 0 to 100; 68 is average, 70 or more is the target.
- **NASA-TLX:** workload on six scales (mental, physical, temporal demand, performance, effort, frustration).
- **Open feedback:** what was hardest, what they would change.

### 17.6 Statistical analysis

- Repeated-measures ANOVA (or the Friedman test if the data are not normal), followed by pairwise tests with Holm correction.
- Effect sizes reported (partial eta squared, or Cohen's d for pairs).
- All analysis scripts are kept in the repository so the numbers can be reproduced.

### 17.7 Ethics

- Approval from the university's ethics committee before any data collection.
- Informed consent, the right to withdraw, and anonymized participant IDs (P01, P02...).
- Extra care for participants with disabilities: sessions at their pace, a caregiver present, stop at any sign of fatigue.

## 18. Non-functional requirements

The whole pipeline must fit in about 60 ms of processing per frame on an ordinary laptop CPU, leaving room under the 100 ms end-to-end target for display refresh and filter lag.

### 18.1 Reference machine

An Intel Core i5 (11th generation) or AMD Ryzen 5 (5000 series) laptop with 8 GB RAM, integrated graphics only, and a 720p built-in webcam, running Windows 11. Every performance target applies to this machine.

### 18.2 Latency budget per frame

| Stage | Budget (ms) |
| --- | --- |
| Camera capture and transfer | 20 |
| Face landmarks (MediaPipe) | 8 |
| Head pose and normalization | 2 |
| Gaze network (M4, INT8) | 8 |
| Gesture network | 1 |
| Filtering, fixation logic, screen mapping | 1 |
| OS cursor update | 1 |
| Display refresh (60 Hz) | 8 to 16 |
| **Subtotal** | **49 to 57** |
| Filter lag during fast movement | 20 to 30 |
| **End-to-end** | **About 70 to 90 (target 100 or less)** |

### 18.3 Other requirements

| Category | Requirement |
| --- | --- |
| Throughput | 30 FPS sustained; if the CPU is busy, drop frames rather than queue them, so latency never builds up |
| Resource use | 30% or less total CPU; 500 MB or less RAM; no GPU needed |
| Start-up | App ready and tracking within 5 s of launch with a saved profile |
| Robustness | Works with glasses; from 40 to 80 cm; head yaw up to ±30 degrees; ordinary indoor lighting |
| Reliability | Runs 2 hours with no crash or memory growth over 50 MB; recovers automatically if the camera disconnects |
| Safety | Auto-pause on face loss; emergency stop hotkey (Ctrl + Alt + G) that always works; never sends keystrokes while paused |
| Privacy | Works fully offline; no frames saved or sent by default (Section 21) |
| Portability | Windows 10/11 required; macOS 13+ and Ubuntu 22.04+ as stretch targets |
| Accessibility | WCAG 2.1 AA contrast; every GazeOS screen operable by gaze alone |
| Maintainability | Modular packages with typed interfaces; 70% or more unit-test coverage of core logic; documented configs |
| Installability | One installer (.exe) or a single pip install; no admin rights needed beyond camera permission |

## 19. Tech stack

Everything is Python, free, and open source: PyTorch for training, ONNX Runtime for fast CPU inference, MediaPipe and OpenCV for vision, and Qt (PySide6) for the on-screen interface.

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11 | One language for ML, vision, and app |
| Vision | OpenCV 4.x, MediaPipe Tasks (Face Landmarker) | Camera capture, solvePnP, warping; fast landmarks |
| Training | PyTorch 2.x, torchvision, timm (pretrained backbones) | Standard research stack |
| Configs and tracking | Hydra, Weights & Biases (free tier) or TensorBoard | Reproducible experiments |
| Inference | ONNX, ONNX Runtime (CPU) with INT8 quantization | Several times faster than PyTorch on CPU; no GPU needed |
| Numerics and classical ML | NumPy, SciPy, scikit-learn (ridge regression, Gaussian processes) | Calibration and personalization |
| Filtering | Own One Euro implementation; filterpy for Kalman filters | Cursor smoothing |
| Language model | KenLM (n-gram), Hugging Face tokenizers and transformers (small GPT training), Ollama (optional local LLM) | Word and sentence prediction |
| Text-to-speech | pyttsx3 (Windows speech engine), Piper (neural, offline) | Speak typed text |
| User interface | PySide6 (Qt 6): transparent, click-through, always-on-top overlay windows | Cursor overlay, dwell ring, keyboard, settings |
| OS input | Windows SendInput through ctypes or pywin32; pynput as the cross-platform fallback | Move cursor, click, type |
| Accessibility tree | uiautomation (Windows UI Automation); pyatspi (Linux); pyobjc Accessibility (macOS) | Target snapping |
| Data | h5py (HDF5 shards), DVC (data versioning) | Fast loading, traceable data |
| Analysis | pandas, statsmodels, pingouin, matplotlib, seaborn | Statistics and charts for the report |
| Testing and CI | pytest, hypothesis (property tests), GitHub Actions | Quality gates on every push |
| Packaging | PyInstaller (Windows .exe) | One-click install for demo and users |
| Documentation | MkDocs (Material theme) | Developer and user docs |

**Compute for training:** Kaggle notebooks (free GPU, about 30 hours a week) as the main option and Google Colab as backup. **Runtime:** the user's own laptop CPU only.

## 20. OS integration

GazeOS controls the computer by injecting ordinary mouse and keyboard events, so it works with every app without plugins. One small interface hides the differences between Windows, macOS, and Linux.

### 20.1 Input backend interface

```python
class InputBackend(Protocol):
    def move(self, x: int, y: int) -> None: ...
    def click(self, button: str = "left", count: int = 1) -> None: ...
    def press(self, button: str) -> None: ...      # drag start
    def release(self, button: str) -> None: ...    # drag end
    def scroll(self, dx: int, dy: int) -> None: ...
    def type_text(self, text: str) -> None: ...
    def hotkey(self, *keys: str) -> None: ...
    def element_at(self, x: int, y: int) -> UIElement | None: ...  # for snapping
```

### 20.2 Windows (primary target)

| Need | How |
| --- | --- |
| Move cursor | SendInput with MOUSEEVENTF\_MOVE, ABSOLUTE, and VIRTUALDESK flags; coordinates scaled to 0 to 65,535 |
| Click and drag | SendInput with LEFTDOWN/LEFTUP, RIGHTDOWN/RIGHTUP |
| Scroll | SendInput with MOUSEEVENTF\_WHEEL, in steps of 120 |
| Type any character | SendInput with KEYEVENTF\_UNICODE, which also handles Hindi and Gujarati text |
| Shortcuts | Virtual-key codes (Ctrl+C, Alt+Tab, Win key) |
| Correct pixels at 125% or 150% scaling | Per-monitor DPI awareness (SetProcessDpiAwarenessContext, per-monitor v2) |
| Overlay that never steals focus | Qt flags: stay on top, frameless, tool window, translucent, transparent to mouse; plus Windows extended styles WS\_EX\_LAYERED, WS\_EX\_TRANSPARENT, WS\_EX\_NOACTIVATE |
| Keyboard that does not take focus | The gaze keyboard window uses WS\_EX\_NOACTIVATE, so typed keys go to the app the user was in |
| Element under the gaze | UI Automation ElementFromPoint; read the bounding rectangle and whether it can be invoked |
| Camera | OpenCV with the Media Foundation or DirectShow backend |

**Known limitation:** Windows blocks injected input into apps running as administrator (User Interface Privilege Isolation). The fix for a full release is a signed build with the uiAccess manifest flag, installed under Program Files. For the project, GazeOS documents the limitation and can optionally run elevated.

### 20.3 macOS (stretch)

- Input through Quartz CGEventPost (pyobjc).
- Needs the user to grant Accessibility and Camera permissions in System Settings; onboarding shows how.
- Element under the gaze through AXUIElementCopyElementAtPosition.

### 20.4 Linux (stretch)

- X11: input through the XTest extension (python-xlib).
- Wayland: blocks synthetic input from ordinary apps, so GazeOS uses a virtual uinput device (python-evdev), which requires the user to be in the input group.
- Element under the gaze through AT-SPI (pyatspi).

## 21. Privacy, security, and ethics

GazeOS watches a person's face continuously and can type and click for them, so privacy and safety are core requirements, not extras. By default nothing leaves the computer and no camera frame is ever saved.

### 21.1 Privacy by design

- **Offline by default:** no network calls at runtime. The optional local LLM also runs on the device.
- **No frames stored:** frames live only in memory for the few milliseconds they are processed.
- **What is stored:** calibration parameters, settings, the personal dictionary, and phrase bank, all in the local user profile folder. A privacy panel lists exactly these and offers "Delete all my data".
- **Research recording:** off by default; turning it on shows a clear banner, saves only normalized eye and face crops, and asks for consent each session.
- **Gaze data is sensitive:** gaze patterns can reveal interests and can even identify a person, so session heatmaps (FR-41) are opt-in and stored locally only.
- **Legal context:** handling of personal data follows India's Digital Personal Data Protection Act, 2023 (consent, purpose limitation, right to erasure).

### 21.2 Security

- GazeOS can inject keystrokes, so it must not become a tool for others to control the machine: it accepts no remote commands, opens no network ports, and has no plugin system in this version.
- Typed text is never logged except in the opt-in experiment logs, where passwords are excluded (no logging while the focused field is a password field, detected through UI Automation).
- Releases are built by CI from tagged commits, with a checksum published for the installer.

### 21.3 Safety

- Auto-pause when the face is lost; an emergency stop hotkey; a visible status chip at all times.
- Confirmation before destructive shortcuts (for example closing a window with unsaved work, Alt+F4).
- Rest reminders to limit eye strain.
- Clear documentation that GazeOS is an assistive tool, not a certified medical device, and should not be the only way a person can call for help.

### 21.4 Ethics and fairness

- **Bias check:** accuracy is reported separately by skin tone (Fitzpatrick scale groups, self-reported), glasses, and age group, to make sure the system does not work worse for some people. Gaps over 0.5 degrees are flagged and investigated.
- **Inclusive design:** people with disabilities are involved in testing where possible, following "nothing about us without us".
- **Study ethics:** ethics committee approval and informed consent (Section 17.7).
- **Honest claims:** the report states limitations plainly, including conditions where GazeOS does not work well.

## 22. Repository structure and engineering practices

One monorepo holds the app, the training code, and the research analysis, split so that training code never ships in the app and the app never depends on PyTorch.

```text
gazeos/
├── app/                      # the desktop application (ships to users)
│   ├── capture/              # camera handling, frame queue
│   ├── vision/               # landmarks, head pose, normalization
│   ├── inference/            # ONNX Runtime wrappers: gaze, gesture, LM
│   ├── calibration/          # protocol, personalization methods P0-P5
│   ├── control/              # filters, fixation detection, mapping, snapping, dwell
│   ├── gestures/             # gesture event logic
│   ├── typing/               # keyboard UI, prediction merge, noisy-channel decoder, TTS
│   ├── os_backends/          # windows.py, macos.py, linux_x11.py, linux_uinput.py
│   ├── ui/                   # PySide6 overlay, action bar, settings, onboarding
│   ├── profiles/             # profile storage, privacy panel
│   └── main.py
├── training/                 # never shipped
│   ├── data/                 # dataset classes, HDF5 packing, augmentation
│   ├── models/               # GazeNet-GOS, Lite, gesture nets, small GPT
│   ├── losses/               # L2CS combined loss, NLL, distillation
│   ├── train.py, distill.py, meta_train.py
│   └── export_onnx.py, quantize.py
├── configs/                  # Hydra YAML for every experiment
├── research/                 # analysis notebooks, stats scripts, figure generation
├── tools/                    # data-collection app, labelling tool, latency profiler
├── tests/                    # unit, integration, replay tests
├── models/                   # exported .onnx files (Git LFS)
├── docs/                     # MkDocs site
└── .github/workflows/        # CI: lint, type-check, tests, build
```

**Practices:**

- Git with feature branches and pull requests; at least one teammate reviews every merge.
- Code style with ruff and black; static types with mypy in strict mode on app/.
- Conventional commit messages; tagged releases (v0.1 prototype, v0.5 alpha, v1.0 final).
- Model files tracked with Git LFS; each .onnx file has a model card (training data, metrics, commit).
- A single settings schema (pydantic) validates every configurable value.
- Architecture decisions recorded as short ADR notes in docs/ (for example "why ONNX Runtime instead of TorchScript").

## 23. Testing and QA

A live camera system is hard to test repeatably, so GazeOS records short sessions once and replays them through the pipeline in every test run. Any change that makes recorded accuracy or latency worse fails CI.

| Level | What is tested | How | Pass rule |
| --- | --- | --- | --- |
| Unit | Normalization maths, One Euro and Kalman filters, fixation detector, dwell state machine, noisy-channel decoder, Fitts' calculations | pytest; hypothesis property tests (for example, normalizing then de-normalizing returns the original gaze vector) | All pass; 70%+ coverage of core logic |
| Model | Exported ONNX models | Parity against PyTorch on 1,000 frames; accuracy on a fixed mini test set | Parity under 0.05 degrees; accuracy within 0.2 degrees of the recorded baseline |
| Replay integration | Whole pipeline from frames to cursor positions | 10 recorded sessions (different people, lighting, glasses) with known targets, replayed frame by frame | Mean error and jitter no worse than the last release by more than 5% |
| Performance | Per-stage and end-to-end timing | Latency profiler on the reference laptop; 10-minute soak test | Meets the budget in Section 18.2; no memory growth over 50 MB |
| OS backend | Mouse, keyboard, and scroll injection | Scripted test against a test window that records received events | Every event arrives, in order, at the right coordinates |
| UI | Overlay, keyboard, settings, onboarding | Manual checklist each release; pytest-qt for key widgets | Checklist complete; no focus stealing |
| Accessibility | Contrast and gaze-only operation of GazeOS's own screens | Contrast checker; a teammate completes every screen with gaze only | WCAG 2.1 AA; no screen needs the mouse |
| Field test | Real use | Each teammate uses GazeOS for 30 minutes a day in the last 3 weeks and logs issues | No open P0 bugs at demo time |

**Bug tracking:** GitHub Issues with labels P0 (blocks demo), P1, and P2; a weekly triage meeting.

## 24. Timeline and milestones

The build runs 14 weeks with four milestone gates. Every P0 feature works by the end of week 8, so the last six weeks go to accuracy, speed, the user study, and the demo rather than to unfinished features.

&#91;embedded content: Build plan · 14 weeks, 11 workstreams, 4 milestones\]

Workstreams overlap on purpose: the vision front-end and control engine start with released L2CS-Net weights, so the app is usable long before the team's own model is trained.

**Open question:** which calendar week is week 1, and when is the final evaluation? If the semester has fewer than 14 weeks left, compress weeks 9 to 13 and cut P2, then P1, items (Section 27).

### 24.1 Milestone gates

- [ ] **Prototype (end of week 4):** the cursor follows gaze live using L2CS-Net weights, a basic 9-point calibration, the One Euro filter, and dwell click on Windows.
- [ ] **Alpha (end of week 8):** every P0 requirement works end to end; GazeNet-GOS (M3) trained; GazeOS-Webcam dataset collected; personalization P0 to P4 working.
- [ ] **Feature freeze (end of week 11):** distilled INT8 model (M4) deployed; all experiments E1 to E12 run; installer builds in CI; no open P0 bugs. Only fixes after this point.
- [ ] **Final demo (end of week 14):** user study analysed, report and slides finished, demo rehearsed twice, backup video recorded.

### 24.2 Week 1 to 2 checklist (do these first)

- [ ] Request ETH-XGaze, Gaze360, and GazeCapture access (approvals can take days)
- [ ] Submit the ethics committee application for the user study
- [ ] Contact one or two rehabilitation centres or NGOs about case-study participants
- [ ] Set up the repository, CI, Kaggle notebooks, and experiment tracking
- [ ] Run L2CS-Net's released model on a webcam to confirm the baseline works

### 24.3 Working rhythm

- Weekly 30-minute sync with a live demo of whatever works (even if rough).
- Weekly progress note to the project guide, with one chart from the experiments.
- Every second week, a teammate uses GazeOS for 20 minutes and files issues.

## 25. Team roles and module ownership

The plan assumes a team of four, each owning one layer of the pipeline end to end, including its experiments. Team size is still an open question; the notes under the table show how to split for three people or one.

| Role | Owns (modules) | Experiments | Main deliverables |
| --- | --- | --- | --- |
| A. ML lead | training/, inference/ (gaze), Sections 10 and 16 | E1, E2, E3, E5, E8, E12 | Trained M3 and M4, ONNX export, benchmark tables |
| B. Vision and personalization | vision/, calibration/, tools/ data-collection app, Sections 9, 11, 15 | E4, E6, E9 | Normalization pipeline, P0 to P5, GazeOS-Webcam dataset |
| C. Interaction and systems | control/, gestures/, os\_backends/, ui/, Sections 7, 12, 13, 20 | E7, E10 | Cursor engine, dwell and zoom-click, gesture model, Windows backend, overlay |
| D. Typing, product, and evaluation | typing/, profiles/, research/, Sections 14, 17, 21 | E11, user study | Keyboard, language models, TTS, user study, statistics, final report |

**Shared by all:** code review, the replay test set, weekly demo, and the final presentation.

**Team of three:** merge role D into B (typing) and C (user study), and drop P5 meta-learning and the GRU gesture model.

**Solo:** keep only P0 requirements; use L2CS-Net weights (M1) fine-tuned, instead of training M3 from scratch; personalization P1 to P4; One Euro filter; rule-based plus CNN blink model; n-gram prediction; a user study of 8 to 10 people. This is still a strong project.

## 26. Budget and resources

The project can be built for Rs 0. Optional spending of about Rs 2,000 to 4,000 buys a better webcam and spare GPU time. Prices are approximate.

| Item | Cost | Needed? |
| --- | --- | --- |
| Laptop with built-in webcam | Already owned | Yes |
| Python libraries, MediaPipe, PyTorch, ONNX Runtime, Qt | Free, open source | Yes |
| Public datasets (ETH-XGaze, MPIIFaceGaze, Gaze360, GazeCapture) | Free for research (registration for some) | Yes |
| Training GPU: Kaggle (about 30 h/week) and Colab free tier | Free | Yes |
| Experiment tracking (Weights & Biases free tier) | Free | Recommended |
| Local language model through Ollama | Free | Optional |
| Printed A4 checkerboard for camera calibration | About Rs 10 | Optional |
| Colab Pro, one month, if free GPU limits slow training | About Rs 1,000 | Optional |
| External 1080p 60 FPS webcam (better in dim rooms) | About Rs 1,500 to 3,000 | Optional |
| Snacks or small gift vouchers for user-study participants | About Rs 500 to 1,000 | Optional |
| Claude or another hosted AI API inside the app | Billed separately from any chat subscription; avoided by using local models | Not needed |

**Time resources:** about 12 to 14 weeks for a team of four at roughly 10 to 12 hours per person per week.

## 27. Risks and mitigations

The biggest risk is accuracy falling short of 2 degrees on real webcams; the design already contains three independent fallbacks for it (snapping, zoom-click, and large-key layouts), so the product stays usable even at 3 to 4 degrees.

| Risk | What happens if it hits | Mitigation | Fallback |
| --- | --- | --- | --- |
| Accuracy above 2 degrees on real webcams | Small targets are hard to hit | Normalization, personalization P3 to P5, implicit recalibration | Zoom-click, snapping, large keys; report honest numbers |
| Real-time speed missed on CPU | Laggy cursor | Distilled M4, INT8 quantization, frame dropping | Run gaze at 15 FPS with filtering at 30 FPS; smaller input size |
| Dataset access delayed (ETH-XGaze approval) | Pre-training slips | Request access in week 1 | Start from released L2CS-Net weights; use MPIIFaceGaze only |
| Not enough free GPU time | Fewer experiments | Plan runs early; subsample ETH-XGaze; smaller backbones | One month of Colab Pro (about Rs 1,000) |
| Poor lighting at demo venue | Tracking drops during the viva | Low-light warning, CLAHE, exposure lock | Bring a small desk lamp and an external webcam |
| Glasses glare | Eye crops wash out | Augmentation with glare; face stream compensates | Ask the user to tilt glasses slightly; report as limitation |
| Accidental clicks while reading | Frustration, lost trust | Dwell ring, cooldown, negative gesture class, easy pause | Longer default dwell; blink-only clicking |
| Windows blocks input to admin apps | Some windows cannot be controlled | Document it; run elevated when needed | uiAccess-signed build as future work |
| Few participants with disabilities available | Weaker real-world evidence | Contact rehab centres and NGOs early (week 2) | Able-bodied study plus simulated constraints (head fixed with chin rest) |
| Ethics approval slow | User study delayed | Submit the application in week 2 | Run pilot tests within the team only until approval |
| Scope creep | Core unfinished at demo | P0/P1/P2 priorities; feature freeze in week 11 | Cut P2, then P1 items |
| Team member unavailable | Module unfinished | Clear module ownership; weekly demos; shared code review | Solo-scope plan (Section 25) |

## 28. Demo plan for evaluation and viva

The demo hands control to a professor: within five minutes, an examiner who has never used GazeOS calibrates, opens a website, and types a sentence with their eyes. That moment does more than any slide.

### 28.1 Live demo script (about 12 minutes)

1. **Hook (1 min):** a teammate sits with hands folded and, using only their eyes, opens the presentation, moves to the next slide, and types "Good morning" which GazeOS speaks aloud.
2. **Under the hood (2 min):** switch to the debug view: live landmarks, normalized eye crops, the 3D gaze ray, the fusion-attention weights, and the confidence circle growing when the eyes are covered.
3. **Examiner takes over (4 min):** a professor calibrates in 30 seconds, sees their validation score, opens YouTube, searches, and plays a video, hands-free.
4. **Typing (2 min):** type a sentence; show predictions and the noisy-channel decoder fixing a near-miss key.
5. **Robustness (1 min):** turn down the lights or put on glasses; show the low-light warning and that tracking continues.
6. **Results (2 min):** the key charts: error by method, error against number of calibration points, and throughput against baselines.

### 28.2 Charts to have ready

- Bar chart: mean angular error of M0, M1, M3, and M4 (experiment E1).
- Line chart: error against calibration points for P0 to P5 (E6), the headline result.
- Speed against accuracy scatter for M3, M4 fp32, M4 INT8 (E8).
- Fitts' throughput and SUS by condition (user study).
- Error breakdown by lighting, glasses, and head angle (E9).
- Reliability diagram showing the model knows when it is wrong (E12).

### 28.3 Backup plan

- A pre-recorded 3-minute video of the full demo, in case the room's lighting or projector fails.
- A spare laptop with GazeOS installed and a calibrated profile.
- An external webcam and a desk lamp.

### 28.4 Deliverables at submission

- [ ] Installer (.exe) and source code repository
- [ ] Trained models with model cards
- [ ] Final report (with all experiment results) and this PRD
- [ ] Presentation slides and demo video
- [ ] GazeOS-Webcam dataset documentation (data itself kept private)
- [ ] Optional: a short paper for a student conference or a workshop

## 29. Future scope

After the semester, GazeOS can grow toward a major project, a research paper, or a real open-source tool used by people who need it.

- **Major project (8th semester):** Indian-language typing (Hindi, Gujarati) with transliteration and prediction; dwell-free gaze-swipe typing; voice plus gaze multimodal control ("click that" while looking).
- **Mobile and tablet versions:** Android app using the front camera and the same ONNX models.
- **Browser extension:** gaze control of web pages without installing a desktop app, using ONNX Runtime Web.
- **Smart-home and caregiver alerts:** look at on-screen tiles to switch lights or fans (through Home Assistant) or send an alert to a caregiver's phone.
- **AAC communication board:** symbol-based boards for users who cannot read, such as young children or people with aphasia.
- **Better models:** vision transformer backbones, self-supervised pre-training on unlabelled webcam video, and test-time adaptation that needs no calibration at all.
- **Clinical partnership:** a longer study with an ALS or spinal-injury rehabilitation centre.
- **Open-source release:** publish GazeOS so anyone can install it free, and invite contributors.
- **Publication:** the personalization comparison (E6) and the GazeOS-Webcam dataset are a natural basis for a workshop or conference paper.

## 30. References and glossary

### 30.1 Sources cited in this document

- [L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments](https://arxiv.org/pdf/2203.03339), Abdelrahman et al., 2022
- [ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation](https://arxiv.org/pdf/2007.15837), Zhang et al., ECCV 2020
- [Eye Tracking for Everyone (GazeCapture and iTracker)](https://gazecapture.csail.mit.edu/index.php), Krafka et al., CVPR 2016
- [WebGazer: Scalable Webcam Eye Tracking Using User Interactions](https://www.ijcai.org/Abstract/16/540), Papoutsaki et al., IJCAI 2016
- [Webcam eye tracking with head movement, PMC study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017)
- [L2CS-Net model card with MPIIGaze and Gaze360 dataset details](https://models.luxonis.com/luxonis/l2cs-net/7051c9d2-78a4-420b-91a8-2d40ecf958dd?backTo=%2F), Luxonis
- [Project Gameface on GitHub](https://github.com/google/project-gameface) and [Google Developers Blog post](https://developers.googleblog.com/en/project-gameface-makes-gaming-accessible-to-everyone/)
- [Introducing Project Gameface](https://blog.google/technology/ai/google-project-gameface/), Google, 2023
- [MediaPipe Face Landmarker overview (478 landmarks, blendshapes)](https://docs.comfy.org/tutorials/utility/face-detection/mediapipe)
- [Tobii Dynavox PCEye 5 listing with estimated cost](https://montech.ruralinstitute.umt.edu/mtdb/item_view.asp?ID=8366), Montana Assistive Technology Program
- [Eye gaze cost history](https://montech.ruralinstitute.umt.edu/eye-gaze-finally-affordable), Montana Assistive Technology Program
- [Measurement of Disability through Census, India 2011](https://unstats.un.org/unsd/demographic-social/meetings/2017/new-york--disability-egm/Session%204/India.pdf), Office of the Registrar General, via UN Statistics Division

### 30.2 Further reading (to collect for the report)

- Zhang, Sugano, Bulling: "Revisiting Data Normalization for Appearance-Based Gaze Estimation" (ETRA 2018)
- Zhang et al.: MPIIGaze and "It's Written All Over Your Face" (MPIIFaceGaze)
- Kellnhofer et al.: Gaze360 (ICCV 2019)
- Park et al.: "Few-Shot Adaptive Gaze Estimation" (FAZE, ICCV 2019)
- Casiez, Roussel, Vogel: "1 Euro Filter" (CHI 2012)
- Soukupová and Čech: "Real-Time Eye Blink Detection using Facial Landmarks" (2016)
- Vertanen and Kristensson: conversational AAC language modelling with crowdsourcing (EMNLP 2011)
- Majaranta and Räihä: surveys on text entry by gaze
- ISO 9241-411 and Soukoreff and MacKenzie on Fitts' law throughput

### 30.3 Glossary

| Term | Meaning |
| --- | --- |
| Appearance-based gaze estimation | Predicting gaze directly from eye and face images with a neural network, instead of modelling the eye's geometry |
| Angular error | The angle in degrees between predicted and true gaze direction |
| AAC | Augmentative and alternative communication: tools for people who cannot speak |
| Calibration | Short procedure where the user looks at known points so the system can adapt to them |
| Dwell click | Clicking by looking at a spot for a set time |
| EAR | Eye Aspect Ratio: a landmark-based measure of how open the eye is |
| Fixation | The eye resting on one spot, typically for a few hundred milliseconds |
| Fitts' throughput | Speed-accuracy measure of a pointing device, in bits per second |
| Head pose | The head's 3D rotation and position relative to the camera |
| Keystroke savings rate | Share of keystrokes saved by word prediction |
| Midas touch | The problem of everything you look at getting clicked |
| Normalization | Warping the face image as if a virtual camera looked straight at it from a fixed distance |
| One Euro filter | Adaptive low-pass filter that smooths slow motion and passes fast motion |
| ONNX Runtime | Fast engine for running exported neural networks on CPU |
| Saccade | A rapid jump of the eye between two points |
| SUS | System Usability Scale, a 10-question usability score out of 100 |
