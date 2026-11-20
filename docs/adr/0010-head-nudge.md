# ADR 0010: Eyes place the cursor, the head nudges it the last bit

**Status:** accepted, not yet tried live · **PRD:** 12 (FR-18), 11.5, 11.6

After the 18 Nov live runs and one on 20 Nov, the user found GazeOS "barely usable": the cursor was
off in a different place each time (near the scroll bar, near Close or P on the keyboard, near the
taskbar), with different calibrations and at different times, and heads do not stay still.

## What the recordings show (20 Nov 2026)

Scripts and data in `D:\GazeOS-cache\calib_study`. Four 9-dot recordings from two days and three
seating positions (27 Oct; 18 Nov at the normal distance, and further back, on light and dark
screens) were run through the app's pipeline with M3.

| Calibration | Leave one dot out | Fit on another recording | ...plus one re-aim dot |
| --- | --- | --- | --- |
| Screen polynomial (current) | 16.2 mm | 28.4 mm | 21.7 mm |
| Screen affine | 16.5 | 27.4 | 20.1 |
| Angle offset only | 132 | 125 | 132 |
| Affine on angles, camera / head / normalized frame | 16.3-16.5 | 28.0-36.6 | 18.4-20.1 |
| Second order on angles | 15.3-16.8 | 30.0-38.9 | 21.3-25.7 |
| Screen affine + head rotation and/or position as inputs | 16.5-17.8 | 28.1-83.6 | 19.9-23.2 |

No method holds better than about 16 mm (1.5 degrees) even within the sitting it was made in, and a
calibration from another sitting is about 28 mm off; a key is about 31 mm. Where the correction is
applied (screen, camera angles, head frame, the network's normalized frame) makes no real difference,
and head pose as an extra input overfits. The network already accounts for where the head is.

In the 18 Nov run 2 the user pressed Fix cursor 15 times. The offsets it found jumped by 60-100 px
between presses less than a minute apart while the head barely moved, so a one-dot re-aim is about as
noisy as the drift it corrects. The vertical part was almost always positive (cursor too high).

The places the user named are all at the screen's edges, outside the outer calibration dots (10% and
90%), where the error is largest and the user cannot look past a target to pull the cursor onto it.

## Decision

Accept that webcam gaze lands near, not on, the target, and let the user finish the pointing with the
head, which is steady:

- **Head nudge** (`gazeos/control/head_nudge.py`, mode `head_fine_tune`, now the default). The nose tip
  in the camera image jitters 0.32 px per frame (about 0.13 degrees of head rotation), against 0.5-1.2
  degrees for the head angles the old fine-tune mode used. While the eyes hold, the nose's movement since
  the gaze arrived moves the cursor: 1.5 screen widths per eye distance, with a dead band of 1% of the
  eye distance for sway. A key's worth of correction is about a 2.5 degree tilt; on recorded fixations,
  sway moves the cursor under a third of a key (95th percentile). The reference is taken 0.4 s after the
  gaze arrives, because the head keeps turning toward a new target for a moment after the eyes land. A
  saccade starts again from the gaze. The nudge is capped at 4 degrees.
- **Moving the head cancels the dwell** (cursor moving by head faster than 150 px/s; sway stays under
  about 120 px/s). Nothing is pressed on the way to the right key, and a full dwell is needed after.
- **Learning from nudges** (`CorrectionField` in `gazeos/calibration/implicit.py`, part of the drift
  corrector). When a dwell fires, the nudge in use is how far off the gaze was there. It is kept as a
  local correction (Gaussian, a quarter of the screen width) that fades with a 5-minute half-life and is
  shrunk toward zero (one nudge moves it two thirds of the way). Unlike ADR 0009's labels, which repeated
  the tracker's own error, a nudge is the user's own correction. A press without a nudge confirms the
  correction in place. A full calibration clears it; Fix cursor keeps it.

## Not done, and why

- Calibration dots nearer the edges: no recording has dots there to check the effect against.
- Head pose in the calibration fit: overfits (table above).
- Measured camera focal length: tried 27 Oct and 18 Nov, no help.

## To check live

Whether the gain and dead band suit the user; how often a nudge is needed after a few minutes of
learning; that sway does not cause false clicks. The trace (`GAZEOS_TRACE`) now has the nose position,
eye distance, the nudge and the learned correction per frame.

## First live run (20 Nov, 10:39, dry run, `D:\GazeOS-cache\live_20nov`)

The eyes often glide to a new spot instead of jumping, so no saccade was detected, and the head turned
along with them. That turn read as a nudge: it pinned the cursor in the bottom-right corner (235 px) and
carried it 190 px past the gaze onto Settings, which was pressed. A genuine nudge also showed up: the gaze
sat about 70 px above Menu and the nudge carried the cursor onto it. Fix: a nudge counts only while the
gaze point stays within 1.5 degrees of where the nudge started (`system.head_nudge_gaze_move_deg`), and only
the part of a nudge that moved the cursor within the screen is learned. Replayed on the run, the two bad
drags fall to 4 px and 1 px; nudges over 100 px fall from 6.3% to 0.7% of frames. Wider limits (2-3
degrees) keep more of a genuine nudge but let 60-70 px of the bad drag back, so 1.5 stays.

## Second live run and change of course (20 Nov, 10:55)

No more drags or false presses outside the keyboard, but the user found the nudge "very confusing:
helped at a few moments, ruined at a few", and needed Fix cursor 11 times: "every time I use Fix cursor
it feels really good for a few seconds before getting ruined again". The Fix cursor presses (30 over
three runs) show Fix itself is accurate (presses seconds apart agree within about 20 px) while the offset
moves with the head: the horizontal error follows the sideways head position (correlation 0.57-0.77
within a sitting). Compensating head position would remove only about 20% (36 -> 28 px horizontally).

**Head nudge is now off by default** (mode `eyes_only`; `head_fine_tune` stays available).

Instead, at the user's choice:
- **Glance aim after each word** (`gazeos/calibration/glance_aim.py`, `typing.aim_after_word`): after
  Space or a chosen prediction a dot shows in the middle of the keys for one second; the gaze from 0.35 s
  on re-aims the cursor. Nothing is pressed meanwhile. A glance whose gaze spread exceeds 1.9 degrees
  (real Fix cursor dots: median 28 px, 90th percentile 67 px, limit 111 px) or whose correction exceeds
  2.5 degrees changes nothing.
- **Learning from corrected near misses** (`MistakeLearner`, `calibration.learn_from_mistakes`): a
  letter, one Del, then a neighbouring letter teaches the local correction (the same fading field as the
  nudges, half-life 3 minutes). Kept because it is safe, but it is rare: one or two clean cases in 128
  recorded presses.

## Runs 3-5 (20 Nov, 11:16-11:42)

- Run 3: Menu could not be reached: the cursor sat 100-150 px left of the gaze at the right-hand bar (a
  calibration from another sitting). The side bar's buttons now respond from up to 2.5 degrees to their
  left (`TargetSync.reach`); run 4 pressed Menu with the cursor 95 px left of the bar.
- Run 4: Calibrate in the menu then could not be reached (the whole cursor about 125 px off). After run 6 the user
  chose calibration at every start (`calibration.at_startup`, hands-free; a poor run keeps the saved one).
- After run 4 the user asked for the head nudge back: default mode `head_fine_tune` again. Presses without
  a real nudge (under 0.3 degrees) are no longer stored, since they diluted the learned corrections.
- Run 5 (sitting closer, about 530 app mm): "deep adatiya" typed with three corrected errors; the cursor
  drifted 100-270 px low within minutes of calibrating, again following the head.

**Posture compensation** (`ControlEngine._posture`, `calibration.posture_*`). Over four sittings (30 Fix
cursor presses) the cursor's error followed the head with the same sign each time: +2.6 to +5.2 px per app
mm sideways, -4.6 to -18.6 vertically. The cursor is now corrected by +3.7 / -7.6 px per app mm (0.66 /
-1.36 screen mm per app mm) for the head's movement since the last aim (calibration, Fix cursor, an
accepted glance), head position smoothed at 0.5 Hz (about 1 px of added jitter), capped at 3 degrees.
Simulated on consecutive Fix presses, the change between them falls from a median of 79 to 58 px (better
in 73% of 30 pairs). Slopes fitted on the other sittings reduce every sitting's spread, so they carry
over between sittings.

## Run 6 (20 Nov, 11:49): "felt much better"

With the head nudge, glance aim, mistake learning, side-bar reach and posture compensation together,
"deep adatiya" was typed with no wrong key: 12 presses in 31 s (run 5: about 2 minutes with three
corrections; run 2: about 4 minutes). Two glance dots re-aimed the cursor; three were rightly ignored
because the user was reading the typed text, not the dot.
