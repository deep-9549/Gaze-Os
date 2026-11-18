# ADR 0006: Saccades by a confirmed jump; no gaze sample while the lids close

**Status:** accepted · **PRD:** 12.4 (deviation), 11.2 step 4 (extension)

## Saccades

The PRD detects saccades with I-VT: angular velocity above 30 deg/s, measured on
the filtered gaze. Each detection resets the cursor filter so the cursor jumps
without trailing.

On the laptop webcam (27 Oct 2026, fixation on a centre dot) the network's
frame-to-frame noise was about 1.8-2.0 deg vertically and 1.1-1.5 deg
horizontally. At 30 fps that alone reads as 40-60 deg/s. A simulation with the
real filter and detector at that noise level flagged 194 of 600 frames as
saccades; each reset restarted the filter at a raw, noisy point, which looked
like another saccade on the next frame. The cursor was effectively unfiltered.

GazeOS now calls a saccade when the raw gaze lands more than 7 deg from the
cursor on two consecutive frames in the same direction, or more than 14 deg on a
single frame. After a saccade the detector restarts from the raw point, so one
reset cannot trigger the next. Smaller moves are left to the One Euro filter,
whose cutoff rises with speed.

| Rule (simulated at the measured noise) | False resets per minute | Time to land within 1.5 deg of a 3 / 5 / 7 / 10 / 15 deg jump |
| --- | --- | --- |
| Velocity > 30 deg/s | 620 | 0-33 ms |
| Jump > 5 deg, 2 frames | 1.2 | 99 / 132 / 33 / 17 / 0 ms |
| Jump > 7 deg, 2 frames (chosen) | below 1 | 99 / 132 / 165 / 33 / 0 ms |

## Lids

Blinks were the other source of jitter. As the lids close, the network reads the
eye as looking down: on a 20 s webcam recording, frames with a MediaPipe
eye-blink score of 0.2-0.3 were 3 deg low on median, 0.3-0.5 were 6-8 deg low.
The existing "eyes closed" flag needs both eyes above 0.5, so it missed the
start and end of every blink (and some whole blinks, where one eye scored lower).
Frames where either eye's score exceeds `quality.gaze_lid_threshold` (0.3), and
`quality.gaze_lid_hold_frames` (1) after, now give no gaze sample: the cursor
holds still and calibration skips them. The raw gaze stays in the frame record.

## Result

Replaying the 20 s webcam recording through the engine (before calibration):

| | RMS sample-to-sample precision | Vertical SD | Saccade resets |
| --- | --- | --- | --- |
| Before (velocity rule, no lid gating) | 1.66 deg | 2.84 deg | 15 |
| Jump > 5 deg + lid gating | 1.04 deg | 1.88 deg | 4 |
| Jump > 7 deg + lid gating (chosen) | 0.68 deg | 1.43 deg | 2 |

The PRD target is 0.5 deg after filtering; the rest is network noise. Lowering the
One Euro minimum cutoff to 0.5 Hz gave 0.62 deg but made 3-7 deg moves take
about 250 ms, so it was not adopted. All numbers come from one person, one room
and one session, before calibration; re-check them on the pilot data.

The replay baseline (`tests/fixtures/replay_baseline.json`) was regenerated: on
its near-noiseless synthetic session mean error rose 6.8% (small target jumps
now glide instead of snapping) and jitter fell 2.7%.

Settings: `fixation.saccade_jump_deg` (7.0), `fixation.saccade_confirm_frames`
(2), `quality.gaze_lid_threshold` (0.3), `quality.gaze_lid_hold_frames` (1).
Profiles saved with the old `saccade_velocity_deg_s` still load; the key is
dropped. Revisit if the gaze model's noise falls well below 1 deg per frame.
