# ADR 0006: Saccades detected by a confirmed jump, not by velocity

**Status:** accepted · **PRD:** 12.4 (deviation)

The PRD detects saccades with I-VT: angular velocity above 30 deg/s, measured on
the filtered gaze. Each detection resets the cursor filter so the cursor jumps
without trailing.

On the laptop webcam (27 Oct 2026, fixation on a centre dot, 290 frames) the
network's frame-to-frame noise was 2.0 deg vertically and 1.25 deg horizontally.
At 30 fps that alone reads as 40-60 deg/s. A simulation with the real filter and
detector at that noise level flagged 194 of 600 frames as saccades; each reset
restarted the filter at a raw, noisy point, which looked like another saccade on
the next frame. The cursor was effectively unfiltered (2.3 deg frame-to-frame
vertical jitter, against 0.47 deg for the filter alone).

GazeOS now calls a saccade when the raw gaze lands more than 5 deg from the
cursor on two consecutive frames in the same direction, or more than 10 deg on a
single frame (more than 4 standard deviations of the measured noise). After a
saccade the detector restarts from the raw point, so one reset cannot trigger the
next.

| Rule (simulated, measured noise) | False resets per minute | Vertical frame-to-frame jitter | Time to land on a 3-15 deg jump |
| --- | --- | --- | --- |
| Velocity > 30 deg/s | 620 | 2.49 deg | 0-33 ms |
| Jump > 5 deg, 2 frames | 1.2 | 0.53 deg | 33-116 ms |

Jumps under 5 deg are left to the One Euro filter, whose cutoff rises with speed.
The replay baseline (`tests/fixtures/replay_baseline.json`) was regenerated: on
its near-noiseless synthetic session mean error fell 16% and jitter rose 5.5%.

Settings: `fixation.saccade_jump_deg` (5.0) and `fixation.saccade_confirm_frames`
(2). Profiles saved with the old `saccade_velocity_deg_s` still load; the key is
dropped. Revisit if the gaze model's noise falls well below 1 deg per frame.
