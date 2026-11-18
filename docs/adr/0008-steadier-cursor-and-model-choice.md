# ADR 0008: Steadier cursor, per-target dwell and automatic model choice

**Status:** accepted · **PRD:** 12.4, 12.7, 10.8, E8 (extensions)

After the fixes of ADR 0006 and 0007 the developer's first calibrated session scored
80+ but felt "shaky", "a bit off target, worse at the edges" and typing was hard even
with large keys. All numbers below come from one guided full-frame recording (9 dots,
3 s each, 640x480, 27 Oct 2026), replayed offline: calibration on the first half of each
dot, scores on the second half.

## Fixation hold

During a fixation the cursor sat at the running median of the last 0.5 s, which still
moved every frame with the network's noise (cursor perfectly still in 13-21% of frames).
The hold now glides to the median (30% of the way per frame) and keeps the cursor where it
is while the median stays within 0.75 deg, once 0.3 s have passed since the median last
left that band (a real move is always followed fully, so the cursor cannot stop short of
a small correction).

| After calibration | Jitter, mm per frame | Still frames | Offset | Mean error from 0.6 s |
| --- | --- | --- | --- | --- |
| M4, previous hold | 2.9 | 13% | 7.7 mm | 12.3 mm |
| M4, new hold | 2.2 | 47% | 7.3 mm | 12.0 mm |
| M3, previous hold | 2.0 | 21% | 5.5 mm | 10.1 mm |
| M3, new hold | 1.3 | 61% | 6.3 mm | 10.3 mm |

Tried and rejected: locking for good once a fixation is 0.3 s old (still 70-84% of
frames, but a 1 deg correction left the cursor 32 px short); a slow creep inside the band
(no frame was still any more). On the near-noiseless synthetic replay the hold trades +12%
mean error for -79% jitter (baseline regenerated).

## Per-target dwell

Dwell restarted whenever the cursor left a 1.5 deg circle around the point where it
began, so wandering inside a 170 px key restarted the ring. On GazeOS's own targets
(keyboard keys, action bar) dwell now counts per target: the dwell clicker sees the
target's centre while the cursor stays on it (with a 0.5 deg margin, at most a quarter of
the target), so a large key works as large. OS targets keep the radius rule, with
snapping and zoom-click at the moment of the click.

## Model choice

Leave-one-dot-out on the same recording (polynomial calibration):

| Model | Held-out error | Edges and corners | Raw spread | Network, ms per frame |
| --- | --- | --- | --- | --- |
| M4 | 1.08 deg | 1.10 deg | 1.59 deg | 3.5 (whole frame 16.6) |
| M3 | 0.79 deg | 0.81 deg | 1.23 deg | 13.4 (whole frame 26.0, p95 30.1) |

The offset-only, Gaussian-process and fine-tune calibrations were no better than the
polynomial (fine-tune overfits: up to 16 deg on one held-out corner). `gaze_model`
defaults to "auto": M3 when this CPU runs it within 15 ms (timed at startup on a blank
input after a warm-up), else M4. M3 exceeds the PRD's 8 ms network budget but the whole
frame still fits 30 fps on the development laptop; slower machines get M4.
