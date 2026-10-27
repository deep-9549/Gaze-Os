# ADR 0007: Lids, blink gestures and profile defaults

**Status:** accepted · **PRD:** 7.2, 11.2 step 4, 13.2 (deviation) · supersedes the lid rule of ADR 0006

## What the webcam showed (27 Oct 2026, REVIEW.md item 12)

Looking down moves the upper lid over the eye, and MediaPipe's eye-blink score rises
with it. On the developer's laptop (eyes at about 67 cm, roughly level with the camera):

| Situation | Larger blink score of the two eyes | Eye crops |
| --- | --- | --- |
| Upper and middle screen | 0.01-0.15 | open |
| Bottom row of the screen | 0.2-0.3 | open, lid lower, iris visible |
| Looking far down (keyboard; leaning back with the chin up) | 0.6-0.7 | iris hidden |
| Real blinks | peaks of 0.72-0.76 | closed |

The relevant angle is the eye's rotation inside the head: below about -25 deg the
score passed 0.5 in 97% of frames. Every eye rule used fixed thresholds, so:

- The fixed 0.3 lid cutoff of ADR 0006 sat inside the normal bottom-of-screen range
  (0.27 on a clean run). In two app calibrations it removed most bottom-row frames
  (6, 0 and 8 of about 30 kept), and the fit guessed the bottom of the screen
  (validation errors 8.1, 6.0 and 3.2 deg; scores 59 and 49). A clean run scored 82.
- "Both eyes closed" also fires when looking far down: in 25 s of such frames the
  rule-based detector produced 2 pause toggles and 4 long-blink clicks.

## Decision

1. **Lid gate** (`LidGate`): a frame gives no gaze sample when either eye's score is
   above `quality.lid_hidden_level` (0.45, the iris is mostly hidden from about 0.4)
   or changes by more than `quality.lid_speed` (0.1) in one frame (a blink starting or
   ending), and for `quality.lid_hold_frames` (2) after. A steadily lowered lid is
   kept. On the blink recording this is as steady as the 0.3 cutoff (cursor RMS
   0.68 deg) with fewer large excursions (4.2% vs 5.2% of frames beyond 3 deg); with
   the bottom-row scores raised by 0.05-0.10 it keeps 100% of bottom-row calibration
   frames where the 0.3 cutoff kept 66-72%.
2. **Eyes-closed pause** is off by default (`blink.eyes_closed_pause`): a 2 s glance
   at the keyboard toggled it. The pause hotkey and the corner still work.
3. **Profiles store only changed settings** (`exclude_defaults`), so improved
   defaults reach existing users; the first app sessions had pinned every default.
   Old keys are migrated on load.

Long-blink clicks remain (FR-20), pending a per-user check that deliberate long
blinks and looking down can be told apart; see REVIEW.md item 12.
