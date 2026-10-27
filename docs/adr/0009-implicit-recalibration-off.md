# ADR 0009: Implicit recalibration off by default

**Status:** accepted · **PRD:** 11.5, 11.6 (FR-12)

In the developer's first typing session with M3 (28 Oct 2026) the keyboard's bottom row
(Space, Del, Enter) felt unreachable, while the other rows were usable. The profile's
calibration file had been rewritten about two minutes after calibration: implicit
recalibration (PRD 11.5) refits after every 20 confirmed clicks, and each gaze-typed key
counted as one.

## What the refit did

The calibration samples are saved with the profile (251 samples, 9 dots), so the
original fit can be rebuilt and compared with the rewritten one. For a gaze that the
original fit puts on the centre of each keyboard row, the rewritten fit moved the cursor
by (up is negative; three positions across the screen):

| Keyboard row (fraction of screen height) | Cursor moved, mm |
| --- | --- |
| Q row (0.41) | -0.6 to -1.0 |
| A row (0.56) | -2.3 to -2.8 |
| Z row (0.71) | -5.1 to -5.8 |
| Space row (0.86) | -9.1 to -10.2 |

On the calibration's own bottom dots the error went from 1.2-2.3 mm to 10.9-12.0 mm.
A key is about 29 mm tall, so on the Space row the cursor sat at the top edge of the key
and ordinary noise carried it into the Z row.

## Why

PRD 11.5 assumes the clicked element tells where the user was looking, independently of
the tracker; that holds for a mouse click. A dwell or blink click lands wherever the
cursor is, so the label comes from the tracker's own estimate:

- The residual check (accept only under twice the current error) always passes, because
  the click happened where the cursor was.
- When the cursor sits too high and the user looks below a key to pull it in, the sample
  says that this lower gaze belongs at the key's centre, which pushes the cursor further
  up. The error grows instead of shrinking.
- The samples cluster where the user types (20 keys at about 9 frames each, against 251
  calibration samples, at full weight), and one screen-wide polynomial carries their
  pull to the rest of the screen.

## Decision

`calibration.implicit_enabled` defaults to false; the code stays for experiments. The
automatic drift warning (PRD 11.6) is built on implicit residuals and is off with it; the
menu's one-point Quick fix remains.

To make implicit recalibration safe later: use only labels that do not come from the
cursor that made the click (for example a key the typing decoder corrected, or a mouse or
switch click), correct locally near the samples instead of refitting the whole screen,
and cap the implicit weight against the calibration samples.
