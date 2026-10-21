# ADR 0005: The One Euro filter runs in degrees

**Status:** accepted · **PRD:** 12.2

The PRD's starting values (minimum cutoff 1 Hz, beta 0.007) assume small units.
Run on screen pixels, normal gaze noise (about 1,400 px/s apparent speed) raised
the cutoff to about 10 Hz and the filter barely smoothed; dwell clicks almost never
completed at 0.8° noise. The engine converts pixels to degrees before filtering,
so the same parameters work on every screen size and resolution. In simulation
(E7, 0.4° noise) this gives about 0.05° jitter during fixations and about 60 ms
lag after saccades.
