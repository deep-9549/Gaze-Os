# ADR 0002: Personalization runs after the network

**Status:** accepted · **PRD:** 8.3, 11.4

P1-P3 correct the network's output (angles or screen points); P4 and P5 learn
an additive linear correction on the network's 256-d penultimate features, in
NumPy, with 100 Adam steps. Nothing is re-exported, and a calibration refit
takes milliseconds to a second.

Two rules found during simulation (E6):

- With fewer than 5 distinct targets, P2 is pulled strongly toward the
  identity and P4 learns only a bias, so 1- and 3-point calibrations degrade to
  an offset correction rather than fitting per-frame noise.
- P3 (Gaussian process) fits one median point per fixation, not raw frames: with
  raw frames its inputs are as noisy as its targets and it learns to "undo"
  noise locally, which extrapolates badly.
