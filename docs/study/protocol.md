# Session protocol (experimenter script)

Before the participant arrives: laptop charged, room lights on, `gazeos` installed, a fresh profile for
the participant code, Latin-square order for this participant noted below.

| Order | Condition |
| --- | --- |
| [ ] | C1 GazeOS eyes only |
| [ ] | C2 GazeOS eyes + face gestures |
| [ ] | C3 baseline (geometric M0, linear calibration): `gazeos --gaze-model none` with settings method "polynomial" |
| [ ] | C4 mouse (upper bound) |

1. Welcome, information sheet, consent form (signed before anything is recorded).
2. Demographics: age group, glasses, optional skin-tone group.
3. For each condition, in the assigned order:
   - **T1 Calibration:** `gazeos --profile P07`; menu > Calibrate; note the score and time.
   - **T2 Target acquisition:** `python -m tools.fitts_task --participant P07 --condition C1`.
   - **T3 Web task:** open a news site, search a given phrase, open the third result, scroll to the end.
     Record time, success and errors.
   - **T4 Typing:** transcribe five phrases from the MacKenzie-Soukoreff set (opt-in experiment
     logging on, so keystrokes and timings are saved).
   - **T5 Reading:** read an article for five minutes with tracking on; count accidental clicks.
   - **T6 Robustness:** participant leans back or shifts; repeat a short T2; then menu > Quick fix and
     repeat once more.
   - SUS and NASA-TLX for this condition.
4. Open feedback: what was hardest, what would you change.
5. Thank the participant; give compensation.

After the session: copy `data/study/` and the profile's `logs/` to the encrypted drive; delete the local
copies. Score questionnaires with `python -m tools.questionnaires`.
