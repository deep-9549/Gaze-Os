# For your review

Decisions and questions collected while the laptop ran unattended (26 Oct 2026,
until 16:00). Each item says what was done in the meantime, so nothing waits on
you unless marked **Blocked**. Newest first. The plan and its status table are
in [PLAN.md](PLAN.md).

| # | Item | What I did meanwhile | Needs from you |
| --- | --- | --- | --- |
| 5 | **Reboot when convenient.** The page file on C: grew to 18 GB during the memory problem (now about 14.5 GB) and only shrinks fully on restart | Nothing; a reboot mid-training would only cost the current epoch (runs resume), but there is no need | Reboot after training, whenever suits you |
| 4 | **RT-GENE s003 excluded from Stage A.** Its labels look normal but its inpainted eyes are too degraded: an MPIIFaceGaze model's predictions correlate with all 16 other subjects' labels and not with s003's | Excluded via `data.exclude_subjects` (one config value to undo) | Agree, or ask me to re-run Stage A with it |
| 3 | **Stage A validation is noisy.** RT-GENE pre-training validates on a single subject; the error swung between 5° and 11° and both runs stopped at epoch 7 (best at epoch 2) | Used the epoch-2 checkpoints as planned. If RT-GENE pre-training wins Phase 4, a cleaner re-run (2-3 validation subjects, lower learning rate) could help further | Whether that re-run is worth about 2 GPU hours |
| 2 | **How Phase 4 picks its winner.** Fixed before seeing all results: lowest mean test error over MPIIFaceGaze folds 0-2; if two arms are within 0.1°, the simpler one wins (fewer stages: a < b < c < d) | Will apply this rule when Phase 4 ends and start Phase 5 with the winner | Nothing, unless you prefer another rule |
| 1 | **Seeds 1 and 2 for the benchmark** (the PRD asks for mean ± SD over 3 seeds): about 10 more GPU hours after Phase 5 | Phase 5 runs seed 0 on all 15 folds first (reusing the winner's folds 0-2) | Whether the laptop can train overnight, or whether those runs go to Colab |
