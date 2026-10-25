# Datasets

| Dataset | Used for | Access | Pack with |
| --- | --- | --- | --- |
| ETH-XGaze (224 px release) | Stage A pre-training | Registration on the ETH-XGaze site; academic use | `packers ethxgaze` |
| MPIIFaceGaze | Fine-tuning, main LOSO benchmark | Direct download; research use | `packers mpiifacegaze` |
| RT-GENE (inpainted crops) | Stage A without ETH-XGaze (PLAN.md) | Direct download; CC BY-NC-SA 4.0 | `packers rtgene` (reads the `sNNN_glasses.tar` files directly) |
| ETH-XGaze ResNet-18 weights (hysts) | Face-stream initialization (`init_face_checkpoint`) | Hugging Face `hysts/ptgaze-eth-xgaze-resnet18`; MIT, trained on ETH-XGaze | not packed; `data/weights/` |
| L2CS-Net Gaze360 weights | M1 baseline | Hugging Face mirror `py-feat/l2cs` (official Drive link is dead) | `convert_l2cs` |
| Gaze360 | Cross-dataset robustness (E5) | Registration form | add a packer when access arrives |
| GazeCapture | Optional extra pre-training | Registration form | add a packer when access arrives |
| GazeOS-Webcam (own) | Personalization (E6), final test, E9 | Collected with consent via `tools/collect.py` | `packers gazeos` |
| Gesture set (own) | B1/B2 (E10) | `tools/label_gestures.py` | used directly by `train_gesture` |
| Eyeblink8, Talking Face | Blink pre-training and test | Public research datasets | convert to the gesture `.npz` format |
| DailyDialog, OpenSubtitles (OPUS), Vertanen-Kristensson AAC | Language models (E11) | Public; licences vary | plain text into `data/corpus/` |

Rules (PRD 15.5, 21): public gaze data is non-commercial research data; it is
never redistributed or committed (`data/` is ignored by Git). GazeOS-Webcam
stays private to the team; only normalized face and eye crops are stored, never
full frames. Every packing run writes `manifest.json` with SHA-256 checksums and
the Git commit, so each result traces back to its exact data.

Splits (PRD 15.3): ETH-XGaze official training split with 10% of subjects held
out for validation; MPIIFaceGaze 15-fold leave-one-subject-out; GazeOS-Webcam
subject-independent, with sessions 1-2 for adaptation and session 3 for testing.
RT-GENE: subject ids 2000-2016; s014-s016 are excluded from pre-training
(`data.exclude_subjects`) and kept as a cross-dataset test.

RT-GENE labels are `[yaw, pitch]` (gaze yaw negative toward image-right, pitch
positive looking up); the packer stores `(pitch, yaw)` like every other shard. Its
`left`/`right` eye folders are the subject's own eyes.
