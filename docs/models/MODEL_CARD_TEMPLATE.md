# Model card: <file name>.onnx

| Field | Value |
| --- | --- |
| Model | M1 / M3 / M4 / B1 / B2 / L2 |
| Commit | output of `git rev-parse HEAD` when trained (also in the checkpoint) |
| Config | `configs/...` plus overrides |
| Training data | datasets, subjects, stride, manifest checksum |
| Input / output | GazeOS ONNX contract (see `gazeos/inference/`) |
| Size / precision | MB, fp32 or INT8 |

## Metrics

| Metric | Value | Data |
| --- | --- | --- |
| Mean angular error (LOSO) | ° ± SD over 3 seeds | MPIIFaceGaze |
| After 9-point calibration | ° | GazeOS-Webcam session 3 |
| Parity vs PyTorch | ° | export check |
| CPU latency (reference laptop) | ms/frame | latency profiler |
| Uncertainty Spearman ρ | | E12 |

## Limitations and fairness

Error by lighting, glasses, distance, head yaw (E9) and by skin-tone group, glasses and age (PRD 21.4);
gaps over 0.5° are flagged. Known failure cases.
