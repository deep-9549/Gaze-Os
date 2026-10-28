# Model card: gazenet_lite.onnx

The deployed gaze model (M4, GazeNet-GOS-Lite), distilled from M3 without ETH-XGaze
(see [PLAN.md](../PLAN.md)). The file itself is not in the repository: it is trained on
non-commercial research data and stays local (`models/` is git-ignored for it).

| Field | Value |
| --- | --- |
| Model | M4: MobileNetV3-Small face stream, shared eye stream, head-pose MLP, transformer fusion, 28 x 3° bins |
| Teacher | M3 (ResNet-18 face stream), Phase 4 arm d: face stream from the ETH-XGaze-trained ResNet-18 (hysts/ptgaze-eth-xgaze-resnet18), Stage A on RT-GENE, Stage B on MPIIFaceGaze |
| Commit | `ff74923` (recorded in both checkpoints) |
| Config | Stage A `experiment=stage_a_rtgene_eth`; Stage B `experiment=stage_b_mpii data.fold=null`; Stage C `training.distill` (T = 4, 20 epochs, best epoch 15); `data.workers=4`, seed 0 |
| Training data | MPIIFaceGaze subjects p00-p05, p07-p12, p14 (13); validation p06, p13. Stage A: RT-GENE s000-s002, s004-s006, s008-s013 (validation s007; s003 excluded for degraded images; s014-s016 held out). Shard manifests with SHA-256 checksums next to the shards |
| Input / output | GazeOS ONNX contract: normalized 224x224 RGB face, two 36x60 eye crops (left mirrored), head pose; pitch/yaw bin logits, log-variance, 256-d features, attention |
| Size / precision | 10.0 MB, fp32 (installed 27 Oct 2026 with the refitted uncertainty head; the previous file is kept as `models/gazenet_lite_prev.onnx`). Static INT8 was rejected: it lost 1.2° and ran slower (2.69 vs 2.02 ms/frame) |

## Metrics

| Metric | Value | Data |
| --- | --- | --- |
| Mean angular error, held-out subjects (mean of subjects) | 3.66° (p06 2.67°, p13 4.64°); teacher M3 3.65° | MPIIFaceGaze p06 + p13, all 4,375 images, through the app's ONNX runtime wrapper |
| Mean angular error, leave-one-subject-out (the benchmark, E1) | M3: 4.78° ± 0.09° over 3 seeds (seed means 4.83, 4.84, 4.68°; SD over subjects about 1.3°); 3-seed ensemble 4.65° | MPIIFaceGaze, 15 folds |
| After calibration, real webcam | 1.08° leave-one-dot-out (polynomial, 9 dots); raw spread 1.59° (M3: 0.79° and 1.23°) | one user, guided 9-dot recording of 27 Oct 2026 (ADR 0008) |
| Parity vs PyTorch | M3 3.9e-6° mean (limit 0.05°) | 1,000 images of p06 |
| CPU latency, model only | 2.02 ms/frame (ONNX Runtime, fp32) | laptop CPU, 1,000 images; full pipeline profile pending |
| Uncertainty: Spearman ρ between predicted σ and error; median σ | ρ +0.21 (was +0.19); median σ 2.7° (was 1.5°, against a 3.3° actual error) | MPIIFaceGaze p06 + p13; `var_head` refitted after training (`training.fit_uncertainty`, 27 Oct 2026), angles bit-identical. Across the 45 leave-one-subject-out M3 models the refitted head reaches only ρ 0.08 ± 0.19 (E12, see the [M3 card](gazenet_m3.md)) |

## Limitations and fairness

- Not yet measured on real webcam users after calibration, by lighting, glasses, distance or
  head yaw (E9), or by skin tone and age (PRD 21.4): that needs the GazeOS-Webcam dataset.
- MPIIFaceGaze has 15 participants; subject-to-subject spread is large (3.2-6.8° across folds).
- RT-GENE images are low resolution with inpainting artifacts; their influence on the final
  model is limited to Stage A initialization.
- Licences: MPIIFaceGaze and RT-GENE are CC BY-NC-SA 4.0 and the ETH-XGaze-trained
  initialization derives from non-commercial data, so this model is for non-commercial
  research and assistive use only and must not be redistributed publicly.
