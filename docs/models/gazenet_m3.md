# Model card: gazenet_m3.onnx

The gaze model the app uses on the development laptop (M3, GazeNet-GOS with a ResNet-18
face stream). `gaze_model: auto` picks it when this CPU runs it within 15 ms, else
[M4](gazenet_lite.md) (ADR 0008). The file itself is not in the repository: it is trained
on non-commercial research data and stays local (`models/` is git-ignored for it).

| Field | Value |
| --- | --- |
| Model | M3: ResNet-18 face stream, shared eye stream, head-pose MLP, transformer fusion, 28 x 3° bins |
| Initialization | Phase 4 arm d: face stream from the ETH-XGaze-trained ResNet-18 (hysts/ptgaze-eth-xgaze-resnet18), Stage A on RT-GENE |
| Commit | `ff74923` (recorded in the checkpoint) |
| Config | Stage A `experiment=stage_a_rtgene_eth`; Stage B `experiment=stage_b_mpii data.fold=null`, `data.workers=4`, seed 0; best epoch 9 of 10. Uncertainty head refitted afterwards (`training.fit_uncertainty`, 27 Oct 2026) |
| Training data | MPIIFaceGaze subjects p00-p05, p07-p12, p14 (13); validation p06, p13. Stage A: RT-GENE s000-s002, s004-s006, s008-s013 (validation s007; s003 excluded for degraded images; s014-s016 held out) |
| Input / output | GazeOS ONNX contract: normalized 224x224 RGB face, two 36x60 eye crops (left mirrored), head pose; pitch/yaw bin logits, log-variance, 256-d features, attention |
| Size / precision | 50.9 MB, fp32 |

## Metrics

| Metric | Value | Data |
| --- | --- | --- |
| Mean angular error, leave-one-subject-out (the benchmark, E1) | 4.78° ± 0.09° over 3 seeds (seed means 4.83, 4.84, 4.68°; SD over subjects about 1.3°); 3-seed ensemble 4.65° | MPIIFaceGaze, 15 folds, same recipe |
| Mean angular error, validation subjects | 3.65° (mean of subjects, through the app's ONNX wrapper); 3.31° per image in training | MPIIFaceGaze p06 + p13. Optimistic: these two subjects also chose the epoch |
| After calibration, real webcam | 0.76-0.79° leave-one-dot-out (polynomial, 9 dots); raw spread 1.23° | one user, guided 9-dot recording of 27 Oct 2026 (ADR 0008) |
| Jitter on a still gaze (E7) | 1.12° unfiltered; 0.13° with the One Euro filter the app uses | same recording, `research.e7_filters` |
| Sensitivity to invisible input changes | 0.16-0.32° (sensor noise SD 2/255, JPEG re-encoding, a 1 px shift) | same recording, `research.webcam_stability` |
| CPU latency | network 13.4 ms per frame, whole frame 26 ms (p95 30 ms) | development laptop (ADR 0008); idle E8 timing pending |
| Uncertainty: Spearman ρ between predicted σ and error | 0.11 on p06 + p13 after the refit (0.15 before). Over the 45 E1 models: 0.08 ± 0.19, pooled 0.00, flat reliability diagram (E12) | the head does not rank errors on new people |

## Tried and not adopted

- Consistency fine-tune (28 Oct 2026): a second, slightly perturbed view of every training
  image with a penalty on different answers. The stronger the penalty, the steadier the
  output and the larger the accuracy cost (weight 10: frame-to-frame change 49% lower,
  0.71° worse leave-one-dot-out, 0.95° worse on p06 + p13). None met the rule fixed before
  the run (`D:\GazeOS-cache\outputs\consistency\decision.json`).

## Limitations and fairness

- Measured on real webcam use for one person only; not yet by lighting, glasses, distance
  or head yaw (E9), or by skin tone and age (PRD 21.4).
- MPIIFaceGaze has 15 participants; subject-to-subject spread is large (3.2-6.8° across folds).
- The error on a new person is mostly a per-person offset that the network cannot see in
  one image (E12), which is why calibration matters more than the network's own accuracy.
- Exceeds the PRD's 8 ms network budget; slower machines get M4 automatically.
- Licences: MPIIFaceGaze and RT-GENE are CC BY-NC-SA 4.0 and the ETH-XGaze-trained
  initialization derives from non-commercial data, so this model is for non-commercial
  research and assistive use only and must not be redistributed publicly.
