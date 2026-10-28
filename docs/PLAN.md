# Implementation plan without ETH-XGaze

A living plan: update the status column and the log at the bottom as steps
finish. Started 25 Oct 2026, after ETH-XGaze access could not be obtained.
It replaces only Stage A of [TRAINING.md](TRAINING.md); Stages B to D, the
app, the user study and every other PRD section are unchanged.

Status legend: **Done** · **In progress** · **Waiting** (on a download, an
approval or a person) · **To do**.

## 1. Data on disk

Raw data, shards and training outputs live in `F:\GazeOS-data`, linked into
the repo as `data/` (a directory junction; `data/` is ignored by Git). Only
that folder is used on F:. `outputs/` is likewise a junction to
`F:\GazeOS-data\outputs`. The repo drive (D:, about 90 GB free) holds the
`.venv` (Python 3.11 managed by uv) and the uv cache at `D:\.uv-cache`.
Packed shards (not raw data) are mirrored to a private Google Drive folder for
cloud training: `G:\My Drive\Minor Project\GazeOS` on the laptop (Google Drive for
desktop), `/content/drive/MyDrive/Minor Project/GazeOS` on Colab, laid out as
`data\shards\<set>`, `data\weights\<model>` and `outputs\`. Only that folder is
used on the Drive.

| Dataset | Location | State (26 Oct 2026) |
| --- | --- | --- |
| 2026-10-28 | 10:33, **word-level transformer trained** (10.8 M parameters, 1.5 M sentences, 1 h): held-out perplexity 45.6 and top-3 43.5% (3-gram: 33.9 and 48%, not directly comparable). E11 keystroke savings: 55.5% with the transformer against 54.6% for the trained 3-gram; it costs 3.4 ms per word. Two predictor fixes found on the way: suggestions after a common word took 0.3-1 s (the 3-gram scored all ~11,000 followers; now the top 100, 5 ms, same top 4 in 99.9% of contexts), and the transformer saw only the last 2 words (now 16: 54.9% -> 55.5%). `typing.use_transformer` stays off so the weekend typing retest measures the calibration fix alone. Model cards added for M3 and the language models; the M4 card has its webcam numbers. E2 seeds 1-2 running (until about 14:00), then E8 timing on the idle machine. |
| 2026-10-28 | 09:33, **E2 done** (input streams, folds 0-2, seed 0; E1 4.08°, noise band 0.35°): eyes only 5.61° (+1.53, a clear loss: the face stream carries most of the signal); face only 4.14° (+0.05: the eye crops add nothing on top of the face); no head pose 3.62° (-0.47, just outside the noise band, one seed); concat fusion 3.81° (-0.28, inside the noise). Seeds 1-2 of no head pose and face only queued after the language model to check the two surprises. |
| 2026-10-28 | 06:46, **E3 done** (loss, folds 0-2, E1's recipe; E1 seed 0 4.08°, seed-to-seed spread on these folds 0.35°): regression 4.03° (-0.05), classification only 4.08° (-0.01), β = 2 4.08° (-0.01). All inside the noise: on MPIIFaceGaze the loss choice does not matter. Table: `python -m research.ablation_table D:/GazeOS-cache/outputs/e2 D:/GazeOS-cache/outputs/e3`. E2 restarted and training. |
| 2026-10-28 | Morning, GPU queue results. **M3 noise fine-tune: not adopted.** No arm met the rule written into the job (calibrated spread down 15% or more, leave-one-dot-out error no more than 0.05° worse, MPIIFaceGaze no more than 0.10° worse): consistency weight 0 changed nothing (spread -2%); weight 2 cut spread 10% and frame-to-frame change 18% but cost 0.18° leave-one-dot-out; weight 10 cut spread 25% and frame change 49% but cost 0.71° leave-one-dot-out and 0.95° on MPIIFaceGaze. M3 stays. **E12 final** (45 models, all refit with the fixed sigma range): Spearman rho 0.08 ± 0.19 (33 of 45 positive), pooled rho 0.00; the reliability diagram is flat (mean error 4.2-4.9° in every decile while predicted sigma goes from 2.3 to 10.4°). The model cannot tell when it is wrong on a new person: the error is mostly a per-person offset, which one image does not reveal. **E2 failed at start** (an ablation changes layer shapes, so the Stage A checkpoint did not load); fixed and requeued after E3. |
| MPIIFaceGaze | `raw/MPIIFaceGaze` | Complete: p00-p14 with calibration files, 37,667 images (the official total), all decode |
| RT-GENE | `raw/rtgene/sNNN_glasses.tar` | Complete: 17 subjects (s000-s016), about 114,600 labelled frames; every tar reads and every label has a face image |
| ETH-XGaze-trained ResNet-18 | `weights/` | To download: [hysts/ptgaze-eth-xgaze-resnet18](https://huggingface.co/hysts/ptgaze-eth-xgaze-resnet18) (MIT, not gated) |
| L2CS-Net Gaze360 weights (M1) | `weights/` | To download: [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) |

### RT-GENE facts the packer relies on

Checked on the real files, not taken from papers:

- Each tar holds `sNNN_glasses/inpainted/{face,left,right}/` PNGs (face 224x224,
  eyes 60x36) and `label_combined.txt`. Use `inpainted/`, not `original/`
  (the originals show the eye-tracking glasses).
- Label line: `frame, [head A, head B], [gaze A, gaze B], timestamp`, angles in radians.
- Gaze B is **pitch** (positive = looking up) and gaze A is **yaw** (negative =
  looking toward image-right). This is the convention of
  `gazeos.vision.geometry.vector_to_pitchyaw`, so the shard label is
  `gaze = (B, A)` with no sign flip (verified on extreme-gaze frames of s012).
- Head pose looks like the same (yaw, pitch) order by its spread; its signs
  still need checking (step 14).
- Frame ids start at 2 and have gaps: match images to labels by id, not by line.
- Faces are low resolution with inpainting artifacts: good for pre-training,
  not a substitute for ETH-XGaze on its own.

## 2. Alternatives to ETH-XGaze

| Rank | Option | Access | Role |
| --- | --- | --- | --- |
| 1 | ETH-XGaze-trained ResNet-18 (hysts) | Direct, MIT | Initialize the M3 face stream: same backbone, 224 px input, ImageNet normalization, same 960 px / 600 mm normalization as the runtime |
| 2 | RT-GENE | Have it; CC BY-NC-SA 4.0 | Pre-train the whole network, including eye and head streams |
| 3 | [GazeGene](https://phi-ai.buaa.edu.cn/project/GazeGene/index.htm) (CVPR 2025, synthetic, 1M images, 56 subjects) | Hugging Face click-through; 28.7 GB; CC BY-NC-SA 4.0 | Optional extra pre-training arm |
| 4 | [UniGaze](https://github.com/ut-vision/UniGaze) (WACV 2026, ViT) | `pip install unigaze`; non-commercial licence; pins torch 2.0.1 and timm 0.3.2, so it needs its own venv | Too heavy for CPU; optional offline teacher for Stage C or pseudo-labels |
| 5 | L2CS-Net Gaze360 weights | [py-feat/l2cs](https://huggingface.co/py-feat/l2cs) mirror (the official Drive link is dead) | M1 baseline via `training/convert_l2cs.py` |
| 6 | Gaze360, EYEDIAP, GazeCapture | Registration forms | Request anyway; Gaze360 is E5's cross-dataset test |

Re-request ETH-XGaze from the guide's institutional email with the guide named
on the form. All of these are non-commercial research data or weights trained
on it; model cards must say so.

## 3. Where training runs

| Where | What |
| --- | --- |
| Laptop (RTX 3050, 4 GB) | Packing, debugging, smoke runs, Stage A arm selection on folds 0-2, export, INT8, latency profiling (the CPU targets are measured here). Train from the SSD copy `D:\GazeOS-cache\shards` (F: is a USB hard drive: about 25 img/s from it) with `data.workers=4`. M3 at batch 64 with AMP peaks at 1.1 GB of GPU memory, so the default batch 128 fits. RAM is the tighter limit (15.7 GB): each Windows loader worker commits about 1.5 GB and the training process about 8 GB, so 6 persistent workers pushed commit past RAM, Windows paged, and the page file on C: grew to 18 GB. Validation and test therefore load in the main process, and 4 training workers is the default |
| Colab + Google Drive (5 TB) | Long runs. Shards live on Drive; copy them to `/content` at session start (HDF5 over the Drive mount is slow); write checkpoints to Drive every epoch |
| Kaggle | Extra free GPU time (about 30 h/week, 12 h sessions) in parallel with Colab; pull shards with `gdown`/`rclone` or a private Kaggle dataset |

Only packed shards go to the cloud, never raw datasets, and Drive folders are
never shared by public link. Estimated total compute: about 30-45 GPU hours.

## 4. Steps

### Phase 0: environment

| # | Step | Status |
| --- | --- | --- |
| 1 | Python 3.11 venv; `pip install -e ".[app,train,research,dev]"`; CUDA build of PyTorch | Done: uv Python 3.11.15, torch 2.11.0+cu128 sees the RTX 3050 |
| 2 | `cmd /c mklink /J data F:\GazeOS-data` so existing config paths work | Done, plus `outputs` -> `F:\GazeOS-data\outputs` |
| 3 | `pytest` and the smoke test in TRAINING.md section 0 pass | Done: 111 passed, 2 skipped (Linux/macOS only); smoke parity 1e-6°; GPU + AMP smoke run works |

Recreate the environment with:

```bash
set UV_CACHE_DIR=D:\.uv-cache
uv venv .venv --python 3.11
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -e ".[app,train,research,dev]"
```

### Phase 1: data

| # | Step | Status |
| --- | --- | --- |
| 4 | Full MPIIFaceGaze (p00-p14) in `raw/MPIIFaceGaze` | Done |
| 5 | RT-GENE `s004` re-downloaded, `s005` and `s006` added; all tars pass `tar -tf` | Done |
| 6 | ETH-XGaze ResNet-18 (`weights/eth-xgaze-resnet18/`, `hf download hysts/ptgaze-eth-xgaze-resnet18`) and L2CS-Net Gaze360 weights (`weights/l2cs/`, `hf download py-feat/l2cs`: the official Google Drive link is dead; this mirror ships `.safetensors`, so `convert_l2cs.py` needs safetensors support, step 9). No L2CS MPIIGaze weights exist online: E1 cites the published 3.92° and uses `m2_face_only` as the same-folds L2CS-style baseline | L2CS done: `weights/l2cs/l2cs_gaze360_resnet50.safetensors`, keys load into `convert_l2cs.L2CS(90)` with nothing missing. ETH-XGaze done: `weights/eth-xgaze-resnet18/model.safetensors`; every backbone key loads into torchvision `resnet18` (only its 2-output `fc` is dropped) |
| 7 | Optional: accept GazeGene terms; Gaze360 and EYEDIAP forms; ETH-XGaze via the guide | To do |

### Phase 2: code

| # | Step | Status |
| --- | --- | --- |
| 8 | `pack_rtgene` in `training/data/packers.py`: inpainted face and eye crops (grayscale + CLAHE, left eye mirrored), `gaze=(B, A)`, `head=(B, A, 0)`, subject ids 2000+n, skip frames without images, drop angles outside the ±42° bins | Done. Eye sides measured by template matching on 50 frames: `right` is always image-left, so the folders name the subject's own eyes |
| 9 | `init_face_checkpoint` in `train.py` loads the ETH-XGaze ResNet-18 straight into the face stream (no separate converter needed); `.safetensors` input for `convert_l2cs.py`; `safetensors` in the `train` extra | Done |
| 10 | Configs `stage_a_rtgene.yaml`, `stage_a_rtgene_eth.yaml`, `stage_b_eth_init.yaml`; `data.exclude_subjects` keeps s014-s016 out; point Stage B at the winner of step 18 | Done (pointing at the winner waits for step 18) |
| 11 | Shard globbing accepts absolute paths (`glob.glob` instead of `Path().glob`) | Done |
| 12 | `train.resume=true`: `last.pt` every epoch (model, optimizer, scheduler, scaler, early-stop state) | Done |
| 13 | `notebooks/train_cloud.ipynb` for Colab and Kaggle: clone the branch, copy shards to local disk, run a job list, resume or skip finished jobs, outputs on Drive | Done (not yet run on Colab or Kaggle) |
| 14 | Tests (fake RT-GENE tar, face init, safetensors L2CS, resume + absolute paths); DATASETS.md, TRAINING.md, HANDOFF.md, PRD 16.1 and 27 updated | Done: 117 passed, ruff and mypy clean |

### Phase 3: pack and check

| # | Step | Status |
| --- | --- | --- |
| 15 | Pack MPIIFaceGaze (15 subjects) and RT-GENE into `data/shards/` | Done: MPIIFaceGaze 37,667 samples (4.8 GB, 74 min); RT-GENE 114,288 samples (13.6 GB, 171 min; 333 frames beyond ±42° dropped). Fixed a packer bug on the way (real MPII lines have a 28th, non-numeric column). Shards rechunked to one sample per image chunk (36x faster random reads) |
| 16 | Checks: image grid with gaze arrows; RT-GENE vs MPII head-pose spread (head signs); an MPII-trained model on RT-GENE has positive per-axis correlation | Done (`tools/inspect_shards.py`). A 3-epoch MPII model reaches 3.89° on 2 held-out subjects. On RT-GENE its predictions correlate with the labels for 16 of 17 subjects (pitch +0.46 to +0.81, yaw +0.37 to +0.81), so gaze signs are right; head-gaze correlations are positive in both datasets, so head signs are right. **s003 fails** (pitch -0.01, yaw -0.30): its labels look normal but its inpainted eyes are badly degraded, so it is excluded from Stage A |
| 17 | Copy shards to `G:\My Drive\Minor Project\GazeOS\data\shards` (Drive for desktop uploads them) | Done: 34 files, 18.4 GB copied (rechunked versions); Drive's cache is on D: |

### Phase 4: replacement Stage A (new E5)

| # | Step | Status |
| --- | --- | --- |
| 18 | MPII folds 0-2, compare: (a) ImageNet only, (b) ETH-XGaze face weights, (c) RT-GENE pre-training, (d) ETH weights then RT-GENE, (e) optional GazeGene. Keep RT-GENE s014-s016 out of pre-training as a cross-dataset test | Done. Mean test error: a 4.29°, b 4.67°, c 4.77°, **d 4.04°** (per fold in REVIEW.md item 6). d wins by the pre-registered rule, but 3 folds cannot separate the arms reliably. Both Stage A runs early-stopped at epoch 7 (best epoch 2, val 5.08° on one RT-GENE subject). (e) skipped |

### Phase 5: benchmark and deployment model

| # | Step | Status |
| --- | --- | --- |
| 19 | Stage B, 15 LOSO folds x 3 seeds from the winning Stage A (E1); M0 and M1 baselines on the same folds. Target 4.5° or less | M1 done: L2CS Gaze360 off the shelf scores 8.01° on MPIIFaceGaze (`training/eval_onnx.py`, every 10th sample, `outputs/baselines/`); published MPIIGaze-trained L2CS 3.92°. Seed 0 done (arm d, 15 folds): **mean 4.83°**, SD 1.26° over subjects, median 4.62°, best 3.22° (p00), worst 6.82° (p03); 7/15 under 4.5°. Target 4.5° missed by 0.33°. Per fold: 3.22, 3.40, 5.64, 6.82, 3.85, 3.86, 3.43, 5.01, 4.22, 4.28, 6.52, 6.25, 4.74, 4.62, 6.68. **3 seeds done (27 Oct): 4.78° ± 0.09°** (mean ± SD of the seed means 4.83, 4.84, 4.68°; Stage A shared, Stage B and its validation subjects vary by seed). **3-seed ensemble (average of predicted angles): 4.65°**, SD 1.38° over subjects, median 4.66°, 7/15 under 4.5°. Per fold (ensemble): 2.45, 2.99, 5.84, 6.92, 3.56, 3.72, 3.01, 5.06, 4.29, 3.97, 6.06, 6.06, 4.81, 4.66, 6.38. Results in `D:\GazeOS-cache\outputs\arms\d_eth_rtgene\e1_*.json` with the script |
| 20 | Deployment model on all 15 subjects (`data.fold=null`) | To do |
| 21 | Ablations E2, E3, E12 on MPII | To do |

### Phase 6: real-time model

| # | Step | Status |
| --- | --- | --- |
| 22 | Stage C distillation, ONNX export (parity < 0.05°), INT8 (rejected at a 0.2° loss); optional UniGaze-B teacher | Done: M4 3.66° vs M3 3.65° on held-out p06 + p13 (distillation cost 0.01°); M3 parity 3.9e-6°. INT8 rejected (lost 1.2°, slower: 2.69 vs 2.02 ms/frame), so fp32 M4 ships. UniGaze teacher not tried |
| 23 | Model card; `models/gazenet_lite.onnx`; `gazeos --dry-run` and the latency profiler on the laptop | Model card `docs/models/gazenet_lite.md`; fp32 model installed at `models/gazenet_lite.onnx` (git-ignored, public repo) and set as the default. Latency profile and `--dry-run` wait until the overnight runs finish (an idle machine) |

### Phase 7: unchanged from HANDOFF.md

| # | Step | Status |
| --- | --- | --- |
| 24 | Ethics approval, then GazeOS-Webcam collection (E6, E9, Stage D) | Waiting (approval) |
| 25 | Gesture dataset (E10), language-model corpora (E11), user study | To do |

### Final week (27 Oct - 2 Nov 2026, solo)

Solo project (PRD 25 solo scope): P5 meta-learning (Stage D) and the GRU gesture model
(B2) are dropped; the 15-20 person user study is likely future work. Friends may record
pilot data once they have signed the consent form (`docs/study/`) and the guide has said
whether formal ethics approval is needed; the user recording themselves needs nothing.

| Day | Work | Needs the user | Status |
| --- | --- | --- | --- |
| 1 | Reboot; seeds 1-2 summary (E1 mean ± SD) and a 3-seed offline ensemble row; latency profile + 10-min soak; first `gazeos --dry-run` on the webcam and fixes | About 30 min | Seeds and ensemble done (05:30): 4.78° ± 0.09°, ensemble 4.65°. Simulated 10-min soak passed (29.3 fps, memory +6.7 MB, limit 50 MB; no camera or network in simulation). Camera latency (27 Oct evening): 28.4 fps, frame-to-cursor median 16 ms / p95 31 ms, gaze net 3.2 ms, landmarks 11.6 ms (over its 8 ms budget), CPU 22%. First dry-run found and fixed: webcam needed DirectShow (Media Foundation gives no frames) with auto exposure switched on; cursor hidden behind other windows; cursor jitter from false saccade resets and blinks (ADR 0006: precision 1.66 -> 0.68 deg before calibration). Camera soak and the calibrated dry-run to do |
| 1-2 | Pilot GazeOS-Webcam data: `tools/collect.py`, 3 sessions of about 15 min per person (user, then friends with consent) | About 1 h | To do |
| 2-3 | E6 personalization on the pilot data: P0-P4 with 1/3/5/9/16 points (the PRD 2° target) | No | To do |
| 2 (night) | E2 and E3 ablations on folds 0-2 (about 6 GPU h) | No | To do |
| 3 | E7 filters, E12 uncertainty, E8 speed/accuracy: run the existing runners, make figures | No | To do. E12 first needs the uncertainty head trained (REVIEW.md item 11: ρ ≈ 0 in all 45 E1 runs) |
| 3-4 | Gestures (E10): record with `tools/label_gestures.py`, train TCN (B1), compare with rules (B0) | About 30 min | To do |
| 4-5 | Typing (E11): corpora, n-gram and small transformer, keystroke savings | No | To do |
| 5 | Optional: pursuit calibration, tested on the pilot data | No | To do |
| 6 | Windows hardware checks (DPI scaling, focus, UI Automation, TTS); installer build | About 1 h | To do |
| 7 | Figures, results tables, model cards, report material; buffer | No | To do |

Out of scope this week (report as limitations or future work): the user study, fairness
breakdowns (need its data), Gaze360 cross-dataset (no access), P2 features (FR-08, FR-31,
FR-32).

### Proposed: smooth-pursuit calibration in onboarding (idea, 26 Oct 2026)

Add a 15-20 s moving-dot phase to onboarding, on top of today's 9 static
shrinking dots (`gazeos/calibration/protocol.py`), which give about 270 samples
clustered at 9 points.

- Why: hundreds of samples spread over the whole screen (P3 and P4 benefit most);
  quicker and less tiring than 9 jumps; a new experiment for the report (static
  vs pursuit vs both: error against calibration time, an E6 extension).
- Care needed: the eye lags the dot by about 50-150 ms and makes catch-up
  saccades, so estimate each user's lag by cross-correlating gaze with the dot
  path and drop saccade samples (I-VT detector); dot speed about 5-10°/s; average
  per path segment before P3 (ADR 0002); keep static 9-point as the fallback for
  users who cannot follow smoothly (nystagmus, late-stage ALS).
- `tools/collect.py` already records a 30 s pursuit phase with per-frame
  timestamps and dot positions, so GazeOS-Webcam data can evaluate this offline
  once collected; confirm lag-corrected labels can be recomputed from what it stores.
- Effort: about 1-2 days. Status: To do, after Phase 6.

## 5. Open decisions

- Team size (HANDOFF section 6). Solo: drop arm (e) and Stage D.
- Whether the 5 TB Google plan includes Colab Pro (longer, more reliable sessions).

## 6. Log

| Date | Change |
| --- | --- |
| 2026-10-28 | Unattended night (02:18 on, about 20 h): GPU queue in `D:\GazeOS-cache\queue` runs E12 refits (45 models), the M3 noise fine-tune (adoption rule fixed in the job), E2 (eyes only, face only, no head pose, concat fusion) and E3 (regression, classification, β = 2) on folds 0-2 with E1's recipe, then the word-level transformer. Done on the CPU meanwhile: E12 fit bug fixed (sigma range); snapping finds 16-48 px buttons (it found only 48 px); `tools.small_targets` for the weekend; trained 3-gram installed (DailyDialog + 1.5M OpenSubtitles sentences: perplexity 34, top-3 next word 48%; the AAC corpus is no longer online); E7 on the guided recording (One Euro best: jitter 0.13° vs 1.12° unfiltered, 0.36° moving average, 0.39° Kalman); E8 INT8 re-evaluated (4.97° vs 3.66° fp32; slower); installer spec fixed (model paths in the frozen app). First E12 numbers: rho 0.03 ± 0.16 over 18 models, so the uncertainty head barely ranks errors even after the refit. E11 (300 held-out DailyDialog sentences, 4 suggestions, ideal user): keystroke savings 43.2% word list only, 43.9% with the seed 3-gram, **54.6% with the trained 3-gram**. Installer: `pyinstaller packaging/gazeos.spec` builds `GazeOS.exe` (594 MB folder, no PyTorch; output in `D:\GazeOS-cache\build\dist`), which starts and lists the cameras. |
| 2026-10-28 | First typing session with M3 and the ADR 0008 fixes: the Q, A and Z rows usable, the bottom row (Space, Del, Enter) unreachable. Cause: implicit recalibration refit the calibration after the first 20 gaze-typed keys and moved the Space row 9-10 mm up, because a gaze click's label is the tracker's own cursor (ADR 0009). Now off by default; the user's original fit rebuilt from its saved samples (bottom dots back from 11-12 mm to within 2.3 mm). Also measured on the guided recording: invisible input changes (sensor noise SD 2/255, JPEG re-encoding, a 1 px shift) move M3's output by 0.16-0.32°, against 0.39-0.54° between consecutive frames, so about half the per-frame shake is the network's sensitivity to input noise. A consistency fine-tune of M3 is the one GPU job with a direct usability payoff. |
| 2026-10-27 | Late night: after a clean onboarding (score 80+) the user found it "usable with a lot of pain": shaky, slightly off target, worse at the edges, typing hard even with large keys. Fixes (ADR 0008): fixation hold with a dead band (cursor still 47-61% of frames instead of 13-21%), per-key dwell on GazeOS targets (wandering inside a large key used to restart the ring), `gaze_model: auto` picks M3 here (leave-one-dot-out 0.79° vs 1.08° for M4; 26 ms per frame). E6 first data point on the same recording: polynomial best; offset-only, GP and fine-tune no better (fine-tune overfits). Open: small Windows targets need snapping and zoom-click, untested so far because dry-run never clicks; next a safe click test with real buttons of Windows sizes. |
| 2026-10-27 | Night, E4 (camera intrinsics), first data point: a phone-checkerboard calibration gives fx 444 px at 640x480 (the default assumes 640), so the user sits at 43 cm, not 63. On the guided recording the measured focal length worsens the offset before calibration (1.60 to 5.43°: at 43 cm the bottom of the screen is 23° down, steeper than MPIIFaceGaze, and the network underestimates it; the wrong default distance cancelled that) and changes nothing after a 9-point fit (1.03 vs 1.00°; jitter 2.9 vs 3.0 mm per frame). The default stays; the full five-term distortion fit broke head pose (24 mm) and the tool now fits k1 only. |
| 2026-10-27 | Night, fixes (REVIEW.md item 12): lid rule and blink gestures (ADR 0007), calibration per-dot minimum with retry, profiles store only changes, refitted uncertainty head installed, camera calibration tool fixed. On a guided full-frame recording the cursor now holds 0.35° RMS per frame at all 9 dots before calibration (PRD 0.5°); M3 would give 0.22° at 12 ms per frame. |
| 2026-10-27 | Night: root causes of the unusable first sessions found (REVIEW.md item 12): lid-driven blink scores with fixed thresholds (including the 0.3 cutoff added this evening, which emptied bottom-row calibration dots), no per-dot minimum in calibration, a network noise floor of about 1.5° per frame that resolution does not change, an unused head-pose filter, and a guessed camera focal length. A clean calibration run scored 82 (1.9°). |
| 2026-10-27 | Late evening: 9-point webcam recording (light screen, before calibration) replayed through the engine gives cursor RMS precision 0.09-0.58° per point (PRD 0.5°) and an uncalibrated offset of 0.3-3°, so the cursor logic now behaves; the live app still felt unsteady, cause not yet isolated. Calibration now runs on a light surface; the cursor circle's size is eased; onboarding says Next can be clicked. |
| 2026-10-27 | Evening, first webcam session: four app fixes (DirectShow fallback and auto exposure; overlay above other windows; saccade jump rule; no gaze sample while the lids close, ADR 0006). Network noise on the webcam is about 1.8 deg per frame vertically, 1.1-1.5 deg horizontally. |
| 2026-10-27 | Seeds 1-2 finished at 05:08 (30 runs, no failures; seed means 4.84° and 4.68°). E1: 4.78° ± 0.09° over 3 seeds; 3-seed ensemble 4.65°. Both still above the 4.5° target; seed-to-seed spread (0.09°) is much smaller than fold-to-fold spread (1.3°), so the subjects, not the seeds, drive the number. |
| 2026-10-25 | Plan written. Data audit of `F:\GazeOS-data`: MPIIFaceGaze 4 of 15 subjects; RT-GENE s004 truncated, s005 and s006 missing. RT-GENE label order verified. Google Drive (5 TB) added for shards and checkpoints. |
| 2026-10-26 | Phase 0 done. MediaPipe resolved to 1.1.1 (newer than the code was written against); the unit tests pass, but check landmarks on a live webcam with `gazeos --dry-run`. Synthetic shards take about 190 KB per sample, so expect roughly 10-20 GB for RT-GENE. |
| 2026-10-26 | Steps 4 and 5 done: MPIIFaceGaze complete (15 subjects, 37,667 images); RT-GENE complete (17 subjects, about 114,600 frames). |
| 2026-10-26 | Step 6 done: L2CS Gaze360 (py-feat mirror) and ETH-XGaze ResNet-18 weights in `weights/`, both checked against the models they load into. GazeGene deferred until Phase 4 shows whether more pre-training helps. |
| 2026-10-26 | 14:27 F: back after a replug (no errors since); outputs backed up to `D:\GazeOS-cache\outputs_from_F`. Phase 5 folds 0-5: 3.22, 3.40, 5.64, 6.82, 3.85, 3.86° (mean 4.47°). Re-running Stage A with identical settings moved fold 0 by 0.5° (2.70 -> 3.22°), so run-to-run noise exceeds the Phase 4 arm gaps. |
| 2026-10-26 | F: (USB drive) failed with hardware I/O errors during Phase 5's first run (REVIEW.md item 8). Training moved entirely to the SSD: weights from the Drive folder in `D:\GazeOS-cache\weights`, outputs in `D:\GazeOS-cache\outputs`; arm d's Stage A re-run there (12:20), then all 15 folds, the deployment model and Phase 6. |
| 2026-10-26 | Phase 4 done at 10:43; arm d (ETH-XGaze weights + RT-GENE) chosen by the pre-registered rule. Phase 5 started: arm d on all 15 folds, then the deployment model. |
| 2026-10-26 | Phase 4 fold 0: a ImageNet 3.15°, b ETH weights 2.86°, c RT-GENE 2.69°, d ETH + RT-GENE 2.70° (test subject p00). Shards copied to the Drive folder (18.4 GB). Training memory fixed (persistent training workers, no validation workers, 4 workers) after the page file on C: grew to 18 GB. |
| 2026-10-26 | Phase 3 steps 15-16 done (results in the table). Training speed fixed: shards copied to the NVMe SSD and rechunked; the loader went from 62 to 464 img/s. Phase 4 started on the laptop GPU. |
| 2026-10-26 | Phase 2 done on branch `feature/no-ethxgaze-pipeline`. Google Drive access for step 17 goes through Google Drive for desktop (a local synced folder); the Drive folder must stay Restricted. Packing started. |
