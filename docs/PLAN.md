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
| Laptop (RTX 3050, 4 GB) | Packing, debugging, smoke runs, Stage A arm selection on folds 0-2, export, INT8, latency profiling (the CPU targets are measured here). Batch 64, mixed precision on |
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
| 15 | Pack MPIIFaceGaze (15 subjects) and RT-GENE into `data/shards/` | In progress |
| 16 | Checks: image grid with gaze arrows; RT-GENE vs MPII head-pose spread (head signs); an MPII-trained model on RT-GENE has positive per-axis correlation | To do |
| 17 | Copy shards to `G:\My Drive\Minor Project\GazeOS\data\shards` (Drive for desktop uploads them) | In progress: layout created, both weight files copied; shards follow step 16 |

### Phase 4: replacement Stage A (new E5)

| # | Step | Status |
| --- | --- | --- |
| 18 | MPII folds 0-2, compare: (a) ImageNet only, (b) ETH-XGaze face weights, (c) RT-GENE pre-training, (d) ETH weights then RT-GENE, (e) optional GazeGene. Keep RT-GENE s014-s016 out of pre-training as a cross-dataset test | To do |

### Phase 5: benchmark and deployment model

| # | Step | Status |
| --- | --- | --- |
| 19 | Stage B, 15 LOSO folds x 3 seeds from the winning Stage A (E1); M0 and M1 baselines on the same folds. Target 4.5° or less | To do |
| 20 | Deployment model on all 15 subjects (`data.fold=null`) | To do |
| 21 | Ablations E2, E3, E12 on MPII | To do |

### Phase 6: real-time model

| # | Step | Status |
| --- | --- | --- |
| 22 | Stage C distillation, ONNX export (parity < 0.05°), INT8 (rejected at a 0.2° loss); optional UniGaze-B teacher | To do |
| 23 | Model card; `models/gazenet_lite_int8.onnx`; `gazeos --dry-run` and the latency profiler on the laptop | To do |

### Phase 7: unchanged from HANDOFF.md

| # | Step | Status |
| --- | --- | --- |
| 24 | Ethics approval, then GazeOS-Webcam collection (E6, E9, Stage D) | Waiting (approval) |
| 25 | Gesture dataset (E10), language-model corpora (E11), user study | To do |

## 5. Open decisions

- Team size (HANDOFF section 6). Solo: drop arm (e) and Stage D.
- Whether the 5 TB Google plan includes Colab Pro (longer, more reliable sessions).

## 6. Log

| Date | Change |
| --- | --- |
| 2026-10-25 | Plan written. Data audit of `F:\GazeOS-data`: MPIIFaceGaze 4 of 15 subjects; RT-GENE s004 truncated, s005 and s006 missing. RT-GENE label order verified. Google Drive (5 TB) added for shards and checkpoints. |
| 2026-10-26 | Phase 0 done. MediaPipe resolved to 1.1.1 (newer than the code was written against); the unit tests pass, but check landmarks on a live webcam with `gazeos --dry-run`. Synthetic shards take about 190 KB per sample, so expect roughly 10-20 GB for RT-GENE. |
| 2026-10-26 | Steps 4 and 5 done: MPIIFaceGaze complete (15 subjects, 37,667 images); RT-GENE complete (17 subjects, about 114,600 frames). |
| 2026-10-26 | Step 6 done: L2CS Gaze360 (py-feat mirror) and ETH-XGaze ResNet-18 weights in `weights/`, both checked against the models they load into. GazeGene deferred until Phase 4 shows whether more pre-training helps. |
| 2026-10-26 | Phase 2 done on branch `feature/no-ethxgaze-pipeline`. Google Drive access for step 17 goes through Google Drive for desktop (a local synced folder); the Drive folder must stay Restricted. Packing started. |
