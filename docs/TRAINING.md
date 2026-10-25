# Training, export and experiments

All runs are Hydra configs in `configs/`; each run saves its resolved config,
the Git commit and `result.json` next to the checkpoint (PRD 16.3). Use 3 seeds
(`seed=0,1,2`) for headline numbers and report mean ± SD.

## 0. Smoke test (any machine, 3 minutes on CPU)

```bash
python -m training.data.packers synthetic - data/shards/synthetic
python -m training.train experiment=smoke_synthetic output_dir=outputs/smoke
python -m training.export_onnx outputs/smoke/best.pt models/smoke.onnx --parity-shards "data/shards/synthetic/*.h5"
```

## 1. Stage A: pre-train M3 on ETH-XGaze (~15-20 GPU h)

```bash
python -m training.train experiment=stage_a_ethxgaze output_dir=outputs/stage_a seed=0
```

### 1b. Stage A without ETH-XGaze (current plan, see [PLAN.md](PLAN.md))

`data/` and `outputs/` are junctions to `F:\GazeOS-data`. Pack RT-GENE straight from its tars,
then compare four starting points on MPIIFaceGaze folds 0-2 with the same Stage B schedule
(on the 4 GB laptop GPU add `train.batch_size=64`):

```bash
python -m training.data.packers rtgene data/raw/rtgene data/shards/rtgene

# Stage A runs for arms c and d (s014-s016 stay out as a cross-dataset test)
python -m training.train experiment=stage_a_rtgene     output_dir=outputs/stage_a_rtgene
python -m training.train experiment=stage_a_rtgene_eth output_dir=outputs/stage_a_rtgene_eth

for f in 0 1 2; do
  python -m training.train experiment=stage_b_mpii data.fold=$f init_checkpoint=null output_dir=outputs/arms/a_imagenet/fold$f
  python -m training.train experiment=stage_b_eth_init data.fold=$f output_dir=outputs/arms/b_eth/fold$f
  python -m training.train experiment=stage_b_mpii data.fold=$f init_checkpoint=outputs/stage_a_rtgene/best.pt output_dir=outputs/arms/c_rtgene/fold$f
  python -m training.train experiment=stage_b_mpii data.fold=$f init_checkpoint=outputs/stage_a_rtgene_eth/best.pt output_dir=outputs/arms/d_eth_rtgene/fold$f
done
```

The winner becomes `init_checkpoint` (or `init_face_checkpoint`) for the full Stage B below and
replaces the ETH-XGaze arm of E5.

## 2. Stage B: fine-tune on MPIIFaceGaze

Benchmark (E1, leave-one-subject-out, 15 folds):

```bash
for f in $(seq 0 14); do
  python -m training.train experiment=stage_b_mpii data.fold=$f output_dir=outputs/stage_b/fold$f
done
```

Each `result.json` has `test.mean_error_deg`; average over folds. Deployment model (all 15 subjects):

```bash
python -m training.train experiment=stage_b_mpii data.fold=null output_dir=outputs/stage_b/all
```

## 3. Ablations

| Exp | Command |
| --- | --- |
| E2 streams | `experiment=e2_streams model.use_face=false` / `model.use_head=false` / `model.fusion=concat`; M2: `experiment=m2_face_only` |
| E3 loss | `experiment=e3_loss train.loss_mode=regression` / `classification`, `train.beta=2` |
| E4 normalization | pack MPIIFaceGaze with and without normalization (crop-only variant: set `W` to a pure crop in `packers.py`) |
| E5 pre-training | `experiment=e5_pretraining` (ImageNet only) vs Stage B (ImageNet + ETH-XGaze) |
| E12 uncertainty | every `result.json` reports `uncertainty_spearman`; reliability diagram via `research.figures` |

## 4. Stage C: distil M3 into M4 Lite, export, quantize (E8)

```bash
python -m training.distill teacher=outputs/stage_b/all/best.pt output_dir=outputs/m4
python -m training.export_onnx outputs/m4/best.pt models/gazenet_lite.onnx --parity-shards "data/shards/mpiifacegaze/*.h5"
python -m training.quantize models/gazenet_lite.onnx models/gazenet_lite_int8.onnx \
    --calib "data/shards/mpiifacegaze/*.h5" --eval "data/shards/mpiifacegaze/*.h5"
```

Export fails if PyTorch/ONNX parity is worse than 0.05°; quantization is rejected if it loses 0.2° or
more. Copy the accepted model to `models/` and write its model card (`docs/models/`).

## 5. Stage D (optional): meta-learned personalization P5

```bash
python -m training.meta_train outputs/m4/best.pt "data/shards/gazeos_webcam/*.h5" models/meta_init.npz
python -m research.e6_personalization --recordings "data/recordings/P*/*.npz" --meta-init models/meta_init.npz \
    --out results/e6.json --figure figures/e6.png
```

## 6. Gestures (E10)

```bash
python -m training.train_gesture "data/gestures/*.npz" models/gesture_tcn.onnx --model tcn
python -m training.train_gesture "data/gestures/*.npz" models/gesture_gru.onnx --model gru
```

## 7. Language models (E11)

```bash
python -m training.train_lm ngram data/corpus/*.txt --out models/ngram_en.json
python -m training.train_lm gpt   data/corpus/*.txt --out models/lm_small.onnx --epochs 3
```

## Cloud training (Colab, Kaggle)

- `notebooks/train_cloud.ipynb` runs on both: edit its settings cell (data folder, job list), run all.
- Upload `data/shards/` (and `data/weights/`) to a private Google Drive folder or a private Kaggle
  dataset; never the raw datasets (licences), and never a public link.
- The notebook copies shards to the local disk first (HDF5 over the Drive mount is slow) and writes
  outputs to Drive on Colab, so a disconnect costs at most one epoch.
- `train.resume=true` continues a run from `output_dir/last.pt`; finished runs (with `result.json`)
  are skipped, so re-running the notebook after a disconnect picks up where it stopped.
- Mixed precision is on by default on GPU (`train.amp=true`); use `data.workers=2`.
