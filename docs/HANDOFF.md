# Your part: what needs you

Everything that could be built without a person, a webcam, private data or
GPUs is built and tested (see STATUS.md). These are the steps only you can do,
in the order the PRD's timeline needs them. Commands assume
`pip install -e ".[app,train,research,dev]"`.

## 0. Repository access (blocking the push)

The commits are made, but the push to GitHub was refused: the Claude GitHub
App does not have write access to `deep-9549/Gaze-Os`. Connect or reconnect
GitHub at https://claude.ai/connect-github and make sure the app is installed
on that repository; then the branch `feature/gazeos-build` can be pushed.

## 1. Week 1-2 (start now: approvals take days)

- [ ] **Request dataset access**: ETH-XGaze, Gaze360 and GazeCapture (registration forms on each project
      site). MPIIFaceGaze is a direct download.
- [ ] **Ethics application**: a draft is in `docs/study/ethics_application_draft.md`. Fill in the names, the
      committee's form and submit it.
- [ ] **Contact one or two rehabilitation centres or NGOs** for case-study participants
      (outreach note in `docs/study/outreach_note.md`).
- [ ] **Choose a licence** for the code (for example Apache 2.0 or MIT; GPL if you want derivatives to stay
      open). Add it as `LICENSE` and to `pyproject.toml`.
- [ ] **Try it on your laptop** (this is the "run L2CS-Net on a webcam" checklist item):
  ```bash
  python -m tools.download_models
  gazeos --dry-run          # watch the overlay follow your eyes; nothing is clicked
  gazeos                    # full control; Ctrl+Alt+G stops everything
  ```
  Then download the L2CS-Net MPIIGaze weights from https://github.com/Ahmednull/L2CS-Net and convert them:
  ```bash
  python -m training.convert_l2cs L2CSNet_mpiigaze.pkl models/l2cs_mpii.onnx --dataset mpiigaze
  gazeos --gaze-model models/l2cs_mpii.onnx
  ```
  Check the debug view
  (menu > Debug view): the gaze arrow should point where you look. If pitch or yaw come out mirrored,
  tell me and I will flip the conversion.
- [ ] **Turn on GitHub Actions** for the repository so CI runs on every push.

## 2. Data (weeks 2-6)

- [ ] **Pack the public datasets** once downloaded:
  ```bash
  python -m training.data.packers mpiifacegaze /path/to/MPIIFaceGaze data/shards/mpiifacegaze
  python -m training.data.packers ethxgaze /path/to/xgaze_224/train data/shards/ethxgaze --stride 3
  ```
- [ ] **Collect GazeOS-Webcam** (after ethics approval; signed consent first, see `docs/study/`):
  `python -m tools.collect --out data/recordings`, three sessions per participant (50/65/80 cm,
  bright/dim, glasses where relevant). Then
  `python -m training.data.packers gazeos data/recordings data/shards/gazeos_webcam`.
- [ ] **Record gestures**: `python -m tools.label_gestures --participant P01` (an operator holds SPACE during
  each gesture).
- [ ] **Language corpora**: download DailyDialog, OpenSubtitles (OPUS) and the Vertanen-Kristensson AAC corpus
  as plain text (one utterance per line) into `data/corpus/`.

## 3. Training on Kaggle or Colab (weeks 4-11, ~50-65 GPU hours)

See `docs/TRAINING.md` for the exact commands per stage (A pre-training, B fine-tuning per fold, C
distillation, D meta-learning, gestures, language models) and how to export and quantize.
Upload the shards (not the raw datasets) to a private Kaggle dataset.

## 4. Real-hardware checks (any time from week 2)

- [ ] Run `python -m tools.latency_profiler --seconds 60 --model models/gazenet_lite_int8.onnx` on the
      reference laptop and a 10-minute `--soak`; compare against PRD 18.2.
- [ ] Windows checks no container can do: clicks land on the right pixel at 125% and 150% display scaling;
      the keyboard and action bar never steal focus; UI Automation snapping works in a browser and in
      Explorer; text-to-speech speaks.
- [ ] Optional: `python -m tools.camera_calibration` with a printed checkerboard (ablation E4).

## 5. User study (weeks 9-13, after approval)

Protocol and forms are in `docs/study/`. Tasks: calibration, `python -m tools.fitts_task`, web task,
transcription, reading, robustness. Score questionnaires with `python -m tools.questionnaires`, analyse
with `research.stats.repeated_measures`, and draw charts with `research.figures`.

## 6. Things to decide or tell me

- Team size (four, three or solo): it changes scope (PRD 25).
- Which calendar week is week 1 and the final evaluation date (PRD 24's open question).
- Anything that behaves oddly on your machine: send the debug view screenshot and the log
  (`gazeos -v`), and I will fix it.
