# Model card: word prediction (ngram_en.json, lm_small.onnx)

The language models behind the keyboard's suggestions (PRD 14.1-14.3). Both stay local:
OpenSubtitles and DailyDialog are research corpora, and `models/ngram_*.json`,
`models/lm_*.onnx` and `models/lm_*.vocab.txt` are git-ignored. Without them the app falls
back to the seed 3-gram built into the repository.

| Field | L1: `ngram_en.json` | L2: `lm_small.onnx` (+ `.vocab.txt`) |
| --- | --- | --- |
| Model | Kneser-Ney 3-gram (`gazeos.text_entry.ngram`) | word-level GPT (`training.train_lm` SmallGPT), 10.8 M parameters, 30,000-word vocabulary, 32-word window |
| Training | `python -m training.train_lm ngram ...` | `python -m training.train_lm gpt ... --max-lines 1500000 --epochs 3` (laptop GPU, 1 h) |
| Data | 1.5 M sentences: all of DailyDialog train (87,170 utterances), then OpenSubtitles English (OPUS) up to the cap; lowercased, runtime tokenizer; 5% held out | same |
| Size | 42.6 MB JSON | 74.2 MB fp32 ONNX + 0.3 MB vocabulary |
| In the app | always, when the file exists | only with `typing.use_transformer: true` (off by default) |

## Metrics

| Metric | L1 3-gram | L2 transformer |
| --- | --- | --- |
| Perplexity, held-out 5% | 33.9 | 45.6 |
| Top-3 next-word accuracy, held-out 5% | 48.0% | 43.5% |
| Keystroke savings (E11), with the word list | 54.6% | 55.5% (54.9% with only 2 words of context) |
| Time per suggestion | 5 ms | + 3.4 ms per new word (one ONNX run per context) |

The two perplexities are not directly comparable (different vocabularies and unknown-word
handling); E11 is the common test. E11: 300 held-out DailyDialog test sentences typed by
an ideal user with 4 suggestions (`research.e11_typing`). For reference, the word list
alone saves 43.2% and the seed 3-gram 43.9%.

## Limitations

- Conversational English from TV subtitles and textbook dialogues, not the phrases of
  AAC users (the AAC corpus of Vertanen and Kristensson is no longer online).
  Personal learning (`Predictor.learn`) adapts the 3-gram to what the user types.
- Subtitles contain profanity and dated phrasing; there is no filter on suggestions.
- English only.
- Licences: DailyDialog is CC BY-NC-SA 4.0; OpenSubtitles is distributed by OPUS for
  research. Non-commercial use only; do not redistribute the files.
