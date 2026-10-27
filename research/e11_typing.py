"""E11 (PRD 16): how much does word prediction save when typing? Offline simulation.

Each test sentence is typed word by word with an ideal user: before each letter, if the
wanted word is among the k suggestions it is chosen (one selection, which also adds the
space); otherwise the next letter is typed. A word never offered is typed in full plus a
space. Punctuation is dropped. Keystroke savings = 1 - keys used / characters needed.
Personal learning is off (`learn` would adapt to the test text).

    python -m research.e11_typing D:/GazeOS-cache/corpora/dailydialog_test.txt --models models --out e11.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from gazeos.text_entry.layouts import get_layout
from gazeos.text_entry.ngram import tokenize
from gazeos.text_entry.predictor import Predictor


def simulate(predictor: Predictor, sentences: list[list[str]]) -> dict[str, float]:
    predictor.learn = lambda _text: None  # type: ignore[method-assign]
    keys = chars = 0
    chosen_words = words = 0
    for sent in sentences:
        text = ""
        for w in sent:
            words += 1
            chars += len(w) + 1
            for i in range(len(w) + 1):
                if w in predictor.suggest(text, w[:i]).words:
                    keys += i + 1
                    chosen_words += 1
                    break
            else:
                keys += len(w) + 1
            text += w + " "
    return {"keystroke_savings": 1 - keys / max(chars, 1), "keys_per_char": keys / max(chars, 1),
            "words_from_predictions": chosen_words / max(words, 1), "sentences": len(sentences), "words": words}


def conditions(models: Path) -> dict[str, Any]:
    layout = get_layout("qwerty")
    out: dict[str, Any] = {}
    base = Predictor.default(layout)
    out["L0 word list"] = Predictor(base.trie, None, layout=layout)
    out["L0+L1 seed 3-gram"] = base
    if (models / "ngram_en.json").exists():
        out["L0+L1 trained 3-gram"] = Predictor.default(layout, ngram_path=models / "ngram_en.json")
        if (models / "lm_small.onnx").exists():
            out["L0+L1+L2 transformer"] = Predictor.default(layout, ngram_path=models / "ngram_en.json",
                                                            transformer_path=models / "lm_small.onnx")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("test_text")
    ap.add_argument("--models", default="models")
    ap.add_argument("--sentences", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = [s for s in (Path(a.test_text).read_text(encoding="utf-8").splitlines()) if s.strip()]
    random.Random(0).shuffle(lines)
    sents = [t for t in (tokenize(line) for line in lines) if t][: a.sentences]
    sents = [[w for w in s if any(ch.isalpha() for ch in w)] for s in sents]
    report = {name: simulate(p, sents) for name, p in conditions(Path(a.models)).items()}
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
