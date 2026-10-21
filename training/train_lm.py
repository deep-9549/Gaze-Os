"""Train the prediction language models (PRD 14.1-14.3, experiment E11).

    # L1: Kneser-Ney 3-gram as JSON (loaded by gazeos.text_entry.ngram)
    python -m training.train_lm ngram corpus/*.txt --out models/ngram_en.json
    # L2: small word-level transformer exported to ONNX (+ .vocab.txt)
    python -m training.train_lm gpt corpus/*.txt --out models/lm_small.onnx --epochs 3

Corpora (download yourself; licences vary, see docs/DATASETS.md): DailyDialog,
OpenSubtitles (OPUS), the Vertanen-Kristensson AAC corpus. Plain text, one
utterance per line. Lines are lowercased and split with the runtime tokenizer.
A held-out 5% gives perplexity and top-3 next-word accuracy.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path

from gazeos.text_entry.ngram import KneserNeyTrigram, sentences

log = logging.getLogger(__name__)


def read_corpus(patterns: list[str], max_lines: int | None = None) -> list[list[str]]:
    out: list[list[str]] = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    out.extend(sentences(line))
                    if max_lines and len(out) >= max_lines:
                        return out
    return out


def split(sents: list[list[str]], seed: int = 0) -> tuple[list[list[str]], list[list[str]]]:
    rng = random.Random(seed)
    s = list(sents)
    rng.shuffle(s)
    n = max(1, len(s) // 20)
    return s[n:], s[:n]


def top3_accuracy(model: object, test: list[list[str]], limit: int = 3000) -> float:
    hits = total = 0
    for sent in test:
        for i in range(1, len(sent)):
            preds = [w for w, _ in model.next_words(sent[:i], 3)]  # type: ignore[attr-defined]
            hits += sent[i] in preds
            total += 1
            if total >= limit:
                return hits / total
    return hits / max(total, 1)


def train_ngram(sents: list[list[str]], out: str) -> dict[str, float]:
    tr, te = split(sents)
    m = KneserNeyTrigram().fit(tr)
    m.save(out)
    return {"perplexity": m.perplexity(te[:2000]), "top3": top3_accuracy(m, te)}


def train_gpt(sents: list[list[str]], out: str, epochs: int = 3, vocab_size: int = 30000, context: int = 32,
              batch: int = 64, lr: float = 3e-4) -> dict[str, float]:
    import torch

    from training.models.gpt import SmallGPT

    tr, te = split(sents)
    counts = Counter(w for s in tr for w in s)
    itos = ["<pad>", "<unk>", "<s>", "</s>"] + [w for w, _ in counts.most_common(vocab_size - 4)]
    stoi = {w: i for i, w in enumerate(itos)}

    def encode(s: list[str]) -> list[int]:
        return [2] + [stoi.get(w, 1) for w in s] + [3]

    def batches(data: list[list[str]], shuffle: bool):  # type: ignore[no-untyped-def]
        seqs = [encode(s)[: context + 1] for s in data if s]
        if shuffle:
            random.shuffle(seqs)
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            L = max(len(c) for c in chunk)
            x = torch.zeros(len(chunk), L, dtype=torch.long)
            for j, c in enumerate(chunk):
                x[j, : len(c)] = torch.tensor(c)
            yield x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallGPT(len(itos), context=context).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    for ep in range(epochs):
        model.train()
        for x in batches(tr, True):
            x = x.to(device)
            logits = model(x[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        log.info("epoch %d loss %.3f", ep, float(loss))
    model.eval()
    nll = n = 0.0
    with torch.no_grad():
        for x in batches(te[:2000], False):
            x = x.to(device)
            logits = model(x[:, :-1])
            nll += float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0, reduction="sum"))
            n += float((x[:, 1:] != 0).sum())
    model = model.cpu()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 8, dtype=torch.long), out, input_names=["tokens"], output_names=["logits"],
                      dynamic_axes={"tokens": {1: "T"}, "logits": {1: "T"}}, opset_version=17, dynamo=False)
    Path(out.removesuffix(".onnx") + ".vocab.txt").write_text("\n".join(itos), encoding="utf-8")
    from gazeos.inference.lm import OnnxWordLM

    return {"perplexity": math.exp(nll / max(n, 1)), "top3": top3_accuracy(OnnxWordLM(out), te, 1000),
            "params": sum(p.numel() for p in model.parameters())}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ngram", "gpt"])
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-lines", type=int, default=None)
    a = ap.parse_args()
    sents = read_corpus(a.corpus, a.max_lines)
    log.info("%d sentences", len(sents))
    report = train_ngram(sents, a.out) if a.kind == "ngram" else train_gpt(sents, a.out, a.epochs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
