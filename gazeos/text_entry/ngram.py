"""Word 3-gram with interpolated Kneser-Ney smoothing (L1, PRD 14.1).

A pure-Python implementation so the app needs no compiled dependency.
KenLM ARPA models trained with training/train_lm.py can be loaded through
`ArpaModel` when available.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9']+|[.!?]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def sentences(text: str) -> list[list[str]]:
    out, cur = [], []
    for tok in tokenize(text):
        if tok in ".!?":
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        out.append(cur)
    return out


class KneserNeyTrigram:
    def __init__(self, discount: float = 0.75) -> None:
        self.d = discount
        self.vocab: set[str] = set()
        self.c3: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.c2_cont: dict[str, Counter[str]] = defaultdict(Counter)  # continuation counts for bigrams
        self.cont_unigram: Counter[str] = Counter()  # number of distinct left contexts
        self.total_bigram_types = 0
        self.unigram: Counter[str] = Counter()

    # -- training --------------------------------------------------------------
    def fit(self, corpus: Iterable[Sequence[str]], max_vocab: int = 30000) -> KneserNeyTrigram:
        sents = [list(s) for s in corpus if s]
        freq = Counter(w for s in sents for w in s)
        self.vocab = {w for w, _ in freq.most_common(max_vocab)}
        bigram_types: set[tuple[str, str]] = set()
        trigram_left: dict[tuple[str, str], set[str]] = defaultdict(set)
        for s in sents:
            toks = [BOS, BOS] + [w if w in self.vocab else UNK for w in s] + [EOS]
            for w in toks[2:]:
                self.unigram[w] += 1
            for i in range(2, len(toks)):
                a, b, c = toks[i - 2], toks[i - 1], toks[i]
                self.c3[(a, b)][c] += 1
                bigram_types.add((b, c))
                trigram_left[(b, c)].add(a)
        for b, c in bigram_types:
            self.cont_unigram[c] += 1
        # Continuation counts for the bigram level: distinct left words of (b, c).
        for (b, c), lefts in trigram_left.items():
            self.c2_cont[b][c] = len(lefts)
        self.total_bigram_types = len(bigram_types)
        self.vocab |= {UNK, EOS}
        return self

    def update(self, sentence: Sequence[str]) -> None:
        """Online update from a sentence the user typed (personalization, PRD 14.4)."""
        toks = [BOS, BOS] + list(sentence) + [EOS]
        for w in sentence:
            self.vocab.add(w)
        for i in range(2, len(toks)):
            a, b, c = toks[i - 2], toks[i - 1], toks[i]
            if self.c3[(a, b)][c] == 0 and self.c2_cont[b][c] == 0:
                self.cont_unigram[c] += 1
                self.total_bigram_types += 1
            self.c3[(a, b)][c] += 1
            self.c2_cont[b][c] += 1
            self.unigram[c] += 1

    # -- probabilities ---------------------------------------------------------
    def _p_uni(self, w: str) -> float:
        n = len(self.vocab) or 1
        return (self.cont_unigram.get(w, 0) + 0.5) / (self.total_bigram_types + 0.5 * n)

    def _p_bi(self, b: str, w: str) -> float:
        counts = self.c2_cont.get(b)
        if not counts:
            return self._p_uni(w)
        total = sum(counts.values())
        types = len(counts)
        return max(counts.get(w, 0) - self.d, 0.0) / total + self.d * types / total * self._p_uni(w)

    def prob(self, w: str, context: Sequence[str]) -> float:
        ctx = [BOS, BOS, *context][-2:]
        w = w if w in self.vocab else UNK
        counts = self.c3.get((ctx[0], ctx[1]))
        if not counts:
            return self._p_bi(ctx[1], w)
        total = sum(counts.values())
        types = len(counts)
        return max(counts.get(w, 0) - self.d, 0.0) / total + self.d * types / total * self._p_bi(ctx[1], w)

    def next_words(self, context: Sequence[str], k: int = 5, prefix: str = "") -> list[tuple[str, float]]:
        ctx = [BOS, BOS, *context][-2:]
        candidates: set[str] = set()
        candidates.update(self.c3.get((ctx[0], ctx[1]), {}).keys())
        candidates.update(self.c2_cont.get(ctx[1], {}).keys())
        if len(candidates) < k * 4:
            candidates.update(w for w, _ in self.cont_unigram.most_common(200))
        candidates -= {UNK, EOS, BOS}
        if prefix:
            candidates = {w for w in candidates if w.startswith(prefix)}
        scored = sorted(((w, self.prob(w, context)) for w in candidates), key=lambda x: -x[1])
        return scored[:k]

    def perplexity(self, corpus: Iterable[Sequence[str]]) -> float:
        log_sum, n = 0.0, 0
        for s in corpus:
            ctx: list[str] = []
            for w in [*s, EOS]:
                log_sum += math.log(max(self.prob(w, ctx), 1e-12))
                ctx.append(w)
                n += 1
        return math.exp(-log_sum / max(n, 1))

    # -- persistence -----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {
            "d": self.d,
            "vocab": sorted(self.vocab),
            "c3": [[a, b, dict(c)] for (a, b), c in self.c3.items()],
            "c2": {b: dict(c) for b, c in self.c2_cont.items()},
            "cu": dict(self.cont_unigram),
            "tbt": self.total_bigram_types,
            "uni": dict(self.unigram),
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> KneserNeyTrigram:
        data = json.loads(Path(path).read_text())
        m = cls(data["d"])
        m.vocab = set(data["vocab"])
        for a, b, c in data["c3"]:
            m.c3[(a, b)] = Counter(c)
        for b, c in data["c2"].items():
            m.c2_cont[b] = Counter(c)
        m.cont_unigram = Counter(data["cu"])
        m.total_bigram_types = data["tbt"]
        m.unigram = Counter(data["uni"])
        return m


class ArpaModel:
    """Optional KenLM model (ARPA or binary) if the `kenlm` module is installed."""

    def __init__(self, path: str | Path, vocab: Iterable[str]) -> None:
        import kenlm  # type: ignore[import-not-found]

        self.model = kenlm.Model(str(path))
        self.vocab = list(vocab)

    def prob(self, w: str, context: Sequence[str]) -> float:
        ctx = " ".join(context[-2:])
        full = self.model.score(f"{ctx} {w}".strip(), bos=True, eos=False)
        base = self.model.score(ctx, bos=True, eos=False) if ctx else 0.0
        return float(10 ** (full - base))

    def next_words(self, context: Sequence[str], k: int = 5, prefix: str = "") -> list[tuple[str, float]]:
        cands = [w for w in self.vocab if w.startswith(prefix)] if prefix else self.vocab
        scored = sorted(((w, self.prob(w, context)) for w in cands[:5000]), key=lambda x: -x[1])
        return scored[:k]
