"""Prediction stack: merge and re-rank suggestions from L0-L3 (PRD 14.1, FR-28).

L0 prefix trie, L1 Kneser-Ney 3-gram, L2 small transformer (ONNX, optional),
L3 local LLM through Ollama (optional, asynchronous). Plus the noisy-channel
decoder for the current word's gaze points.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from gazeos.text_entry.decoder import NoisyChannelDecoder
from gazeos.text_entry.layouts import Layout
from gazeos.text_entry.ngram import KneserNeyTrigram, tokenize
from gazeos.text_entry.trie import PrefixTrie

DATA_DIR = Path(__file__).parent / "data"


class NextWordModel(Protocol):
    def next_words(self, context: Sequence[str], k: int = 5, prefix: str = "") -> list[tuple[str, float]]: ...
    def prob(self, w: str, context: Sequence[str]) -> float: ...


@dataclass
class Suggestions:
    words: list[str]
    sentence: str | None = None  # L3 sentence completion, when confident
    sources: dict[str, str] = field(default_factory=dict)


class Predictor:
    def __init__(
        self,
        trie: PrefixTrie,
        ngram: KneserNeyTrigram | None = None,
        transformer: NextWordModel | None = None,
        layout: Layout | None = None,
        k: int = 4,
        weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
    ) -> None:
        self.trie = trie
        self.ngram = ngram
        self.transformer = transformer
        self.k = k
        self.w_trie, self.w_ngram, self.w_tf = weights
        self.decoder = NoisyChannelDecoder(layout, trie) if layout is not None else None
        self.llm: LocalLLM | None = None
        self._sentence: str | None = None

    @classmethod
    def default(cls, layout: Layout | None = None, k: int = 4) -> Predictor:
        """Bundled English word list and seed conversational n-gram."""
        trie = PrefixTrie.from_file(DATA_DIR / "words_en.txt")
        seed = DATA_DIR / "seed_corpus.txt"
        ngram = None
        if seed.exists():
            from gazeos.text_entry.ngram import sentences

            ngram = KneserNeyTrigram().fit(sentences(seed.read_text(encoding="utf-8")))
            for w, c in ngram.unigram.items():
                if w.isalpha() and w not in trie:
                    trie.add(w, c)
        return cls(trie, ngram, layout=layout, k=k)

    def lm_prob(self, w: str, context: Sequence[str]) -> float:
        p_trie = max(self.trie.count(w) / (self.trie.total or 1.0), 1e-9)
        if self.ngram is None and self.transformer is None:
            return p_trie
        p = self.w_trie * p_trie
        wsum = self.w_trie
        if self.ngram is not None:
            p += self.w_ngram * self.ngram.prob(w, context)
            wsum += self.w_ngram
        if self.transformer is not None:
            p += self.w_tf * self.transformer.prob(w, context)
            wsum += self.w_tf
        return p / wsum

    def suggest(
        self,
        text_before: str,
        current_word: str,
        gaze_points: Sequence[tuple[float, float]] = (),
    ) -> Suggestions:
        context = tokenize(text_before)[-2:]
        prefix = current_word.lower()
        scores: dict[str, float] = {}
        sources: dict[str, str] = {}

        def add(word: str, score: float, src: str) -> None:
            if not word or word == prefix and not prefix:
                return
            if score > scores.get(word, -math.inf):
                scores[word] = score
                sources[word] = src

        if prefix or gaze_points:
            if self.decoder is not None and gaze_points:
                for c in self.decoder.decode(gaze_points, context, self.lm_prob, k=self.k * 2):
                    add(c.word, c.log_score, "decoder")
            for w, _ in self.trie.complete(prefix, self.k * 3):
                add(w, math.log(self.lm_prob(w, context)) - 2.0 if gaze_points else math.log(self.lm_prob(w, context)), "trie")
        else:
            if self.ngram is not None:
                for w, _p in self.ngram.next_words(context, self.k * 2):
                    add(w, math.log(self.lm_prob(w, context)), "ngram")
            if self.transformer is not None:
                for w, _p in self.transformer.next_words(context, self.k * 2):
                    add(w, math.log(self.lm_prob(w, context)), "transformer")
            if not scores:
                for w, _ in self.trie.complete("", self.k):
                    add(w, math.log(self.lm_prob(w, context)), "trie")
        ranked = sorted(scores, key=lambda w: -scores[w])[: self.k]
        if self.llm is not None:
            self.llm.request(text_before + current_word)
            self._sentence = self.llm.latest()
        return Suggestions(ranked, self._sentence, {w: sources[w] for w in ranked})

    def learn(self, text: str) -> None:
        """Personal dictionary: new words are added and frequent ones boosted (PRD 14.4)."""
        toks = tokenize(text)
        for w in toks:
            if w.isalpha() or "'" in w:
                self.trie.add(w, 5.0)
        if self.ngram is not None and toks:
            self.ngram.update([t for t in toks if t not in ".!?"])


class LocalLLM:
    """L3: whole-sentence completion from a local Ollama server (P2, opt-in).

    Talks only to localhost, runs asynchronously, and never blocks typing.
    """

    def __init__(self, model: str = "qwen2.5:1.5b", url: str = "http://127.0.0.1:11434/api/generate",
                 min_chars: int = 12) -> None:
        if not url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("LocalLLM only talks to a local server")
        self.model, self.url, self.min_chars = model, url, min_chars
        self._latest: str | None = None
        self._busy = False
        self._for: str = ""

    def request(self, text: str) -> None:
        if self._busy or len(text) < self.min_chars or text == self._for:
            return
        self._busy, self._for = True, text
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text: str) -> None:
        import json
        import urllib.request

        prompt = (
            "Complete this message typed by a person who uses eye-gaze typing. "
            "Reply with only the completed sentence.\n\nMessage: " + text
        )
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"num_predict": 24, "temperature": 0.2}}).encode()
        try:
            req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                out = json.loads(resp.read())["response"].strip().splitlines()[0]
            self._latest = out if out.lower().startswith(text.lower()[:8]) else None
        except Exception:
            self._latest = None
        finally:
            self._busy = False

    def latest(self) -> str | None:
        return self._latest
