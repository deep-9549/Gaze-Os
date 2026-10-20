"""Noisy-channel decoding of gaze keystrokes (PRD 14.2).

P(w | g_1..n) is proportional to P_LM(w | context) * prod_i exp(-|g_i - k(w_i)|^2 / 2 sigma^2),
with sigma from the user's calibration error. It fixes near-miss key selections
the way phone keyboards fix fat-finger typos.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gazeos.text_entry.layouts import Layout
from gazeos.text_entry.trie import PrefixTrie, _Node

LMScore = Callable[[str, Sequence[str]], float]  # P(word | context)


@dataclass(frozen=True)
class Candidate:
    word: str
    log_score: float
    complete: bool  # True if len(word) == number of key presses


class NoisyChannelDecoder:
    def __init__(
        self,
        layout: Layout,
        trie: PrefixTrie,
        sigma_keys: float = 0.5,
        beam: int = 64,
        lm_weight: float = 1.0,
    ) -> None:
        self.layout = layout
        self.trie = trie
        self.sigma = sigma_keys  # gaze error in key units
        self.beam = beam
        self.lm_weight = lm_weight
        self._centers = layout.letter_centers()

    def set_sigma_from_px(self, error_px: float, key_unit_px: float) -> None:
        self.sigma = max(error_px / max(key_unit_px, 1.0), 0.15)

    def _key_loglik(self, point: tuple[float, float], letter: str) -> float:
        c = self._centers.get(letter)
        if c is None:
            return -1e9
        d2 = (point[0] - c[0]) ** 2 + (point[1] - c[1]) ** 2
        return -d2 / (2.0 * self.sigma**2)

    def decode(
        self,
        points: Sequence[tuple[float, float]],
        context: Sequence[str] = (),
        lm: LMScore | None = None,
        k: int = 5,
        allow_completions: bool = True,
    ) -> list[Candidate]:
        """Best words for the key-press points of the current word."""
        if not points:
            return []
        # Beam search over trie paths, scoring keys and prefix frequency.
        beam: list[tuple[float, str, _Node]] = [(0.0, "", self.trie.root)]
        for p in points:
            nxt: list[tuple[float, str, _Node]] = []
            for score, prefix, node in beam:
                for ch, child in node.children.items():
                    kl = self._key_loglik(p, ch)
                    if kl < -18.0:  # beyond ~6 sigma: impossible
                        continue
                    nxt.append((score + kl, prefix + ch, child))
            if not nxt:
                return []
            # Rank partial paths by key fit plus how common words below them are.
            nxt.sort(key=lambda x: -(x[0] + 0.3 * math.log1p(x[2].best[0][0] if x[2].best else 0.0)))
            beam = nxt[: self.beam]

        total = self.trie.total or 1.0
        out: dict[str, Candidate] = {}
        for score, prefix, node in beam:
            words: list[tuple[str, bool]] = []
            if node.count > 0:
                words.append((prefix, True))
            if allow_completions:
                words.extend((w, False) for _, w in node.best[:3] if w != prefix)
            for w, complete in words:
                if lm is not None:
                    p_lm = max(lm(w, context), 1e-9)
                else:
                    p_lm = max(self.trie.count(w) / total, 1e-9)
                # Completions skip key evidence for unseen letters; a small penalty
                # keeps exact-length words ahead when they fit equally well.
                s = score + self.lm_weight * math.log(p_lm) - (0.0 if complete else 1.0)
                if w not in out or out[w].log_score < s:
                    out[w] = Candidate(w, s, complete)
        return sorted(out.values(), key=lambda c: -c.log_score)[:k]
