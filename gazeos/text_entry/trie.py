"""Prefix trie with word frequencies: instant completion of the current word (L0, PRD 14.1)."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from pathlib import Path


class _Node:
    __slots__ = ("children", "count", "best")

    def __init__(self) -> None:
        self.children: dict[str, _Node] = {}
        self.count = 0.0
        # Cached top completions below this node: list of (count, word).
        self.best: list[tuple[float, str]] = []


class PrefixTrie:
    def __init__(self, cache_size: int = 8) -> None:
        self.root = _Node()
        self.cache_size = cache_size
        self.total = 0.0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, word: str, count: float = 1.0) -> None:
        word = word.lower()
        if not word:
            return
        path = [self.root]
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
            path.append(node)
        if node.count == 0:
            self._size += 1
        node.count += count
        self.total += count
        total = node.count
        for n in path:
            # Keep the cached best list sorted and unique.
            entries = [e for e in n.best if e[1] != word]
            entries.append((total, word))
            entries.sort(key=lambda e: (-e[0], e[1]))
            n.best = entries[: self.cache_size]

    def count(self, word: str) -> float:
        node = self._find(word.lower())
        return node.count if node is not None else 0.0

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and self.count(word) > 0

    def _find(self, prefix: str) -> _Node | None:
        node = self.root
        for ch in prefix:
            nxt = node.children.get(ch)
            if nxt is None:
                return None
            node = nxt
        return node

    def complete(self, prefix: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-k words starting with prefix, with relative frequency."""
        node = self._find(prefix.lower())
        if node is None:
            return []
        if k <= self.cache_size:
            items = node.best[:k]
        else:
            items = heapq.nlargest(k, self._walk(node, prefix.lower()))
        total = self.total or 1.0
        return [(w, c / total) for c, w in items]

    def _walk(self, node: _Node, prefix: str) -> Iterable[tuple[float, str]]:
        stack = [(node, prefix)]
        while stack:
            n, p = stack.pop()
            if n.count > 0:
                yield (n.count, p)
            for ch, child in n.children.items():
                stack.append((child, p + ch))

    def words(self) -> Iterable[tuple[str, float]]:
        for c, w in self._walk(self.root, ""):
            yield w, c

    @classmethod
    def from_file(cls, path: str | Path) -> PrefixTrie:
        """Lines of `word<TAB>count`; lines starting with # are comments."""
        trie = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            trie.add(parts[0], float(parts[1]) if len(parts) > 1 else 1.0)
        return trie
