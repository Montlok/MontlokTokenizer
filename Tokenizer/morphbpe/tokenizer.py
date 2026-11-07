# -*- coding: utf-8 -*-
"""Morphology-constrained BPE tokenizer for traditional Mongolian."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from typing import Any

from Tokenizer.traditional_mongolian.stemmer import MongolStemmer
from Tokenizer.traditional_mongolian.unicode_norm import CTRL_ALL, strip_all_with_map

from . import serialization
from .offsets import MorphToken, Piece, split_on_ascii_space


LONE_VOWELS = frozenset("\u1820\u1821\u1822\u1823\u1824\u1825\u1826")


class MorphBPETokenizer:
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: dict[tuple[str, str], tuple[str, int]] | None = None,
        stemmer: MongolStemmer | None = None,
        min_boundary_confidence: float = 0.60,
        seed_alphabet: bool = False,
        retired: Iterable[str] = (),
        dictionary: Iterable[str] = (),
    ):
        self.vocab = vocab or {"<unk>": 0}
        self.merges = merges or {}
        self.stemmer = stemmer
        self.min_boundary_confidence = min_boundary_confidence
        self.seed_alphabet = seed_alphabet
        self._id_to_token = {i: tok for tok, i in self.vocab.items()}
        # Retired tokens still form inside merges, so tokens built on them keep
        # their segmentation, but a word never ends up with one: a retired final
        # piece is split back into the pair that formed it.
        self.retired = frozenset(retired)
        self._formed_by: dict[str, tuple[str, str]] = {}
        for pair, (merged, _rank) in self.merges.items():
            if merged in self.retired:
                if merged in self._formed_by:
                    raise ValueError(f"retired token {merged!r} is formed by several merges")
                self._formed_by[merged] = pair
        missing = sorted(self.retired - self._formed_by.keys())
        if missing:
            raise ValueError(f"retired tokens without a merge to split them: {missing[:5]}")
        # MVS/FVS preservation is gated on the vocab: vocabularies that carry
        # the control characters as tokens (v3+ builds) keep them in the id
        # stream; older vocabularies fold them exactly as before, so v2
        # bundles stay bit-identical.
        self._preserve_controls = all(ch in self.vocab for ch in CTRL_ALL)
        # Words of a dictionary stay whole, and a dictionary word followed by
        # suffixes splits only there. Other words split where the stemmer finds
        # a suffix, except a lone final vowel: without a root to confirm it,
        # that vowel is as likely part of the root.
        self.dictionary = frozenset(dictionary)
        self._suffixes: frozenset[str] | None = None
        self._longest_suffix = 0
        self._reach_of: tuple[frozenset[str], int] | None = None
        self._spans_of: dict[str, list[tuple[int, int]]] = {}

    # ---- io ----

    @classmethod
    def from_file(
        cls, path: str, stemmer: MongolStemmer | None = None
    ) -> "MorphBPETokenizer":
        payload = serialization.load(path)
        merges = serialization.merges_from_payload(payload)
        return cls(
            vocab={str(k): int(v) for k, v in payload.get("vocab", {}).items()},
            merges=merges,
            stemmer=stemmer,
            min_boundary_confidence=float(
                payload.get("config", {}).get("min_boundary_confidence", 0.60)
            ),
            seed_alphabet=bool(payload.get("config", {}).get("seed_alphabet", False)),
            retired=payload.get("config", {}).get("retired", ()),
            dictionary=payload.get("config", {}).get("dictionary", ()),
        )

    def save(self, path: str, extra_config: dict[str, Any] | None = None) -> None:
        serialization.dump(self, path, extra_config=extra_config)

    # ---- encode ----

    def encode(self, text: str) -> list[int]:
        return [tok.id for tok in self.encode_with_offsets(text)]

    def encode_with_offsets(self, text: str, base_start: int = 0) -> list[MorphToken]:
        tokens: list[MorphToken] = []
        for start, end in split_on_ascii_space(text):
            for tok in self.tokenize_word(text[start:end], base_start + start):
                tokens.append(tok)
        return tokens

    def tokenize_word(self, word: str, base_start: int = 0) -> list[MorphToken]:
        if not self._preserve_controls:
            return self._tokenize_clean_word(word, base_start)

        # Controls are hard segmentation boundaries: each control character
        # becomes its own token at its original position and BPE runs on the
        # control-free segments between them. MVS only ever marks the
        # stem/separated-suffix junction, so splitting there is exactly the
        # morpheme boundary MorphBPE wants anyway; FVS/NIRUGU are rare (2.4‰)
        # word-internal marks where the fertility cost is negligible. Within
        # a clean segment skeleton offsets equal original offsets, which also
        # sidesteps the skeleton-vs-original coordinate drift the legacy path
        # has for multi-boundary words with embedded controls.
        tokens: list[MorphToken] = []
        seg_start = 0
        for i, ch in enumerate(word):
            if ch not in CTRL_ALL:
                continue
            if i > seg_start:
                tokens.extend(
                    self._tokenize_clean_word(word[seg_start:i], base_start + seg_start)
                )
            tokens.append(
                MorphToken(ch, self.vocab[ch], base_start + i, base_start + i + 1)
            )
            seg_start = i + 1
        if seg_start < len(word):
            tokens.extend(
                self._tokenize_clean_word(word[seg_start:], base_start + seg_start)
            )
        return tokens

    def _tokenize_clean_word(self, word: str, base_start: int = 0) -> list[MorphToken]:
        skeleton, boundary_map = strip_all_with_map(word)
        if not skeleton:
            return []
        if self.dictionary:
            return self._tokenize_by_dictionary(skeleton, boundary_map, base_start)

        analysis = self.stemmer.analyze(word) if self.stemmer else None
        forbidden: set[int] = set()
        if analysis is not None and analysis.confidence >= self.min_boundary_confidence:
            forbidden = set(analysis.skeleton_boundaries[1:-1])

        pieces = [
            Piece(ch, boundary_map[i], boundary_map[i + 1])
            for i, ch in enumerate(skeleton)
        ]
        pieces = self._apply_merges(pieces, forbidden)
        if self.retired:
            pieces = self._split_retired(pieces, boundary_map)
        return [
            MorphToken(
                p.text,
                self._piece_id(p.text),
                base_start + p.start,
                base_start + p.end,
            )
            for p in pieces
        ]

    def _tokenize_by_dictionary(
        self, skeleton: str, boundary_map: list[int], base_start: int
    ) -> list[MorphToken]:
        tokens: list[MorphToken] = []
        for start, end in self.dictionary_spans(skeleton):
            text = skeleton[start:end]
            if text in self.vocab and text not in self.retired:
                pieces = [Piece(text, boundary_map[start], boundary_map[end])]
            else:
                pieces = self._apply_merges(
                    [
                        Piece(skeleton[i], boundary_map[i], boundary_map[i + 1])
                        for i in range(start, end)
                    ],
                    set(),
                )
                if self.retired:
                    pieces = self._split_retired(pieces, boundary_map)
            for piece in pieces:
                tokens.append(
                    MorphToken(
                        piece.text,
                        self._piece_id(piece.text),
                        base_start + piece.start,
                        base_start + piece.end,
                    )
                )
        return tokens

    def dictionary_spans(self, text: str) -> list[tuple[int, int]]:
        """Stretches of ``text`` that stay apart: a dictionary word and its suffixes, or the guarded stemmer's cuts."""

        cached = self._spans_of.get(text)
        if cached is None:
            cached = self._find_spans(text)
            if len(self._spans_of) < 2_000_000:
                self._spans_of[text] = cached
        return cached

    def root_end(self, text: str, headwords: Iterable[str] = (), *, shortest: int = 3) -> int:
        """Where the root of ``text`` ends.

        A word listed in ``headwords`` is its own root. Otherwise the root is the longest headword, then the longest
        dictionary word, it starts with whose remainder is made of known suffixes: a dictionary of attested forms lists
        inflected words too, and those still have a shorter root. A root of fewer than ``shortest`` letters is more often a
        chance match than a root. Failing that, it is the stretch the tokenizer cuts first.
        """

        headwords = frozenset(headwords)
        if text in headwords:
            return len(text)
        for known in (headwords, self.dictionary):
            for cut in range(len(text) - 1, shortest - 1, -1):
                if text[:cut] in known and self._suffix_chain(text, cut) is not None:
                    return cut
        return self.dictionary_spans(text)[0][1]

    def _longest_dictionary_word(self) -> int:
        if self._reach_of is None or self._reach_of[0] is not self.dictionary:
            self._reach_of = (self.dictionary, max((len(word) for word in self.dictionary), default=0))
        return self._reach_of[1]

    def _find_spans(self, text: str) -> list[tuple[int, int]]:
        if text in self.dictionary:
            return [(0, len(text))]
        for cut in range(min(len(text) - 1, self._longest_dictionary_word()), 0, -1):
            if text[:cut] in self.dictionary:
                chain = self._suffix_chain(text, cut)
                if chain is not None:
                    return [(0, cut)] + chain
        analysis = self.stemmer.analyze(text) if self.stemmer else None
        if analysis is None or analysis.confidence < self.min_boundary_confidence:
            return [(0, len(text))]
        cuts = list(analysis.skeleton_boundaries)
        spans: list[tuple[int, int]] = []
        for start, end in zip(cuts, cuts[1:]):
            if end - start == 1 and text[start:end] in LONE_VOWELS and spans:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
        return spans or [(0, len(text))]

    def _suffix_chain(self, text: str, start: int) -> list[tuple[int, int]] | None:
        """Spans that cover ``text[start:]`` with known suffixes, longest first; None when they cannot."""

        if self._suffixes is None:
            # A lone vowel is left out: suffixes that are one vowel follow a
            # control character, which already ends the stretch.
            self._suffixes = frozenset(
                item.skeleton
                for item in getattr(self.stemmer, "lookup", ())
                if len(item.skeleton) > 1
            )
            self._longest_suffix = max((len(suffix) for suffix in self._suffixes), default=0)
        n = len(text)
        if start == n:
            return []
        # Right to left: each position takes the longest suffix whose remainder can be covered.
        step = [0] * (n + 1)
        step[n] = n
        covered = [False] * (n + 1)
        covered[n] = True
        for i in range(n - 1, start - 1, -1):
            for end in range(min(n, i + self._longest_suffix), i, -1):
                if covered[end] and text[i:end] in self._suffixes:
                    step[i], covered[i] = end, True
                    break
        if not covered[start]:
            return None
        spans: list[tuple[int, int]] = []
        i = start
        while i < n:
            spans.append((i, step[i]))
            i = step[i]
        return spans

    def _apply_merges(self, pieces: list[Piece], forbidden: set[int]) -> list[Piece]:
        """Merge adjacent pieces by rank, the leftmost first among equal pairs; pairs across a forbidden boundary stay
        apart."""

        if not self.merges or len(pieces) < 2:
            return pieces
        n = len(pieces)
        text = [piece.text for piece in pieces]
        start = [piece.start for piece in pieces]
        end = [piece.end for piece in pieces]
        right = list(range(1, n)) + [-1]
        left = [-1] + list(range(n - 1))
        stamp = [0] * n
        heap: list[tuple[int, int, int, int, int, int, str]] = []
        merges = self.merges

        def push(i: int) -> None:
            j = right[i]
            if j < 0 or end[i] in forbidden:
                return
            merge = merges.get((text[i], text[j]))
            if merge is not None:
                heapq.heappush(heap, (merge[1], start[i], i, j, stamp[i], stamp[j], merge[0]))

        for i in range(n - 1):
            push(i)
        while heap:
            _rank, _start, i, j, stamp_i, stamp_j, merged = heapq.heappop(heap)
            if stamp[i] != stamp_i or stamp[j] != stamp_j or right[i] != j:
                continue
            text[i] = merged
            end[i] = end[j]
            right[i] = right[j]
            if right[j] >= 0:
                left[right[j]] = i
            stamp[i] += 1
            stamp[j] = -1
            if left[i] >= 0:
                push(left[i])
            push(i)
        out: list[Piece] = []
        i = 0
        while i >= 0:
            out.append(Piece(text[i], start[i], end[i]))
            i = right[i]
        return out

    def _split_retired(self, pieces: list[Piece], boundary_map: list[int]) -> list[Piece]:
        skeleton_index = {offset: i for i, offset in enumerate(boundary_map)}
        out: list[Piece] = []
        stack = pieces[::-1]
        while stack:
            piece = stack.pop()
            if piece.text not in self.retired:
                out.append(piece)
                continue
            left, right = self._formed_by[piece.text]
            middle = boundary_map[skeleton_index[piece.start] + len(left)]
            stack.append(Piece(right, middle, piece.end))
            stack.append(Piece(left, piece.start, middle))
        return out

    def _piece_id(self, piece: str) -> int:
        if piece in self.vocab:
            return self.vocab[piece]
        return self.vocab.get("<unk>", 0)

    # ---- decode ----

    def decode(self, ids: list[int]) -> str:
        if not self._id_to_token:
            self._id_to_token = {i: tok for tok, i in self.vocab.items()}
        out: list[str] = []
        for i in ids:
            tok = self._id_to_token.get(i, "")
            if tok == "<unk>":
                out.append("\ufffd")
                continue
            out.append(tok)
        return "".join(out)
