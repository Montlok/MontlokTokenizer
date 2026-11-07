# -*- coding: utf-8 -*-
"""General multilingual byte-level BPE track.

A wrapper around a HuggingFace ``tokenizers`` byte-level BPE model for every script outside the traditional Mongolian
track: Chinese, English, Japanese, Cyrillic, digits, punctuation and symbols. Byte-level coverage gives an exact round
trip for any text.

``tokenizers`` is required for this track (``pip install tokenizers``).
"""

from __future__ import annotations

from typing import Iterable

# Pieces the BPE merges within: a word with the space before it (contractions stay with the word, a change from
# lower to upper case starts a new word), a single digit with the space before it, a run of punctuation or symbols,
# a run of whitespace.
PRE_TOKENIZER_PATTERN = (
    r" ?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"| ?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"| ?\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _require_tokenizers():
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without dep.
        raise ImportError(
            "the 'tokenizers' package is required for the general BPE track; "
            "install it with: pip install 'tokenizers>=0.19'"
        ) from exc
    return tokenizers


def byte_level_pre_tokenizer():
    """Split text with ``PRE_TOKENIZER_PATTERN``, then map each piece to byte-level symbols."""
    _require_tokenizers()
    from tokenizers import Regex, pre_tokenizers

    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRE_TOKENIZER_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )


class GeneralBPEModel:
    """Byte-level BPE model wrapper with deterministic local ids and offsets."""

    def __init__(self, tk) -> None:
        self._tk = tk

    # ---- construction ----

    @classmethod
    def minimal(cls) -> "GeneralBPEModel":
        """A byte-level model with the 256 byte tokens and no merges: one token per byte, full coverage."""
        _require_tokenizers()
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers

        alphabet = pre_tokenizers.ByteLevel.alphabet()
        vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
        tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
        tk.pre_tokenizer = byte_level_pre_tokenizer()
        tk.decoder = decoders.ByteLevel()
        return cls(tk)

    @classmethod
    def load(cls, path: str) -> "GeneralBPEModel":
        _require_tokenizers()
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))

    def save(self, path: str) -> None:
        self._tk.save(path)

    # ---- vocab ----

    def get_vocab(self) -> dict[str, int]:
        return self._tk.get_vocab()

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    # ---- encode / decode ----

    def encode_pieces(self, text: str) -> list[tuple[int, str, int, int]]:
        """Return ``(local_id, token_str, char_start, char_end)`` per piece."""
        enc = self._tk.encode(text)
        return [
            (local_id, token, offset[0], offset[1])
            for local_id, token, offset in zip(enc.ids, enc.tokens, enc.offsets)
        ]

    def decode(self, local_ids: list[int]) -> str:
        return self._tk.decode(local_ids, skip_special_tokens=False)


class GeneralBPETrainer:
    """Train a byte-level BPE ``GeneralBPEModel`` from raw text."""

    def __init__(
        self,
        vocab_size: int = 40000,
        min_frequency: int = 2,
        show_progress: bool = False,
    ) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.show_progress = show_progress

    def train(self, texts: Iterable[str]) -> GeneralBPEModel:
        """Train on ``texts``, each string being text the tokenizer routes to this track."""
        _require_tokenizers()
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = byte_level_pre_tokenizer()
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=self.show_progress,
        )
        tk.train_from_iterator(texts, trainer=trainer)
        return GeneralBPEModel(tk)
