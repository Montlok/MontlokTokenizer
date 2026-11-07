# MontlokTokenizer

MontlokTokenizer is a tokenizer for autoregressive language models, optimized for Mongolian written in the traditional
Uyghur-derived script (Hudum Mongol bichig). The same vocabulary also covers Chinese, English, Cyrillic and every other
script.

## Features

- **Morphology-aware Mongolian segmentation.** Words are cut into stem and suffixes before BPE, using a suffix inventory
  with vowel-harmony rules, and merges stay inside these parts.
- **Dictionary words as single tokens.** 2,569 Mongolian dictionary words are whole tokens.
- **Multilingual.** Chinese, English, Cyrillic, digits, punctuation and symbols are handled by a 40,000-token byte-level
  BPE in the same id space.
- **Exact round trip.** Decoding reproduces the input text exactly, including the Mongolian free variation selectors
  (FVS1–FVS4), the Mongolian vowel separator (MVS) and the narrow no-break space (NNBSP), so Mongolian text keeps its
  GB/T 25914-2023 encoding.
- **Character offsets.** Every token carries the span of the input it covers and the part of the tokenizer that
  produced it.
- **Special tokens for model inputs.** Sequence markers and reserved tokens for image, video, audio, OCR and layout
  spans.

## Installation

```bash
git clone https://github.com/Montlok/MontlokTokenizer.git
cd MontlokTokenizer
pip install -e .
```

Requires Python 3.10 or newer and [`tokenizers`](https://github.com/huggingface/tokenizers).

## Quick Start

```python
from Tokenizer.unified.bundle import TokenizerBundle

tokenizer = TokenizerBundle.from_dir(".").tokenizer

text = "ᠮᠣᠩᠭᠣᠯ᠎ᠤᠨ ᠪᠢᠴᠢᠭ᠎ᠤᠨ"
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text

print([tokenizer.decode([i]) for i in ids])
# ['ᠮᠣᠩᠭᠣᠯ', '\u180e', 'ᠤᠨ', ' ', 'ᠪᠢᠴᠢᠭ', '\u180e', 'ᠤᠨ']
```

Each stem and the suffix after the vowel separator are separate tokens, and both words share the token of the suffix
`ᠤᠨ`.

For a language model, add the sequence markers:

```python
ids = tokenizer.encode(text, add_bos=True, add_eos=True)
```

Offsets map each token back to the input:

```python
result = tokenizer.encode_with_spans("ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ, 蒙古文")
for token in result.tokens:
    print(token.id, token.track, token.start, token.end)
```

## Vocabulary

| Part | Tokens |
|---|---:|
| Mongolian script, morphology-aware BPE | 25,280 |
| of which dictionary words | 2,569 |
| Other scripts, byte-level BPE | 40,000 |
| Special tokens | 26 |
| Total | 65,305 |

All ids fit in 16 bits. Mongolian letters, Mongolian digits and the Mongolian format controls go to the morphology-aware
BPE, with each control as its own token. All other characters, Mongolian punctuation included, go to the byte-level
BPE. A space before Mongolian text is its own token; a space before other text joins the byte-level token of the word
after it.

| Group | Special tokens |
|---|---|
| Sequence | `<pad>` 0, `<unk>` 1, `<bos>` 2, `<eos>` 3 |
| Image | `<img>` 4, `<image>` 5, `<image_start>` 6, `<image_patch>` 7, `<image_end>` 8 |
| Video | `<video>` 9, `<video_start>` 10, `<video_patch>` 11, `<video_end>` 12 |
| Document | `<bbox>` 13, `<ocr>` 14, `<ocr_start>` 15, `<ocr_end>` 16, `<doc>` 19, `<table>` 20, `<layout>` 21 |
| Audio | `<audio>` 22, `<audio_start>` 23, `<audio_patch>` 24, `<audio_end>` 25 |
| Boundaries | `▁` 17 (word), `◈` 18 (morpheme) |

## API

| Method | Returns |
|---|---|
| `encode(text, add_bos=False, add_eos=False)` | token ids |
| `decode(ids)` | text |
| `encode_with_spans(text, add_bos=False, add_eos=False)` | `input_ids` and `tokens`, each token with `id`, `token`, `track`, `start`, `end` |
| `encode_plain_text(text)` | token ids, with special-token strings in `text` encoded as ordinary text |
| `vocab` | token-to-id mapping |

`encode` reads special-token strings such as `<bos>` in the text as special tokens.

## Training a New Vocabulary

```python
from Tokenizer.generic_bpe import GeneralBPETrainer
from Tokenizer.morphbpe.trainer import MorphBPETrainer
from Tokenizer.unified.bundle import TokenizerBundle

texts = [...]  # training text
MorphBPETrainer(vocab_size=4096).train_to_file(texts, "morphbpe.json")
GeneralBPETrainer(vocab_size=40000).train(texts).save("general.json")
TokenizerBundle.from_files("morphbpe.json", general_path="general.json").save_dir("my_tokenizer")
```

`TokenizerBundle.from_dir("my_tokenizer")` then loads the new tokenizer.

## License

The code is licensed under Apache-2.0 ([LICENSE](LICENSE)). The vocabulary files `config.json`, `general.json`,
`morphbpe.json` and `vocab.json` are licensed under CC BY-SA 4.0; they include words from Wiktionary (CC BY-SA 4.0) and
from gege-converter (MIT).
