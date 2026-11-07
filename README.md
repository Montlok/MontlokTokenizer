# MontlokTokenizer

A tokenizer for traditional Mongolian.

## Installation

```bash
git clone https://github.com/Montlok/MontlokTokenizer.git
cd MontlokTokenizer
pip install -e .
```

## Usage

```python
from Tokenizer.unified.bundle import TokenizerBundle

tokenizer = TokenizerBundle.from_dir(".").tokenizer
ids = tokenizer.encode("ᠮᠣᠩᠭᠣᠯ ᠬᠡᠯᠡ")
text = tokenizer.decode(ids)
```

## License

The code is licensed under Apache-2.0. The vocabulary files (`config.json`, `general.json`, `morphbpe.json`,
`vocab.json`) are licensed under CC BY-SA 4.0 and contain words from Wiktionary (CC BY-SA 4.0) and gege-converter
(MIT).
