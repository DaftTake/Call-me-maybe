"""Tests for the recoded tokenizer bonus."""

from __future__ import annotations

import json
from pathlib import Path

from src.tokenizer import RecodedTokenizer, _bytes_to_unicode


def _make_tokenizer(tmp_path: Path) -> RecodedTokenizer:
    """Build a tiny tokenizer from synthetic vocab/merges files."""
    p = tmp_path
    vocab = {
        "hello": 0,
        "Ġworld": 1,
        "Ġ": 2,
        "!": 3,
        "h": 4,
        "e": 5,
        "l": 6,
        "o": 7,
        "w": 8,
        "r": 9,
        "d": 10,
    }
    (p / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (p / "merges.txt").write_text(
        "#version: 0.2\nh e\nhe l\nhel l\nhell o\n",
        encoding="utf-8",
    )
    return RecodedTokenizer(str(p / "vocab.json"), str(p / "merges.txt"))


def test_bytes_to_unicode_maps_all_bytes() -> None:
    """Every byte 0-255 maps to a unique unicode string."""
    mapping = _bytes_to_unicode()
    assert len(mapping) == 256
    assert len(set(mapping.values())) == 256


def test_encode_known_tokens(tmp_path: Path) -> None:
    """Known tokens encode to their expected IDs."""
    tokenizer = _make_tokenizer(tmp_path)
    assert tokenizer.encode("hello") == [0]


def test_decode_roundtrip(tmp_path: Path) -> None:
    """Decoding the encoded text returns the original text."""
    tokenizer = _make_tokenizer(tmp_path)
    ids = tokenizer.encode("hello world!")
    assert tokenizer.decode(ids) == "hello world!"
