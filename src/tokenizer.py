"""Recoded tokenizer independent of the SDK's encode/decode.

A public, stand-alone implementation of the byte-level BPE tokenizer
used by Qwen models. It reads the model's vocab.json and merges.txt
files and provides encode(text) and decode(token_ids) methods.
"""



import json
import re
from typing import Dict, List, Tuple


def _bytes_to_unicode() -> Dict[int, str]:
    """Map each byte to a unicode string, like the GPT-2 byte-level BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: Tuple[str, ...]) -> set[Tuple[str, str]]:
    """Return all adjacent bigrams of a token word."""
    return {
        (word[i], word[i + 1]) for i in range(len(word) - 1)
    }


class RecodedTokenizer:
    """A simplified byte-level BPE tokenizer built from vocab/merges files."""

    def __init__(self, vocab_path: str, merges_path: str) -> None:
        """Load vocab.json and merges.txt and build the ranking tables."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder: Dict[str, int] = {
            v: k for k, v in self.byte_encoder.items()
        }
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        with open(merges_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 2:
                    self.bpe_ranks[(parts[0], parts[1])] = i
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| """
            r"""?[^\s\w]+|\s+(?!\S)|\s+"""
        )

    def _bpe(self, token: str) -> List[str]:
        """Apply the merge rules to a single already-byte-encoded token."""
        word = tuple(token)
        if len(word) == 1:
            return [token]
        pairs = _get_pairs(word)
        while pairs:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                if (
                    i < len(word) - 1
                    and word[i] == first
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        return list(word)

    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs."""
        token_ids: List[int] = []
        for raw_token in self.pat.findall(text):
            token = "".join(
                self.byte_encoder[b] for b in raw_token.encode("utf-8")
            )
            for bpe_token in self._bpe(token):
                token_id = self.encoder.get(bpe_token)
                if token_id is None:
                    for char in bpe_token:
                        token_ids.append(self.encoder[char])
                else:
                    token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        text = "".join(self.decoder.get(i, "") for i in token_ids)
        decoded = bytes(
            self.byte_decoder[char]
            for char in text
            if char in self.byte_decoder
        )
        return decoded.decode("utf-8", errors="replace")
