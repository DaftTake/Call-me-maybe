"""Typed wrapper around the llm_sdk Small_LLM_Model."""

from __future__ import annotations

from typing import Any, List, Tuple

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, PrivateAttr


class LLMWrapper(BaseModel):
    """Wrapper around the llm_sdk Small_LLM_Model."""

    model_name: str
    _model: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the underlying SDK model."""
        self._model = Small_LLM_Model(self.model_name)

    def encode(self, text: str) -> List[int]:
        """Encode text into a flat list of token IDs."""
        encoded = self._model.encode(text)[0].tolist()
        return encoded  # type: ignore[no-any-return]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        decoded = self._model.decode(token_ids)
        return decoded  # type: ignore[no-any-return]

    def get_logits(self, input_ids: List[int]) -> List[float]:
        """Return raw logits for the next token."""
        logits = self._model.get_logits_from_input_ids(input_ids)
        return logits  # type: ignore[no-any-return]

    def prefill_logits(
        self, input_ids: List[int]
    ) -> Tuple[List[float], Any]:
        """Run a prefix and return logits plus cache state."""
        result = self._model.prefill_logits(input_ids)
        return result  # type: ignore[no-any-return]

    def advance_logits(
        self, token_id: int, past_key_values: Any
    ) -> Tuple[List[float], Any]:
        """Advance cached decoding by one token."""
        result = self._model.advance_logits(
            token_id, past_key_values
        )
        return result  # type: ignore[no-any-return]

    def next_token(self, logits: List[float]) -> int:
        """Return the token ID with the highest logit."""
        return max(enumerate(logits), key=lambda x: x[1])[0]
