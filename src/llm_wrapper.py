"""Wrapper for the Small LLM Model from the llm_sdk.

This module provides a convenient, typed abstraction over the raw SDK,
handling tensor flattening and device details transparently.
"""

from typing import Any, List, Tuple
from pydantic import BaseModel, PrivateAttr
from llm_sdk import Small_LLM_Model


class LLMWrapper(BaseModel):
    """A convenient wrapper around the llm_sdk Small_LLM_Model."""

    model_name: str
    _model: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the underlying SDK model after Pydantic validation."""
        self._model = Small_LLM_Model(self.model_name)

    def encode(self, text: str) -> List[int]:
        """Encode text into a flat list of token IDs.

        Args:
            text: The string to tokenize.

        Returns:
            A list of token IDs.
        """
        # The SDK returns a 2D tensor, we flatten it for ease of use.
        tensor = self._model.encode(text)
        return tensor[0].tolist()

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into text.

        Args:
            token_ids: The token IDs to decode.

        Returns:
            The decoded string.
        """
        return self._model.decode(token_ids)

    def get_logits(self, input_ids: List[int]) -> List[float]:
        """Get the raw logits for the next token given a sequence of IDs.

        Args:
            input_ids: The context sequence.

        Returns:
            A list of floats representing the logits for each vocabulary token.
        """
        return self._model.get_logits_from_input_ids(input_ids)

    def prefill_logits(self, input_ids: List[int]) -> Tuple[List[float], Any]:
        """Run a prefix once and return logits plus cache state."""
        return self._model.prefill_logits(input_ids)

    def advance_logits(self, token_id: int, past_key_values: Any) -> Tuple[List[float], Any]:
        """Advance cached decoding by one token."""
        return self._model.advance_logits(token_id, past_key_values)

    def next_token(self, logits: List[float]) -> int:
        """Find the token ID with the highest logit score.

        Args:
            logits: The list of token logits.

        Returns:
            The index (token ID) of the maximum logit.
        """
        return max(enumerate(logits), key=lambda x: x[1])[0]
