"""Typed wrapper around the llm_sdk Small_LLM_Model."""



from typing import Any, List, Tuple

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class LLMWrapper:
    """Wrapper around the llm_sdk Small_LLM_Model."""

    def __init__(self, model_name: str) -> None:
        """Initialize the underlying SDK model."""
        self.model_name = model_name
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

    def get_vocab_path(self) -> str:
        """Return the path to the model's vocab.json file."""
        path = self._model.get_path_to_vocab_file()
        return path  # type: ignore[no-any-return]

    def get_merges_path(self) -> str:
        """Return the path to the model's merges.txt file."""
        path = self._model.get_path_to_merges_file()
        return path  # type: ignore[no-any-return]
