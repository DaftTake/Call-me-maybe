"""Quick smoke test for the llm_sdk model."""

from __future__ import annotations

import sys
import traceback

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

MODEL = "Qwen/Qwen3-0.6B"

try:
    print(f"Initializing model: {MODEL}")
    m = Small_LLM_Model(MODEL)
    print("Model initialized successfully.")
    ids = m.encode("hello world")
    print("Encoded length:", ids.shape if hasattr(ids, "shape") else len(ids))
    print("Decode test:", m.decode(ids[0] if hasattr(ids, "shape") else ids))
except Exception:
    print("Exception during model init:")
    traceback.print_exc()
    sys.exit(2)

print("Done.")
