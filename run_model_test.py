import traceback
import sys
from llm_sdk import Small_LLM_Model

MODEL = "Qwen/Qwen3-0.6B"

try:
    print(f"Initializing model: {MODEL}")
    m = Small_LLM_Model(MODEL)
    print("Model initialized successfully.")
    # Try a small encode/decode to exercise tokenizer
    ids = m.encode("hello world")
    print("Encoded length:", ids.shape if hasattr(ids, 'shape') else len(ids))
    print("Decode test:", m.decode(ids[0] if hasattr(ids, 'shape') else ids))
except Exception as e:
    print("Exception during model init:")
    traceback.print_exc()
    sys.exit(2)

print("Done.")
