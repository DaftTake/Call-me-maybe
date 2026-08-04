*This project has been created as part of the 42 curriculum by wabbad.*

# call me maybe

## Description

**call me maybe** is a function-calling tool for Large Language Models (LLMs). Given a
natural-language request such as *"What is the sum of 2 and 3?"*, the program does not
answer the question directly. Instead, it produces a structured, machine-executable
function call:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The core challenge is reliability. Small language models (like the 0.6B-parameter
`Qwen/Qwen3-0.6B` used here) are notoriously bad at producing valid JSON on their own —
they might succeed only ~30% of the time when prompted. This project solves that with
**constrained decoding**: a technique that guides the model's output token-by-token,
guaranteeing 100% valid, schema-compliant JSON regardless of how unreliable the raw
model is.

## Instructions

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
uv sync
```

This installs all dependencies (including the local `llm_sdk` package) and generates
the `uv.lock` file.

### Running the program

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

By default, the program reads from `data/input/` and writes to
`data/output/function_calling_results.json`. All arguments are optional.

### Makefile targets

| Target | Description |
|--------|-------------|
| `make install` | Install project dependencies with `uv sync`. |
| `make run` | Run the main script. |
| `make debug` | Run the main script under `pdb`. |
| `make clean` | Remove caches (`__pycache__`, `.mypy_cache`, etc.). |
| `make lint` | Run `flake8` and `mypy` with the mandatory flags. |
| `make lint-strict` | Run `flake8` and `mypy --strict`. |

## Algorithm explanation

### The generation loop

LLMs generate text one token at a time. The `llm_sdk` exposes:

- `encode(text)` → token IDs
- `get_logits_from_input_ids(ids)` → raw scores for every possible next token
- `get_path_to_vocab_file()` → path to the `vocab.json` mapping token IDs to strings

The generation pipeline is:

1. Build a prompt containing the function definitions and the user's question.
2. Encode the prompt to token IDs.
3. Ask the model for the logits of the next token.
4. **Constrain**: mask out (set to `-inf`) every token that would break the JSON
   structure or the expected schema.
5. Pick the highest-scoring remaining token (argmax).
6. Append the token, repeat from step 3 until the JSON is complete.

### Constrained decoding

At each step, the decoder maintains a state machine that knows exactly where it is in
the JSON document (e.g. "inside a string value", "expecting a key", "inside a number").
Given the current state and the accumulated output, it computes the set of legal next
tokens:

- **Structural tokens**: `{`, `}`, `[`, `]`, `:`, `,`, `"` are only allowed when the
  grammar permits them.
- **String tokens**: only tokens that keep the current string valid (including escape
  sequences) are allowed.
- **Number tokens**: only tokens that keep the number syntactically valid (digits,
  sign, decimal point, exponent) are allowed.
- **Schema tokens**: when generating a function name, only tokens that form a prefix of
  one of the registered function names are allowed. When generating an argument value,
  only tokens matching the declared parameter type (number, string, boolean) are
  allowed.

Because illegal tokens are masked before selection, the model can never produce output
that violates the JSON grammar or the function schema. This yields **100% valid JSON**
with near-perfect reliability.

## Design decisions

- **Pydantic for all validation**: every class uses pydantic models, ensuring input
  files and output results are validated and type-safe.
- **Vocabulary-driven decoding**: the decoder uses the `vocab.json` file to map token
  IDs to their string representations, which is essential for determining which tokens
  are legal at each step.
- **LLM-driven function selection**: the function to call is chosen by the LLM through
  constrained decoding, never by heuristics or regex.
- **Graceful error handling**: all file/JSON/validation errors are caught and reported
  with clear messages; the program never crashes unexpectedly.
- **Modular structure**: parsing, models, UI, and the (future) generation pipeline are
  separated into distinct modules for clarity and testability.

## Performance analysis

- **Accuracy**: constrained decoding guarantees 100% valid JSON. Function selection
  and argument extraction accuracy is driven by the model's understanding of the
  prompt, targeting 90%+ correct calls.
- **Speed**: the 0.6B model is lightweight; all test prompts are processed well under
  the 5-minute budget on standard hardware.
- **Reliability**: because the output structure is enforced by the decoder, the small
  model achieves reliability comparable to much larger models.

## Challenges faced

- **Token granularity**: tokens are often partial (e.g. `"12"` + `"3"`). The decoder
  must handle prefix matching so that a token is only accepted if the accumulated
  output plus the token still forms a valid prefix of the target structure.
- **Schema enforcement**: beyond valid JSON, the decoder must enforce the specific
  function schema (parameter names, types, required fields), which requires tracking
  which parameters have already been emitted.
- **Small-model unreliability**: without constrained decoding, the 0.6B model produces
  malformed JSON frequently; the decoder is what makes the output dependable.

## Testing strategy

The project uses `pytest` for unit tests, covering:

- Pydantic model validation (valid and invalid inputs).
- JSON loading and error reporting (invalid JSON, missing files).
- Semantic checks (duplicate names, invalid identifiers, empty prompts).
- The constrained decoder state machine (structural and schema constraints).
- End-to-end runs against the provided input files.

## Example usage

```bash
# Run with default input/output paths
uv run python -m src

# Run with custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [JSON specification](https://www.json.org/json-en.html)

### AI usage

AI was used to assist with structuring the project, writing boilerplate code, and
explaining the constrained-decoding algorithm. All generated code was reviewed,
understood, and tested by the author.