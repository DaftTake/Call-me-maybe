# Call-Me-Maybe: Zero-to-Hero Project Run Down

This document is a full walkthrough of the project from the idea level down to the implementation details, the problems encountered, the reasoning used to solve them, and the final shape of the codebase.

It is written as a development narrative, not as source code commentary. The goal is to explain how the system works, why the design looks the way it does, and what had to be fixed along the way.

## 1. What the project does

Call-Me-Maybe is a natural-language-to-function-call system.

The input is:
- a list of available function definitions
- a list of user prompts

The output is:
- a JSON file containing one structured function call per prompt

Each output object contains:
- the original prompt
- the selected function name
- the generated parameters with the correct types

The key requirement is that the output must remain valid JSON and must respect the schema of the available functions.

In practice, that means the program is not supposed to answer questions directly. It is supposed to turn human language into a machine-usable call such as:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## 2. The core idea behind the solution

The project uses constrained decoding.

That is the central technical idea. Instead of letting the model freely generate text and then trying to clean up the result afterward, the program masks illegal next tokens before choosing the next token. This means the model can only continue along paths that remain compatible with the expected JSON structure and with the schema of the selected function.

The practical benefit is important:
- invalid JSON is prevented before it is produced
- impossible parameter values are filtered early
- the UI can show live generation without risking malformed output

This approach is used in [src/constrained_decoder.py](src/constrained_decoder.py) and is the heart of the project.

## 3. High-level architecture

The current codebase is organized into a small set of focused modules:

- [src/__main__.py](src/__main__.py) orchestrates the full run
- [src/parser.py](src/parser.py) loads arguments and validates input files
- [src/models.py](src/models.py) defines the Pydantic models
- [src/llm_wrapper.py](src/llm_wrapper.py) wraps the SDK model behind a simple interface
- [src/constrained_decoder.py](src/constrained_decoder.py) performs token-by-token constrained generation
- [src/ui.py](src/ui.py) renders the live terminal interface
- [src/errors.py](src/errors.py) centralizes expected application errors

The project is intentionally modular. That matters because the most difficult parts are not all in one place:
- parsing and file handling are separate from generation
- model loading is separate from token masking
- UI rendering is separate from the actual decoding logic
- error handling is separate from the normal success path

That separation makes the system easier to test and easier to debug.

## 4. The execution flow from start to finish

The runtime flow is simple in concept, but each step has a specific job.

1. Parse CLI arguments and load the two input JSON files.
2. Validate the function registry and prompt list.
3. Prepare the output path.
4. Initialize the LLM wrapper.
5. For each prompt:
   - build a text context containing the registry and prompt
   - generate a function name token by token
   - look up the selected function definition
   - generate each parameter using the correct type-specific logic
   - append the final result to the output list
   - write the output file
6. Print a summary at the end.

The orchestration code for that flow lives in [src/__main__.py](src/__main__.py).

## 5. Input validation and parsing

The parser is responsible for making sure the program does not start from broken input.

The responsibilities are in [src/parser.py](src/parser.py):
- parse command-line arguments
- load JSON safely
- validate the function registry schema
- validate prompts
- ensure the output directory exists
- create the output file path before generation begins
- convert any expected failures into friendly error messages

### Why the parser matters

Without early validation, the generator would have to deal with many bad states later:
- missing registry files
- malformed JSON
- empty prompt lists
- invalid function names
- duplicate names
- unusable output paths

The parser prevents those problems before the model is even loaded.

### What the parser checks

It checks both syntax and semantics:

- JSON syntax errors are caught and reported with line and column information
- function definitions are validated as Pydantic models
- function names must be valid identifiers
- duplicate function names are rejected
- empty prompt lists are rejected
- empty prompt strings are rejected

That is important because a structured generation system only works if the input schema itself is trustworthy.

## 6. The data models

The models in [src/models.py](src/models.py) define the internal structure of the project.

They include:
- `ParameterType`
- `ReturnType`
- `Function`
- `FunctionRegistry`
- `Prompt`
- `FunctionCallResult`
- `FunctionCallResults`

### Why Pydantic is used

Pydantic gives the project a strong validation layer with very little boilerplate.

It helps with:
- schema enforcement
- type safety
- readable error messages
- predictable output serialization

For a project like this, that matters because the generation layer is only useful if the surrounding data shape is stable.

## 7. The LLM wrapper

The raw Hugging Face model is wrapped in [src/llm_wrapper.py](src/llm_wrapper.py).

That wrapper is there for two reasons:

1. It hides the SDK details from the rest of the application.
2. It gives the rest of the code a small, consistent interface.

The wrapper exposes:
- `encode`
- `decode`
- `get_logits`
- `prefill_logits`
- `advance_logits`
- `next_token`

### Why the wrapper matters

The generation code should not need to know how the model is loaded or how tensor shapes are handled.

The wrapper makes the rest of the system easier to read because the decoder can focus on token constraints instead of on low-level model housekeeping.

### Performance improvement in the wrapper

One of the important later improvements was adding cache-aware decoding support. Instead of recomputing logits from the full prefix every time, the code now supports incremental advancement using cached model state.

That changed the runtime from "technically working but slow" to "comfortably under the time limit." It is one of the most meaningful performance improvements in the project.

## 8. Constrained decoding logic

The constrained decoder in [src/constrained_decoder.py](src/constrained_decoder.py) is the technical center of the project.

It handles several output shapes:
- function names
- string parameters
- float parameters
- integer parameters
- boolean parameters

### Function name generation

The decoder precomputes token sequences for all valid function names and builds a prefix map.

At generation time:
- it looks at the current partial name
- it checks which next tokens still keep the prefix valid
- it masks every other token
- it chooses the best remaining token

This guarantees that the name stays within the set of allowed function names.

### String parameter generation

String parameters are treated more freely, but the decoder still waits for a closing quote and keeps the generation inside the expected quoted span.

### Float and integer parameter generation

Numeric parameters are tricky because tokens can be partial:
- a token may contain only `1`
- or `12`
- or `3.14`
- or `-`

The decoder therefore validates the growing numeric prefix at each step.

That is why separate helpers exist for floats and integers. They do not simply accept any number-shaped output. They check whether the current prefix is still syntactically valid.

### Boolean parameter generation

Booleans are restricted to the valid prefixes of `true` and `false`.

That means the decoder is not guessing semantically. It is constraining the token stream to legal continuations only.

## 9. The user interface layer

The live terminal UI is in [src/ui.py](src/ui.py).

It exists for visibility:
- to show the loaded registry
- to show the prompts being processed
- to show the evolving JSON output in real time
- to show summary information at the end

### Why the UI matters

Without the live UI, the project would still work, but it would be much harder to understand what the model is doing while it runs.

The UI is useful for:
- debugging token-by-token behavior
- checking whether the function name is drifting
- seeing parameter generation evolve live
- making the generation process easier to demonstrate

That makes it a quality-of-life feature, but also a practical development tool.

## 10. The problems that came up during development

This project is not just about having a model. The difficult part is making the model obey a strict format.

### Problem 1: The model can generate invalid output

Raw LLM output is not trustworthy enough for structured JSON.

#### Reasoning

If the model is left unconstrained, it can:
- produce prose instead of JSON
- invent a function name
- insert invalid punctuation
- stop in the middle of a field
- create malformed numeric values

That would fail the assignment requirements immediately.

#### Solution

Use constrained decoding.

The decoder masks invalid tokens before selection, so the model never gets the chance to choose illegal continuations.

### Problem 2: Numeric decoding can become slow or look stuck

The square-root and numeric prompts were especially expensive because the decoder was repeatedly running full-prefix scoring and token masking.

#### Reasoning

Numbers are deceptively hard because the decoder must still consider token granularity:
- a single number can be split into multiple tokens
- partial prefixes must remain legal
- closing delimiters must not appear too early

The first version was correct, but it was too slow.

#### Solution

The decoder was optimized to reuse cached model state instead of recomputing the whole prefix on every step.

That turned the previously long square-root case into a much faster run and brought the full test set under the time budget.

### Problem 3: Impossible decode states needed explicit handling

Even with masking, there are states where no valid token remains.

#### Reasoning

If every token is masked out:
- the loop has no safe continuation
- the model cannot recover on its own
- silent fallback would hide a real logic bug

That needs to be treated as an application error, not as a normal generation step.

#### Solution

Add explicit decoding errors in [src/errors.py](src/errors.py).

The decoder now raises a controlled `DecodingError` if the current state cannot continue safely.

### Problem 4: The selected function may not exist in the registry

The generation process can produce a function name that does not map back to a real function definition.

#### Reasoning

If that happens and the program continues anyway:
- parameter generation has no schema to follow
- output becomes inconsistent
- the final JSON may be structurally valid but logically meaningless

That should not be treated as a normal success.

#### Solution

The main pipeline now checks the function lookup result explicitly and raises `UnknownFunctionError` if the selected name is not present in the registry.

### Problem 5: Error handling had to be graceful, not abrupt

The project should not crash with a raw traceback for expected problems.

#### Reasoning

The assignment expects friendly failures for:
- malformed JSON
- missing files
- invalid schema input
- unsupported model behavior

That means the application should fail cleanly, with a readable error message.

#### Solution

Expected failures are now routed through [src/errors.py](src/errors.py) and converted into clear console output from [src/__main__.py](src/__main__.py).

## 11. The debugging and validation process

The work was not done by guessing. The main validation pattern was:

1. reproduce the problem with a concrete prompt or command
2. inspect the local code path that actually controls the behavior
3. identify the smallest plausible fault surface
4. make the smallest change that directly addresses that fault
5. run focused tests first
6. only then re-run the full project command

### Example: the square-root slowdown

The prompt about square root of 16 looked like a hang.

The debugging process was:
- reproduce the exact prompt
- observe that the program was still inside generation
- inspect the numeric decoder, not the parser
- confirm that numeric generation was the hot path
- reduce repeated recomputation by caching model advancement
- verify the change with a focused decoder test
- re-run the full pipeline and measure wall-clock time

That is the kind of reasoning that turned a working but slow prototype into a usable project.

### Example: impossible decode and unknown function

The next issue was making failure explicit.

The reasoning was:
- if there is no valid next token, the decoder should stop with a controlled error
- if the model names a function that does not exist, the main pipeline should not continue into parameter generation

Those two checks prevent silent corruption of the output state.

## 12. The tools used

The following tools and libraries were important to the project:

- `uv` for dependency management and execution
- `pytest` for focused regression testing
- `pydantic` for validation and schema modeling
- `rich` for the live terminal UI
- Hugging Face `transformers` for model loading and inference
- the local `llm_sdk` package for model abstraction

### Why these tools fit the problem

The project is a structured generation task, so it benefits from:
- strict data validation
- reproducible execution
- controlled model inference
- visible debugging output

These tools match those needs well.

## 13. The final code path

At the end of the work, the code path is effectively:

1. Parse input files and validate them.
2. Initialize the model wrapper.
3. Show the registry and prompts.
4. For each prompt, generate a function name with token masking.
5. Resolve the selected function against the registry.
6. Generate each parameter using the declared type.
7. Detect impossible decode states and unknown functions early.
8. Write the final JSON output.
9. Print a summary.

That gives the project a clear flow from input validation to final output.

## 14. What the final version is good at

The finished project is strong in a few specific ways:

- it produces valid structured output rather than free-form text
- it uses type-aware generation for multiple parameter types
- it validates input files carefully
- it fails gracefully for expected errors
- it shows live progress while generating
- it runs fast enough to meet the time requirement

## 15. What is still intentionally limited

The project is not a universal function-calling compiler.

It is intentionally scoped to the assignment’s problem shape:
- flat function schemas
- a known function registry
- simple scalar parameter types
- a single generation loop per prompt

It does not try to solve every possible structured-output problem on earth.

That is a good thing. The implementation is simpler, easier to reason about, and closer to the assignment’s actual requirements.

## 16. Short version of the whole journey

The project started as a constrained decoding function-calling pipeline and ended as a more polished version with:
- better validation
- clearer error handling
- faster generation
- live terminal visualization
- tests for the important failure modes

The biggest technical lesson was that correctness alone was not enough. The system also had to be debuggable, fast enough to finish, and explicit about failures.

That is what turned the project from a basic prototype into a reliable final submission.