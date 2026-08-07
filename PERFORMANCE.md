# Why Is This Program So Fast?

This program turns natural-language questions into structured function calls using a
small AI model. It processes 11 questions in about 37 seconds on a normal laptop CPU.
That might not sound fast, but for an AI model it is *very* fast — and it's fast
because of a few clever design choices.

This document explains those choices, but first it explains the basics so the
technical terms make sense.

---

## Part 1: How an LLM works (the 30-second version)

An **LLM** (Large Language Model) is a program that predicts the next word. It doesn't
"think" — it just guesses, one word at a time, which word is most likely to come next.

### Tokens

An LLM doesn't read words directly. It breaks text into small pieces called **tokens**.
A token is usually a word or part of a word. For example:

```
"hello world"  ->  ["hello", " world"]
```

Each token has a number (its **token ID**). The model works with these numbers, not
with the actual letters.

### The forward pass (the expensive part)

When you give the model a list of token IDs, it does a big mathematical calculation
called a **forward pass**. This is the slow, heavy step — it's where the model "thinks".

The result of a forward pass is a list of scores called **logits**. There is one score
for every possible next token in the model's vocabulary (about 150,000 of them). A
higher score means the model thinks that token is more likely to come next.

### Generating text, step by step

To generate text, the model repeats this loop:

1. Do a forward pass on everything so far.
2. Look at the logits to see which token is most likely next.
3. Pick that token.
4. Add it to the text.
5. Go back to step 1.

So generating a sentence means doing many forward passes, one per token.

---

## Part 2: The technical terms, explained simply

| Term | What it means |
|------|---------------|
| **Token** | A small piece of text (a word or part of a word) that the model reads. |
| **Token ID** | The number that represents a token. |
| **Forward pass** | The slow mathematical calculation the model does to predict the next token. |
| **Logits** | The list of scores the model produces, one per possible next token. |
| **Vocabulary** | All the tokens the model knows about (~150,000 of them). |
| **Tokenizer** | The tool that converts text to token IDs and back. |
| **KV cache** | A saved copy of the model's intermediate calculations, so it doesn't have to redo them. |
| **Argmax** | Just "pick the biggest number". |
| **Constrained decoding** | Forcing the model to only pick tokens that keep the output valid. |
| **Masking** | Setting a token's score to "impossible" so the model can't pick it. |

---

## Part 3: The optimizations

### 1. The KV cache (the biggest win)

**The problem:** In the naive loop above, every step re-does the forward pass on the
*entire* text so far. If the text is 50 tokens long, step 1 processes 1 token, step 2
processes 2 tokens, step 3 processes 3 tokens... up to 50. That's a lot of repeated
work.

**The fix:** The model's forward pass works in layers. Each layer produces something
called **key** and **value** states. The clever part: when you add one new token, the
old tokens' key/value states don't change. So instead of recomputing them, we **save**
them and reuse them.

This saved data is the **KV cache**. The code does this:

```python
# Run the whole context once, and keep the cache.
def _prefill(model, context):
    return model.prefill_logits(model.encode(context))

# Each new step only processes the ONE new token, using the saved cache.
logits, past_key_values = model.advance_logits(next_token_id, past_key_values)
```

Now each step is a tiny calculation on one token instead of a big calculation on all
of them. This is the single biggest speedup.

---

### 2. Constraint checking with token IDs (no string work)

**The problem:** To make sure the model only outputs valid function names, we need to
check which tokens are allowed at each step. The naive way is to convert every one of
the ~150,000 candidate tokens to text and check it. That's 150,000 conversions per
step — very slow.

**The fix:** We work with token **IDs** (numbers) instead of text. Before generating,
we convert each valid function name to its token-ID sequence once:

```python
prefix_allowed: dict[tuple, set[int]] = {}
for seq in (model.encode(n) for n in valid_names):
    for pos in range(len(seq)):
        prefix_allowed.setdefault(tuple(seq[:pos]), set()).add(seq[pos])
```

This builds a map: "if I've generated these token IDs so far, which token IDs are
allowed next?" Then at each step, finding the allowed tokens is just a dictionary
lookup:

```python
allowed_next = prefix_allowed.get(tuple(generated_ids), set())
```

No text conversion, no string parsing — just a fast lookup. The same trick is used
for booleans (`"true"` / `"false"`).

---

### 3. Token-string cache (don't decode the same token twice)

**The problem:** Sometimes we *do* need to know what a token's text is (for example,
to check if a number is valid). Converting a token ID to text calls the tokenizer,
which is slow.

**The fix:** We remember the answer. The first time we decode a token, we store it in
a dictionary. Next time we need it, we just look it up:

```python
_TOKEN_STR_CACHE: dict[tuple[int, int], str] = {}

def _token_str(model, token_id):
    key = (id(model), token_id)
    if key not in _TOKEN_STR_CACHE:
        _TOKEN_STR_CACHE[key] = model.decode([token_id])
    return _TOKEN_STR_CACHE[key]
```

Each token is decoded at most once for the whole run, instead of once per step.

---

### 4. Precomputed candidate sets for numbers

**The problem:** For a number parameter, we need to check that the output stays a
valid number. Checking all ~150,000 tokens every step is wasteful.

**The fix:** We scan the vocabulary **once** and keep only the token IDs that could
possibly be part of a number (digits, signs, decimal points) or that are delimiters
(`,` and `}`):

```python
def _digit_token_ids(model, logits, charset):
    numeric_ids, delimiter_ids = set(), set()
    for tid in range(len(logits)):
        tstr = _token_str(model, tid)
        if tstr in (",", "}"):
            delimiter_ids.add(tid)
        elif tstr and all(c in charset for c in tstr):
            numeric_ids.add(tid)
    return numeric_ids, delimiter_ids
```

Now each step only carefully checks this small set (a few dozen tokens) instead of the
whole vocabulary. Everything else is masked in one cheap loop.

---

### 5. Greedy decoding (argmax, no sampling)

**The problem:** Some models use fancy sampling to pick the next token, which can
involve randomness and multiple candidate paths (beam search). That's slow.

**The fix:** We always pick the single highest-scoring token:

```python
def next_token(self, logits):
    return max(enumerate(logits), key=lambda x: x[1])[0]
```

This is called **argmax** — just "pick the biggest number". One forward pass per
token, no branching, no randomness.

---

### 6. Constrained decoding produces short, exact output

**The problem:** A normal model might ramble, add extra words, or produce broken
output that needs to be retried.

**The fix:** Because the model can only pick tokens that keep the JSON valid, it:

- never rambles or adds extra text,
- never produces broken output that needs a retry,
- stops as soon as the structure is complete (closing quote, `,`, or `}`).

A typical function call is only ~10–20 tokens. No wasted generation, no re-generation
loop.

---

### 7. Throttled UI updates

**The problem:** The terminal display (using the `rich` library) is updated after
every token. Rendering the terminal is surprisingly slow.

**The fix:** We only update the display every 4 tokens:

```python
if len(generated_ids) % 4 == 0:
    live_update_function_call(live, result)
```

The display still looks smooth, but we don't waste time redrawing it constantly.

---

### 8. Small model

`Qwen/Qwen3-0.6B` has only about 600 million parameters. That's small for an LLM.
Each forward pass is cheap compared to models with billions of parameters, and it
fits easily in a laptop's memory. The constrained decoding is what makes such a small
model reliable enough to use.

---

## Summary Table

| Technique | Where in code | What it does |
|-----------|---------------|--------------|
| KV cache | `_prefill`, `advance_logits` | Reuses saved calculations; each step is tiny instead of redoing everything |
| Token-ID prefix maps | `decode_function_name`, `decode_parameter_boolean` | Finds allowed tokens with a fast lookup, not slow text conversion |
| Token-string cache | `_token_str` | Decodes each token to text at most once |
| Precomputed number candidates | `_digit_token_ids` | Checks only a few dozen tokens per step, not ~150k |
| Greedy argmax | `next_token` | Picks the best token with no search or randomness |
| Exact short output | whole decoder | No rambling, no retries |
| Throttled UI | `% 4` updates | Less terminal redrawing |
| Small 0.6B model | `llm_sdk` | Cheap forward passes |

The two biggest reasons the program is fast are the **KV cache** (reusing saved
calculations) and **token-ID-level constraint checking** (avoiding slow text
conversion). The rest are smaller but still important optimizations on top.