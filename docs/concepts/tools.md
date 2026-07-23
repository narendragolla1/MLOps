# Tools

A tool is a typed Python function the model is allowed to invoke. OmniAI's design goal: **the type signature is the contract** — schema generation, documentation, and validation all derive from it, so they can't drift apart.

## From signature to schema

`@tool` builds a Pydantic model from the function's parameters (`create_model` over the type hints), which yields:

- `tool.json_schema` — JSON Schema of the arguments (defaults → optional fields, hints → types, docstring → description);
- `tool.to_openai()` — the OpenAI function-calling definition providers consume (Anthropic's `input_schema` is derived from the same source).

## Validated execution

`await tool.execute(arguments)` never calls your function with unvetted input:

1. JSON strings are parsed (malformed JSON → `ToolValidationError`).
2. Arguments are validated **and coerced** by the generated Pydantic model (`"2"` → `2` for an `int` param; missing required fields or uncoercible values → `ToolValidationError`).
3. The function runs; async functions are awaited transparently.

The distinction matters in agent loops: a `ToolValidationError` means *the model* produced bad arguments — recoverable by telling it so — while an exception from your function body means the tool itself failed. The [agent executor](agents.md) reports the two differently.

## Design guidance

- Keep parameters primitive (str/int/bool/lists) — small models conform to flat schemas far better than to nested unions.
- Docstrings are model-facing prompt text, not comments: say *when* to use the tool, not just what it does.
- Tools that touch the outside world should enforce their own authorization — the model's choice to call a tool is never a security boundary (see [security](security.md)).
