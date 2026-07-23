# How to return structured output

Get validated Pydantic objects out of any `ChatModel` instead of free text.

## Basic usage

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = await model.with_structured_output(Person).invoke("Who was Ada Lovelace?")
assert isinstance(person, Person)
```

`invoke` accepts a plain string or a full message list.

## How it works

1. The target JSON Schema is injected as a system instruction ("respond ONLY with JSON conforming to …").
2. The reply is extracted (code fences and surrounding prose are stripped) and validated with `schema.model_validate_json`.
3. On failure, the validation error is appended to the conversation and the model retries — up to `max_retries` times (default 2).
4. If no attempt validates, `StructuredOutputError` is raised with the last error.

```python
from omniai.models import StructuredOutputError

try:
    person = await model.with_structured_output(Person, max_retries=3).invoke("...")
except StructuredOutputError as exc:
    ...  # all attempts failed; exc explains the last validation error
```

## Tips

- Field descriptions and constraints on your Pydantic model flow into the schema the model sees — use them to steer output.
- Keep schemas small; deeply nested unions reduce conformance on smaller models.
- For extract-a-tool-call use cases, prefer [native tool calling](tool_calling.md) — providers enforce those schemas server-side.
