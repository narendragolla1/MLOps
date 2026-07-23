# omniai.prompts

## `PromptTemplate`

```python
PromptTemplate(template: str, partial_variables: dict | None = None)
```

- `input_variables: set[str]` — `{variable}` slots minus partials.
- `format(**kwargs) -> str` — raises `PromptError` on missing variables.
- `partial(**kwargs) -> PromptTemplate` — bind some variables now.

## `ChatPromptTemplate`

```python
ChatPromptTemplate.from_messages(items)
```

`items` may contain, in order:

- `(role, template_str)` tuples — formatted with the call's kwargs;
- `MessagesPlaceholder(variable_name, optional=False)` — splices a list of message dicts or `OmniMessage`s;
- any object with `format_messages(**kwargs)` (nesting; e.g. few-shot below).

Methods: `format_messages(**kwargs) -> list[dict]`, `input_variables`, `partial(**kwargs)`. Missing variables/placeholders raise `PromptError`.

## `FewShotChatPromptTemplate`

```python
FewShotChatPromptTemplate(examples: list[dict], input_key="input", output_key="output")
```

Renders each example as a user/assistant turn pair; consumes no variables; nestable in `from_messages`.

## `PromptError`

Subclass of `KeyError` raised on any missing variable or placeholder.
