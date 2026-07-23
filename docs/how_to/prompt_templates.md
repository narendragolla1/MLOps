# How to use prompt templates

## String templates

```python
from omniai.prompts import PromptTemplate

template = PromptTemplate("Translate {text} into {language}.")
template.input_variables                      # {"text", "language"}
template.format(text="hi", language="French") # "Translate hi into French."
```

Formatting with a variable missing raises `PromptError` naming the missing variables — templates fail loudly, not with half-filled prompts.

## Partial application

```python
german = template.partial(language="German")
german.format(text="hello")     # only {text} remains required
```

## Chat templates

```python
from omniai.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {profession}."),
    MessagesPlaceholder("history", optional=True),
    ("user", "{question}"),
])

messages = prompt.format_messages(
    profession="pirate",
    history=[{"role": "user", "content": "ahoy"}, {"role": "assistant", "content": "arr"}],
    question="where's the treasure?",
)
await model.generate(messages)
```

`MessagesPlaceholder` splices in a list of message dicts **or** [`OmniMessage`](../concepts/messages.md) objects (converted automatically). Mark it `optional=True` to allow empty history; otherwise a missing variable raises `PromptError`.

## Composing templates

Any object with `format_messages()` can be nested as an item — that's how [few-shot examples](few_shot_examples.md) plug in.
