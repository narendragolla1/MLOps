# How to use few-shot examples

Showing the model worked examples is often more effective than describing the task. `FewShotChatPromptTemplate` renders input/output pairs as alternating user/assistant turns and nests inside any `ChatPromptTemplate`.

```python
from omniai.prompts import ChatPromptTemplate, FewShotChatPromptTemplate

few_shot = FewShotChatPromptTemplate(
    examples=[
        {"input": "2+2", "output": "4"},
        {"input": "3+3", "output": "6"},
    ]
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer with just the number."),
    few_shot,
    ("user", "{question}"),
])

prompt.format_messages(question="5+5")
# system, user("2+2"), assistant("4"), user("3+3"), assistant("6"), user("5+5")
```

Custom example keys work via `input_key=` / `output_key=`. Examples are self-contained — they consume no template variables, so `prompt.input_variables` stays `{"question"}`.

**When to use it:** steering output format, teaching label taxonomies for classification, and demonstrating tool-use patterns to smaller models. Keep examples short and representative — they're paid for on every request (self-hosted SGLang serves the shared prefix from cache; see [serving engines](../concepts/serving_engines.md)).
