"""Prompt templates: variable-checked prompts and chat message scaffolds.

- :class:`PromptTemplate` — a single string with ``{variable}`` slots,
  validated at format time; supports partial application.
- :class:`ChatPromptTemplate` — an ordered list of (role, template) pairs
  plus :class:`MessagesPlaceholder` slots for injecting message history.
- :class:`FewShotChatPromptTemplate` — renders example input/output pairs
  as alternating user/assistant turns; nestable inside a ChatPromptTemplate.
"""

from __future__ import annotations

import string
from typing import Any

from omniai.models.base import omni_to_openai
from omniai.protocol import OmniMessage


class PromptError(KeyError):
    """A template was formatted without all of its required variables."""


def _variables_of(template: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(template) if field}


class PromptTemplate:
    def __init__(self, template: str, partial_variables: dict[str, Any] | None = None):
        self.template = template
        self.partial_variables = dict(partial_variables or {})

    @property
    def input_variables(self) -> set[str]:
        return _variables_of(self.template) - set(self.partial_variables)

    def partial(self, **kwargs: Any) -> PromptTemplate:
        return PromptTemplate(self.template, {**self.partial_variables, **kwargs})

    def format(self, **kwargs: Any) -> str:
        values = {**self.partial_variables, **kwargs}
        missing = _variables_of(self.template) - set(values)
        if missing:
            raise PromptError(f"missing template variables: {sorted(missing)}")
        return self.template.format(**values)


class MessagesPlaceholder:
    """Slot in a ChatPromptTemplate filled with a list of messages."""

    def __init__(self, variable_name: str, optional: bool = False):
        self.variable_name = variable_name
        self.optional = optional


def _normalize_messages(value: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, OmniMessage):
            messages.extend(omni_to_openai([item]))
        elif isinstance(item, dict):
            messages.append(item)
        else:
            raise TypeError(f"placeholder items must be dicts or OmniMessage, got {type(item)}")
    return messages


class ChatPromptTemplate:
    """Ordered chat scaffold of role templates, placeholders, and nestables."""

    def __init__(self, items: list[Any]):
        self.items = items

    @classmethod
    def from_messages(cls, items: list[Any]) -> ChatPromptTemplate:
        """Items: ``(role, template_str)`` tuples, MessagesPlaceholder
        instances, or nested templates exposing ``format_messages()``."""
        return cls(list(items))

    @property
    def input_variables(self) -> set[str]:
        variables: set[str] = set()
        for item in self.items:
            if isinstance(item, tuple):
                variables |= _variables_of(item[1])
            elif isinstance(item, MessagesPlaceholder):
                variables.add(item.variable_name)
            elif hasattr(item, "input_variables"):
                variables |= item.input_variables
        return variables

    def partial(self, **kwargs: Any) -> ChatPromptTemplate:
        items: list[Any] = []
        for item in self.items:
            if isinstance(item, tuple):
                role, template = item
                bound = PromptTemplate(template, kwargs)
                items.append((role, bound))
            else:
                items.append(item)
        return ChatPromptTemplate(items)

    def format_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for item in self.items:
            if isinstance(item, tuple):
                role, template = item
                if isinstance(template, PromptTemplate):
                    messages.append({"role": role, "content": template.format(**kwargs)})
                    continue
                missing = _variables_of(template) - set(kwargs)
                if missing:
                    raise PromptError(f"missing template variables: {sorted(missing)}")
                messages.append({"role": role, "content": template.format(**kwargs)})
            elif isinstance(item, MessagesPlaceholder):
                if item.variable_name not in kwargs:
                    if item.optional:
                        continue
                    raise PromptError(f"missing placeholder variable: {item.variable_name!r}")
                messages.extend(_normalize_messages(kwargs[item.variable_name]))
            elif hasattr(item, "format_messages"):
                messages.extend(item.format_messages(**kwargs))
            else:
                raise TypeError(f"unsupported chat template item: {item!r}")
        return messages


class FewShotChatPromptTemplate:
    """Examples rendered as alternating user/assistant demonstration turns."""

    def __init__(
        self,
        examples: list[dict[str, str]],
        input_key: str = "input",
        output_key: str = "output",
    ):
        self.examples = examples
        self.input_key = input_key
        self.output_key = output_key

    @property
    def input_variables(self) -> set[str]:
        return set()  # examples are self-contained

    def format_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for example in self.examples:
            messages.append({"role": "user", "content": example[self.input_key]})
            messages.append({"role": "assistant", "content": example[self.output_key]})
        return messages
