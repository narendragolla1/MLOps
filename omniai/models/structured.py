"""Structured output: force any ChatModel to return validated Pydantic objects.

The wrapper injects the target JSON Schema into the conversation, parses the
model's reply, and validates it. On a parse/validation failure it feeds the
error back to the model and retries — the LangChain "output fixing" loop —
up to ``max_retries`` times before raising.
"""

from __future__ import annotations

import json
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from omniai.models.base import ChatModel

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """The model never produced schema-conforming JSON."""


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from a model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


class StructuredOutput(Generic[T]):
    """Callable wrapper returning validated ``schema`` instances."""

    def __init__(self, model: ChatModel, schema: type[T], max_retries: int = 2):
        self.model = model
        self.schema = schema
        self.max_retries = max_retries

    def _instruction(self) -> dict[str, str]:
        schema_json = json.dumps(self.schema.model_json_schema(), indent=2)
        return {
            "role": "system",
            "content": (
                "Respond ONLY with a JSON object that conforms to this JSON "
                f"Schema — no prose, no code fences:\n{schema_json}"
            ),
        }

    async def invoke(self, messages: list[dict[str, Any]] | str, **kwargs: Any) -> T:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        conversation = [self._instruction(), *messages]
        last_error = "no attempts made"
        for _ in range(self.max_retries + 1):
            result = await self.model.generate(conversation, **kwargs)
            candidate = _extract_json(result.content)
            try:
                return self.schema.model_validate_json(candidate)
            except Exception as exc:
                last_error = str(exc)
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": result.content},
                    {
                        "role": "user",
                        "content": (
                            f"That response was invalid: {last_error}\n"
                            "Respond again with ONLY valid JSON matching the schema."
                        ),
                    },
                ]
        raise StructuredOutputError(
            f"no valid {self.schema.__name__} after {self.max_retries + 1} attempts: {last_error}"
        )
