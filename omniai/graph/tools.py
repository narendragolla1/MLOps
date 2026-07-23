"""@tool decorator: type hints -> JSON Schema -> constrained LLM output.

Decorating a function produces a :class:`Tool` that carries an
OpenAI-function-calling-compatible JSON schema derived from the signature.
``Tool.execute`` validates LLM-produced JSON against that schema (via a
generated Pydantic model) before touching the underlying function.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable, get_type_hints

from pydantic import BaseModel, create_model


class ToolValidationError(Exception):
    """LLM-produced arguments failed schema validation."""


class Tool:
    """A callable wrapped with an auto-generated argument schema."""

    def __init__(self, fn: Callable[..., Any], name: str | None = None, description: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or inspect.getdoc(fn) or ""
        self.args_model = self._build_args_model()

    def _build_args_model(self) -> type[BaseModel]:
        hints = get_type_hints(self.fn)
        hints.pop("return", None)
        fields: dict[str, Any] = {}
        for param_name, param in inspect.signature(self.fn).parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            annotation = hints.get(param_name, Any)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param_name] = (annotation, default)
        return create_model(f"{self.name}_args", **fields)

    @property
    def json_schema(self) -> dict[str, Any]:
        """JSON Schema of the tool's parameters."""
        return self.args_model.model_json_schema()

    def to_openai(self) -> dict[str, Any]:
        """OpenAI function-calling tool definition."""
        schema = self.json_schema
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def validate_args(self, arguments: dict[str, Any] | str) -> dict[str, Any]:
        """Validate raw (possibly JSON-string) arguments against the schema."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ToolValidationError(f"Arguments are not valid JSON: {exc}") from exc
        try:
            model = self.args_model.model_validate(arguments)
        except Exception as exc:
            raise ToolValidationError(str(exc)) from exc
        return model.model_dump()

    async def execute(self, arguments: dict[str, Any] | str) -> Any:
        """Validate then run the tool; awaits async functions transparently."""
        kwargs = self.validate_args(arguments)
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator: ``@tool`` or ``@tool(name=..., description=...)``."""
    if fn is not None:
        return Tool(fn)

    def wrapper(f: Callable[..., Any]) -> Tool:
        return Tool(f, name=name, description=description)

    return wrapper


def render_tool_prompt(tools: list[Tool]) -> str:
    """System-prompt fragment forcing schema-conforming JSON tool output."""
    specs = json.dumps([t.to_openai() for t in tools], indent=2)
    return (
        "You have access to the following tools. To call one, respond with "
        'ONLY a JSON object of the form {"tool": "<name>", "arguments": {...}} '
        "where arguments conform to the tool's JSON Schema.\n\n"
        f"Available tools:\n{specs}"
    )
