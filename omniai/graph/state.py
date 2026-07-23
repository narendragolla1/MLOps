"""Generic graph state built on Pydantic.

Subclass :class:`State` to declare the fields your workflow carries between
nodes. Nodes return partial updates (dicts or new State instances) that are
merged immutably, so every step yields a fresh, validated snapshot.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from omniai.protocol import OmniMessage


class State(BaseModel):
    """Base state flowing through graph nodes.

    ``messages`` accumulates conversation history (list-append semantics on
    merge); every other field is replaced by the incoming update.
    """

    model_config = ConfigDict(extra="allow")

    messages: list[OmniMessage] = Field(default_factory=list)

    def merge(self, update: "State | dict[str, Any] | None") -> Self:
        """Return a new state with ``update`` applied on top of this one."""
        if update is None:
            return self
        if isinstance(update, State):
            update = update.model_dump(exclude_unset=True)
        data = self.model_dump()
        for key, value in update.items():
            if key == "messages" and isinstance(value, list):
                data["messages"] = data["messages"] + [
                    m.model_dump() if isinstance(m, OmniMessage) else m for m in value
                ]
            else:
                data[key] = value
        return type(self).model_validate(data)
