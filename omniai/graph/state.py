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

    def merge(self, update: State | dict[str, Any] | None) -> Self:
        """Return a new state with ``update`` applied on top of this one.

        Avoids re-serializing the whole state per node (which made long
        message histories O(n²) over a run): existing messages are carried
        by reference and only the appended ones are validated. Non-message
        fields are applied via ``model_copy`` — node updates are trusted to
        match the declared field types.
        """
        if update is None:
            return self
        if isinstance(update, State):
            update = update.model_dump(exclude_unset=True)
        updates = dict(update)
        appended = updates.pop("messages", None)
        if appended:
            normalized = [
                m if isinstance(m, OmniMessage) else OmniMessage.model_validate(m) for m in appended
            ]
            updates["messages"] = [*self.messages, *normalized]
        return self.model_copy(update=updates)
