"""Multi-tenancy: many agents, one base model, per-request LoRA routing.

Each business unit is an :class:`AgentProfile` — a LoRA adapter (behavior),
a system prompt, and an optional retriever (facts). One engine keeps the base
model in memory once; the profile's adapter is selected **per request** via
the OpenAI ``model`` field, which vLLM/SGLang resolve to a loaded LoRA in
milliseconds. Dozens of distinct agents share one GPU.

Wire-up::

    registry = AgentRegistry(default="assistant")
    registry.register(AgentProfile(
        name="sales",
        lora="sales-lora-v3",
        system_prompt="You are an enthusiastic sales assistant. Use bullet points.",
        retriever=Retriever(product_store),
    ))
    router = GatewayRouter(handler=TenantHandler(registry, engine), ...)

Clients pick their agent with ``{"metadata": {"agent": "sales"}}``; the
handler records the agent and retrieved context on the reply's metadata, so
the telemetry buffer captures the full curation tuple (system prompt, RAG
context, query, response, feedback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from omniai.protocol import OmniMessage
from omniai.rag.store import Retriever

logger = logging.getLogger("omniai.tenancy")


@dataclass
class AgentProfile:
    """One tenant: behavior in the LoRA, facts in the retriever."""

    name: str
    lora: str | None = None  # None = serve the base model
    system_prompt: str = ""
    retriever: Retriever | None = None
    chat_kwargs: dict[str, Any] = field(default_factory=dict)  # temperature etc.

    def build_messages(self, query: str) -> tuple[list[dict[str, str]], str]:
        """(chat messages, rag context used) for one request."""
        context = self.retriever.render_context(query) if self.retriever else ""
        system = "\n\n".join(part for part in (self.system_prompt, context) if part)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": query})
        return messages, context


class AgentRegistry:
    """Named profiles with a default fallback for unrouted traffic."""

    def __init__(self, default: str | None = None):
        self.profiles: dict[str, AgentProfile] = {}
        self.default = default

    def register(self, profile: AgentProfile) -> AgentProfile:
        if profile.name in self.profiles:
            raise ValueError(f"Agent {profile.name!r} already registered")
        self.profiles[profile.name] = profile
        if self.default is None:
            self.default = profile.name
        return profile

    def resolve(self, name: str | None) -> AgentProfile:
        if name is not None and name in self.profiles:
            return self.profiles[name]
        if name is not None:
            logger.warning("unknown agent %r, falling back to default", name)
        if self.default is None or self.default not in self.profiles:
            raise KeyError("No agents registered")
        return self.profiles[self.default]


class TenantHandler:
    """GatewayRouter handler routing each message to its agent's LoRA.

    The reply's metadata carries the resolved agent and the RAG context that
    was shown to the model — exactly what the curation pipeline needs to
    judge the interaction later.
    """

    def __init__(self, registry: AgentRegistry, engine):
        self.registry = registry
        self.engine = engine

    async def __call__(self, message: OmniMessage) -> OmniMessage:
        profile = self.registry.resolve(message.metadata.get("agent"))
        messages, context = profile.build_messages(message.content)
        kwargs = dict(profile.chat_kwargs)
        if profile.lora is not None:
            kwargs["model"] = profile.lora  # per-request LoRA routing
        text = await self.engine.chat_text(messages, **kwargs)
        reply = message.reply(text)
        reply.metadata.update(
            {
                "agent": profile.name,
                "lora": profile.lora,
                "rag_context": context,
            }
        )
        return reply
