"""Production application factory.

Builds the fully wired gateway from environment settings — engine (attached
to the external serving container), DB-backed interaction buffer, continuous
learner, guardrails, and a default chat graph. This is the container
entrypoint:

    uvicorn omniai.app:create_app --factory --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from fastapi import FastAPI

from omniai.engine import ModelEngine
from omniai.gateway import GatewayRouter
from omniai.graph import END, START, Graph, State
from omniai.guardrails import PromptGuard
from omniai.memory import ContinuousLearner, InteractionBuffer, LoRATrainer
from omniai.settings import OmniSettings, get_settings


def build_chat_graph(engine: ModelEngine) -> Graph:
    """Default single-node chat graph; replace with your own for agents."""
    graph = Graph(State)

    async def chat(state: State) -> dict:
        reply = await engine.chat_text([m.to_openai() for m in state.messages])
        return {"messages": [state.messages[-1].reply(reply)]}

    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph


def create_app(settings: OmniSettings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_security()

    engine = ModelEngine.create(
        {
            "model": "served-model",
            "managed": settings.engine_managed,
            "external_base_url": settings.engine_base_url,
            "request_timeout_s": settings.request_timeout_s,
            "retries": settings.engine_retries,
            "breaker_failure_threshold": settings.breaker_failure_threshold,
            "breaker_reset_s": settings.breaker_reset_s,
        }
    )

    buffer = InteractionBuffer(settings.database_url, threshold=1000)
    learner = ContinuousLearner(buffer, LoRATrainer("served-model"), engine=engine)
    buffer.on_threshold = learner.trigger

    router = GatewayRouter(
        handler=build_chat_graph(engine).compile().as_handler(),
        interceptors=[PromptGuard()],
        observers=[buffer],
        settings=settings,
        engine=engine,
        buffer=buffer,
        shutdown_hooks=[engine.stop, buffer.aclose],
    )
    if router.metrics is not None:
        learner.on_report = lambda report: router.metrics.learning_cycles.labels(
            report["status"]
        ).inc()
    return router.app
