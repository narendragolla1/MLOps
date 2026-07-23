"""Prebuilt tool-calling agent: the model⇄tools loop as a compiled graph.

``create_tool_agent(model, tools)`` returns a CompiledGraph implementing the
standard agent loop: the model is called with the tool schemas; when it
responds with tool calls they are executed (schema-validated, errors fed
back as tool output rather than crashing the run) and the results are
appended to the conversation; the loop repeats until the model answers
without tool calls or ``max_steps`` model turns elapse.
"""

from __future__ import annotations

import json
from typing import Any

from omniai.graph.graph import END, CompiledGraph, Graph
from omniai.graph.state import State
from omniai.graph.tools import Tool, ToolValidationError
from omniai.models.base import ChatModel, omni_to_openai
from omniai.protocol import OmniMessage, Role


class AgentState(State):
    steps: int = 0


def create_tool_agent(
    model: ChatModel,
    tools: list[Tool],
    system_prompt: str | None = None,
    max_steps: int = 8,
    **model_kwargs: Any,
) -> CompiledGraph:
    """Build the model -> tools -> model loop over the given ChatModel."""
    tool_map: dict[str, Tool] = {t.name: t for t in tools}

    async def call_model(state: AgentState) -> dict[str, Any]:
        messages = omni_to_openai(state.messages)
        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": system_prompt}, *messages]
        result = await model.generate(messages, tools=tools, **model_kwargs)
        reply = OmniMessage(
            role=Role.ASSISTANT,
            content=result.content,
            tool_calls=result.tool_calls,
            session_id=state.messages[-1].session_id if state.messages else "default",
        )
        return {"messages": [reply], "steps": state.steps + 1}

    async def run_tools(state: AgentState) -> dict[str, Any]:
        outputs: list[OmniMessage] = []
        for call in state.messages[-1].tool_calls:
            tool = tool_map.get(call.name)
            if tool is None:
                content = f"error: unknown tool {call.name!r}"
            else:
                try:
                    result = await tool.execute(call.arguments)
                    content = result if isinstance(result, str) else json.dumps(result)
                except ToolValidationError as exc:
                    # Feed schema errors back so the model can correct itself.
                    content = f"error: invalid arguments: {exc}"
                except Exception as exc:
                    content = f"error: {type(exc).__name__}: {exc}"
            outputs.append(
                OmniMessage(
                    role=Role.TOOL,
                    content=content,
                    session_id=state.messages[-1].session_id,
                    metadata={"tool_call_id": call.id, "tool_name": call.name},
                )
            )
        return {"messages": outputs}

    def route(state: AgentState) -> str:
        last = state.messages[-1]
        if last.tool_calls and state.steps < max_steps:
            return "tools"
        return END

    graph = Graph(AgentState)
    # Graph isn't generic over the state type, so mypy can't see that these
    # nodes' narrower AgentState parameter is the same one passed above.
    graph.add_node("model", call_model)  # type: ignore[arg-type]
    graph.add_node("tools", run_tools)  # type: ignore[arg-type]
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", route)  # type: ignore[arg-type]
    graph.add_edge("tools", "model")
    return graph.compile(max_iterations=2 * max_steps + 2)
