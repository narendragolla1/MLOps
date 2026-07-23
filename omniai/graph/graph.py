"""LangGraph-style graph builder and executor.

Nodes are plain Python callables (sync or async) taking the current State and
returning a partial update. Edges are static or conditional (any callable of
the state — lambdas work). Cycles are allowed and bounded by
``max_iterations``, making agent loops (think -> act -> observe) first-class.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from omniai.graph.state import State
from omniai.telemetry import traced_span

START = "__start__"
END = "__end__"

NodeFn = Callable[[State], State | dict[str, Any] | None | Awaitable[State | dict[str, Any] | None]]
RouterFn = Callable[[State], str]


class GraphError(Exception):
    pass


class Graph:
    """Mutable builder; call :meth:`compile` to get an executable graph."""

    def __init__(self, state_type: type[State] = State):
        self.state_type = state_type
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, tuple[RouterFn, dict[str, str] | None]] = {}
        self.entry_point: str | None = None

    def add_node(self, name: str, fn: NodeFn | None = None) -> Any:
        """Register a node; usable directly or as ``@graph.add_node("name")``."""
        if fn is None:

            def decorator(f: NodeFn) -> NodeFn:
                self.add_node(name, f)
                return f

            return decorator
        if name in (START, END):
            raise GraphError(f"{name!r} is reserved")
        if name in self.nodes:
            raise GraphError(f"Node {name!r} already exists")
        self.nodes[name] = fn
        return fn

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def add_edge(self, source: str, target: str) -> None:
        if source == START:
            self.set_entry_point(target)
            return
        if source in self.conditional_edges:
            raise GraphError(f"Node {source!r} already has a conditional edge")
        self.edges[source] = target

    def add_conditional_edges(
        self,
        source: str,
        router: RouterFn,
        path_map: dict[str, str] | None = None,
    ) -> None:
        """Route from ``source`` via ``router(state)``; lambdas are fine.

        The router returns either a node name (or END), or — when ``path_map``
        is given — a key that the map translates to a node name.
        """
        if source in self.edges:
            raise GraphError(f"Node {source!r} already has a static edge")
        self.conditional_edges[source] = (router, path_map)

    def compile(self, max_iterations: int = 25) -> CompiledGraph:
        if self.entry_point is None:
            raise GraphError("No entry point set (use set_entry_point or add_edge(START, ...))")
        referenced = {self.entry_point, *self.edges.values()}
        for _, path_map in self.conditional_edges.values():
            if path_map:
                referenced.update(path_map.values())
        unknown = {r for r in referenced if r != END and r not in self.nodes}
        if unknown:
            raise GraphError(f"Edges reference unknown nodes: {sorted(unknown)}")
        return CompiledGraph(self, max_iterations=max_iterations)


class CompiledGraph:
    """Immutable, executable view of a Graph."""

    def __init__(self, graph: Graph, max_iterations: int = 25):
        self.graph = graph
        self.max_iterations = max_iterations

    async def _run_node(self, name: str, state: State) -> State:
        fn = self.graph.nodes[name]
        with traced_span(f"graph.node.{name}", {"node": name}):
            result = fn(state)
            if inspect.isawaitable(result):
                result = await result
        return state.merge(result)

    def _next(self, current: str, state: State) -> str:
        if current in self.graph.conditional_edges:
            router, path_map = self.graph.conditional_edges[current]
            target = router(state)
            if path_map is not None:
                if target not in path_map:
                    raise GraphError(f"Router at {current!r} returned {target!r}, not in path map")
                target = path_map[target]
            return target
        return self.graph.edges.get(current, END)

    async def ainvoke(self, state: State | dict[str, Any]) -> State:
        """Execute the graph from the entry point until END."""
        if isinstance(state, dict):
            state = self.graph.state_type.model_validate(state)
        assert isinstance(state, State)
        current = self.graph.entry_point
        for _ in range(self.max_iterations):
            if current == END:
                return state
            if current not in self.graph.nodes:
                raise GraphError(f"Unknown node {current!r}")
            state = await self._run_node(current, state)
            current = self._next(current, state)
        if current == END:
            return state
        raise GraphError(
            f"Graph exceeded max_iterations={self.max_iterations} (stuck at {current!r})"
        )

    def invoke(self, state: State | dict[str, Any]) -> State:
        """Synchronous convenience wrapper around :meth:`ainvoke`."""
        import asyncio

        return asyncio.run(self.ainvoke(state))

    def as_handler(self) -> Callable[[Any], Awaitable[Any]]:
        """Adapt this graph to the GatewayRouter handler signature.

        Wraps an inbound OmniMessage in a fresh State, runs the graph, and
        returns the last message in the final state as the reply.
        """
        from omniai.protocol import OmniMessage

        async def handler(message: OmniMessage) -> OmniMessage:
            final = await self.ainvoke(self.graph.state_type(messages=[message]))
            if not final.messages:
                return message.reply("")
            last = final.messages[-1]
            reply = message.reply(last.content)
            reply.tool_calls = last.tool_calls
            return reply

        return handler
