import pytest

from omniai.graph import END, START, Graph, State
from omniai.graph.graph import GraphError
from omniai.protocol import OmniMessage


class CounterState(State):
    count: int = 0
    result: str = ""


def test_linear_graph():
    g = Graph(CounterState)
    g.add_node("a", lambda s: {"count": s.count + 1})
    g.add_node("b", lambda s: {"result": f"count={s.count}"})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    final = g.compile().invoke({"count": 10})
    assert final.count == 11
    assert final.result == "count=11"


def test_conditional_edges_with_lambda_router():
    g = Graph(CounterState)
    g.add_node("inc", lambda s: {"count": s.count + 1})
    g.add_node("done", lambda s: {"result": "finished"})
    g.set_entry_point("inc")
    g.add_conditional_edges("inc", lambda s: "done" if s.count >= 3 else "inc")
    g.add_edge("done", END)
    final = g.compile().invoke({"count": 0})
    assert final.count == 3
    assert final.result == "finished"


def test_conditional_edges_with_path_map():
    g = Graph(CounterState)
    g.add_node("check", lambda s: None)
    g.add_node("low", lambda s: {"result": "low"})
    g.add_node("high", lambda s: {"result": "high"})
    g.set_entry_point("check")
    g.add_conditional_edges(
        "check",
        lambda s: "big" if s.count > 5 else "small",
        {"big": "high", "small": "low"},
    )
    compiled = g.compile()
    assert compiled.invoke({"count": 10}).result == "high"
    assert compiled.invoke({"count": 1}).result == "low"


async def test_async_nodes():
    async def work(s: CounterState):
        return {"count": s.count + 100}

    g = Graph(CounterState)
    g.add_node("work", work)
    g.add_edge(START, "work")
    final = await g.compile().ainvoke({"count": 1})
    assert final.count == 101


def test_cycle_bounded_by_max_iterations():
    g = Graph(CounterState)
    g.add_node("loop", lambda s: {"count": s.count + 1})
    g.set_entry_point("loop")
    g.add_conditional_edges("loop", lambda s: "loop")
    with pytest.raises(GraphError, match="max_iterations"):
        g.compile(max_iterations=5).invoke({})


def test_messages_accumulate():
    g = Graph(State)
    g.add_node("reply", lambda s: {"messages": [s.messages[-1].reply("pong")]})
    g.add_edge(START, "reply")
    final = g.compile().invoke(State(messages=[OmniMessage(content="ping")]))
    assert len(final.messages) == 2
    assert final.messages[-1].content == "pong"


def test_compile_validates_structure():
    g = Graph()
    with pytest.raises(GraphError, match="entry point"):
        g.compile()
    g.add_node("a", lambda s: None)
    g.set_entry_point("a")
    g.add_edge("a", "missing")
    with pytest.raises(GraphError, match="missing"):
        g.compile()


def test_node_decorator_form():
    g = Graph(CounterState)

    @g.add_node("bump")
    def bump(s: CounterState):
        return {"count": s.count + 1}

    g.add_edge(START, "bump")
    assert g.compile().invoke({}).count == 1


async def test_as_handler_bridges_gateway():
    g = Graph(State)
    g.add_node("reply", lambda s: {"messages": [s.messages[-1].reply("handled")]})
    g.add_edge(START, "reply")
    handler = g.compile().as_handler()
    reply = await handler(OmniMessage(content="x", session_id="s1"))
    assert reply.content == "handled"
    assert reply.session_id == "s1"
