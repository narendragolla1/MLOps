# How to build graphs with conditional edges and cycles

## Nodes and state

Declare a state class, register nodes (sync or async callables taking state, returning a partial update), and connect them:

```python
from omniai.graph import Graph, State, START, END

class CounterState(State):
    count: int = 0
    result: str = ""

g = Graph(CounterState)
g.add_node("increment", lambda s: {"count": s.count + 1})
g.add_node("report", lambda s: {"result": f"count={s.count}"})
g.add_edge(START, "increment")
g.add_edge("increment", "report")
g.add_edge("report", END)

final = g.compile().invoke({"count": 10})    # or: await compiled.ainvoke(...)
```

Updates merge immutably: `messages` appends, every other field replaces — see [graph concepts](../concepts/graphs.md) for the semantics.

The decorator form also works:

```python
@g.add_node("increment")
def increment(s: CounterState):
    return {"count": s.count + 1}
```

## Conditional edges

Route with any callable of the state — lambdas are idiomatic:

```python
g.add_conditional_edges("increment", lambda s: "report" if s.count >= 3 else "increment")
```

Or keep routing keys separate from node names with a path map:

```python
g.add_conditional_edges(
    "check",
    lambda s: "big" if s.count > 5 else "small",
    {"big": "handle_large", "small": "handle_small"},
)
```

A router returning a key missing from the path map raises `GraphError` at runtime; edges referencing unknown nodes fail at `compile()` time.

## Cycles

Cycles are first-class (that's how agent loops work) and always bounded:

```python
compiled = g.compile(max_iterations=25)   # raises GraphError if exceeded
```

## Plugging into the gateway

`compiled.as_handler()` adapts any graph to the `GatewayRouter` handler signature: the inbound `OmniMessage` seeds `state.messages`, and the final state's last message becomes the reply.
