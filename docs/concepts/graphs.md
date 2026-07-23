# Graphs

The graph layer is a small LangGraph-style state machine: **nodes** are Python callables, **edges** (static or conditional) decide what runs next, and a validated **state** object threads through it all.

## State and merging

State is a Pydantic model. Nodes receive the current state and return a *partial update* — never a mutated state:

```python
class MyState(State):
    count: int = 0

def bump(s: MyState):
    return {"count": s.count + 1}
```

Merging rules (see `State.merge`):

- `messages` **appends** — conversation history only grows, which is what agent trajectories need.
- Every other field **replaces**.
- Appended messages are validated to `OmniMessage`; other fields are applied via `model_copy` without re-validating the whole state — that keeps a step O(new data) instead of O(entire history), which matters over long conversations.

Immutability means each step yields a fresh snapshot: no aliasing bugs between nodes, and trajectories are inspectable after the fact.

## Edges

- `add_edge("a", "b")` — unconditional.
- `add_conditional_edges("a", router)` — `router(state)` returns the next node name (or `END`); lambdas are idiomatic. An optional path map translates routing keys to node names so routing logic and topology stay decoupled.
- `add_edge(START, "a")` sets the entry point.

## Compilation

`graph.compile()` validates the topology up front — a missing entry point or an edge to an unknown node fails **before** anything runs, not at step 7 of a production request. The result is an immutable `CompiledGraph` with `invoke` / `ainvoke` and `as_handler()` for the gateway.

## Cycles, bounded

Cycles are legal and essential (agent loops), but every run is capped by `max_iterations` — exceeding it raises `GraphError` naming the stuck node. Prefer an explicit loop bound in your routing (like the agent's `max_steps`) and treat `max_iterations` as the backstop.
