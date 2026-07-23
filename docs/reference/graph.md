# omniai.graph

## `State`

Pydantic base (extra fields allowed). Subclass to declare workflow fields.

- `messages: list[OmniMessage]` — appended on merge; all other fields replace.
- `merge(update: State | dict | None) -> State` — immutable merge; appended messages validated to `OmniMessage`.

## `Graph`

```python
Graph(state_type: type[State] = State)
```

- `add_node(name, fn=None)` — register a node (`fn(state) -> State | dict | None`, sync or async); decorator form without `fn`. Reserved names `START`/`END` and duplicates raise `GraphError`.
- `add_edge(source, target)` — static edge; `add_edge(START, x)` sets the entry point.
- `add_conditional_edges(source, router, path_map=None)` — `router(state) -> str` returns a node name/`END`, or a `path_map` key.
- `set_entry_point(name)`.
- `compile(max_iterations=25) -> CompiledGraph` — validates topology (`GraphError` on missing entry point or unknown edge targets).

## `CompiledGraph`

- `await ainvoke(state: State | dict) -> State` / `invoke(...)` (sync wrapper).
- Exceeding `max_iterations` raises `GraphError` naming the stuck node.
- `as_handler() -> async (OmniMessage) -> OmniMessage` — gateway bridge: inbound message seeds `state.messages`; the final state's last message is the reply.

## `tool` / `Tool`

```python
@tool                      # or @tool(name=..., description=...)
def fn(x: int, y: str = "a") -> str: ...
```

`Tool` attributes/methods: `name`, `description`, `json_schema`, `to_openai()`, `validate_args(dict | json_str)`, `await execute(dict | json_str)` (validates → coerces → runs; async transparent; failures raise `ToolValidationError`), plain `__call__` passthrough. `render_tool_prompt(tools)` (from `omniai.graph.tools`) builds a JSON-tool-calling system-prompt fragment for models without native support.

## `create_tool_agent`

```python
create_tool_agent(model: ChatModel, tools: list[Tool], system_prompt: str | None = None,
                  max_steps: int = 8, **model_kwargs) -> CompiledGraph
```

Prebuilt model⇄tools loop over `AgentState` (`State` + `steps: int`); tool errors are fed back as observations ([concept](../concepts/agents.md)).
