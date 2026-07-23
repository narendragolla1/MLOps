# Agents

An agent is a model in a loop with tools: the model decides *what* to do, the framework does it, and the observation goes back to the model until it can answer. In OmniAI this loop is not a black box — `create_tool_agent` builds it from the public [graph](graphs.md) API, and you can read or replace it.

## The loop

```
        ┌────────────┐   tool_calls?   ┌───────────┐
START ─▶│   model    │───────yes──────▶│   tools   │
        └────────────┘                 └───────────┘
              │  ▲                            │
           no │  └────────── results ─────────┘
              ▼
             END
```

- **model** node: converts `state.messages` to the wire format, injects the system prompt once, calls `ChatModel.generate(tools=...)`, appends the assistant reply (with any `tool_calls`), increments `steps`.
- **tools** node: executes every requested call via `Tool.execute`, appending one `TOOL` message per call.
- **routing**: a conditional edge loops while the last message has tool calls *and* `steps < max_steps`.

## Errors are observations

The tools node never lets a failure crash the run — the model sees it and can adapt:

| Failure | What the model sees |
| --- | --- |
| Arguments fail schema validation | `error: invalid arguments: <details>` → it can re-call with fixed args |
| Tool name doesn't exist | `error: unknown tool 'x'` |
| Tool body raises | `error: RuntimeError: ...` → it can try another approach or report honestly |

This is the practical difference between an agent that recovers and one that 500s.

## Termination

Two independent bounds prevent runaway loops: `max_steps` caps **model turns** (routing exits even if the model still wants tools), and the compiled graph's `max_iterations` caps total node executions as a hard backstop. There is no unbounded agent in this framework.

## Where agents run

A compiled agent is just a graph: `agent.ainvoke(...)` in scripts, or `agent.as_handler()` to serve it through the [gateway](gateway.md) on every channel at once.
