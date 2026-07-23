# Security model

LLM applications have two attack surfaces: the classic API surface, and the model itself (untrusted natural language in, possibly-untrusted actions out). OmniAI layers defenses for both; no single layer is trusted alone.

## Layer 1 — the network edge

- **Fail-closed auth**: the app refuses to boot with auth enabled and no API keys; running open requires an explicit `OMNIAI_AUTH_DISABLED=true`. Keys are compared in constant time. WebSockets authenticate at handshake (close 4401).
- **Rate limiting** per key (token bucket, 429 + `Retry-After`).
- **Body caps** enforced both by `Content-Length` and while streaming — chunked encoding is not a bypass.
- **CORS** closed by default.

## Layer 2 — the prompt

[`PromptGuard`](../how_to/guardrails.md) screens inbound content: known injection patterns are blocked; PII is redacted (with Luhn validation on card-shaped numbers to limit false positives). These are **heuristics** — they raise the bar, they are not a proof. Assume a motivated attacker can get text past them, and design layers 3–4 accordingly.

## Layer 3 — the model's actions

- The model can only invoke tools you registered, with **schema-validated** arguments.
- The model's decision to call a tool is *never* an authorization decision — tools touching real systems must check permissions themselves, scoped to the session's user.
- Generated code runs only in the [sandbox](../how_to/sandbox_code_execution.md): fresh container, no network, read-only FS, memory/CPU/time caps, non-root.

## Layer 4 — outputs and operations

- Error responses never leak stack traces or internal detail.
- Adapters from continuous learning pass an [eval gate](../how_to/evaluate_adapters.md) before deployment — training data can't silently degrade tool-calling behavior.
- Everything is observable: request IDs, structured logs, metrics, and readiness that reports *why* it's failing.

## Residual risks to own

Prompt injection via *tool outputs* (a fetched webpage instructing the model) passes layer 2 — mitigate with least-privilege tools and output review for high-stakes actions. Multi-replica rate limiting needs a shared store. Secrets management (key rotation, vaults) is deliberately left to your platform.

Reporting vulnerabilities: see the [security policy](../security.md).
