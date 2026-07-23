# How to configure guardrails

`PromptGuard` screens inbound content before it reaches your graph: prompt-injection attempts are **blocked**, PII is **redacted in place**.

## Attach it

```python
from omniai.guardrails import PromptGuard
from omniai.gateway import GatewayRouter

router = GatewayRouter(handler=..., interceptors=[PromptGuard()])
```

Blocked messages become HTTP 400 (`prompt rejected by guardrails: <pattern names>`); redacted messages continue through the pipeline with `metadata["pii_redacted"]` listing what was stripped.

## What's detected out of the box

- **Injection (blocking):** instruction-override phrasing ("ignore all previous instructions"), system-prompt probing, role hijacks ("developer mode"), fake system-turn markup (`[system]`, `<|im_start|>`), secret-exfiltration asks.
- **PII (redacting):** emails, US SSNs, phone numbers, IPv4 addresses, and card numbers — card-shaped digit runs are redacted **only if they pass the Luhn checksum**, so order/tracking numbers survive.

Redactions render as `[REDACTED:email]` etc. (template configurable).

## Customize the policy

```python
from omniai.guardrails import GuardrailPolicy, PromptGuard

policy = GuardrailPolicy()
policy.injection_patterns["competitor_probe"] = r"(?i)what model are you"
policy.pii_patterns.pop("ipv4")                 # allow IPs through
policy.block_on_injection = False               # detect + annotate without blocking
policy.redact_template = "<removed:{kind}>"

guard = PromptGuard(policy)
result = guard.check("some text")               # use directly, outside the gateway
# result.blocked, result.sanitized, result.injection_hits, result.pii_hits
```

## Honest limitations

These are **regex heuristics** — a determined attacker can phrase around them, and no pattern list catches novel injections. Treat `PromptGuard` as one layer: combine it with [auth](auth_rate_limiting.md), least-privilege tools, and the [sandbox](sandbox_code_execution.md) for generated code. For higher assurance, chain an LLM-based classifier as an additional interceptor — interceptors compose in order.
