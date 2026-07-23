# omniai.guardrails

## `PromptGuard`

```python
PromptGuard(policy: GuardrailPolicy | None = None)
```

- `check(text) -> GuardrailResult` — direct use.
- Calling the instance with an `OmniMessage` (gateway-interceptor form) raises `GuardrailViolation` on blocked content, else returns the message with PII redacted and `metadata["pii_redacted"]` set.

## `GuardrailPolicy` (dataclass)

| Field | Default | Meaning |
| --- | --- | --- |
| `injection_patterns` | 5 built-in patterns | name → regex; a match marks the message (instruction override, system-prompt probe, role hijack, fake system markup, secret exfiltration). |
| `pii_patterns` | email, ssn, credit_card, phone, ipv4 | name → regex applied as redactions. `credit_card` matches only Luhn-valid digit runs. |
| `block_on_injection` | `True` | `False` = detect and record without blocking. |
| `redact_template` | `"[REDACTED:{kind}]"` | Replacement text. |

## `GuardrailResult` (dataclass)

`blocked: bool`, `sanitized: str`, `injection_hits: list[str]`, `pii_hits: list[str]`.

`GuardrailViolation` lives in `omniai.gateway.router`; the gateway maps it to HTTP 400.
