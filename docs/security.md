# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately via GitHub Security Advisories on this repository ("Report a vulnerability"), or to the repository owner directly. Please do **not** open public issues for security reports. Include reproduction steps and impact; you can expect an acknowledgment within a few business days.

## Scope

In scope:

- Authentication/authorization bypasses in the gateway (API keys, WebSocket handshake, exempt paths).
- Guardrail bypasses that are *systemic* (e.g. a class of injection the blocking layer claims to catch), not individual clever prompts.
- Sandbox escapes from `SandboxExecution`'s container lockdown.
- Injection into the persistence layer, or PII leaking into logs/metrics after redaction claims.
- Denial-of-service vectors that defeat the rate limiter, body caps, or backpressure by design flaw.

Out of scope:

- Prompt content that convinces a model to produce undesirable *text* (model-behavior issues belong upstream with the model provider).
- Vulnerabilities in dependencies (report upstream; do tell us if our default configuration makes one exploitable).
- Deployments that disable the shipped protections (`OMNIAI_AUTH_DISABLED=true`, sandbox with `network=True`, etc.).

## Deployment expectations

The framework is secure-by-default only when the defaults are kept: auth fail-closed, guardrails attached, generated code confined to the sandbox, and secrets provided via your platform's secret store — see the [security model](concepts/security.md) for the full layering and the residual risks you own.
