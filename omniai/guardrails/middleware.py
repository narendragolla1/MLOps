"""Gateway guardrails: prompt-injection screening and PII redaction.

``PromptGuard`` is a GatewayRouter interceptor. It scans inbound content
against a small library of injection heuristics (blocking on match) and
redacts PII in place before the message reaches the graph. Both pattern sets
are extensible per deployment via :class:`GuardrailPolicy`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from omniai.gateway.router import GuardrailViolation
from omniai.protocol import OmniMessage

# Heuristic markers of prompt-injection / jailbreak attempts.
DEFAULT_INJECTION_PATTERNS: dict[str, str] = {
    "override_instructions": r"(?i)\b(ignore|disregard|forget)\s+(all\s+)?(your\s+|previous\s+|prior\s+|above\s+)*(instructions|prompts?|rules)\b",
    "system_prompt_probe": r"(?i)\b(reveal|show|print|repeat)\b.{0,40}\b(system\s+prompt|hidden\s+instructions)\b",
    "role_hijack": r"(?i)\byou\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak|god)\s*mode\b",
    "fake_system_turn": r"(?i)(\[/?(system|inst)\]|<\|im_start\|>\s*system|</?system>)",
    "exfiltrate_secrets": r"(?i)\b(print|dump|reveal)\b.{0,40}\b(api[_\s-]?keys?|credentials|passwords?|secrets?)\b",
}

# PII patterns applied as redactions (never block, just strip).
DEFAULT_PII_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "phone": r"(?<![\d-])(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?![\d-])",
    "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def _luhn_valid(candidate: str) -> bool:
    """Luhn checksum over the digits in ``candidate`` (card-number check)."""
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for idx, digit in enumerate(reversed(digits)):
        if idx % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


@dataclass
class GuardrailPolicy:
    injection_patterns: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_INJECTION_PATTERNS)
    )
    pii_patterns: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PII_PATTERNS))
    block_on_injection: bool = True
    redact_template: str = "[REDACTED:{kind}]"


@dataclass
class GuardrailResult:
    blocked: bool
    sanitized: str
    injection_hits: list[str] = field(default_factory=list)
    pii_hits: list[str] = field(default_factory=list)


class PromptGuard:
    """Screens text for injection attempts and strips PII.

    Use directly (``guard.check(text)``) or attach to a router:
    ``GatewayRouter(handler=..., interceptors=[guard])``.
    """

    def __init__(self, policy: GuardrailPolicy | None = None):
        self.policy = policy or GuardrailPolicy()
        self._injection = {
            name: re.compile(pat) for name, pat in self.policy.injection_patterns.items()
        }
        self._pii = {name: re.compile(pat) for name, pat in self.policy.pii_patterns.items()}

    def check(self, text: str) -> GuardrailResult:
        injection_hits = [name for name, rx in self._injection.items() if rx.search(text)]
        sanitized = text
        pii_hits = []
        for name, rx in self._pii.items():
            replacement = self.policy.redact_template.format(kind=name)
            if name == "credit_card":
                # A bare 13-16 digit run is often an order/tracking number;
                # only redact sequences that pass the Luhn checksum.
                n = 0

                def _redact_if_card(match: re.Match[str]) -> str:
                    nonlocal n
                    if _luhn_valid(match.group()):
                        n += 1
                        return replacement
                    return match.group()

                sanitized = rx.sub(_redact_if_card, sanitized)
            else:
                sanitized, n = rx.subn(replacement, sanitized)
            if n:
                pii_hits.append(name)
        blocked = bool(injection_hits) and self.policy.block_on_injection
        return GuardrailResult(
            blocked=blocked,
            sanitized=sanitized,
            injection_hits=injection_hits,
            pii_hits=pii_hits,
        )

    def __call__(self, message: OmniMessage) -> OmniMessage:
        """GatewayRouter interceptor: block on injection, redact PII."""
        result = self.check(message.content)
        if result.blocked:
            raise GuardrailViolation(
                f"prompt rejected by guardrails: {', '.join(result.injection_hits)}"
            )
        if result.pii_hits:
            message = message.model_copy(
                update={
                    "content": result.sanitized,
                    "metadata": {**message.metadata, "pii_redacted": result.pii_hits},
                }
            )
        return message
