"""Skill ingestion: parse skill.md files and pre-cache them at boot.

A skill file is Markdown with optional YAML-ish frontmatter::

    ---
    name: web-search
    description: Search the web for fresh information.
    ---
    # Instructions
    ...capability details...

At boot, ``SkillLoader.load_directory`` parses every ``*.skill.md`` /
``skill.md`` file and ``compose_system_prompt`` folds them into one system
prompt. Installing that prompt via ``ModelEngine.set_system_prompt`` means
the shared prefix is prefilled once and served from cache thereafter
(RadixAttention on SGLang, prefix caching on vLLM).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Skill(BaseModel):
    name: str
    description: str = ""
    body: str = ""
    source: str = ""

    def render(self) -> str:
        header = f"## Skill: {self.name}"
        if self.description:
            header += f"\n{self.description}"
        return f"{header}\n\n{self.body}".strip()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    meta: dict[str, str] = {}
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[idx + 1:])
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return {}, text  # unterminated frontmatter: treat whole file as body


class SkillLoader:
    """Loads skill files and composes the engine's system prompt."""

    def __init__(self, preamble: str = "You are a capable assistant with the following skills."):
        self.preamble = preamble
        self.skills: list[Skill] = []

    def load_file(self, path: str | Path) -> Skill:
        path = Path(path)
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        skill = Skill(
            name=meta.get("name", path.stem.removesuffix(".skill")),
            description=meta.get("description", ""),
            body=body.strip(),
            source=str(path),
        )
        self.skills.append(skill)
        return skill

    def load_directory(self, directory: str | Path) -> list[Skill]:
        directory = Path(directory)
        paths = sorted({*directory.rglob("*.skill.md"), *directory.rglob("skill.md")})
        return [self.load_file(p) for p in paths]

    def compose_system_prompt(self) -> str:
        sections = [self.preamble] + [s.render() for s in self.skills]
        return "\n\n".join(sections)

    def install(self, engine) -> str:
        """Pre-cache the composed prompt into a ModelEngine."""
        prompt = self.compose_system_prompt()
        engine.set_system_prompt(prompt)
        return prompt
