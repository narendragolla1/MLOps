"""LoRARegistry: the engine's record of adapters and their lifecycle.

The serving backends only know "load" and "unload"; everything an operator
actually needs on top — which adapter is live, which one was live before
(rollback), which loaded adapter to evict when the server's slot cap is
reached, and surviving restarts — lives here. The registry is a plain
in-memory table with optional JSON persistence, so a restarted engine (or a
supervisor) can re-apply the active adapter.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AdapterRecord:
    name: str
    path: str
    loaded_at: float = field(default_factory=time.time)


class LoRARegistry:
    """Tracks loaded adapters, the active one, and rollback history."""

    def __init__(self, persist_path: str | Path | None = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self.loaded: dict[str, AdapterRecord] = {}
        self.active: str | None = None
        self.previous: str | None = None
        if self.persist_path and self.persist_path.exists():
            self._restore()

    def register(self, name: str, path: str) -> AdapterRecord:
        record = AdapterRecord(name=name, path=path)
        self.loaded[name] = record
        self._persist()
        return record

    def activate(self, name: str) -> None:
        if name not in self.loaded:
            raise KeyError(f"Adapter {name!r} is not loaded")
        if self.active != name:
            self.previous = self.active
        self.active = name
        self._persist()

    def deactivate(self) -> None:
        """Drop back to the base model (keeps rollback history)."""
        if self.active is not None:
            self.previous = self.active
        self.active = None
        self._persist()

    def remove(self, name: str) -> None:
        self.loaded.pop(name, None)
        if self.active == name:
            self.active = None
        if self.previous == name:
            self.previous = None
        self._persist()

    def eviction_candidate(self, capacity: int) -> str | None:
        """Oldest loaded adapter that is neither active nor the rollback
        target, once ``capacity`` slots are (or would be) full."""
        if len(self.loaded) < capacity:
            return None
        protected = {self.active, self.previous}
        candidates = [r for r in self.loaded.values() if r.name not in protected]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.loaded_at).name

    # -- persistence --------------------------------------------------------

    def _persist(self) -> None:
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "loaded": [asdict(r) for r in self.loaded.values()],
            "active": self.active,
            "previous": self.previous,
        }
        self.persist_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _restore(self) -> None:
        state = json.loads(self.persist_path.read_text(encoding="utf-8"))
        self.loaded = {r["name"]: AdapterRecord(**r) for r in state.get("loaded", [])}
        self.active = state.get("active")
        self.previous = state.get("previous")
