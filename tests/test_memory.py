import asyncio

import httpx

from omniai.engine import ModelEngine
from omniai.memory import (
    ContinuousLearner,
    InteractionBuffer,
    LoRATrainer,
    SkillLoader,
    format_training_pairs,
)
from omniai.protocol import OmniMessage, Role

# -- skills ----------------------------------------------------------------

SKILL_MD = """---
name: web-search
description: Search the web.
---
Use the search tool when facts may be stale.
"""


def test_skill_loader_parses_frontmatter(tmp_path):
    (tmp_path / "search.skill.md").write_text(SKILL_MD)
    (tmp_path / "plain.skill.md").write_text("Just instructions, no frontmatter.")
    loader = SkillLoader()
    skills = loader.load_directory(tmp_path)
    assert {s.name for s in skills} == {"web-search", "plain"}
    prompt = loader.compose_system_prompt()
    assert "## Skill: web-search" in prompt
    assert "Search the web." in prompt
    assert "Just instructions" in prompt


def test_skill_loader_installs_into_engine(tmp_path):
    (tmp_path / "s.skill.md").write_text(SKILL_MD)
    loader = SkillLoader()
    loader.load_directory(tmp_path)
    engine = ModelEngine.create({"model": "m"})
    loader.install(engine)
    assert "web-search" in engine.system_prompt


# -- interaction buffer ----------------------------------------------------


async def test_buffer_logs_and_fetches(tmp_path):
    buffer = InteractionBuffer(tmp_path / "log.db")
    await buffer.log(OmniMessage(content="hi", session_id="a", role=Role.USER))
    await buffer.log(OmniMessage(content="hello!", session_id="a", role=Role.ASSISTANT))
    assert await buffer.count() == 2
    rows = await buffer.fetch(session_id="a")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"] == "hi"
    buffer.close()


async def test_buffer_threshold_trigger(tmp_path):
    fired = []
    buffer = InteractionBuffer(
        tmp_path / "log.db", threshold=3, on_threshold=lambda: fired.append(1)
    )
    for i in range(7):
        await buffer.log(OmniMessage(content=str(i)))
    assert len(fired) == 2  # at 3 and at 6
    buffer.close()


# -- training pair formatting ----------------------------------------------


def _row(role, content, session="s"):
    return {"session_id": session, "role": role, "content": content}


def test_format_training_pairs_with_tool_context():
    logs = [
        _row("user", "What's the weather?"),
        _row("tool", "22C sunny"),
        _row("assistant", "It's 22C and sunny."),
        _row("user", "Thanks"),
        _row("assistant", "Anytime!"),
        _row("assistant", "orphan reply", session="other"),  # no user turn: dropped
    ]
    pairs = format_training_pairs(logs, system_prompt="sys")
    assert len(pairs) == 2
    assert pairs[0]["prompt"] == "What's the weather?\n[tool output] 22C sunny"
    assert pairs[0]["completion"] == "It's 22C and sunny."
    assert pairs[0]["system"] == "sys"
    assert pairs[1] == {"prompt": "Thanks", "completion": "Anytime!", "system": "sys"}


# -- continuous learning cycle ---------------------------------------------


def fake_train(base_model, pairs, output_dir, **hp):
    return output_dir  # pretend the adapter was written


class SwapRecordingEngine:
    system_prompt = None

    def __init__(self):
        self.swaps = []

    async def load_lora_adapter(self, name, path, activate=True):
        self.swaps.append((name, path))
        return True


async def _seed(buffer):
    await buffer.log(OmniMessage(content="q1", role=Role.USER, session_id="s"))
    await buffer.log(OmniMessage(content="a1", role=Role.ASSISTANT, session_id="s"))


async def test_full_cycle_trains_and_hot_swaps(tmp_path):
    buffer = InteractionBuffer(tmp_path / "log.db")
    await _seed(buffer)
    engine = SwapRecordingEngine()
    trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    learner = ContinuousLearner(buffer, trainer, engine=engine)

    report = await learner.run_cycle()
    assert report["status"] == "deployed"
    assert report["pairs"] == 1
    assert engine.swaps == [(report["adapter"], report["path"])]
    assert "-lora-" in report["adapter"]
    buffer.close()


async def test_adapter_names_unique_across_trainer_instances(tmp_path):
    pairs = [{"prompt": "p", "completion": "c"}]
    names = set()
    for _ in range(2):  # fresh trainer each time simulates a process restart
        trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
        name, _ = await trainer.train(pairs)
        names.add(name)
    assert len(names) == 2


async def test_incremental_training_uses_high_water_mark(tmp_path):
    trained_batches = []

    def spy_train(base_model, pairs, output_dir, **hp):
        trained_batches.append(len(pairs))
        return output_dir

    buffer = InteractionBuffer(tmp_path / "log.db")
    await _seed(buffer)
    engine = SwapRecordingEngine()
    trainer = LoRATrainer("base/model", tmp_path / "adapters", executor=None, train_fn=spy_train)
    # thread-friendly: spy closure can't cross a process boundary
    from concurrent.futures import ThreadPoolExecutor

    trainer.executor = ThreadPoolExecutor(max_workers=1)
    learner = ContinuousLearner(buffer, trainer, engine=engine)

    assert (await learner.run_cycle())["status"] == "deployed"
    # No new data: second cycle must skip instead of retraining everything.
    assert (await learner.run_cycle())["status"] == "skipped"

    await buffer.log(OmniMessage(content="q2", role=Role.USER, session_id="s2"))
    await buffer.log(OmniMessage(content="a2", role=Role.ASSISTANT, session_id="s2"))
    report = await learner.run_cycle()
    assert report["status"] == "deployed"
    assert report["pairs"] == 1  # only the new pair, not the full history
    assert trained_batches == [1, 1]
    buffer.close()


async def test_cycle_skips_without_pairs(tmp_path):
    buffer = InteractionBuffer(tmp_path / "log.db")
    trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    learner = ContinuousLearner(buffer, trainer, engine=SwapRecordingEngine())
    report = await learner.run_cycle()
    assert report["status"] == "skipped"
    buffer.close()


async def test_eval_gate_rejects_bad_adapter(tmp_path):
    buffer = InteractionBuffer(tmp_path / "log.db")
    await _seed(buffer)
    engine = SwapRecordingEngine()
    trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    learner = ContinuousLearner(buffer, trainer, engine=engine, evaluator=lambda name, path: False)
    report = await learner.run_cycle()
    assert report["status"] == "rejected"
    assert engine.swaps == []
    buffer.close()


async def test_threshold_wires_to_learner(tmp_path):
    buffer = InteractionBuffer(tmp_path / "log.db", threshold=2)
    engine = SwapRecordingEngine()
    trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    learner = ContinuousLearner(buffer, trainer, engine=engine)
    buffer.on_threshold = learner.trigger

    await buffer.log(OmniMessage(content="q", role=Role.USER, session_id="s"))
    await buffer.log(OmniMessage(content="a", role=Role.ASSISTANT, session_id="s"))
    await asyncio.sleep(0.2)  # let the fire-and-forget cycle finish
    assert len(engine.swaps) == 1
    buffer.close()


async def test_lora_swap_uses_vllm_rest_api(tmp_path):
    """End-to-end: learner drives a real ModelEngine's LoRA REST call."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"status": "ok"})

    engine = ModelEngine.create({"model": "base/model", "backend": "vllm"})
    engine._client = httpx.AsyncClient(
        base_url=engine.config.base_url, transport=httpx.MockTransport(handler)
    )
    buffer = InteractionBuffer(tmp_path / "log.db")
    await _seed(buffer)
    trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    learner = ContinuousLearner(buffer, trainer, engine=engine)
    report = await learner.run_cycle()
    assert report["status"] == "deployed"
    assert seen["path"] == "/v1/load_lora_adapter"
    assert engine.active_lora == report["adapter"]
    buffer.close()
