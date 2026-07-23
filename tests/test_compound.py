"""Compound AI architecture: RAG, tenancy, curation, rehearsal, shadow gate."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from omniai.memory import (
    ContinuousLearner,
    InteractionBuffer,
    InteractionJudge,
    LoRATrainer,
    RehearsalBuffer,
)
from omniai.protocol import OmniMessage, Role
from omniai.rag import Document, InMemoryVectorStore, Retriever, chunk_text
from omniai.tenancy import AgentProfile, AgentRegistry, TenantHandler

# -- RAG ----------------------------------------------------------------------


def _catalog_store():
    store = InMemoryVectorStore()
    store.add_texts(
        [
            "The Widget Pro costs $49 and ships in blue or red.",
            "Vacation policy: employees accrue 1.5 days per month.",
            "The Gadget Max costs $99 and includes a warranty.",
        ]
    )
    return store


def test_vector_store_retrieves_relevant_documents():
    store = _catalog_store()
    hits = store.search("how much does the widget pro cost", k=1)
    assert "Widget Pro" in hits[0].document.text
    hits = store.search("vacation days accrual policy", k=1)
    assert "Vacation policy" in hits[0].document.text


def test_retriever_renders_grounding_context_or_nothing():
    retriever = Retriever(_catalog_store(), k=2)
    context = retriever.render_context("widget pro price")
    assert "Widget Pro" in context
    assert "don't know" in context  # anti-hallucination instruction
    empty = Retriever(InMemoryVectorStore(), k=2).render_context("anything")
    assert empty == ""


def test_chunking_splits_and_overlaps():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(10))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_chunk_size_must_exceed_overlap():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("some text", chunk_size=50, overlap=50)


def test_pdf_style_document_ingestion():
    store = InMemoryVectorStore()
    ids = store.add_document_text(
        "\n\n".join(["Intro section. " * 20, "Pricing: the Mega costs $10. " * 10]),
        metadata={"source": "catalog.pdf"},
        chunk_size=300,
    )
    assert len(ids) > 1
    hit = store.search("mega price", k=1)[0]
    assert hit.document.metadata["source"] == "catalog.pdf"


def test_vector_store_delete_removes_document():
    store = InMemoryVectorStore()
    (doc_id,) = store.add([Document(text="ephemeral fact")])
    assert len(store) == 1
    store.delete([doc_id])
    assert len(store) == 0
    assert store.search("ephemeral fact") == []


# -- multi-tenancy -------------------------------------------------------------


class ChatRecordingEngine:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []

    async def chat_text(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.reply


async def test_tenant_handler_routes_lora_and_injects_rag():
    engine = ChatRecordingEngine(reply="The Widget Pro is $49.")
    registry = AgentRegistry()
    registry.register(
        AgentProfile(
            name="sales",
            lora="sales-lora-v3",
            system_prompt="Be enthusiastic. Use bullet points.",
            retriever=Retriever(_catalog_store(), k=1),
        )
    )
    registry.register(AgentProfile(name="hr", lora="hr-lora-v1"))

    handler = TenantHandler(registry, engine)
    message = OmniMessage(content="widget pro price?", metadata={"agent": "sales"})
    reply = await handler(message)

    call = engine.calls[0]
    assert call["kwargs"]["model"] == "sales-lora-v3"  # per-request routing
    system = call["messages"][0]
    assert system["role"] == "system"
    assert "enthusiastic" in system["content"]
    assert "Widget Pro" in system["content"]  # RAG context in the prompt
    assert reply.metadata["agent"] == "sales"
    assert "Widget Pro" in reply.metadata["rag_context"]  # telemetry tuple


async def test_tenant_registry_falls_back_to_default():
    engine = ChatRecordingEngine()
    registry = AgentRegistry()
    registry.register(AgentProfile(name="assistant"))  # base model, no LoRA
    handler = TenantHandler(registry, engine)
    await handler(OmniMessage(content="hi", metadata={"agent": "nope"}))
    assert "model" not in engine.calls[0]["kwargs"]  # base model served


def test_duplicate_agent_rejected():
    registry = AgentRegistry()
    registry.register(AgentProfile(name="a"))
    with pytest.raises(ValueError):
        registry.register(AgentProfile(name="a"))


def test_registry_with_no_agents_raises():
    with pytest.raises(KeyError):
        AgentRegistry().resolve(None)


# -- curation (anti-poisoning) --------------------------------------------------


def _logs(*turns):
    rows = []
    for role, content, *meta in turns:
        rows.append(
            {
                "session_id": "s",
                "role": role,
                "content": content,
                "metadata": json.dumps(meta[0] if meta else {}),
            }
        )
    return rows


async def test_judge_discards_low_quality_pairs():
    scores = {"good answer": 0.9, "hallucinated nonsense": 0.1}
    judge = InteractionJudge(judge_fn=lambda instr, out: scores[out], min_score=0.7)
    pairs = await judge.curate(
        _logs(
            ("user", "q1"),
            ("assistant", "good answer"),
            ("user", "q2"),
            ("assistant", "hallucinated nonsense"),
        )
    )
    assert [p["completion"] for p in pairs] == ["good answer"]


async def test_explicit_feedback_short_circuits_the_judge():
    def judge_fn(instr, out):  # pragma: no cover - must not be called
        raise AssertionError("judge should be skipped for explicit feedback")

    judge = InteractionJudge(judge_fn=judge_fn)
    pairs = await judge.curate(
        _logs(
            ("user", "q1"),
            ("assistant", "kept", {"feedback": "thumbs_up"}),
            ("user", "q2"),
            ("assistant", "dropped", {"feedback": "thumbs_down"}),
        )
    )
    assert [p["completion"] for p in pairs] == ["kept"]


async def test_llm_judge_parses_scores_and_unparseable_means_reject():
    class JudgeEngine:
        def __init__(self, replies):
            self.replies = list(replies)

        async def chat_text(self, messages, **kwargs):
            return self.replies.pop(0)

    engine = JudgeEngine(['{"score": 0.95, "reason": "solid"}', "I refuse to answer"])
    judge = InteractionJudge(engine=engine, min_score=0.7)
    pairs = await judge.curate(
        _logs(
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
        )
    )
    assert [p["completion"] for p in pairs] == ["a1"]


def test_judge_requires_engine_or_judge_fn():
    with pytest.raises(ValueError):
        InteractionJudge()


# -- rehearsal (anti-forgetting) ------------------------------------------------


def test_rehearsal_mix_is_mostly_golden_data(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(json.dumps({"instruction": f"g{i}", "output": f"o{i}"}) for i in range(500))
    )
    buffer = RehearsalBuffer.from_jsonl(golden, new_data_ratio=0.2, seed=7)
    new = [{"prompt": f"n{i}", "completion": f"c{i}"} for i in range(10)]
    mixed = buffer.mix(new)
    assert len(mixed) == 50  # 10 new + 40 golden = 20% new
    assert sum(1 for p in mixed if p["prompt"].startswith("n")) == 10


def test_rehearsal_capped_by_available_golden_data():
    buffer = RehearsalBuffer([{"prompt": "g", "completion": "o"}] * 3, new_data_ratio=0.2)
    mixed = buffer.mix([{"prompt": "n", "completion": "c"}] * 10)
    assert len(mixed) == 13


def test_rehearsal_ratio_must_be_valid():
    with pytest.raises(ValueError):
        RehearsalBuffer([], new_data_ratio=0)


def test_rehearsal_mix_of_no_new_pairs_is_empty():
    buffer = RehearsalBuffer([{"prompt": "g", "completion": "o"}])
    assert buffer.mix([]) == []


# -- watermarking + shadow gate --------------------------------------------------


def fake_train(base_model, pairs, output_dir, **hp):
    return output_dir


class GateRecordingEngine:
    system_prompt = None

    def __init__(self):
        self.loads = []
        self.unloads = []
        self.activated = []

    async def load_lora_adapter(self, name, path, activate=True):
        self.loads.append((name, activate))
        if activate:
            self.activated.append(name)
        return True

    async def unload_lora_adapter(self, name):
        self.unloads.append(name)
        return True

    def activate_lora(self, name):
        self.activated.append(name)


async def _seed(buffer, n=1):
    for i in range(n):
        await buffer.log(OmniMessage(content=f"q{i}", role=Role.USER, session_id="s"))
        await buffer.log(OmniMessage(content=f"a{i}", role=Role.ASSISTANT, session_id="s"))


async def test_watermark_prevents_retraining_on_old_data(tmp_path):
    buffer = InteractionBuffer(tmp_path / "log.db")
    await _seed(buffer)
    trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    learner = ContinuousLearner(buffer, trainer, engine=GateRecordingEngine())

    first = await learner.run_cycle()
    assert first["status"] == "deployed" and first["new_pairs"] == 1

    # No new data since the watermark -> nothing to train on.
    second = await learner.run_cycle()
    assert second["status"] == "skipped"

    # New data trains only on the delta.
    await _seed(buffer, n=2)
    third = await learner.run_cycle()
    assert third["status"] == "deployed" and third["new_pairs"] == 2
    buffer.close()


async def test_watermark_survives_restart(tmp_path):
    db = tmp_path / "log.db"
    buffer = InteractionBuffer(db)
    await _seed(buffer)
    trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    await ContinuousLearner(buffer, trainer, engine=GateRecordingEngine()).run_cycle()
    buffer.close()

    reopened = InteractionBuffer(db)
    trainer2 = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    report = await ContinuousLearner(reopened, trainer2, engine=GateRecordingEngine()).run_cycle()
    assert report["status"] == "skipped"  # nothing new after restart
    reopened.close()


async def test_shadow_gate_accepts_and_activates(tmp_path):
    buffer = InteractionBuffer(tmp_path / "log.db")
    await _seed(buffer)
    engine = GateRecordingEngine()
    trainer = LoRATrainer("base/model", tmp_path / "adapters", train_fn=fake_train)
    learner = ContinuousLearner(buffer, trainer, engine=engine, evaluator=lambda name, path: True)
    report = await learner.run_cycle()
    assert report["status"] == "deployed"
    assert engine.loads == [(report["adapter"], False)]  # shadow load first
    assert engine.activated == [report["adapter"]]  # then promoted
    buffer.close()


async def test_shadow_gate_rejects_unloads_and_destroys_weights(tmp_path):
    def train_writing_weights(base_model, pairs, output_dir, **hp):
        out = tmp_path / "adapters" / output_dir.split("/")[-1]
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_model.bin").write_bytes(b"weights")
        return str(out)

    buffer = InteractionBuffer(tmp_path / "log.db")
    await _seed(buffer)
    engine = GateRecordingEngine()
    trainer = LoRATrainer(
        "base/model",
        tmp_path / "adapters",
        train_fn=train_writing_weights,
        executor=ThreadPoolExecutor(max_workers=1),  # closures don't pickle
    )
    learner = ContinuousLearner(buffer, trainer, engine=engine, evaluator=lambda name, path: False)
    report = await learner.run_cycle()
    assert report["status"] == "rejected"
    assert engine.activated == []
    assert engine.unloads == [report["adapter"]]
    assert not (tmp_path / "adapters" / report["adapter"]).exists()  # destroyed
    # Rejection does not advance the watermark: the data gets another chance.
    assert (await buffer.get_watermark()) is None
    buffer.close()


async def test_full_compound_cycle_curates_mixes_and_deploys(tmp_path):
    buffer = InteractionBuffer(tmp_path / "log.db")
    await buffer.log(OmniMessage(content="q-good", role=Role.USER, session_id="s"))
    await buffer.log(OmniMessage(content="a-good", role=Role.ASSISTANT, session_id="s"))
    await buffer.log(OmniMessage(content="q-bad", role=Role.USER, session_id="s"))
    await buffer.log(OmniMessage(content="a-bad", role=Role.ASSISTANT, session_id="s"))

    judge = InteractionJudge(
        judge_fn=lambda instr, out: 0.9 if "good" in out else 0.1, min_score=0.7
    )
    rehearsal = RehearsalBuffer(
        [{"prompt": f"g{i}", "completion": "o"} for i in range(100)], seed=1
    )
    trained = {}

    def capture_train(base_model, pairs, output_dir, **hp):
        trained["pairs"] = pairs
        return output_dir

    engine = GateRecordingEngine()
    learner = ContinuousLearner(
        buffer,
        LoRATrainer(
            "base/model",
            tmp_path / "adapters",
            train_fn=capture_train,
            executor=ThreadPoolExecutor(max_workers=1),  # closures don't pickle
        ),
        engine=engine,
        evaluator=lambda name, path: True,
        curator=judge.curate,
        rehearsal=rehearsal,
    )
    report = await learner.run_cycle()
    assert report["status"] == "deployed"
    assert report["new_pairs"] == 1  # a-bad was judged out
    completions = [p["completion"] for p in trained["pairs"]]
    assert "a-good" in completions and "a-bad" not in completions
    assert len(trained["pairs"]) == 5  # 1 new + 4 rehearsal (20% mix)
    assert engine.activated  # gate passed, adapter promoted
    buffer.close()
