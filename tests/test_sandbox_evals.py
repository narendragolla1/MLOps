import httpx
import pytest

from omniai.engine import ModelEngine
from omniai.evals import AdapterGate, EvalVerdict, GoldenCase, GoldenDataset
from omniai.memory import ContinuousLearner, InteractionBuffer, LoRATrainer
from omniai.protocol import OmniMessage, Role
from omniai.sandbox import SandboxExecution, SandboxResult

# -- sandbox ---------------------------------------------------------------


def test_docker_command_is_locked_down():
    sandbox = SandboxExecution(image="python:3.11-slim", memory_limit="128m")
    cmd = sandbox.build_command("print(1)", "python")
    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert tuple(cmd[cmd.index("--network") :][:2]) == ("--network", "none")
    assert tuple(cmd[cmd.index("--memory") :][:2]) == ("--memory", "128m")
    assert "--read-only" in cmd
    assert cmd[-3:] == ["python3", "-c", "print(1)"]


def test_bash_supported_unknown_language_rejected():
    sandbox = SandboxExecution()
    assert sandbox.build_command("ls", "bash")[-3:] == ["bash", "-c", "ls"]
    with pytest.raises(ValueError):
        sandbox.build_command("x", "ruby")


async def test_execute_uses_injected_runner():
    captured = {}

    async def fake_runner(cmd):
        captured["cmd"] = cmd
        return SandboxResult(exit_code=0, stdout="42\n", stderr="")

    sandbox = SandboxExecution(runner=fake_runner)
    result = await sandbox.execute("print(42)")
    assert result.ok and result.stdout == "42\n"
    assert captured["cmd"][-1] == "print(42)"


async def test_timeout_flagged_not_ok():
    result = SandboxResult(exit_code=-1, stdout="", stderr="t", timed_out=True)
    assert not result.ok


# -- evals -----------------------------------------------------------------


def _engine_with_responses(responses: dict[str, str]) -> ModelEngine:
    """Engine whose mock backend answers per (model, prompt)."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        key = f"{body['model']}:{body['messages'][-1]['content']}"
        content = responses.get(key, "I don't know")
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
        )

    engine = ModelEngine.create({"model": "base", "backend": "vllm"})
    engine._client = httpx.AsyncClient(
        base_url=engine.config.base_url, transport=httpx.MockTransport(handler)
    )
    return engine


GOLDEN = GoldenDataset(
    cases=[
        GoldenCase("weather in Paris?", "get_weather", {"city": "Paris"}),
        GoldenCase("search for llamas", "web_search", None),
    ]
)

GOOD = '{"tool": "get_weather", "arguments": {"city": "Paris"}}'
SEARCH = '{"tool": "web_search", "arguments": {"q": "llamas"}}'


async def test_gate_accepts_adapter_matching_baseline():
    engine = _engine_with_responses(
        {
            "base:weather in Paris?": GOOD,
            "base:search for llamas": SEARCH,
            "cand:weather in Paris?": GOOD,
            "cand:search for llamas": SEARCH,
        }
    )
    gate = AdapterGate(engine, GOLDEN)
    verdict = await gate.evaluate("cand")
    assert isinstance(verdict, EvalVerdict)
    assert verdict.baseline == 1.0
    assert verdict.accuracy == 1.0
    assert verdict.accepted


async def test_gate_rejects_degraded_adapter():
    engine = _engine_with_responses(
        {
            "base:weather in Paris?": GOOD,
            "base:search for llamas": SEARCH,
            "cand:weather in Paris?": "The weather is nice!",  # stopped tool-calling
            "cand:search for llamas": SEARCH,
        }
    )
    gate = AdapterGate(engine, GOLDEN)
    verdict = await gate.evaluate("cand")
    assert verdict.baseline == 1.0
    assert verdict.accuracy == 0.5
    assert not verdict.accepted
    assert verdict.failures == ["weather in Paris?"]


async def test_tolerance_allows_slack():
    engine = _engine_with_responses(
        {
            "base:weather in Paris?": GOOD,
            "base:search for llamas": SEARCH,
            "cand:weather in Paris?": GOOD,
            "cand:search for llamas": "no tools today",
        }
    )
    gate = AdapterGate(engine, GOLDEN, tolerance=0.5)
    assert (await gate.evaluate("cand")).accepted


def test_golden_dataset_from_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"prompt": "p1", "expected_tool": "t1", "expected_args": {"a": 1}}\n'
        '{"prompt": "p2", "expected_tool": "t2"}\n'
    )
    ds = GoldenDataset.from_jsonl(path)
    assert len(ds.cases) == 2
    assert ds.cases[0].expected_args == {"a": 1}
    assert ds.cases[1].expected_args is None


def _fake_train(base_model, pairs, output_dir, **hp):
    return output_dir


async def test_learner_rejects_via_gate_end_to_end(tmp_path):
    """CI/CD path: adapter that degrades accuracy never reaches the engine."""
    engine = _engine_with_responses(
        {"base:weather in Paris?": GOOD, "base:search for llamas": SEARCH}
        # candidate adapter answers nothing correctly -> accuracy 0
    )
    loads, unloads = [], []

    async def spy_load(name, path, activate=True):
        loads.append((name, activate))

    async def spy_unload(name):
        unloads.append(name)

    engine.load_lora_adapter = spy_load
    engine.unload_lora_adapter = spy_unload

    buffer = InteractionBuffer(tmp_path / "log.db")
    await buffer.log(OmniMessage(content="q", role=Role.USER, session_id="s"))
    await buffer.log(OmniMessage(content="a", role=Role.ASSISTANT, session_id="s"))

    gate = AdapterGate(engine, GOLDEN)
    learner = ContinuousLearner(
        buffer,
        LoRATrainer("base", tmp_path / "adapters", train_fn=_fake_train),
        engine=engine,
        evaluator=gate.evaluator,
    )
    report = await learner.run_cycle()
    assert report["status"] == "rejected"
    assert report["reason"] == "failed_eval_gate"
    # Shadow gate: loaded invisibly for scoring, never activated, then purged.
    assert loads == [(report["adapter"], False)]
    assert unloads == [report["adapter"]]
    assert engine.active_lora is None
    buffer.close()
