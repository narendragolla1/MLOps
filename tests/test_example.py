import pathlib


def test_example_is_under_50_lines():
    source = (pathlib.Path(__file__).parent.parent / "examples" / "basic_agent.py").read_text()
    code_lines = [ln for ln in source.splitlines() if ln.strip()]
    assert len(code_lines) <= 50


def test_example_imports_and_wires(monkeypatch):
    import runpy

    module = runpy.run_path(
        str(pathlib.Path(__file__).parent.parent / "examples" / "basic_agent.py"),
        run_name="example",
    )
    router = module["router"]
    engine = module["engine"]
    assert "concise-helper" in engine.system_prompt
    assert "get_weather" in engine.system_prompt
    assert module["buffer"].threshold == 1000
    assert router.interceptors and router.observers
    module["buffer"].close()
    pathlib.Path("interactions.db").unlink(missing_ok=True)
