import pytest

from omniai.graph import Tool, tool
from omniai.graph.tools import ToolValidationError, render_tool_prompt


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"22 {unit} in {city}"


def test_decorator_produces_tool():
    assert isinstance(get_weather, Tool)
    assert get_weather.name == "get_weather"
    assert "weather" in get_weather.description.lower()


def test_schema_from_type_hints():
    schema = get_weather.json_schema
    assert schema["properties"]["city"]["type"] == "string"
    assert schema["required"] == ["city"]  # unit has a default
    assert schema["properties"]["unit"]["default"] == "celsius"


def test_openai_tool_definition():
    spec = get_weather.to_openai()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "get_weather"
    assert "parameters" in spec["function"]


async def test_execute_validates_and_runs():
    assert await get_weather.execute({"city": "Paris"}) == "22 celsius in Paris"
    assert await get_weather.execute('{"city": "Oslo", "unit": "F"}') == "22 F in Oslo"


async def test_execute_rejects_bad_args():
    with pytest.raises(ToolValidationError):
        await get_weather.execute({})  # missing required city
    with pytest.raises(ToolValidationError):
        await get_weather.execute("not json {")


async def test_type_coercion_and_rejection():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert await add.execute({"a": "2", "b": 3}) == 5  # pydantic coerces "2"
    with pytest.raises(ToolValidationError):
        await add.execute({"a": "not-a-number", "b": 3})


async def test_async_tools():
    @tool(name="fetcher", description="Fetch a URL.")
    def fetch(url: str) -> str:
        return f"content of {url}"

    assert fetch.name == "fetcher"

    @tool
    async def slow(x: int) -> int:
        """Double slowly."""
        return x * 2

    assert await slow.execute({"x": 4}) == 8


def test_render_tool_prompt_mentions_schema():
    prompt = render_tool_prompt([get_weather])
    assert "get_weather" in prompt
    assert "JSON" in prompt


def test_plain_call_still_works():
    assert get_weather("Rome") == "22 celsius in Rome"
