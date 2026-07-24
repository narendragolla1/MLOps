"""Comprehensive tests for StructuredOutput retry logic and validation."""

import json
from unittest import mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from omniai.models.structured import (
    StructuredOutput,
    StructuredOutputError,
    _extract_json,
)


# -- Test Models ---------------------------------------------------


class Person(BaseModel):
    """Simple person model for testing."""

    name: str
    age: int = Field(ge=0, le=150)


class Address(BaseModel):
    """Nested model for testing."""

    street: str
    city: str


class PersonWithAddress(BaseModel):
    """Model with nested structure."""

    name: str
    address: Address


class Config(BaseModel):
    """Model with literal/enum-like fields."""

    mode: str = Field(description="Mode", pattern="^(dev|prod|test)$")
    enabled: bool


# -- JSON Extraction Tests -----------------------------------------------


class TestJsonExtraction:
    """Tests for _extract_json helper function."""

    def test_extract_json_plain_object(self):
        """Verify extraction of plain JSON object."""
        text = '{"name": "Alice", "age": 30}'
        result = _extract_json(text)
        assert result == '{"name": "Alice", "age": 30}'

    def test_extract_json_with_markdown_fence(self):
        """Verify extraction from markdown code fence."""
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        result = _extract_json(text)
        assert result == '{"name": "Alice", "age": 30}'

    def test_extract_json_with_markdown_no_language(self):
        """Verify extraction from markdown without language tag."""
        text = '```\n{"name": "Alice", "age": 30}\n```'
        result = _extract_json(text)
        assert result == '{"name": "Alice", "age": 30}'

    def test_extract_json_with_surrounding_text(self):
        """Verify extraction of JSON from text with surrounding content."""
        text = 'Here is the response:\n{"name": "Alice", "age": 30}\nEnd of response'
        result = _extract_json(text)
        assert result == '{"name": "Alice", "age": 30}'

    def test_extract_json_nested_objects(self):
        """Verify extraction of nested JSON objects."""
        text = '{"name": "Alice", "address": {"city": "NYC", "country": "USA"}}'
        result = _extract_json(text)
        assert result == '{"name": "Alice", "address": {"city": "NYC", "country": "USA"}}'

    def test_extract_json_with_arrays(self):
        """Verify extraction of JSON with arrays."""
        text = '{"name": "Alice", "hobbies": ["reading", "coding"]}'
        result = _extract_json(text)
        assert result == '{"name": "Alice", "hobbies": ["reading", "coding"]}'

    def test_extract_json_no_json(self):
        """Verify extraction when no JSON present."""
        text = 'No JSON here, just text'
        result = _extract_json(text)
        assert result == 'No JSON here, just text'

    def test_extract_json_empty_string(self):
        """Verify extraction from empty string."""
        result = _extract_json('')
        assert result == ''

    def test_extract_json_multiple_objects(self):
        """Verify extraction takes first to last brace."""
        text = '{"name": "Alice"} {"age": 30}'
        result = _extract_json(text)
        # Should extract from first { to last }
        assert result.startswith('{"name"')
        assert result.endswith('}')

    def test_extract_json_with_escaped_quotes(self):
        """Verify extraction handles escaped quotes."""
        text = '{"name": "Alice \\"Developer\\"", "age": 30}'
        result = _extract_json(text)
        assert 'Alice' in result and 'Developer' in result

    def test_extract_json_with_whitespace(self):
        """Verify extraction handles leading/trailing whitespace."""
        text = '  \n\t{"name": "Alice", "age": 30}\n  '
        result = _extract_json(text)
        assert result == '{"name": "Alice", "age": 30}'


# -- StructuredOutput Initialization Tests -----------------------


class TestStructuredOutputInit:
    """Tests for StructuredOutput initialization."""

    async def test_init_stores_parameters(self):
        """Verify init stores model, schema, and max_retries."""
        mock_model = mock.AsyncMock()
        structured = StructuredOutput(mock_model, Person, max_retries=3)
        assert structured.model is mock_model
        assert structured.schema is Person
        assert structured.max_retries == 3

    async def test_init_default_max_retries(self):
        """Verify default max_retries is 2."""
        mock_model = mock.AsyncMock()
        structured = StructuredOutput(mock_model, Person)
        assert structured.max_retries == 2

    async def test_instruction_includes_schema(self):
        """Verify _instruction includes JSON schema."""
        mock_model = mock.AsyncMock()
        structured = StructuredOutput(mock_model, Person)
        instruction = structured._instruction()
        assert instruction["role"] == "system"
        assert "JSON" in instruction["content"]
        assert "name" in instruction["content"]  # Schema fields
        assert "age" in instruction["content"]


# -- Success Path Tests -----------------------------------------------


class TestStructuredOutputSuccess:
    """Tests for successful structured output cases."""

    async def test_invoke_with_valid_json_first_try(self):
        """Verify invoke succeeds with valid JSON on first try."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "Alice", "age": 30}'
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person)
        result = await structured.invoke("Who is Alice?")

        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30
        assert mock_model.generate.call_count == 1

    async def test_invoke_with_string_input(self):
        """Verify invoke converts string to message list."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "Bob", "age": 25}'
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person)
        result = await structured.invoke("Who is Bob?")

        # Check that model.generate was called with proper message structure
        call_args = mock_model.generate.call_args
        messages = call_args[0][0]
        assert any(m["role"] == "user" for m in messages)

    async def test_invoke_with_message_list(self):
        """Verify invoke handles message list input."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "Charlie", "age": 35}'
        mock_model.generate.return_value = mock_response

        messages = [{"role": "user", "content": "Who is Charlie?"}]
        structured = StructuredOutput(mock_model, Person)
        result = await structured.invoke(messages)

        assert result.name == "Charlie"
        assert result.age == 35

    async def test_invoke_with_markdown_json(self):
        """Verify invoke extracts JSON from markdown code fence."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '```json\n{"name": "Diana", "age": 28}\n```'
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person)
        result = await structured.invoke("Who is Diana?")

        assert result.name == "Diana"
        assert result.age == 28

    async def test_invoke_nested_model_valid(self):
        """Verify invoke works with nested models."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "Eve", "address": {"street": "123 Main", "city": "NYC"}}'
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, PersonWithAddress)
        result = await structured.invoke("Where does Eve live?")

        assert result.name == "Eve"
        assert result.address.street == "123 Main"
        assert result.address.city == "NYC"

    async def test_invoke_with_kwargs(self):
        """Verify invoke passes kwargs to model.generate."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "Frank", "age": 40}'
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person)
        result = await structured.invoke("test", temperature=0.5, max_tokens=100)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100


# -- Retry Path Tests -----------------------------------------------


class TestStructuredOutputRetries:
    """Tests for retry logic on validation failures."""

    async def test_invoke_retries_on_invalid_json(self):
        """Verify invoke retries when model returns invalid JSON."""
        mock_model = mock.AsyncMock()
        mock_response_bad = mock.MagicMock()
        mock_response_bad.content = "not valid json"

        mock_response_good = mock.MagicMock()
        mock_response_good.content = '{"name": "Alice", "age": 30}'

        mock_model.generate.side_effect = [mock_response_bad, mock_response_good]

        structured = StructuredOutput(mock_model, Person, max_retries=2)
        result = await structured.invoke("test")

        assert result.name == "Alice"
        # Should be called twice: once for initial, once for retry
        assert mock_model.generate.call_count == 2

    async def test_invoke_retries_on_schema_mismatch(self):
        """Verify invoke retries when schema validation fails."""
        mock_model = mock.AsyncMock()

        # First response: invalid schema (missing required field)
        mock_response_bad = mock.MagicMock()
        mock_response_bad.content = '{"age": 30}'  # missing 'name'

        # Second response: valid schema
        mock_response_good = mock.MagicMock()
        mock_response_good.content = '{"name": "Bob", "age": 25}'

        mock_model.generate.side_effect = [mock_response_bad, mock_response_good]

        structured = StructuredOutput(mock_model, Person, max_retries=2)
        result = await structured.invoke("test")

        assert result.name == "Bob"
        assert mock_model.generate.call_count == 2

    async def test_invoke_retries_on_field_validation_error(self):
        """Verify invoke retries on field validation constraints."""
        mock_model = mock.AsyncMock()

        # First response: age violates constraint (>150)
        mock_response_bad = mock.MagicMock()
        mock_response_bad.content = '{"name": "Charlie", "age": 200}'

        # Second response: valid
        mock_response_good = mock.MagicMock()
        mock_response_good.content = '{"name": "Charlie", "age": 35}'

        mock_model.generate.side_effect = [mock_response_bad, mock_response_good]

        structured = StructuredOutput(mock_model, Person, max_retries=2)
        result = await structured.invoke("test")

        assert result.name == "Charlie"
        assert result.age == 35
        assert mock_model.generate.call_count == 2

    async def test_invoke_multiple_retries(self):
        """Verify invoke can retry multiple times."""
        mock_model = mock.AsyncMock()

        responses = [
            mock.MagicMock(content="invalid"),
            mock.MagicMock(content='{"age": 30}'),  # missing name
            mock.MagicMock(content='{"name": "Dave", "age": 40}'),  # valid
        ]
        mock_model.generate.side_effect = responses

        structured = StructuredOutput(mock_model, Person, max_retries=3)
        result = await structured.invoke("test")

        assert result.name == "Dave"
        assert mock_model.generate.call_count == 3

    async def test_retry_includes_error_feedback(self):
        """Verify retry conversation includes error message."""
        mock_model = mock.AsyncMock()

        mock_response_bad = mock.MagicMock()
        mock_response_bad.content = "not json"

        mock_response_good = mock.MagicMock()
        mock_response_good.content = '{"name": "Eve", "age": 28}'

        mock_model.generate.side_effect = [mock_response_bad, mock_response_good]

        structured = StructuredOutput(mock_model, Person, max_retries=2)
        await structured.invoke("test")

        # Check second call includes error message
        second_call_messages = mock_model.generate.call_args_list[1][0][0]
        error_message_found = any(
            "invalid" in msg.get("content", "").lower()
            for msg in second_call_messages
            if msg.get("role") == "user"
        )
        assert error_message_found


# -- Failure Path Tests -----------------------------------------------


class TestStructuredOutputFailure:
    """Tests for max retries exhaustion."""

    async def test_invoke_exhausts_retries_and_raises(self):
        """Verify invoke raises StructuredOutputError when retries exhausted."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = "always invalid"
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person, max_retries=1)

        with pytest.raises(StructuredOutputError) as exc_info:
            await structured.invoke("test")

        assert "Person" in str(exc_info.value)
        assert "no valid" in str(exc_info.value).lower()
        # Should attempt max_retries + 1 times (1 initial + 1 retry)
        assert mock_model.generate.call_count == 2

    async def test_max_retries_zero_no_retry(self):
        """Verify max_retries=0 means only one attempt."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = "invalid"
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person, max_retries=0)

        with pytest.raises(StructuredOutputError):
            await structured.invoke("test")

        assert mock_model.generate.call_count == 1

    async def test_error_includes_last_error(self):
        """Verify error message includes the last validation error."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "Frank", "age": 500}'  # age too high
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person, max_retries=0)

        with pytest.raises(StructuredOutputError) as exc_info:
            await structured.invoke("test")

        error_msg = str(exc_info.value)
        assert "500" in error_msg or "less than or equal to" in error_msg

    async def test_exhaustion_attempts_count_correct(self):
        """Verify error message shows correct attempt count."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = "bad"
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person, max_retries=3)

        with pytest.raises(StructuredOutputError) as exc_info:
            await structured.invoke("test")

        # 4 attempts: 1 initial + 3 retries
        assert "4 attempts" in str(exc_info.value)


# -- Edge Cases Tests -----------------------------------------------


class TestStructuredOutputEdgeCases:
    """Tests for edge cases and special scenarios."""

    async def test_invoke_with_empty_string_response(self):
        """Verify invoke handles empty string from model."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = ""
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person, max_retries=0)

        with pytest.raises(StructuredOutputError):
            await structured.invoke("test")

    async def test_invoke_with_response_containing_json_prefix(self):
        """Verify invoke extracts JSON after 'json' keyword."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = 'json {"name": "Grace", "age": 32}'
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person)
        result = await structured.invoke("test")

        assert result.name == "Grace"
        assert result.age == 32

    async def test_invoke_with_unicode_content(self):
        """Verify invoke handles unicode in fields."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "青田", "age": 25}'  # Japanese name
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person)
        result = await structured.invoke("test")

        assert result.name == "青田"
        assert result.age == 25

    async def test_invoke_with_special_characters(self):
        """Verify invoke handles special characters in JSON."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "Alice \\"The Great\\"", "age": 30}'
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person)
        result = await structured.invoke("test")

        assert "Alice" in result.name
        assert "The Great" in result.name

    async def test_invoke_schema_with_constraints(self):
        """Verify invoke respects field constraints."""
        mock_model = mock.AsyncMock()

        # First: violates constraint
        mock_response_bad = mock.MagicMock()
        mock_response_bad.content = '{"mode": "invalid", "enabled": true}'

        # Second: valid
        mock_response_good = mock.MagicMock()
        mock_response_good.content = '{"mode": "prod", "enabled": true}'

        mock_model.generate.side_effect = [mock_response_bad, mock_response_good]

        structured = StructuredOutput(mock_model, Config, max_retries=1)
        result = await structured.invoke("test")

        assert result.mode == "prod"
        assert result.enabled is True

    async def test_conversation_accumulation(self):
        """Verify conversation accumulates through retries."""
        mock_model = mock.AsyncMock()

        mock_response_bad = mock.MagicMock()
        mock_response_bad.content = "bad"

        mock_response_good = mock.MagicMock()
        mock_response_good.content = '{"name": "Hank", "age": 45}'

        mock_model.generate.side_effect = [mock_response_bad, mock_response_good]

        structured = StructuredOutput(mock_model, Person, max_retries=2)
        await structured.invoke("test")

        # Second call should have more messages than first
        first_messages = mock_model.generate.call_args_list[0][0][0]
        second_messages = mock_model.generate.call_args_list[1][0][0]
        assert len(second_messages) > len(first_messages)

    async def test_instruction_always_first(self):
        """Verify schema instruction is always first message."""
        mock_model = mock.AsyncMock()
        mock_response = mock.MagicMock()
        mock_response.content = '{"name": "Ivy", "age": 27}'
        mock_model.generate.return_value = mock_response

        structured = StructuredOutput(mock_model, Person)
        await structured.invoke("test")

        messages = mock_model.generate.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "JSON" in messages[0]["content"]
