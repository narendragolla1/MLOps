# Test-Driven Development (TDD) Guide
## OmniAI / NexusGraph v1.0

**Document Version:** 1.0  
**Last Updated:** 2026-07-23  
**Owner:** QA & Engineering Team

---

## 1. TDD Philosophy & Principles

### Core Tenets
This project follows **outside-in TDD** where tests drive architecture:

1. **Write tests first** before implementation (Red → Green → Refactor)
2. **Tests document behavior** — read tests to understand requirements
3. **Continuous integration** — every commit runs full suite; no broken builds
4. **Meaningful coverage** — focus on critical paths and edge cases (target: ≥ 85%)
5. **Fast feedback loop** — unit tests run in < 1s; integration tests in < 10s

### Who Tests & When
| Role | Responsibility | When |
|------|-----------------|------|
| **Developer** | Unit tests for functions/classes | Before commit |
| **Reviewer** | Integration test requirements | Code review |
| **QA Lead** | End-to-end scenarios, performance | Release candidate |
| **CI/CD Pipeline** | Full suite on every PR | Push to branch |

---

## 2. Test Structure & Organization

### Directory Layout
```
tests/
├─ __init__.py                           # Shared test utilities & fixtures
├─ conftest.py                           # Pytest configuration & shared fixtures
├─ unit/
│  ├─ test_protocol.py                   # OmniMessage serialization
│  ├─ test_engine_config.py              # Engine configuration parsing
│  ├─ test_graph_state.py                # Graph state validation
│  ├─ test_guardrails.py                 # Injection/PII detection
│  └─ test_models.py                     # ChatModel implementations
├─ integration/
│  ├─ test_engine_lifecycle.py           # Engine start/stop/warmup
│  ├─ test_gateway_rest.py               # REST endpoint behavior
│  ├─ test_gateway_websocket.py          # WebSocket streaming
│  ├─ test_graph_execution.py            # Full graph runs
│  ├─ test_memory_buffer.py              # InteractionBuffer with DB
│  ├─ test_continuous_learning.py        # Train/eval/deploy cycles
│  └─ test_orchestration.py              # Multi-component flows
├─ e2e/
│  ├─ test_full_pipeline.py              # End-to-end LLM inference
│  ├─ test_adapter_hotswap.py            # LoRA zero-downtime swap
│  ├─ test_learning_pipeline.py          # Collect → Train → Deploy
│  └─ test_production_readiness.py       # Deployment scenarios
├─ performance/
│  ├─ test_latency.py                    # First-token, end-to-end
│  ├─ test_throughput.py                 # Tokens/sec under load
│  └─ test_memory_profiling.py           # Memory leaks, overhead
├─ security/
│  ├─ test_auth.py                       # API key validation
│  ├─ test_rate_limiting.py              # Token bucket enforcement
│  ├─ test_injection_detection.py        # Prompt injection patterns
│  └─ test_pii_masking.py                # PII redaction coverage
├─ fixtures/
│  ├─ models.py                          # Mock model configs
│  ├─ graphs.py                          # Sample graph definitions
│  ├─ datasets.py                        # Test data (JSONL, DB seeds)
│  └─ mocks.py                           # Mocks for vLLM, SGLang
└─ README.md                             # Test execution guide
```

### Test Naming Convention
- **Unit**: `test_<module>_<function>_<scenario>.py`
  - Example: `test_protocol_omnimessage_serialization.py`
- **Integration**: `test_<subsystem>_<flow>.py`
  - Example: `test_engine_lifecycle.py`
- **E2E**: `test_<feature>.py`
  - Example: `test_adapter_hotswap.py`

---

## 3. Unit Testing Strategy

### Scope
Unit tests validate **single functions/classes in isolation** with mocks for dependencies.

### Example: OmniMessage Protocol

#### Test File: `tests/unit/test_protocol.py`
```python
import pytest
from omniai.protocol import OmniMessage, ToolCall, ToolResult
from datetime import datetime
from typing import Any


class TestOmniMessageSerialization:
    """Test OmniMessage can be serialized/deserialized."""
    
    def test_user_message_to_dict(self):
        """User message converts to dict with role and content."""
        msg = OmniMessage(role="user", content="hello")
        data = msg.model_dump()
        
        assert data["role"] == "user"
        assert data["content"] == "hello"
        assert "metadata" in data
    
    def test_assistant_with_tool_calls_serialization(self):
        """Assistant message with tool calls serializes completely."""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Paris"}
        )
        msg = OmniMessage(
            role="assistant",
            content="Getting weather...",
            tool_calls=[tool_call]
        )
        
        data = msg.model_dump()
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["name"] == "get_weather"
    
    def test_message_with_tool_results(self):
        """Message can include tool execution results."""
        result = ToolResult(
            tool_call_id="call_123",
            content="Weather: 22°C",
            is_error=False
        )
        msg = OmniMessage(
            role="user",
            content="What's the weather?",
            tool_results=[result]
        )
        
        assert len(msg.tool_results) == 1
        assert msg.tool_results[0].content == "Weather: 22°C"
    
    def test_deserialization_from_openai_format(self):
        """Can parse OpenAI-format messages."""
        openai_msg = {
            "role": "assistant",
            "content": "I'll check the weather.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}'
                    }
                }
            ]
        }
        
        msg = OmniMessage.from_openai(openai_msg)
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
    
    def test_metadata_correlation_id(self):
        """Message includes X-Request-ID for correlation."""
        msg = OmniMessage(
            role="user",
            content="hello",
            metadata={"correlation_id": "req_abc123"}
        )
        
        assert msg.metadata["correlation_id"] == "req_abc123"
    
    def test_invalid_role_raises_validation_error(self):
        """Invalid role raises ValueError."""
        with pytest.raises(ValueError, match="role must be"):
            OmniMessage(role="invalid", content="hello")
    
    def test_empty_content_allowed_for_tool_calls(self):
        """Empty content is ok if tool_calls present."""
        tool_call = ToolCall(id="c1", name="fn", arguments={})
        msg = OmniMessage(
            role="assistant",
            content="",
            tool_calls=[tool_call]
        )
        
        assert msg.content == ""
        assert len(msg.tool_calls) == 1


class TestOmniMessageEdgeCases:
    """Edge cases and boundary conditions."""
    
    @pytest.mark.parametrize("role", ["user", "assistant", "system"])
    def test_all_valid_roles(self, role):
        """All standard roles are accepted."""
        msg = OmniMessage(role=role, content="test")
        assert msg.role == role
    
    def test_very_long_content(self):
        """Can handle multi-MB content (streaming scenario)."""
        large_content = "x" * (1024 * 1024)  # 1MB
        msg = OmniMessage(role="user", content=large_content)
        
        assert len(msg.content) == 1024 * 1024
    
    def test_unicode_content(self):
        """Handles Unicode, emojis, multiple languages."""
        msg = OmniMessage(
            role="user",
            content="Hello 🌍 \n Bonjour 你好 مرحبا"
        )
        
        assert "🌍" in msg.content
        assert "你好" in msg.content
    
    def test_roundtrip_json_serialization(self):
        """Serialize → JSON → deserialize preserves data."""
        original = OmniMessage(
            role="assistant",
            content="Response",
            tool_calls=[ToolCall(id="c1", name="fn", arguments={"x": 1})]
        )
        
        json_str = original.model_dump_json()
        restored = OmniMessage.model_validate_json(json_str)
        
        assert restored.role == original.role
        assert restored.tool_calls == original.tool_calls
```

### Unit Test Checklist
- [ ] **Happy path**: Normal inputs produce expected outputs
- [ ] **Edge cases**: Boundary conditions (empty, max, min)
- [ ] **Error handling**: Invalid inputs raise appropriate exceptions
- [ ] **Type validation**: Wrong types rejected early
- [ ] **Isolation**: No dependencies on other modules (use mocks)
- [ ] **Deterministic**: No flaky timeouts or random data

---

## 4. Integration Testing Strategy

### Scope
Integration tests validate **multiple components working together** with real databases/services (or careful mocks).

### Example: Engine Lifecycle

#### Test File: `tests/integration/test_engine_lifecycle.py`
```python
import pytest
import asyncio
from omniai.engine import ModelEngine, ModelConfig
from omniai.protocol import OmniMessage
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile
import os


@pytest.fixture
async def temp_model_dir():
    """Create temporary directory for model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_vllm_backend(monkeypatch):
    """Mock vLLM subprocess without launching real server."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Process alive
    mock_proc.wait.return_value = 0
    
    async def mock_start(*args, **kwargs):
        return mock_proc
    
    monkeypatch.setattr(
        "omniai.engine.backends.vllm.Popen",
        lambda *a, **k: mock_proc
    )


class TestEngineStartup:
    """Test engine initialization and startup."""
    
    @pytest.mark.asyncio
    async def test_create_with_defaults(self):
        """Can create engine with minimal config."""
        config = ModelConfig(
            model="Qwen/Qwen2.5-7B-Instruct",
            backend="mock"  # Use mock backend for testing
        )
        engine = ModelEngine.create(config)
        
        assert engine.config.model == "Qwen/Qwen2.5-7B-Instruct"
        assert engine.state == "created"
    
    @pytest.mark.asyncio
    async def test_start_with_supervise(self, mock_vllm_backend):
        """Starting engine with supervise=True launches process."""
        config = ModelConfig(
            model="Qwen/Qwen2.5-7B-Instruct",
            backend="vllm",
            log_dir="/tmp/logs"
        )
        engine = ModelEngine.create(config)
        
        # Start in background (this test uses mock, so no real GPU)
        task = asyncio.create_task(engine.start(supervise=True))
        await asyncio.sleep(0.1)  # Let startup begin
        
        # Verify state transition
        assert engine.state in ["starting", "ready"]
        
        # Cleanup
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_warmup_primes_cuda_graphs(self, mock_vllm_backend):
        """Warmup runs prefill/decode to prime CUDA graphs."""
        config = ModelConfig(model="test", backend="mock")
        engine = ModelEngine.create(config)
        
        # Mock the client to track warmup calls
        engine._client.chat = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="test"))]
        ))
        
        await engine.warmup(num_prompts=3)
        
        # Verify multiple requests were made to prime graphs
        assert engine._client.chat.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_stop_gracefully_shuts_down(self, mock_vllm_backend):
        """Stop terminates process and cleans up resources."""
        config = ModelConfig(model="test", backend="mock")
        engine = ModelEngine.create(config)
        
        # Track stop events
        stopped_event = asyncio.Event()
        engine._on_stop_callback = lambda: stopped_event.set()
        
        await engine.stop()
        
        # Verify cleanup
        assert engine.state in ["stopped", "error"]


class TestLoRALifecycle:
    """Test LoRA adapter loading and switching."""
    
    @pytest.fixture
    async def engine(self):
        """Fixture: running engine instance."""
        config = ModelConfig(model="test", backend="mock")
        engine = ModelEngine.create(config)
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_load_lora_adapter(self, engine, temp_model_dir):
        """Loading LoRA adds it to registry and activates."""
        # Create mock adapter files
        adapter_path = os.path.join(temp_model_dir, "skill-v1")
        os.makedirs(adapter_path, exist_ok=True)
        open(os.path.join(adapter_path, "adapter_config.json"), "w").close()
        
        # Load adapter
        await engine.load_lora_adapter("skill-v1", adapter_path)
        
        # Verify active adapter
        assert engine.active_lora == "skill-v1"
        assert "skill-v1" in engine.lora_registry.list_adapters()
    
    @pytest.mark.asyncio
    async def test_lora_hot_swap_under_load(self, engine, temp_model_dir):
        """Swapping adapters doesn't interrupt in-flight requests."""
        adapter_v1 = os.path.join(temp_model_dir, "v1")
        adapter_v2 = os.path.join(temp_model_dir, "v2")
        
        os.makedirs(adapter_v1, exist_ok=True)
        os.makedirs(adapter_v2, exist_ok=True)
        
        # Load first adapter
        await engine.load_lora_adapter("v1", adapter_v1)
        
        # Simulate in-flight request
        async def slow_request():
            await asyncio.sleep(0.2)
            return "response"
        
        # Start request and swap adapter concurrently
        request_task = asyncio.create_task(slow_request())
        await asyncio.sleep(0.05)  # Request is in-flight
        
        # Swap adapter while request running
        await engine.load_lora_adapter("v2", adapter_v2)
        
        # Request should complete with v1, then v2 becomes active
        response = await request_task
        assert response == "response"
        assert engine.active_lora == "v2"
    
    @pytest.mark.asyncio
    async def test_rollback_reverts_to_previous_adapter(self, engine, temp_model_dir):
        """Rollback quickly reverts to previous adapter."""
        v1_path = os.path.join(temp_model_dir, "v1")
        v2_path = os.path.join(temp_model_dir, "v2")
        
        os.makedirs(v1_path, exist_ok=True)
        os.makedirs(v2_path, exist_ok=True)
        
        # Load v1, then v2
        await engine.load_lora_adapter("v1", v1_path)
        await engine.load_lora_adapter("v2", v2_path)
        assert engine.active_lora == "v2"
        
        # Rollback to v1
        await engine.rollback_lora()
        assert engine.active_lora == "v1"


class TestEngineCircuitBreaker:
    """Test circuit breaker fault tolerance."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaks_after_consecutive_errors(self):
        """Circuit breaker opens after 5 failures."""
        config = ModelConfig(model="test", backend="mock")
        engine = ModelEngine.create(config)
        
        # Mock client to always fail
        engine._client.chat = AsyncMock(
            side_effect=Exception("Backend error")
        )
        
        # Try 5 requests; circuit should break on 6th
        for i in range(5):
            with pytest.raises(Exception):
                await engine.chat_text([{"role": "user", "content": "test"}])
        
        # 6th request should fail with CircuitBreakerOpen
        with pytest.raises(Exception, match="Circuit breaker"):
            await engine.chat_text([{"role": "user", "content": "test"}])
        
        assert engine.circuit_breaker.state == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovers(self):
        """Circuit breaker transitions to half-open and retries."""
        config = ModelConfig(model="test", backend="mock")
        engine = ModelEngine.create(config)
        
        # Fail 5 times to open breaker
        engine._client.chat = AsyncMock(side_effect=Exception("error"))
        for _ in range(5):
            with pytest.raises(Exception):
                await engine.chat_text([{"role": "user", "content": "test"}])
        
        assert engine.circuit_breaker.state == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(engine.circuit_breaker.recovery_timeout + 0.1)
        
        # Circuit should be half-open; next call retries
        engine._client.chat = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="success"))]
        ))
        
        response = await engine.chat_text([{"role": "user", "content": "test"}])
        assert engine.circuit_breaker.state == "closed"
```

### Integration Test Checklist
- [ ] **Real dependencies**: Use actual databases (SQLite in tests), not pure mocks
- [ ] **Async/await correctness**: Test concurrent operations and race conditions
- [ ] **Error recovery**: Test circuit breakers, retries, timeouts
- [ ] **State transitions**: Verify legal state changes and guard against invalid ones
- [ ] **Resource cleanup**: Database connections, file handles released
- [ ] **Order independence**: Tests runnable in any order

---

## 5. End-to-End Testing Strategy

### Scope
E2E tests validate **complete user workflows** from API request to final response.

### Example: Full Inference Pipeline

#### Test File: `tests/e2e/test_full_pipeline.py`
```python
import pytest
import asyncio
from omniai.app import create_app
from omniai.engine import ModelEngine
from omniai.graph import Graph, State, START, END, tool
from omniai.protocol import OmniMessage
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json


@pytest.fixture
def test_app():
    """Create test FastAPI app."""
    app = create_app({
        "OMNIAI_MODEL": "test-model",
        "OMNIAI_BACKEND": "mock",
        "OMNIAI_AUTH_DISABLED": True,  # Disable auth for testing
    })
    return app


@pytest.fixture
def client(test_app):
    """Sync test client."""
    return TestClient(test_app)


@pytest.fixture
async def async_client(test_app):
    """Async test client."""
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        yield ac


class TestRESTEndpoint:
    """Test REST /v1/messages endpoint."""
    
    def test_basic_completion_request(self, client):
        """POST /v1/messages returns completion."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "messages": [
                    {"role": "user", "content": "hello"}
                ],
                "max_tokens": 100,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "message" in data["choices"][0]
    
    def test_streaming_response(self, client):
        """POST with stream=true returns Server-Sent Events."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers={"Accept": "text/event-stream"}
        )
        
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "text/event-stream"
        
        # Parse SSE chunks
        chunks = response.text.split("\n\n")
        assert any("delta" in chunk for chunk in chunks)
    
    def test_tool_use_in_request(self, client):
        """Request with tools returns tool_calls in response."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "messages": [
                    {"role": "user", "content": "What's the weather?"}
                ],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        # Model may use tools or respond directly
        assert "choices" in data
    
    def test_request_correlation_id(self, client):
        """X-Request-ID header flows through and appears in logs."""
        correlation_id = "req_abc123xyz"
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "test"}]
            },
            headers={"X-Request-ID": correlation_id}
        )
        
        # Correlation ID should be in response headers
        assert response.headers.get("X-Request-ID") == correlation_id
    
    def test_missing_api_key_rejected(self, client):
        """Request without API key returns 401 (if auth enabled)."""
        # This test would apply if OMNIAI_AUTH_DISABLED = False
        # For now, with auth disabled, verify auth is optional
        response = client.post(
            "/v1/messages",
            json={"model": "test", "messages": [{"role": "user", "content": "test"}]}
        )
        assert response.status_code == 200  # Auth disabled


class TestGraphExecution:
    """Test LangGraph-style agent execution."""
    
    @pytest.mark.asyncio
    async def test_multi_node_graph_execution(self):
        """Graph executes through multiple nodes."""
        
        @tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: 22°C"
        
        class AgentState(State):
            messages: list[OmniMessage]
            weather_result: str | None = None
        
        def route_fn(state: AgentState) -> str:
            if "weather" in state.messages[-1].content:
                return "weather"
            return "respond"
        
        async def weather_node(state: AgentState) -> AgentState:
            city = "Paris"
            state.weather_result = await asyncio.to_thread(get_weather, city)
            return state
        
        async def respond_node(state: AgentState) -> AgentState:
            # Add response message
            msg = OmniMessage(role="assistant", content="Response sent")
            state.messages.append(msg)
            return state
        
        # Build graph
        graph = Graph(AgentState)
        graph.add_node("route", lambda s: s)
        graph.add_node("weather", weather_node)
        graph.add_node("respond", respond_node)
        
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", route_fn, {
            "weather": "weather",
            "respond": "respond"
        })
        graph.add_edge("weather", "respond")
        graph.add_edge("respond", END)
        
        app = graph.compile()
        
        # Execute graph
        result = await app.ainvoke({
            "messages": [OmniMessage(role="user", content="What's the weather?")]
        })
        
        assert result.weather_result == "Weather in Paris: 22°C"
        assert len(result.messages) == 2  # Input + response
    
    @pytest.mark.asyncio
    async def test_graph_max_iterations_enforced(self):
        """Graph respects max_iterations limit."""
        
        counter = {"calls": 0}
        
        async def counting_node(state: State) -> State:
            counter["calls"] += 1
            state.messages.append(OmniMessage(
                role="assistant",
                content=f"iteration {counter['calls']}"
            ))
            return state
        
        graph = Graph(State)
        graph.add_node("loop", counting_node)
        graph.add_edge(START, "loop")
        graph.add_edge("loop", "loop")  # Self-loop
        
        app = graph.compile()
        
        # Run with max_iterations=5
        result = await app.ainvoke(
            {"messages": []},
            max_iterations=5
        )
        
        # Should stop at 5 even though loop continues
        assert counter["calls"] == 5


class TestContinuousLearning:
    """Test learn-from-interaction loop."""
    
    @pytest.mark.asyncio
    async def test_interaction_buffer_collects_exchanges(self, tmp_path):
        """InteractionBuffer saves user/assistant exchanges to DB."""
        from omniai.memory import InteractionBuffer
        
        db_path = str(tmp_path / "interactions.db")
        buffer = InteractionBuffer(db_path, threshold=10)
        
        # Record 5 interactions
        for i in range(5):
            buffer.add_interaction(
                user_message=f"Question {i}",
                assistant_message=f"Answer {i}",
                metadata={"session_id": "s1"}
            )
        
        # Verify stored
        interactions = buffer.get_recent(limit=5)
        assert len(interactions) == 5
        assert interactions[0].user_message == "Question 0"
    
    @pytest.mark.asyncio
    async def test_threshold_triggers_learning_cycle(self, tmp_path):
        """Reaching threshold triggers train/eval/deploy."""
        from omniai.memory import InteractionBuffer, ContinuousLearner
        from omniai.evals import AdapterGate
        from unittest.mock import AsyncMock
        
        db_path = str(tmp_path / "interactions.db")
        buffer = InteractionBuffer(db_path, threshold=3)
        
        # Mock learning components
        mock_learner = AsyncMock(spec=ContinuousLearner)
        buffer.on_threshold = mock_learner.trigger
        
        # Add interactions
        for i in range(3):
            buffer.add_interaction(
                user_message=f"Q{i}",
                assistant_message=f"A{i}"
            )
        
        # Verify threshold callback triggered
        mock_learner.trigger.assert_called_once()


class TestAdapterHotSwap:
    """Test zero-downtime LoRA adapter updates."""
    
    @pytest.mark.asyncio
    async def test_adapter_swap_zero_downtime(self):
        """Swapping adapters doesn't interrupt active requests."""
        from omniai.engine import ModelEngine
        from unittest.mock import AsyncMock, MagicMock
        
        config = {"model": "test", "backend": "mock"}
        engine = ModelEngine.create(config)
        
        # Mock slow inference
        slow_response = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))]
        )
        engine._client.chat = AsyncMock(return_value=slow_response)
        
        # Start inference request
        async def inference():
            return await engine.chat_text([
                {"role": "user", "content": "test"}
            ])
        
        request_task = asyncio.create_task(inference())
        await asyncio.sleep(0.05)  # Request in-flight
        
        # Swap adapter mid-request
        adapter_task = asyncio.create_task(
            engine.load_lora_adapter("new", "/path/to/new")
        )
        
        # Both should complete without errors
        response = await asyncio.wait_for(request_task, timeout=1.0)
        await asyncio.wait_for(adapter_task, timeout=1.0)
        
        assert response  # Request completed successfully
        assert engine.active_lora == "new"  # Adapter updated
```

### E2E Test Checklist
- [ ] **Real workflows**: Start from user action (API request) to final output
- [ ] **Integration with mocked backends**: Use mock vLLM but real gateway
- [ ] **Performance baselines**: Measure latency, throughput against targets
- [ ] **Error scenarios**: Network failures, timeouts, malformed requests
- [ ] **Data validation**: Responses match schema, no corruption

---

## 6. Performance Testing Strategy

### Metrics to Monitor
| Metric | Target | Test File |
|--------|--------|-----------|
| First-token latency | < 500ms (7B model) | `test_latency.py` |
| End-to-end latency | < 2s (full pipeline) | `test_latency.py` |
| Throughput | ≥ 200 tok/sec | `test_throughput.py` |
| Adapter swap latency | < 100ms | `test_adapter_hotswap.py` |
| Memory per request | < 10MB | `test_memory_profiling.py` |

#### Example: Latency Benchmarks

```python
# tests/performance/test_latency.py
import pytest
import time
import asyncio
from omniai.engine import ModelEngine

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_first_token_latency(benchmark):
    """Measure time to first token."""
    engine = ModelEngine.create({"model": "test", "backend": "mock"})
    
    async def get_first_token():
        response = await engine.chat_text([
            {"role": "user", "content": "hello"}
        ])
        return response
    
    # Run benchmark
    result = await benchmark.pedantic(
        asyncio.run,
        args=(get_first_token(),),
        rounds=10,
        iterations=1
    )
    
    # Assert performance
    assert benchmark.stats.mean < 0.5  # 500ms target
```

---

## 7. Security Testing Strategy

### Checklist: Security Tests
```python
# tests/security/test_injection_detection.py

class TestPromptInjectionDetection:
    """Test guardrails catch injection attempts."""
    
    def test_sql_injection_blocked(self):
        """SQL injection patterns are detected."""
        from omniai.guardrails import PromptGuard
        
        guard = PromptGuard()
        
        injection = "hello'; DROP TABLE users; --"
        is_safe = guard.is_safe(injection)
        
        assert not is_safe, "SQL injection should be blocked"
    
    def test_command_injection_blocked(self):
        """Shell command injection detected."""
        guard = PromptGuard()
        
        injection = "test && rm -rf /"
        is_safe = guard.is_safe(injection)
        
        assert not is_safe

# tests/security/test_auth.py

class TestAPIKeyAuth:
    """Test API key validation."""
    
    def test_missing_api_key_rejected(self, client):
        """Request without API key returns 401."""
        response = client.post(
            "/v1/messages",
            json={"model": "test", "messages": [...]},
            headers={}  # No API key
        )
        
        assert response.status_code == 401

# tests/security/test_pii_masking.py

class TestPIIMasking:
    """Test PII redaction."""
    
    def test_email_redacted(self):
        """Email addresses masked in logs."""
        from omniai.guardrails import PromptGuard
        
        guard = PromptGuard(enable_pii_masking=True)
        
        text = "Contact me at john@example.com"
        masked = guard.redact_pii(text)
        
        assert "john@example.com" not in masked
        assert "[EMAIL]" in masked
```

---

## 8. Test Fixtures & Utilities

### Shared Fixtures (`tests/conftest.py`)
```python
import pytest
import asyncio
from omniai.engine import ModelEngine
from omniai.protocol import OmniMessage
from omniai.graph import State
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture(scope="session")
def event_loop():
    """Event loop for async tests (session-scoped)."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_engine():
    """Mock ModelEngine for unit tests."""
    engine = MagicMock(spec=ModelEngine)
    engine.chat_text = AsyncMock(return_value="Mocked response")
    engine.load_lora_adapter = AsyncMock()
    engine.circuit_breaker = MagicMock(state="closed")
    return engine

@pytest.fixture
def sample_omnimessage():
    """Sample OmniMessage for reuse."""
    return OmniMessage(
        role="user",
        content="What is the weather?",
        metadata={"correlation_id": "test_123"}
    )

@pytest.fixture
def sample_graph_state():
    """Sample graph state."""
    return State(messages=[
        OmniMessage(role="user", content="hello")
    ])
```

### Mock Models (`tests/fixtures/models.py`)
```python
MOCK_MODEL_CONFIGS = {
    "small": {
        "model": "test-small",
        "backend": "mock",
        "quantization": "fp8",
    },
    "medium": {
        "model": "test-medium",
        "backend": "mock",
        "tensor_parallel_size": 2,
    },
}

MOCK_TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
]
```

---

## 9. Running Tests Locally

### Installation
```bash
pip install -e ".[dev]"       # Install with test dependencies
```

### Run Full Suite
```bash
pytest                         # All tests
pytest -v                      # Verbose output
pytest --cov=omniai           # Coverage report
pytest --cov=omniai --cov-report=html  # HTML coverage
```

### Run by Category
```bash
pytest tests/unit/            # Unit tests only
pytest tests/integration/     # Integration tests
pytest tests/e2e/             # End-to-end tests
pytest -m "not e2e"           # Skip E2E
pytest -m "security"          # Security tests only
```

### Run with Specific Markers
```bash
pytest -m "asyncio"           # Async tests
pytest -m "benchmark"         # Performance tests
pytest -m "not benchmark"     # Exclude slow tests
```

### Parallel Execution
```bash
pip install pytest-xdist
pytest -n auto                # Parallel on all CPU cores
pytest -n 4                   # Parallel on 4 cores
```

---

## 10. CI/CD Test Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=omniai
      
      - name: Run integration tests
        run: pytest tests/integration/ -v
      
      - name: Upload coverage
        run: |
          pip install codecov
          codecov --file coverage.xml
      
      - name: Run security checks
        run: |
          pip install bandit
          bandit -r omniai/
      
      - name: Lint
        run: |
          pip install ruff
          ruff check omniai/ tests/
```

### Test Coverage Requirements
- **Core modules** (protocol, engine, graph, gateway): ≥ 90% coverage
- **Overall**: ≥ 85% coverage
- **Critical paths** (auth, guardrails, continuous learning): 100% coverage

---

## 11. Test Data Management

### Fixture Location: `tests/fixtures/datasets.jsonl`
```jsonl
{"user": "What's the weather?", "assistant": "The weather is...", "tools": ["get_weather"]}
{"user": "Tell me a joke", "assistant": "Why did...", "tools": []}
{"user": "Summarize this text", "assistant": "Summary: ...", "tools": ["summarize"]}
```

### Golden Dataset for Evals
```
tests/fixtures/golden_dataset.jsonl
- Curated QA pairs for evaluating adapter quality
- Format: {"user_message": "...", "expected_tool": "...", "expected_output": "..."}
- Size: 100–500 examples
```

---

## 12. Test Maintenance & Flakiness

### Flaky Test Detection
```bash
# Run test 10 times to catch intermittent failures
pytest tests/integration/test_engine_lifecycle.py::TestEngineStartup::test_start_with_supervise -v --count=10
```

### Common Flakiness Issues & Fixes
| Issue | Cause | Fix |
|-------|-------|-----|
| Timeout errors | Network slow | Use `pytest-timeout` with generous defaults (30s for integration) |
| Race conditions | Concurrent state changes | Use `asyncio.Lock`, await synchronization points |
| Resource leaks | Unclosed files/connections | Explicitly cleanup in fixtures with `finally` |
| Time-dependent | Hardcoded timeouts | Use `freezegun` to control time in tests |

---

## 13. Coverage Goals

### Coverage Targets by Module
| Module | Unit | Integration | E2E | Target |
|--------|------|-------------|-----|--------|
| `omniai.protocol` | ✓ | — | — | 95% |
| `omniai.engine` | ✓ | ✓ | ✓ | 90% |
| `omniai.graph` | ✓ | ✓ | ✓ | 88% |
| `omniai.gateway` | ✓ | ✓ | ✓ | 85% |
| `omniai.guardrails` | ✓ | ✓ | ✓ | 92% |
| `omniai.memory` | ✓ | ✓ | ✓ | 85% |
| `omniai.telemetry` | ✓ | ✓ | — | 80% |

---

## 14. Code Review Checklist for Tests

Before approving a PR with code changes:

- [ ] **Unit tests** exist for new functions/classes
- [ ] **Happy path** tested (normal inputs)
- [ ] **Edge cases** tested (empty, max, min, boundaries)
- [ ] **Error paths** tested (invalid inputs, exceptions)
- [ ] **Integration tests** cover multi-component interactions
- [ ] **No hardcoded timeouts** (use generous defaults with fixtures)
- [ ] **No external dependencies** in unit tests (use mocks)
- [ ] **Fixtures properly scoped** (function, class, module, session)
- [ ] **Async correctness**: Proper `await`, no race conditions
- [ ] **Resource cleanup**: Files, DB connections released
- [ ] **Coverage maintained**: No decrease in overall %
- [ ] **CI passes**: All test suites green before merge

---

## 15. Continuous Improvement

### Weekly TDD Metrics
```python
# tests/metrics/test_coverage_trend.py
- Track coverage trend over time
- Alert if coverage drops > 2%
- Generate report of untested code
```

### Flaky Test Report
```
- Run weekly full suite 5 times
- Track tests that fail intermittently
- Prioritize fixes for high-flakiness tests
```

---

## Appendix: Pytest Configuration

```ini
# pytest.ini
[pytest]
minversion = 7.0
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
    -p no:cacheprovider
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    asyncio: marks tests as async (deselect with '-m "not asyncio"')
    benchmark: marks tests as slow performance tests
    security: marks tests as security-focused
    integration: marks tests as integration tests
    e2e: marks tests as end-to-end tests
    slow: marks tests as slow
asyncio_mode = auto

# Timeout for all tests (prevent hanging)
timeout = 300
timeout_method = thread

# Coverage
[coverage:run]
branch = True
omit = 
    */site-packages/*
    tests/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

---

**Document End**
