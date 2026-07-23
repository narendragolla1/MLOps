"""OmniAI end-to-end demo: engine + skills + graph + tools + gateway + learning.

Run:  python examples/basic_agent.py   then POST to http://localhost:8080/v1/messages
"""
import json, pathlib, uvicorn
from omniai.engine import ModelEngine
from omniai.gateway import GatewayRouter
from omniai.graph import END, START, Graph, State, tool
from omniai.graph.tools import render_tool_prompt
from omniai.guardrails import PromptGuard
from omniai.memory import ContinuousLearner, InteractionBuffer, LoRATrainer, SkillLoader

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"22C and sunny in {city}"

# 1. Engine: vLLM backend with fp8 quantization (start() launches the server).
engine = ModelEngine.create({"model": "Qwen/Qwen2.5-7B-Instruct", "backend": "vllm",
                             "quantization": "fp8", "tensor_parallel_size": 1})

# 2. Skills: parsed from skill.md files and pre-cached into the system prompt.
loader = SkillLoader()
loader.load_directory(pathlib.Path(__file__).parent / "skills")
engine.set_system_prompt(loader.compose_system_prompt() + "\n\n" + render_tool_prompt([get_weather]))

# 3. Graph: think -> (maybe act) -> respond, with a lambda-routed edge.
graph = Graph(State)
async def think(s: State):
    reply = await engine.chat_text([m.to_openai() for m in s.messages])
    return {"messages": [s.messages[-1].reply(reply)]}
async def act(s: State):
    call = json.loads(s.messages[-1].content)
    result = await get_weather.execute(call["arguments"])
    return {"messages": [s.messages[-1].reply(f"[tool output] {result}")]}
graph.add_node("think", think)
graph.add_node("act", act)
graph.add_edge(START, "think")
graph.add_conditional_edges("think", lambda s: "act" if s.messages[-1].content.lstrip().startswith('{"tool"') else END)
graph.add_edge("act", "think")

# 4. Continuous learning: log every message; fine-tune + hot-swap LoRA at 1000.
buffer = InteractionBuffer("interactions.db", threshold=1000)
learner = ContinuousLearner(buffer, LoRATrainer(engine.config.model), engine=engine)
buffer.on_threshold = learner.trigger

# 5. Gateway: REST + WebSocket + Discord, guarded and observed.
router = GatewayRouter(handler=graph.compile().as_handler(),
                       interceptors=[PromptGuard()], observers=[buffer])

if __name__ == "__main__":
    uvicorn.run(router.app, host="0.0.0.0", port=8080)
