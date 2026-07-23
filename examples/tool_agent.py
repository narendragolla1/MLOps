"""LangChain-style agent in OmniAI: providers, prompts, tools, structured output.

Set OPENAI_API_KEY or ANTHROPIC_API_KEY (or point at your own engine) and run:
    python examples/tool_agent.py
"""

import asyncio
import os

from pydantic import BaseModel

from omniai.graph import create_tool_agent, tool
from omniai.models import AnthropicChatModel, OpenAIChatModel
from omniai.prompts import ChatPromptTemplate
from omniai.protocol import OmniMessage


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22C and sunny in {city}"


class TripPlan(BaseModel):
    city: str
    activity: str
    packing_list: list[str]


def pick_model():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicChatModel("claude-sonnet-5", api_key=os.environ["ANTHROPIC_API_KEY"])
    return OpenAIChatModel("gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY", ""))
    # Self-hosted instead? EngineChatModel(ModelEngine.create({...}))


async def main() -> None:
    model = pick_model()

    # 1. Prompt template with validated variables.
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a {style} travel assistant."), ("user", "{question}")]
    )
    messages = prompt.format_messages(style="concise", question="Weather in Lisbon?")

    # 2. Prebuilt tool-calling agent: model <-> tools loop, bounded steps.
    agent = create_tool_agent(model, [get_weather], system_prompt="Use tools for live data.")
    final = await agent.ainvoke({"messages": [OmniMessage(content=messages[-1]["content"])]})
    print("agent:", final.messages[-1].content)

    # 3. Structured output: validated Pydantic objects with retry-on-error.
    plan = await model.with_structured_output(TripPlan).invoke("Plan a sunny afternoon in Lisbon.")
    print("plan:", plan.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
