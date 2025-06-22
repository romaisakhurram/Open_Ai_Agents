import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import os

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
    model="google/gemini-2.0-flash-exp:free",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

writer_agent = Agent(
    name="Writer Agent",
    instructions="You are writer agent, Generate a essay ",
)

@cl.on_chat_start
async def on_chart_start():
    cl.user_session.set("chat history" , [])
    await cl.Message(content = "***Welcome to the writer Agent*** \n\n**Please tell me what you want to generate a Essay**").send()


@cl.on_message
async def main(message: cl.Message):
    topic = message.content.strip()
    if not topic:
        await cl.Message(content="working...").send()
        return

    agent_result = Runner.run_sync(
        writer_agent,
        topic,
        run_config=config
    )
    await cl.Message(content=agent_result.final_output).send()