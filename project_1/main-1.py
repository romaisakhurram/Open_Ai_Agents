from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI , RunConfig, Runner
from dotenv import load_dotenv
import os
import chainlit as cl
import json

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Check if the API key is present; if not, raise an error
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
    model="deepseek/deepseek-r1-0528:free",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

tranlate_agent = Agent(
    name = "Translate Agent" ,
    instructions = "You are a translation agent, Translate the text to English. ",
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat history", [])
    await cl.Message(content="***Welcome to the Translate Agent*** \n\n**Please tell me what you want to translate**").send()

@cl.on_message
async def main(message: cl.Message):
    text = message.strip()
    if not text:
        await cl.Message(content="Please enter a translate").send()
        return

    agent_result = Runner.run_sync(
        tranlate_agent,
        text,
        run_config=config
    )
    
    # Convert the result to JSON format
    result_json = json.dumps({"translated_text": agent_result.final_output}, ensure_ascii=False)
    
    await cl.Message(content=result_json).send()
# This code sets up a Chainlit application that uses an agent to translate text into English.