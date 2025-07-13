from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import os
import chainlit as cl
from typing import cast
from agents.run import RunConfig

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set. Please ensure it is defined in your .env file.")

@ cl.on_chat_start
async def on_chat_start():
    external_client = AsyncOpenAI(
        api_key= openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    model = OpenAIChatCompletionsModel(
        model="google/gemma-3-27b-it:free",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("chat history" , [])
    cl.user_session.set("config", config)

    agent : Agent = Agent(
        name="Writer Agent",
        instructions="You are Greeting agent, you will greet the user and ssys to Salam and help them.",
        model = model
    )

    cl.user_session.set("agent" , agent)
    await cl.Message(content = "Welcome to the Greeting Agent").send()

@ cl.on_message
async def on_message(message:cl.Message):
    msg = cl.Message(content = "Thinking...")
    await msg.send()

    agent: Agent = cast(Agent , cl.user_session.get("agent"))

    config : RunConfig = cast(RunConfig , cl.user_session.get("config"))

    history = cl.user_session.get("chat history") or []
    history.append({"role" : "user" , "content" : message.content})

    try:
        print("\n [CALLING-AGENT-WITH-CONTEXT]\n" , history , "\n")

        result = Runner.run_sync(
            starting_agent = agent,
            input = history,
            run_config = RunConfig
        )

        response_content = result.final_output
        msg.content = response_content
        await msg.update()

        cl.user_session.set("chat history" , result.to_input_list())

        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")
    
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")


