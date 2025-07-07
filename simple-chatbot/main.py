import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent , Runner , AsyncOpenAI , OpenAIChatCompletionsModel
from agents.run import RunConfig

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

@ cl.on_chat_start
async def on_chat_start():
    #Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
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
        name = "Assistant",
        instructions = "You are a helpful assistant. ",
        model = model
    )

    cl.user_session.set("agent" , agent)
    await cl.Message(content = "Welcome to the  Romi! AI Assistant how can you help today?").send()

@ cl.on_message
async def on_message(message:cl.Message):

    # history = cl.user_session.get("chat history") or []    #streamed chat history
    # history.append({"role" : "user" , "content" : message.content})

    msg = cl.Message(content = "Thinking...")
    await msg.send()

    agent: Agent = cast(Agent , cl.user_session.get("agent"))

    config : RunConfig = cast(RunConfig , cl.user_session.get("config"))

    history = cl.user_session.get("chat history") or []      #non-streamed chat history
    history.append({"role" : "user" , "content" : message.content})

    try:
        print("\n [CALLING-AGENT-WITH-CONTEXT]\n" , history , "\n")

        result = Runner.run_sync(       #non-streamed response
            starting_agent = agent,
            input = history,
            run_config = RunConfig
        )

        response_content = result.final_output
        msg.content = response_content
        await msg.update()

        cl.user_session.set("chat history" , result.to_input_list())

        # result = Runner.run_streamed(agent , history , run_config = config)   #streamed response
        # async for event in result.stream_events():
        #     if event.type ==  "raw_response_event" and hasattr(event.data , "delta"):
        #         token = event.data.delta
        #         await msg.stream_token(token)

        # history.append({"role": "assistant", "content": msg.content})
        # cl.user_session.set("chat history" , history)

        print(f"User: {message.content}")
        print(f"Assistant: {msg.content}")
    
    except Exception as e:
        msg.content = f"Error: {str(e)}"     #non-streamed error message
        await msg.update()
        # await msg.update(content=f"Error: {str(e)}")    #streming error message
        print(f"Error: {str(e)}")


