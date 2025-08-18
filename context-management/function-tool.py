from agents import Agent , Runner , AsyncOpenAI, OpenAIChatCompletionsModel, function_tool , set_tracing_disabled
import os
from dotenv import load_dotenv
import requests
import random

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

@function_tool
def how_many_jokes():
    """ Get random for jokes."""
    return random.randint(1, 10)

@function_tool
def get_weather(city:str)-> str:
    """Get the weather for given a city"""
    try:
        result = requests.get(
            f"https://api.weatherapi.com/v1/current.json?key=8e3aca2b91dc4342a1162608252604&q"
        )
        data = result.json()
        return f"The current weather in {city} is {data['current']['temp_c']}°C with {data['current']['condition']['text']}."

    except Exception as e:
        return f"Sorry, I couldn't fetch the weather data at the moment {e}."


agent = Agent(
    name="Assistant",
    instructions="""
    if the user asks the jokes first call 'how_many_jokes' function, then tell that jokes with numbers .
    If the user asks for the weather, call 'get_weather' function with the city name.
    """,
    model=model,
    tools=[how_many_jokes , get_weather],
)

result = Runner.run_sync(
    agent,
    input="tell me jokes",
)

print(result.final_output)