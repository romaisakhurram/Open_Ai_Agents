from dotenv import load_dotenv
from agents import Agent , Runner , function_tool , AsyncOpenAI , OpenAIChatCompletionsModel, RunConfig 
import os
import requests

load_dotenv()

@function_tool

def get_info_details():
    response = requests.get("https://api.gemini.com/v1/pubticker/btcusd")
    result = response.json()
    return result

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

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
     

Greeting_agent = Agent(
    name = "Greeting Agent",
    instructions = "You are a greeting agent. your task is says to Salam , when someone says hello , hi or anything similar",
)

info_details_agent = Agent(
    name = "Info Agent",
    instructions = "You are an info agent. your task is to provide information about the agents.",
    tools = [get_info_details]
)

cordinate_agent = Agent(
    name = "Cordinate Agent",
    instructions = "You are a cordinate agent. your task is to provide the cordinates of the agents.",
    handoffs = [Greeting_agent, info_details_agent]
)

input_value = input("Enter a question: ")

agent_result = Runner.run_sync(
    cordinate_agent ,
    input_value ,
    run_config = config
)

print(agent_result.final_output) 