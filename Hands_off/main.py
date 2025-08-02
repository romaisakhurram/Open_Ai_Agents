from dotenv import load_dotenv
from agents import Agent , Runner , function_tool , AsyncOpenAI , OpenAIChatCompletionsModel, RunConfig 
import os
import requests

load_dotenv()

# @function_tool

# def get_info_details():
#     response = requests.get("https://api.gemini.com/v1/pubticker/btcusd")
#     result = response.json()
#     return result

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
     

frontened_agent = Agent(
    name = "Frontened Agent",
    instructions = "You are a frontened agent. your help with UI/Ux design , HTML , CSS and Javascript Do not ralated backened questions",
)

backened_agent = Agent(
    name = "Backened Ageny",
    instructions = "You are an backened agent. your help with database , integration , Django , APIs and authentications do not ralted with frontened questions",
)

cordinate_agent = Agent(
    name = "Cordinate Agent",
    instructions = "You are a cordinate agent who decide whether a frontened or backened. If the user ask ask UI/Ux , HTML , CSS and Javascript handsoffs to the frontened , If the user ask  database , integration , Django , APIs and authentications  handsoffs to the backened and If it's unrelated question to plicy decline",
    handoffs = [frontened_agent, backened_agent],
)

input_value = input("Enter a question: ")

agent_result = Runner.run_sync(
    cordinate_agent ,
    input_value ,
    run_config = config
)

print(agent_result.final_output) 