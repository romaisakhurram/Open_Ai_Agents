from dotenv import load_dotenv
from agents import Agent , Runner , AsyncOpenAI , OpenAIChatCompletionsModel, RunConfig , handoff
import os
import asyncio

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
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

billing_agent = Agent(
    name = "Billing Agent",
    instructions = "You are a billing agent. Your help with billing related questions, payment issues, and subscription management. Do not handle technical or product-related inquiries.",
)

refund_agent = Agent(
    name = "Refund Agent",
    instructions = "You are a refund agent. Your help with refund requests, return policies, and customer service issues. Do not handle technical or billing inquiries.",
)

custom_refund_handoff= handoff(
    agent = refund_agent,
    tool_name_override = "custom_refund_handoff",
    tool_description_override = "Handle user refund requests with extra care"
)

late_delivery_refund_handoff= handoff(
    agent = refund_agent,
    tool_name_override = "late_delivery_refund_handoff",
    tool_description_override = "Handle refund due to late delivery"
)

damaged_refund_handoff= handoff(
    agent = refund_agent,
    tool_name_override = "damaged_refund_handoff",
    tool_description_override = "Handle refund due to damaged item"
)

triage_agent = Agent(
    name = "Triage Agent",
    instructions = """ You are a triage agent who decides whether a question should be handled by the billing
     agent or the 'refund agent'. If the user asks about billing, handoff to the 'billing agent'. If the user asks
     about refunds, handoff to the refund agent. If it's unrelated, decline.
    """,
    handoffs = [billing_agent, refund_agent , custom_refund_handoff , late_delivery_refund_handoff, damaged_refund_handoff],
)

async def main():
    result = await Runner.run(
        triage_agent,
        "my product arrived broken. I want a refund",
        run_config=config,
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())