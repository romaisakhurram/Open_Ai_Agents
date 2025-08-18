import asyncio
from agents import Agent, Runner, function_tool, RunConfig , AsyncOpenAI , OpenAIChatCompletionsModel
from typing import Dict
import os

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

# Inventory storage (simple in-memory dict)
inventory: Dict[str, int] = {}

# Define tools
@function_tool
def manage_inventory(id: str, quantity: int, operation: str) -> str:
    """
    Manage inventory by adding, deleting, or updating item quantities.
    
    Args:
        id (str): Item ID
        quantity (int): Quantity of item
        operation (str): "add", "delete", or "update"
    """
    if operation == "add":
        if id in inventory:
            inventory[id] += quantity
        else:
            inventory[id] = quantity
        return f"✅ Added {quantity} units of {id}. Current stock: {inventory[id]}"

    elif operation == "update":
        if id in inventory:
            inventory[id] = quantity
            return f"✏️ Updated {id} to {quantity} units."
        else:
            return f"⚠️ Item {id} not found in inventory."

    elif operation == "delete":
        if id in inventory:
            del inventory[id]
            return f"🗑️ Deleted {id} from inventory."
        else:
            return f"⚠️ Item {id} not found."

    else:
        return "❌ Invalid operation. Use 'add', 'update', or 'delete'."

# Create the agent
agent = Agent(
    name="InventoryAgent",
    tools=[manage_inventory],
    instructions="You are an inventory manager. Use tools to manage stock."
)

async def main():
    runner = await Runner.run(
        agent,
        "Add 10 apples",     
        config=RunConfig
    )
    
if __name__ == "__main__":
    asyncio.run(main())
