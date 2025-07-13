import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

def main():

    response = completion(
        model = "gemini/gemini-2.0-flash",
        messages = [
            {
                "role": "user",
                "content": "who is the founder of Pakistan?"
            }
        ]
    )
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
