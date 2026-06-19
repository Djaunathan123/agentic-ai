import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

react_identity = """
You are a Research Assistant. You have access to the following tools:

1. get_weather(city): Returns current temperature.
2. get_stock_price(symbol): Returns current stock price.

If you need a tool, respond ONLY with:
TOOL: [tool_name]([params])

Once you have the info, provide the final answer.
"""


def simulated_react_agent(user_query):

    print(f"\n[USER]: {user_query}")

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        config={
            "system_instruction": react_identity
        },
        contents=user_query
    )

    if "TOOL:" in response.text:

        tool_call = response.text.split("TOOL:")[1].strip()

        print(f"[ACTION] Agent requested tool: {tool_call}")

        observation = (
            "OBSERVATION: The temperature in Manila is 32°C and sunny."
        )

        print(f"[OBSERVE] Providing data: {observation}")

        final_response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            config={
                "system_instruction": react_identity
            },
            contents=[
                user_query,
                response.text,
                observation
            ]
        )

        print(f"\n[FINAL ANSWER]: {final_response.text}")

    else:

        print(f"\n[FINAL ANSWER]: {response.text}")


simulated_react_agent(
    "What should I wear in Manila today?"
)