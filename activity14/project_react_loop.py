"""
Activity 14 — Bridging Memory and Action
ReAct Loop wired to real Qdrant search (replaces mock KB from activity13)
Run: python project_react_loop.py
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import real Qdrant-backed search_documents from bridging_search_tool
from bridging_search_tool import search_documents

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

# search_documents is imported from bridging_search_tool — real Qdrant + improvements
# calculate and clarify remain unchanged

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def clarify(question: str) -> str:
    """Ask for clarification when the request is ambiguous."""
    return f"[Clarify] Could you please clarify: {question}"


# ---------------------------------------------------------------------------
# TOOL REGISTRY
# ---------------------------------------------------------------------------

TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "Search the persistent Qdrant knowledge base for factual information "
                "about course topics, user preferences, or stored documents. Use this "
                "when the question requires knowledge not already in the conversation, "
                "such as questions about rainbows, the solar system, weather, or any "
                "topic from the uploaded documents."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "Evaluate a mathematical expression and return the numeric result. "
                "Use for any arithmetic or numeric computation, such as "
                "'45 * 12', '15% of 2000', '2 to the power of 10', or "
                "'what is 100 divided by 4'."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"expression": types.Schema(type="STRING")},
                required=["expression"],
            ),
        ),
        types.FunctionDeclaration(
            name="clarify",
            description=(
                "Ask the user a clarifying question when their request is too "
                "ambiguous or vague to handle with any other tool. Use this as "
                "a fallback when you cannot determine what the user needs — for "
                "example 'Help me with that thing' or 'Do the stuff I asked'."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"question": types.Schema(type="STRING")},
                required=["question"],
            ),
        ),
    ]
)

AVAILABLE_FUNCTIONS = {
    "search_documents": search_documents,
    "calculate": calculate,
    "clarify": clarify,
}

# ---------------------------------------------------------------------------
# REACT LOOP
# ---------------------------------------------------------------------------

def _find_parts(response):
    """
    gemini-3.1-flash-lite is a thinking model whose response may contain:
        [Part(thought=True, text="..."), Part(function_call=...)]
    Scan all parts and return (fc_part, text_part).
    """
    fc_part = None
    text_part = None
    for p in response.candidates[0].content.parts:
        if fc_part is None and p.function_call:
            fc_part = p
        if text_part is None and p.text and not getattr(p, "thought", False):
            text_part = p
    return fc_part, text_part


def react_loop(
    question: str,
    max_iterations: int = 5,
    system_prompt: str = "You are a helpful assistant with access to tools.",
) -> list[dict]:
    """
    Run a ReAct (Reason → Act → Observe) loop.
    Returns a labeled transcript list.
    """
    transcript = [{"phase": "USER", "content": question}]
    history = [types.Content(role="user", parts=[types.Part(text=question)])]

    for _turn in range(max_iterations):
        # REASON
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[TOOLS],
            ),
        )

        fc_part, text_part = _find_parts(response)

        # ACT
        if fc_part is not None:
            fc = fc_part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args)

            transcript.append({
                "phase": "ACTION",
                "tool": tool_name,
                "content": f"{tool_name}({tool_args})",
            })

            func = AVAILABLE_FUNCTIONS.get(tool_name)
            result = func(**tool_args) if func else f"Unknown tool: {tool_name}"

            # OBSERVE
            transcript.append({"phase": "OBSERVE", "content": result})

            # Append raw Content to preserve thought_signature
            history.append(response.candidates[0].content)
            history.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": result},
                            )
                        )
                    ],
                )
            )
        else:
            # ANSWER
            answer = text_part.text if text_part else ""
            transcript.append({"phase": "ANSWER", "content": answer})
            return transcript

    transcript.append({
        "phase": "SYSTEM",
        "content": f"Max iterations ({max_iterations}) reached without a final answer.",
    })
    return transcript


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

def print_transcript(transcript: list[dict]):
    print(f"\n{'=' * 60}")
    for entry in transcript:
        phase = entry["phase"]
        content = entry["content"]
        if phase == "ACTION":
            print(f"  [{phase:7}] 🔧 {content}")
        elif phase == "OBSERVE":
            print(f"  [{phase:7}] 👁  {content}")
        elif phase == "ANSWER":
            print(f"  [{phase:7}] ✅ {content}")
        elif phase == "USER":
            print(f"  [{phase:7}] 💬 {content}")
        else:
            print(f"  [{phase:7}] {content}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# MAIN — Requirement A demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Activity 14 — ReAct Loop with Real Qdrant Search")
    print("=" * 60)

    queries = [
        "What is a rainbow?",               # real Qdrant search
        "What causes rain?",                # real Qdrant search
        "How many planets are there?",      # real Qdrant search
        "Calculate 15% of 2000",            # calculate tool
        "Help me with that thing",          # clarify tool
        "Hello!",                           # direct answer (no tool)
    ]

    for q in queries:
        print(f"\n>>> QUERY: {q}")
        t = react_loop(q)
        print_transcript(t)
