"""
Activity 14 — Bridging Memory and Action
ReAct Loop wired to real Qdrant search (replaces mock KB)
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Real Qdrant-backed search with Query Expansion + Hybrid Search
from bridging_search_tool import search_documents

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------
# search_documents — imported from bridging_search_tool (real Qdrant)

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
                "when the question requires knowledge from uploaded files such as "
                "questions about rainbows, the solar system, weather, or any stored topic."
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
    gemini-3.1-flash-lite is a thinking model.
    The response parts list may look like:
        [Part(thought=True, text="..."), Part(function_call=...) ]
    or just:
        [Part(text="final answer")]

    Scan every part and return (fc_part, text_part) where:
        fc_part  — first part that has a function_call (or None)
        text_part — first non-thought text part (or None)
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

    Returns a labeled transcript where each entry has:
        {"phase": "USER"|"ACTION"|"OBSERVE"|"ANSWER"|"SYSTEM", ...}

    NOTE on gemini-3.1-flash-lite thought_signature:
    The thinking model attaches a thought_signature to each function_call part.
    We MUST append response.candidates[0].content (the full raw Content object)
    back into history without rebuilding it — that preserves the signature.
    Rebuilding the Content manually drops the signature and causes a 400 error.
    """
    transcript = [{"phase": "USER", "content": question}]
    history = [types.Content(role="user", parts=[types.Part(text=question)])]

    for _turn in range(max_iterations):
        # --- REASON ---
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[TOOLS],
            ),
        )

        fc_part, text_part = _find_parts(response)

        # --- ACT ---
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

            # --- OBSERVE ---
            transcript.append({"phase": "OBSERVE", "content": result})

            # Append the RAW model Content — thought_signature stays intact
            history.append(response.candidates[0].content)

            # Append tool result as a user turn
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
            # --- ANSWER ---
            answer = text_part.text if text_part else ""
            transcript.append({"phase": "ANSWER", "content": answer})
            return transcript

    transcript.append({
        "phase": "SYSTEM",
        "content": f"Max iterations ({max_iterations}) reached without a final answer.",
    })
    return transcript


# ---------------------------------------------------------------------------
# TRANSCRIPT DISPLAY
# ---------------------------------------------------------------------------

def print_transcript(transcript: list[dict]):
    """Pretty-print a labeled ReAct transcript."""
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


def demo_query(question: str):
    """Run one query through the ReAct loop and print the transcript."""
    print(f"\n>>> QUERY: {question}")
    transcript = react_loop(question)
    print_transcript(transcript)
    return transcript


# ---------------------------------------------------------------------------
# MANUAL CHECKPOINT DEMOS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Week 5 — ReAct Loop Demo (Requirement A)")
    print("=" * 60)

    demo_query("What is ReAct?")
    demo_query("What is the travel budget?")
    demo_query("Calculate 15% of 2000")
    demo_query("Help me with that thing")
    demo_query("Hello!")
    demo_query("What's the budget and what is 15% of it?")
