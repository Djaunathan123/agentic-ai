"""
Week 5 — Requirement A
ReAct Loop + 3 Tools + Labeled Transcript Logging
Run: python project_react_loop.py
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def search_documents(query: str) -> str:
    """Search the knowledge base for factual information."""
    kb = {
        "ReAct": "ReAct stands for Reasoning + Acting. It interleaves thought and tool calls.",
        "Qdrant": "Qdrant is a vector database for long-term agent memory.",
        "chunking": "Semantic chunking splits documents at natural boundaries.",
        "budget": "The travel budget is $2000 for flights and $500 for accommodations.",
        "RAG Triad": "The RAG Triad evaluates context relevance, groundedness, and answer relevance.",
        "agentic": "Agentic AI refers to AI systems that can take autonomous actions to complete goals.",
        "embedding": "Embeddings are numerical vector representations of text used for semantic search.",
        "llm": "A Large Language Model (LLM) is a neural network trained on large text corpora to generate human-like text.",
        "prompt": "A prompt is the input text given to an LLM to guide its response.",
        "vector": "A vector database stores high-dimensional vectors for fast similarity search.",
    }
    for key, doc in kb.items():
        if key.lower() in query.lower():
            return f"[Found] {doc}"
    return "[Not found] No relevant information in the knowledge base."


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
                "Search the knowledge base for factual information about "
                "course topics, budgets, or stored knowledge. Use this when "
                "the question asks about specific content, definitions, or "
                "stored facts such as 'What is ReAct?', 'What is the budget?', "
                "or 'Tell me about Qdrant'."
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
    print("  Week 5 — Requirement A: ReAct Loop Demo")
    print("=" * 60)

    queries = [
        "What is ReAct?",                              # search_documents
        "What is the travel budget?",                  # search_documents
        "Calculate 15% of 2000",                       # calculate
        "Help me with that thing",                     # clarify
        "Hello!",                                      # direct answer
        "What's the budget and what is 15% of it?",   # multi-step
    ]

    for q in queries:
        print(f"\n>>> QUERY: {q}")
        t = react_loop(q)
        print_transcript(t)
