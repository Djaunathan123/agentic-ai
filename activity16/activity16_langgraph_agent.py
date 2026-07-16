"""
Activity 16 — From ReAct Loop to LangGraph
Self-Correcting RAG Agent using a LangGraph State Graph

Architecture:
    generate → evaluate → (should_retry) → END
                                         ↘ rewrite → generate

Run: python activity16_langgraph_agent.py
"""

import os
import time
import random
from functools import wraps
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END

load_dotenv()

# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    _test_qdrant = QdrantClient(url="http://localhost:6333", timeout=3)
    _test_qdrant.get_collections()
    qdrant = _test_qdrant
    QDRANT_AVAILABLE = True
    print("  Qdrant connected OK.")
except Exception:
    qdrant = None
    QDRANT_AVAILABLE = False
    print("  ⚠️  Qdrant not reachable — search_documents will return a placeholder.")

COLLECTION_NAME = "simple_rag"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

THRESHOLD        = 0.7
MAX_ITERATIONS   = 3
LLM_MODEL        = "gemini-3.1-flash-lite"   # matches the working model from activity15
INTER_CALL_DELAY = 4                          # seconds between LLM calls (rate-limit guard)


# ---------------------------------------------------------------------------
# STEP 1 — AgentState TypedDict
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question:           str
    original_question:  str
    retrieved_chunk:    str
    answer:             str
    context_relevance:  float
    groundedness:       float
    answer_relevance:   float
    iteration:          int
    log:                list
    route_decision:     str   # "accept" | "accept_maxed" | "rewrite"


# ---------------------------------------------------------------------------
# RETRY DECORATOR
# ---------------------------------------------------------------------------

def with_backoff(max_retries: int = 4, base_delay: float = 3.0):
    """Exponential back-off for 429 / RESOURCE_EXHAUSTED errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries, last_exc = 0, None
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and retries < max_retries:
                        delay = base_delay * (2 ** retries) + random.uniform(0, 2)
                        print(f"  ⚠️  Rate limited — retrying in {delay:.1f}s "
                              f"(attempt {retries + 1}/{max_retries})...")
                        time.sleep(delay)
                        retries += 1
                        last_exc = e
                    else:
                        raise
            raise Exception(f"Max retries exceeded: {last_exc}")
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# STEP 2 — Tool Functions
# ---------------------------------------------------------------------------

@with_backoff()
def _embed(text: str) -> list:
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def search_documents(query: str) -> str:
    """Search Qdrant — returns the best matching chunk as a plain string."""
    if not query.strip():
        return "No query provided."
    if not QDRANT_AVAILABLE:
        return (
            "Qdrant is offline. Start it with: "
            "docker run -p 6333:6333 qdrant/qdrant"
        )
    try:
        query_vector = _embed(query)
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3,
        )
        results = response.points
        if not results:
            return "No relevant information found."
        chunk = (
            results[0].payload.get("text")
            or results[0].payload.get("text_segment")
            or ""
        )
        return chunk if chunk else "Found a result but no text payload."
    except Exception as e:
        return f"Search error: {e}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def clarify(question: str) -> str:
    """Fallback — ask the user for more information."""
    return f"[Clarify] {question}"


# ---------------------------------------------------------------------------
# RAG Triad Judge helper
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    score:  float
    reason: str


@with_backoff()
def judge_metric(criterion: str, reference: str, target: str) -> JudgeVerdict:
    """Score a single RAG Triad metric using an LLM judge."""
    prompt = (
        f"You are an evaluation agent. Score on a scale of 0.0 to 1.0.\n\n"
        f"Criterion: {criterion}\n\n"
        f"Reference:\n{reference}\n\n"
        f"Text to evaluate:\n{target}\n\n"
        'Return JSON: {"score": <float>, "reason": "<one sentence>"}'
    )
    time.sleep(INTER_CALL_DELAY)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
        ),
    )
    return JudgeVerdict.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# STEP 3 — Generate Node
# ---------------------------------------------------------------------------

def generate_node(state: AgentState) -> AgentState:
    """Retrieve context from Qdrant and generate an answer via LLM."""
    question = state["question"]
    chunk    = search_documents(question)

    prompt = (
        "You are a helpful assistant. "
        "Answer the question using ONLY the context below.\n\n"
        f"Question: {question}\n\n"
        f"Context: {chunk}\n\n"
        "Answer:"
    )

    time.sleep(INTER_CALL_DELAY)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    answer = response.text

    state["retrieved_chunk"] = chunk
    state["answer"]          = answer
    state["iteration"]       = state.get("iteration", 0) + 1

    log = state.get("log", [])
    log.append({
        "iteration":      state["iteration"],
        "question":       question,
        "chunk_preview":  chunk[:100] if chunk else "",
        "answer_preview": answer[:100] if answer else "",
    })
    state["log"] = log

    print(f"  [Generate] Iteration {state['iteration']} | Q: {question[:60]}")
    return state


# ---------------------------------------------------------------------------
# STEP 4 — Evaluate Node
# ---------------------------------------------------------------------------

def evaluate_node(state: AgentState) -> AgentState:
    """Score the current answer using three RAG Triad judges."""
    cr = judge_metric(
        "Is the retrieved chunk relevant to answering the question?",
        state["question"],
        state["retrieved_chunk"],
    )
    gr = judge_metric(
        "Is the answer factually supported by the retrieved chunk?",
        state["retrieved_chunk"],
        state["answer"],
    )
    ar = judge_metric(
        "Does the answer correctly address the user's question?",
        state["question"],
        state["answer"],
    )

    state["context_relevance"] = cr.score
    state["groundedness"]      = gr.score
    state["answer_relevance"]  = ar.score

    # Determine route decision here so it's saved into graph state
    if state["iteration"] >= MAX_ITERATIONS:
        state["route_decision"] = "accept_maxed"
    elif min(cr.score, gr.score, ar.score) >= THRESHOLD:
        state["route_decision"] = "accept"
    else:
        state["route_decision"] = "rewrite"

    if state["log"]:
        state["log"][-1].update({
            "context_relevance": cr.score,
            "groundedness":      gr.score,
            "answer_relevance":  ar.score,
            "cr_reason":         cr.reason,
            "gr_reason":         gr.reason,
            "ar_reason":         ar.reason,
        })

    print(
        f"  [Evaluate] CR: {cr.score:.2f} | "
        f"GR: {gr.score:.2f} | AR: {ar.score:.2f}"
    )
    return state


# ---------------------------------------------------------------------------
# STEP 5 — Conditional Edge
# ---------------------------------------------------------------------------

def should_retry(state: AgentState) -> str:
    """Route to 'end' if scores pass or max iterations reached, else 'rewrite'."""
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"  [Decision] Max iterations ({MAX_ITERATIONS}) reached. Accepting.")
        return "end"

    min_score = min(
        state["context_relevance"],
        state["groundedness"],
        state["answer_relevance"],
    )

    if min_score >= THRESHOLD:
        print(f"  [Decision] All scores >= {THRESHOLD:.1f}. Accepting.")
        return "end"

    print(
        f"  [Decision] Min score {min_score:.2f} < {THRESHOLD:.1f}. "
        "Rewriting query..."
    )
    return "rewrite"


# ---------------------------------------------------------------------------
# STEP 6 — Rewrite Node
# ---------------------------------------------------------------------------

def rewrite_node(state: AgentState) -> AgentState:
    """Use the LLM to rewrite a poor-performing query for better retrieval."""
    original = state["question"]
    chunk    = state["retrieved_chunk"]

    rewrite_prompt = (
        "The following query did not retrieve good context from the vector database.\n"
        "Rewrite it to be more specific and likely to find relevant information.\n\n"
        f"Original query: {original}\n"
        f"Retrieved chunk (not helpful): {chunk[:200]}\n\n"
        "Return ONLY the rewritten query, nothing else."
    )

    time.sleep(INTER_CALL_DELAY)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=rewrite_prompt,
    )
    rewritten = response.text.strip()
    state["question"] = rewritten

    print(f"  [Rewrite]  '{original[:50]}' → '{rewritten[:50]}'")
    return state


# ---------------------------------------------------------------------------
# STEP 7 — Build Graph
# ---------------------------------------------------------------------------

def build_graph():
    """Assemble and compile the self-correcting RAG graph."""
    graph = StateGraph(AgentState)

    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("rewrite",  rewrite_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        should_retry,
        {"end": END, "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "generate")

    return graph.compile()


# ---------------------------------------------------------------------------
# STEP 8 — run_agent
# ---------------------------------------------------------------------------

def run_agent(question: str) -> dict:
    """Run the self-correcting LangGraph agent on a single question."""
    app = build_graph()

    initial_state: AgentState = {
        "question":           question,
        "original_question":  question,
        "retrieved_chunk":    "",
        "answer":             "",
        "context_relevance":  0.0,
        "groundedness":       0.0,
        "answer_relevance":   0.0,
        "iteration":          0,
        "log":                [],
        "route_decision":     "",
    }

    return app.invoke(initial_state, config={"recursion_limit": 10})


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

def print_result(result: dict):
    print(f"\nFinal Answer    : {result['answer'][:200]}")
    print(f"Iterations used : {result['iteration']}")
    print(
        f"Final Scores    — "
        f"CR: {result['context_relevance']:.2f} | "
        f"GR: {result['groundedness']:.2f} | "
        f"AR: {result['answer_relevance']:.2f}"
    )
    print(f"Route Decision  : {result.get('route_decision', 'N/A')}")

    print("\n  Iteration Log:")
    for entry in result.get("log", []):
        cr = entry.get("context_relevance", "?")
        gr = entry.get("groundedness",      "?")
        ar = entry.get("answer_relevance",  "?")
        cr_s = f"{cr:.2f}" if isinstance(cr, float) else str(cr)
        gr_s = f"{gr:.2f}" if isinstance(gr, float) else str(gr)
        ar_s = f"{ar:.2f}" if isinstance(ar, float) else str(ar)
        print(
            f"    #{entry['iteration']}: \"{entry['question'][:40]}...\" "
            f"CR={cr_s} GR={gr_s} AR={ar_s}"
        )


# ---------------------------------------------------------------------------
# MAIN — Requirement A + B demo (6 questions: 3 happy-path, 3 tricky)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Happy-path questions (expect 1 iteration, scores >= 0.7) ───────────
    happy_path = [
        "What is a rainbow?",
        "What causes rain and precipitation?",
        "How many planets are in the solar system?",
    ]

    # ── Tricky / vague questions (expect rewrite cycle or max-iteration stop)
    tricky = [
        "Tell me about that thing from class",
        "What did the course say?",
        "Do the stuff",
    ]

    all_questions = happy_path + tricky

    print("=" * 60)
    print("  Activity 16 — LangGraph Self-Correcting RAG Agent")
    print("=" * 60)

    for q in all_questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {q}")
        print(f"{'=' * 60}")
        result = run_agent(q)
        print_result(result)
        time.sleep(2)   # light throttle between questions
