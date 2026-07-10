"""
Activity 14 — Bridging Memory and Action
Requirement A + B: Qdrant-wrapped search_documents with improvements

Improvements implemented:
  1. Query Expansion  — counters asymmetric embedding (short query vs long doc)
  2. Hybrid Search    — counters embedding dilution (dense + keyword re-scoring)

Run standalone:
    python bridging_search_tool.py
"""

import os
import time
import random
from functools import wraps
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
qdrant = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "simple_rag"   # same collection used by the chatbot backend
EMBEDDING_MODEL  = "gemini-embedding-2"
REACT_MODEL      = "gemini-3.1-flash-lite"

# ---------------------------------------------------------------------------
# RETRY DECORATOR
# ---------------------------------------------------------------------------

def with_backoff(max_retries=4, base_delay=2.0):
    """Exponential backoff for 429 / RESOURCE_EXHAUSTED errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exc = None
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and retries < max_retries:
                        delay = (base_delay * (2 ** retries)) + random.uniform(0, 1)
                        print(f"  ⚠️  Rate limited — retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        retries += 1
                        last_exc = e
                    else:
                        raise e
            raise Exception(f"Max retries exceeded: {last_exc}")
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# IMPROVEMENT 1 — QUERY EXPANSION
# Counters asymmetric retrieval: short queries land in a different vector
# region than long documents. Expanding them before embedding closes the gap.
# ---------------------------------------------------------------------------

@with_backoff()
def expand_query(short_query: str) -> str:
    """
    Rewrite a short user query into a document-style passage so its embedding
    lands in the same vector-space region as the stored document chunks.
    """
    prompt = (
        "Rewrite the following short query as a detailed, paragraph-style "
        "statement suitable for semantic similarity search. Include key terms "
        "and relevant context without adding facts not implied by the query.\n\n"
        f"Original query: {short_query}\n\nExpanded passage:"
    )
    response = gemini_client.models.generate_content(
        model=REACT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# IMPROVEMENT 2 — HYBRID SEARCH
# Counters embedding dilution: long document vectors average specific details
# into a generic mean. Adding keyword overlap re-scores candidates so exact
# terms are not lost.
# alpha=0.7 → 70% dense similarity, 30% keyword overlap (good default)
# ---------------------------------------------------------------------------

def hybrid_search(query: str, dense_vector: list, alpha: float = 0.7) -> str:
    """
    Retrieve top-10 by dense similarity, then re-rank by weighted hybrid score
    (dense cosine + keyword overlap). Returns the top-1 chunk text.
    """
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vector,
        limit=10,
    )
    dense_results = results.points

    if not dense_results:
        return "No relevant information found in the knowledge base."

    query_terms = set(query.lower().split())
    scored = []

    for point in dense_results:
        dense_score = point.score
        text = point.payload.get("text", "")
        text_lower = text.lower()

        doc_terms = set(text_lower.split())
        overlap = len(query_terms & doc_terms)
        if query_terms:
            tf_sum = sum(text_lower.count(t) for t in query_terms)
            keyword_score = (overlap / len(query_terms)) * min(1.0, tf_sum / 10.0)
        else:
            keyword_score = 0.0

        hybrid = alpha * dense_score + (1 - alpha) * keyword_score
        scored.append((hybrid, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_text = scored[0][1]
    return best_text if best_text else "Found a result but no text payload."


# ---------------------------------------------------------------------------
# REQUIREMENT A — MAIN search_documents TOOL (with edge-case handling)
# Combines Query Expansion + Hybrid Search
# ---------------------------------------------------------------------------

@with_backoff()
def _embed(text: str) -> list:
    """Embed text with gemini-embedding-2."""
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def search_documents(query: str) -> str:
    """
    Search the persistent Qdrant knowledge base for factual information.

    Pipeline:
      1. Validate input
      2. Expand the query (Improvement 1 — counters asymmetric retrieval)
      3. Embed the expanded query with gemini-embedding-2
      4. Hybrid search: dense + keyword re-scoring (Improvement 2 — counters dilution)
      5. Return a clean string — never a raw Qdrant object or exception traceback
    """
    # Edge case: empty query
    if not query.strip():
        return "No query provided."

    # Step 1: Query Expansion
    try:
        expanded = expand_query(query)
    except Exception as e:
        expanded = query          # fall back to original if expansion fails
        print(f"  ⚠️  Query expansion failed ({e}), using original query.")

    # Step 2: Embed
    try:
        query_vector = _embed(expanded)
    except Exception as e:
        return f"Embedding failed: {e}"

    # Step 3: Hybrid search
    try:
        result = hybrid_search(query, query_vector, alpha=0.7)
        return result
    except Exception as e:
        return f"Qdrant search error: {e}"


# ---------------------------------------------------------------------------
# ORIGINAL (BASELINE) — plain Qdrant search, no improvements
# Used by search_comparison.py to measure before/after delta
# ---------------------------------------------------------------------------

def original_search_documents(query: str) -> str:
    """
    Baseline: embed query directly and return top-1 result by cosine similarity.
    No query expansion, no hybrid re-scoring.
    """
    if not query.strip():
        return "No query provided."
    try:
        query_vector = _embed(query)
    except Exception as e:
        return f"Embedding failed: {e}"
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=1,
        )
        if not results.points:
            return "No relevant information found in the knowledge base."
        return results.points[0].payload.get("text", "Found a result but no text payload.")
    except Exception as e:
        return f"Qdrant search error: {e}"


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Activity 14 — bridging_search_tool.py smoke test")
    print("=" * 60)

    test_queries = [
        "What is a rainbow?",
        "What causes rain?",
        "How does the solar system work?",
    ]

    for q in test_queries:
        print(f"\n>>> QUERY : {q}")
        print(f"    RESULT : {search_documents(q)[:200]}")
