"""
Week 5 — Requirement C
RAG Triad Wrapper around the ReAct Loop + Self-Correction
Run: python evaluation_integration.py
"""

import os
import time
import random
from functools import wraps
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from project_react_loop import react_loop, print_transcript

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------------------------------------------------------
# RAG TRIAD EVALUATION (self-contained, no backend imports)
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
    reason: str = Field(description="Brief explanation justifying the score")


def with_exponential_backoff(max_retries=4, base_delay=2.0):
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
                        print(f"⚠️  Rate limited. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        retries += 1
                        last_exc = e
                    else:
                        raise e
            raise Exception(f"Max retries exceeded: {last_exc}")
        return wrapper
    return decorator


@with_exponential_backoff()
def evaluate_metric(criterion: str, source_text: str, generated_answer: str) -> EvaluationResult:
    prompt = f"""
You are an impartial judge evaluating a generated answer based on a specific criterion.

Criterion:
{criterion}

Source Text:
{source_text}

Generated Answer:
{generated_answer}

Provide your evaluation as a JSON object with 'score' (float 0.0–1.0) and 'reason' (string).
"""
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluationResult,
            temperature=0.1,
        ),
    )
    return response.parsed if response.parsed else EvaluationResult.model_validate_json(response.text)


def evaluate_rag_triad(question: str, context: str, answer: str, threshold: float = 0.7) -> dict:
    print("\n" + "=" * 60)
    print("  RAG TRIAD EVALUATION")
    print("=" * 60)

    # Leg 1 — Context Relevance
    print("\n📋 Leg 1/3: Context Relevance...")
    cr = evaluate_metric(
        f"Is this source text relevant for answering '{question}'? "
        "(0.0 = irrelevant, 1.0 = perfectly relevant)",
        context, question
    )
    print(f"   Score: {cr.score:.2f} | {cr.reason}")

    # Leg 2 — Groundedness
    print("\n📋 Leg 2/3: Groundedness...")
    gr = evaluate_metric(
        "Does the answer rely ONLY on the source text, with no hallucinations? "
        "(0.0 = not grounded, 1.0 = fully grounded)",
        context, answer
    )
    print(f"   Score: {gr.score:.2f} | {gr.reason}")

    # Leg 3 — Answer Relevance
    print("\n📋 Leg 3/3: Answer Relevance...")
    ar = evaluate_metric(
        f"Does the answer directly address '{question}'? "
        "(0.0 = irrelevant, 1.0 = perfectly answers)",
        context, answer
    )
    print(f"   Score: {ar.score:.2f} | {ar.reason}")

    decision = "ACCEPT"
    reason = "All three triad metrics passed!"
    if cr.score < threshold:
        decision, reason = "RE-RETRIEVE", "Context is not relevant enough."
    elif gr.score < threshold:
        decision, reason = "RE-GENERATE", "Answer contains hallucinations."
    elif ar.score < threshold:
        decision, reason = "RE-PROMPT", "Answer doesn't address the question."

    print(f"\n  Decision: {decision} — {reason}")

    return {
        "context_relevance": cr.score,
        "groundedness": gr.score,
        "answer_relevance": ar.score,
        "average_score": (cr.score + gr.score + ar.score) / 3,
        "decision": decision,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# SELF-CORRECTION
# ---------------------------------------------------------------------------

def self_correct_answer(question: str, context: str, original_answer: str, evaluation: dict) -> str:
    weak_leg = min(
        [("groundedness", evaluation["groundedness"]),
         ("answer_relevance", evaluation["answer_relevance"])],
        key=lambda x: x[1]
    )[0]

    prompt = f"""
The previous answer failed the RAG Triad quality check on: {weak_leg}.

Question: {question}
Context: {context}
Failed answer: {original_answer}

Write a corrected answer that:
- Uses ONLY the provided context
- Directly and completely answers the question
- Is concise and accurate
"""
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_with_evaluation(question: str, verbose: bool = True) -> dict:
    """ReAct loop → RAG Triad evaluation → optional self-correction."""

    # Step 1: ReAct loop
    transcript = react_loop(question)
    if verbose:
        print_transcript(transcript)

    # Step 2: Extract answer and context
    answer, context = "", ""
    for entry in transcript:
        if entry["phase"] == "ANSWER":
            answer = entry["content"]
        if entry["phase"] == "OBSERVE":
            context = entry["content"]

    # Step 3: Evaluate
    evaluation = evaluate_rag_triad(question, context, answer)

    passed = (
        evaluation["context_relevance"] >= 0.7
        and evaluation["groundedness"] >= 0.7
        and evaluation["answer_relevance"] >= 0.7
    )

    result = {
        "question": question,
        "answer": answer,
        "transcript": transcript,
        **evaluation,
        "passed": passed,
        "was_corrected": False,
        "corrected_answer": None,
    }

    # Step 4: Self-correct if needed
    if not passed and evaluation["decision"] in ("RE-GENERATE", "RE-PROMPT"):
        print("\n🔄 Self-correcting answer...")
        corrected = self_correct_answer(question, context, answer, evaluation)
        result["corrected_answer"] = corrected
        result["was_corrected"] = True
        if verbose:
            print(f"\n  ✏️  Corrected answer: {corrected}")

    return result


def print_result(result: dict):
    def badge(s): return "✅" if s >= 0.7 else "❌"
    print("\n" + "─" * 60)
    print("  RAG Triad Scores")
    print("─" * 60)
    print(f"  Context Relevance : {result['context_relevance']:.2f}  {badge(result['context_relevance'])}")
    print(f"  Groundedness      : {result['groundedness']:.2f}  {badge(result['groundedness'])}")
    print(f"  Answer Relevance  : {result['answer_relevance']:.2f}  {badge(result['answer_relevance'])}")
    print(f"  Average           : {result['average_score']:.2f}")
    print(f"  Decision          : {result['decision']}")
    print(f"  Quality Gate      : {'PASS ✅' if result['passed'] else 'FAIL ❌'}")
    if result["was_corrected"]:
        print(f"\n  🔄 Answer was self-corrected.")
        print(f"  Corrected: {result['corrected_answer']}")
    print("─" * 60)


# ---------------------------------------------------------------------------
# MAIN — Requirement C demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Week 5 — Requirement C: RAG Triad Integration Demo")
    print("=" * 60)

    test_questions = [
        "What is ReAct?",
        "What is the travel budget?",
        "Calculate 100 * 5",
    ]

    for q in test_questions:
        print(f"\n\n{'#' * 60}")
        print(f"  QUERY: {q}")
        print(f"{'#' * 60}")
        result = run_with_evaluation(q, verbose=True)
        print_result(result)
