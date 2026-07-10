"""
Week 5 — RAG Triad Integration
Requirement C: Wrap the ReAct loop with RAG Triad quality gate + self-correction
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from project_react_loop import react_loop, print_transcript
from evaluation import evaluate_rag_triad

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------------------------------------------------------
# SELF-CORRECTION
# ---------------------------------------------------------------------------

def self_correct_answer(
    question: str,
    context: str,
    original_answer: str,
    evaluation: dict,
) -> str:
    """
    Ask the LLM to produce a better answer given the quality feedback.
    Triggered only when groundedness or answer_relevance fail.
    """
    weak_leg = min(
        [
            ("groundedness", evaluation["groundedness"]),
            ("answer_relevance", evaluation["answer_relevance"]),
        ],
        key=lambda x: x[1],
    )[0]

    correction_prompt = f"""
The previous answer failed the RAG Triad quality check on: {weak_leg}.

Original question: {question}
Retrieved context:
{context}

Original answer (which failed): {original_answer}

Please write a corrected answer that:
- Is ONLY based on the provided context (no outside information)
- Directly and completely addresses the question
- Is concise and accurate
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=correction_prompt,
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# MAIN INTEGRATION FUNCTION
# ---------------------------------------------------------------------------

def run_with_evaluation(question: str, verbose: bool = True) -> dict:
    """
    Full pipeline: ReAct loop → RAG Triad evaluation → optional self-correction.

    Returns a result dict with:
        question, answer, transcript,
        context_relevance, groundedness, answer_relevance,
        passed, (optionally) corrected_answer, was_corrected
    """
    # 1. Run ReAct loop
    transcript = react_loop(question)

    if verbose:
        print_transcript(transcript)

    # 2. Extract answer and retrieved context from transcript
    answer = None
    context = None
    for entry in transcript:
        if entry["phase"] == "ANSWER":
            answer = entry["content"]
        if entry["phase"] == "OBSERVE":
            context = entry["content"]

    # Graceful fallback if no answer or context
    answer = answer or ""
    context = context or ""

    # 3. RAG Triad evaluation
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
        "context_relevance": evaluation["context_relevance"],
        "groundedness": evaluation["groundedness"],
        "answer_relevance": evaluation["answer_relevance"],
        "average_score": evaluation["average_score"],
        "decision": evaluation["decision"],
        "passed": passed,
        "was_corrected": False,
        "corrected_answer": None,
    }

    # 4. Self-correct if groundedness or answer_relevance failed
    if not passed and evaluation["decision"] in ("RE-GENERATE", "RE-PROMPT"):
        print("\n🔄 Self-correcting answer due to quality gate failure...")
        corrected = self_correct_answer(question, context, answer, evaluation)
        result["corrected_answer"] = corrected
        result["was_corrected"] = True
        if verbose:
            print(f"\n  Corrected answer:\n  {corrected}")

    return result


def print_result(result: dict):
    """Display the evaluation scores and quality gate decision."""
    print("\n" + "─" * 60)
    print("  RAG Triad Scores")
    print("─" * 60)

    def badge(score):
        return "✅" if score >= 0.7 else "❌"

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
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Week 5 — RAG Triad Integration Demo (Requirement C)")
    print("=" * 60)

    test_questions = [
        "What is ReAct?",
        "What is the travel budget?",
        "Calculate 100 * 5",
    ]

    for q in test_questions:
        print(f"\n\n>>> QUERY: {q}")
        result = run_with_evaluation(q, verbose=True)
        print_result(result)
