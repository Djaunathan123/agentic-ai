"""
RAG Triad Evaluation Framework
Evaluates retrieval-augmented generation quality across three dimensions:
1. Context Relevance: Is the retrieved chunk relevant to the question?
2. Groundedness: Is the answer supported by the retrieved context?
3. Answer Relevance: Does the answer address the original question?
"""

import time
import random
from functools import wraps
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


class EvaluationResult(BaseModel):
    """Structured evaluation result with score and justification."""
    score: float = Field(description="Score between 0.0 and 1.0")
    reason: str = Field(description="Brief explanation justifying the score")


def with_exponential_backoff(max_retries=5, base_delay=2.0):
    """
    Decorator for exponential backoff retry logic.
    Handles HTTP 429 (Too Many Requests) errors gracefully.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    # Check for rate limit error (429)
                    if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and retries < max_retries:
                        delay = (base_delay * (2 ** retries)) + random.uniform(0, 1)
                        print(f"⚠️  Rate limited. Retrying in {delay:.2f}s... (Attempt {retries + 1}/{max_retries})")
                        time.sleep(delay)
                        retries += 1
                        last_exception = e
                    else:
                        raise e
            
            # If we exhausted retries
            if last_exception:
                raise Exception(f"Max retries ({max_retries}) exceeded: {last_exception}")
            
        return wrapper
    return decorator


@with_exponential_backoff(max_retries=4, base_delay=2.0)
def evaluate_metric(criterion: str, source_text: str, generated_answer: str) -> EvaluationResult:
    """
    Uses Gemini as an impartial judge to score a specific RAG metric.
    """
    prompt = f"""
You are an impartial judge evaluating a generated answer based on a specific criterion.

Criterion:
{criterion}

Source Text:
{source_text}

Generated Answer:
{generated_answer}

Provide your evaluation as a JSON object containing a 'score' (float 0.0 to 1.0) and a 'reason' (string).
"""
    
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluationResult,
            temperature=0.1,
        ),
    )
    
    if response.parsed:
        return response.parsed
    else:
        return EvaluationResult.model_validate_json(response.text)


def evaluate_rag_triad(question: str, context: str, answer: str, threshold: float = 0.7) -> dict:
    """
    Runs full RAG Triad evaluation with three checks.
    
    Returns:
        dict with scores, reasons, and routing decision
    """
    print("\n" + "="*60)
    print("STARTING RAG TRIAD EVALUATION")
    print("="*60)
    
    # Check 1: Context Relevance
    print("\n📋 Evaluating Leg 1/3: Context Relevance...")
    context_criterion = (
        f"Is the provided source text highly relevant for answering "
        f"the user's query: '{question}'? "
        "Does it contain necessary information? "
        "(0.0 = completely irrelevant, 1.0 = perfectly relevant)"
    )
    context_relevance = evaluate_metric(context_criterion, context, question)
    print(f"   ✓ Context Relevance: {context_relevance.score:.2f}")
    print(f"     Reason: {context_relevance.reason}")
    
    # Check 2: Groundedness
    print("\n📋 Evaluating Leg 2/3: Groundedness...")
    groundedness_criterion = (
        "Does the generated answer rely ONLY on the provided source text? "
        "Check for hallucinations or outside assumptions. "
        "(0.0 = not grounded at all, 1.0 = fully grounded)"
    )
    groundedness = evaluate_metric(groundedness_criterion, context, answer)
    print(f"   ✓ Groundedness: {groundedness.score:.2f}")
    print(f"     Reason: {groundedness.reason}")
    
    # Check 3: Answer Relevance
    print("\n📋 Evaluating Leg 3/3: Answer Relevance...")
    relevance_criterion = (
        f"Does the generated answer directly and completely address "
        f"the user's query: '{question}'? "
        "(0.0 = completely irrelevant, 1.0 = perfectly answers)"
    )
    answer_relevance = evaluate_metric(relevance_criterion, context, answer)
    print(f"   ✓ Answer Relevance: {answer_relevance.score:.2f}")
    print(f"     Reason: {answer_relevance.reason}")
    
    # Determine routing decision
    print("\n" + "="*60)
    print("ROUTING DECISION")
    print("="*60)
    
    decision = "ACCEPT"
    reason = ""
    
    if context_relevance.score < threshold:
        decision = "RE-RETRIEVE"
        reason = "Vector DB pulled irrelevant context. Generation was compromised from start."
    elif groundedness.score < threshold:
        decision = "RE-GENERATE"
        reason = "Good context retrieved but answer contains hallucinations."
    elif answer_relevance.score < threshold:
        decision = "RE-PROMPT"
        reason = "Answer is grounded but missed the point of the question."
    else:
        reason = "All three triad metrics passed quality checks!"
    
    print(f"\n>>> Final Decision: {decision}")
    print(f"    Reason: {reason}\n")
    
    return {
        "context_relevance": context_relevance.score,
        "groundedness": groundedness.score,
        "answer_relevance": answer_relevance.score,
        "decision": decision,
        "reason": reason,
        "average_score": (
            context_relevance.score + groundedness.score + answer_relevance.score
        ) / 3
    }
