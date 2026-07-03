"""
Example Usage Script for RAG Chat App with RAG Triad Evaluation

This script demonstrates how to interact with the new evaluation endpoints.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def upload_document(file_path):
    """Upload a document to the RAG system."""
    print(f"\n📤 Uploading document: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print(f"✓ Response: {response.json()}")
    return response.json()


def chat_basic(question):
    """Get a basic answer without evaluation."""
    print(f"\n💬 Basic Chat: {question}")
    
    response = requests.get(
        f"{BASE_URL}/chat",
        params={"question": question}
    )
    
    result = response.json()
    print(f"✓ Answer: {result['answer']}")
    return result


def chat_with_evaluation(question):
    """Get an answer with full RAG Triad evaluation."""
    print(f"\n📊 Chat with Evaluation: {question}")
    
    response = requests.get(
        f"{BASE_URL}/chat",
        params={
            "question": question,
            "evaluate": True
        }
    )
    
    result = response.json()
    print(f"✓ Answer: {result['answer']}")
    
    if 'evaluation' in result:
        eval_data = result['evaluation']
        print(f"\n📈 Evaluation Results:")
        print(f"   Context Relevance:  {eval_data['context_relevance']:.2f} {'✓' if eval_data['context_relevance'] >= 0.7 else '✗'}")
        print(f"   Groundedness:       {eval_data['groundedness']:.2f} {'✓' if eval_data['groundedness'] >= 0.7 else '✗'}")
        print(f"   Answer Relevance:   {eval_data['answer_relevance']:.2f} {'✓' if eval_data['answer_relevance'] >= 0.7 else '✗'}")
        print(f"   Average Score:      {eval_data['average_score']:.2%}")
        print(f"   Decision:           {eval_data['decision']}")
        print(f"   Reason:             {eval_data['reason']}")
    
    return result


def evaluate_answer(question, answer):
    """Evaluate a specific answer."""
    print(f"\n🔍 Manual Evaluation")
    print(f"   Question: {question}")
    print(f"   Answer:   {answer}")
    
    response = requests.get(
        f"{BASE_URL}/evaluate",
        params={
            "question": question,
            "answer": answer
        }
    )
    
    result = response.json()
    
    if 'error' in result:
        print(f"✗ Error: {result['error']}")
        return result
    
    eval_data = result['evaluation']
    print(f"\n📈 Evaluation Results:")
    print(f"   Context Relevance:  {eval_data['context_relevance']:.2f} {'✓' if eval_data['context_relevance'] >= 0.7 else '✗'}")
    print(f"   Groundedness:       {eval_data['groundedness']:.2f} {'✓' if eval_data['groundedness'] >= 0.7 else '✗'}")
    print(f"   Answer Relevance:   {eval_data['answer_relevance']:.2f} {'✓' if eval_data['answer_relevance'] >= 0.7 else '✗'}")
    print(f"   Average Score:      {eval_data['average_score']:.2%}")
    print(f"   Decision:           {eval_data['decision']}")
    print(f"   Reason:             {eval_data['reason']}")
    
    return result


def compare_answers(question, answers):
    """Compare multiple answers to the same question."""
    print(f"\n🏆 Comparing Answers for: {question}")
    print("=" * 70)
    
    results = []
    for i, answer in enumerate(answers, 1):
        print(f"\nCandidate {i}:")
        result = evaluate_answer(question, answer)
        if 'evaluation' in result:
            results.append({
                'answer': answer,
                'score': result['evaluation']['average_score']
            })
            print("-" * 70)
    
    # Rank answers
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"\n🏅 Ranking:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.2%}")
        print(f"   Answer: {result['answer'][:80]}...")


def main():
    """Run example demonstrations."""
    print("=" * 70)
    print("RAG Chat App - RAG Triad Evaluation Examples")
    print("=" * 70)
    
    # Example 1: Basic chat
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Chat (No Evaluation)")
    print("=" * 70)
    chat_basic("What is machine learning?")
    
    # Example 2: Chat with evaluation
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Chat with Full Evaluation")
    print("=" * 70)
    chat_with_evaluation("What is machine learning?")
    
    # Example 3: Evaluate different answer qualities
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Evaluating Answer Quality")
    print("=" * 70)
    
    question = "What is Python?"
    
    # Good answer
    good_answer = "Python is a high-level, interpreted programming language known for its simple syntax and widespread use in data science, AI, and web development."
    
    # Hallucinated answer
    hallucinated_answer = "Python is only used for web development and is written exclusively in C++."
    
    # Off-topic answer
    offtopic_answer = "Java is a programming language that runs on the Java Virtual Machine."
    
    evaluate_answer(question, good_answer)
    time.sleep(1)
    evaluate_answer(question, hallucinated_answer)
    time.sleep(1)
    evaluate_answer(question, offtopic_answer)
    
    # Example 4: Compare answers
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Comparing Multiple Answers")
    print("=" * 70)
    
    compare_answers(
        "What is artificial intelligence?",
        [
            "AI is the simulation of human intelligence by machines.",
            "AI is a technology used only in robots.",
            "AI refers to intelligent behavior demonstrated by machines.",
        ]
    )


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server.")
        print("Make sure the backend is running:")
        print("  cd backend")
        print("  uvicorn app:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
