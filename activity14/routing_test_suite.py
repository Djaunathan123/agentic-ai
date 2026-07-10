"""
Week 5 — Requirement B
10-Query Routing Test Suite + Accuracy Report
Run: python routing_test_suite.py
"""

import time
from project_react_loop import react_loop

# ---------------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------------

ROUTING_TESTS = [
    # Factual recall → search_documents
    ("What is ReAct?",                    "search_documents"),
    ("What is the travel budget?",        "search_documents"),
    ("What did we learn about chunking?", "search_documents"),
    # Math → calculate
    ("Calculate 45 * 12",                "calculate"),
    ("What is 15% of 3000?",             "calculate"),
    ("What is 2 to the power of 10?",    "calculate"),
    # Ambiguous → clarify
    ("Help me with that thing",          "clarify"),
    ("Do the stuff I asked",             "clarify"),
    # Greeting / small-talk → no tool
    ("Hello!",                           None),
    ("Thank you!",                       None),
]

# ---------------------------------------------------------------------------
# TEST RUNNER
# ---------------------------------------------------------------------------

def test_routing_accuracy():
    print("\n" + "=" * 65)
    print("  Week 5 — Requirement B: Routing Accuracy Test Suite")
    print("=" * 65)
    print(f"  {'Status':<6} {'Expected':<22} {'Got':<22} Query")
    print("-" * 65)

    correct = 0

    for query, expected in ROUTING_TESTS:
        transcript = react_loop(query)

        actual_tool = None
        for entry in transcript:
            if entry.get("phase") == "ACTION":
                actual_tool = entry.get("tool")
                break

        match = actual_tool == expected
        status = "✓" if match else "✗"
        correct += 1 if match else 0

        print(
            f"  {status:<6} "
            f"{str(expected):<22} "
            f"{str(actual_tool):<22} "
            f"{query[:40]}"
        )

        # Free tier: 15 req/min — 6s delay keeps us at ~10 req/min safely
        time.sleep(6)

    total = len(ROUTING_TESTS)
    accuracy = correct / total * 100
    print("-" * 65)
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0f}%)")

    if accuracy < 80:
        print("  ⚠  Below 80% — review tool descriptions in project_react_loop.py")
    else:
        print("  ✅ Passed target of ≥ 80% routing accuracy")

    print("=" * 65)
    return correct, total


if __name__ == "__main__":
    test_routing_accuracy()
