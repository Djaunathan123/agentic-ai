"""
Week 5 — Routing Test Suite
Requirement B: 10-query test suite with accuracy report
"""

from project_react_loop import react_loop

# ---------------------------------------------------------------------------
# TEST CASES  (query, expected_first_tool_or_None)
# ---------------------------------------------------------------------------

ROUTING_TESTS = [
    # Factual recall → search_documents
    ("What is ReAct?",                   "search_documents"),
    ("What is the travel budget?",       "search_documents"),
    ("What did we learn about chunking?","search_documents"),
    # Math → calculate
    ("Calculate 45 * 12",               "calculate"),
    ("What is 15% of 3000?",            "calculate"),
    ("What is 2 to the power of 10?",   "calculate"),
    # Ambiguous → clarify
    ("Help me with that thing",         "clarify"),
    ("Do the stuff I asked",            "clarify"),
    # Greeting / small-talk → no tool (None)
    ("Hello!",                          None),
    ("Thank you!",                      None),
]


def test_routing_accuracy():
    """
    Run all routing tests and print a per-query result table
    followed by an overall accuracy score.
    """
    print("\n" + "=" * 65)
    print("  Week 5 — Routing Accuracy Test Suite (Requirement B)")
    print("=" * 65)
    print(f"  {'Status':<6} {'Expected':<22} {'Got':<22} Query")
    print("-" * 65)

    correct = 0

    for query, expected in ROUTING_TESTS:
        transcript = react_loop(query)

        # Find the first tool called (if any)
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

    total = len(ROUTING_TESTS)
    accuracy = correct / total * 100
    print("-" * 65)
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0f}%)")

    if accuracy < 80:
        print("\n  ⚠  Below 80% — review tool descriptions in project_react_loop.py")
    else:
        print("\n  ✅ Passed target of ≥ 80% routing accuracy")

    print("=" * 65)
    return correct, total


if __name__ == "__main__":
    test_routing_accuracy()
