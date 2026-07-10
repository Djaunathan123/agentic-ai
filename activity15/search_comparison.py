"""
Activity 14 — Requirement C
Before/After Search Comparison Script

Runs 10 queries against:
  1. Baseline  — plain Qdrant cosine search (no improvements)
  2. Improved  — Query Expansion + Hybrid Search
  3. Oracle    — expected keyword that should appear in the result

Run:
    python search_comparison.py
"""

import time
from bridging_search_tool import search_documents, original_search_documents

# ---------------------------------------------------------------------------
# TEST QUERIES
# Topics drawn from the uploaded documents in the Qdrant collection:
#   - "What is a Rainbow.txt"
#   - "About Solar System.txt"
#   - "weather_and_precipitation_transcript.txt"
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "query": "What is a rainbow?",
        "expected_chunk": "rainbow",
        "topics": ["rainbow", "light"],
    },
    {
        "query": "How are rainbows formed?",
        "expected_chunk": "light",
        "topics": ["rainbow", "refraction"],
    },
    {
        "query": "What colors appear in a rainbow?",
        "expected_chunk": "color",
        "topics": ["rainbow", "spectrum"],
    },
    {
        "query": "What is the solar system?",
        "expected_chunk": "solar",
        "topics": ["solar system", "planets"],
    },
    {
        "query": "How many planets are in the solar system?",
        "expected_chunk": "planet",
        "topics": ["solar system", "planets"],
    },
    {
        "query": "What is the Sun made of?",
        "expected_chunk": "sun",
        "topics": ["solar system", "sun"],
    },
    {
        "query": "What causes rain?",
        "expected_chunk": "rain",
        "topics": ["weather", "precipitation"],
    },
    {
        "query": "How does precipitation form?",
        "expected_chunk": "water",
        "topics": ["weather", "precipitation"],
    },
    {
        "query": "What is evaporation?",
        "expected_chunk": "evapor",
        "topics": ["weather", "water cycle"],
    },
    {
        "query": "Explain the water cycle",
        "expected_chunk": "water",
        "topics": ["weather", "water cycle"],
    },
]

# ---------------------------------------------------------------------------
# COMPARISON RUNNER
# ---------------------------------------------------------------------------

def compare_search(queries: list[dict]):
    print("\n" + "=" * 80)
    print("  Activity 14 — Requirement C: Before/After Search Comparison")
    print("=" * 80)

    baseline_hits = 0
    improved_hits = 0
    results_table = []

    for i, item in enumerate(queries, 1):
        q        = item["query"]
        expected = item["expected_chunk"].lower()

        print(f"\n[{i:02d}] Query   : {q}")
        print(f"      Expected: contains '{expected}'")

        # --- Baseline ---
        baseline = original_search_documents(q)
        b_hit = expected in baseline.lower() if baseline else False
        baseline_hits += 1 if b_hit else 0
        print(f"      Baseline: {'✓ HIT' if b_hit else '✗ MISS'}")
        print(f"        → {baseline[:120].strip()}...")

        # Small delay to respect free-tier RPM limit
        time.sleep(6)

        # --- Improved ---
        improved = search_documents(q)
        i_hit = expected in improved.lower() if improved else False
        improved_hits += 1 if i_hit else 0
        print(f"      Improved: {'✓ HIT' if i_hit else '✗ MISS'}")
        print(f"        → {improved[:120].strip()}...")

        results_table.append({
            "query": q,
            "expected": expected,
            "baseline_hit": b_hit,
            "improved_hit": i_hit,
            "delta": (1 if i_hit else 0) - (1 if b_hit else 0),
        })

        # Delay between queries
        time.sleep(6)

    # --- Summary ---
    total = len(queries)
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Query':<45} {'Expected':<18} {'Baseline':^10} {'Improved':^10} {'Delta':^6}")
    print("-" * 80)
    for r in results_table:
        b = "✓" if r["baseline_hit"] else "✗"
        im = "✓" if r["improved_hit"] else "✗"
        d  = f"+{r['delta']}" if r["delta"] > 0 else str(r["delta"])
        print(f"  {r['query']:<45} {r['expected']:<18} {b:^10} {im:^10} {d:^6}")
    print("-" * 80)
    print(f"  Baseline accuracy : {baseline_hits}/{total} ({baseline_hits/total*100:.0f}%)")
    print(f"  Improved accuracy : {improved_hits}/{total} ({improved_hits/total*100:.0f}%)")
    net = improved_hits - baseline_hits
    print(f"  Net improvement   : {'+' if net >= 0 else ''}{net} queries")
    print("=" * 80)


if __name__ == "__main__":
    compare_search(TEST_QUERIES)
