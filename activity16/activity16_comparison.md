# Activity 16 — ReAct Loop vs. LangGraph: Comparison Report

---

## 1. Architecture Diagrams

### Week 5 — ReAct Loop (activity15)

```
User Question
     │
     ▼
┌─────────────────────────────┐
│  react_loop()               │
│                             │
│  for turn in range(max):    │
│    ┌─────────────────────┐  │
│    │  REASON             │  │
│    │  LLM picks a tool   │  │
│    └────────┬────────────┘  │
│             │               │
│    ┌────────▼────────────┐  │
│    │  ACT                │  │
│    │  call tool()        │  │
│    └────────┬────────────┘  │
│             │               │
│    ┌────────▼────────────┐  │
│    │  OBSERVE            │  │
│    │  append to history  │  │
│    └────────┬────────────┘  │
│             │               │
│        if no tool call:     │
│          break → ANSWER     │
└─────────────────────────────┘
     │
     ▼
run_with_evaluation()
  → RAG Triad (3 judges)
  → if fail: self_correct_answer()  ← one manual answer rewrite
     │
     ▼
 Final Answer
```

Key routing logic is scattered across `if fc_part is not None` inside `react_loop()`,
and a separate `if not passed` block inside `run_with_evaluation()`.

---

### Week 6 — LangGraph (activity16)

```
User Question
     │
     ▼
┌────────────┐
│  generate  │  ← search_documents() + LLM answer
└─────┬──────┘
      │
      ▼
┌────────────┐
│  evaluate  │  ← 3 RAG Triad judges score CR / GR / AR
│            │    sets route_decision in state
└─────┬──────┘
      │
 ┌────┴────┐
 │         │
[accept] [rewrite]  ← should_retry() reads state, returns route string
 │         │
 ▼         ▼
END    ┌──────────┐
       │  rewrite │  ← LLM rewrites the query for better retrieval
       └────┬─────┘
            │
            └──► back to generate
```

All routing logic lives in one place: `should_retry()`.

---

## 2. Comparison Table — 5 Questions Through Both Architectures

| # | Question | ReAct Answer (W5) | LangGraph Answer (W16) | ReAct Iters | LangGraph Iters | Which was better? |
|---|----------|-------------------|------------------------|-------------|-----------------|-------------------|
| 1 | What is a rainbow? | A rainbow is an optical phenomenon caused by reflection, refraction, and dispersion of light in water droplets. | A rainbow is a natural phenomenon that appears when sunlight passes through tiny water droplets — refracted, reflected, and dispersed into different colors. | 1 | 1 (CR=1.00 GR=1.00 AR=1.00) | Comparable |
| 2 | What causes rain and precipitation? | Rain is caused by water droplets in clouds becoming heavy enough to fall. | Precipitation occurs when tiny water droplets or ice crystals in clouds become too heavy to stay suspended. Rain specifically forms when liquid droplets combine and grow heavy enough to fall. | 1 | 1 (CR=1.00 GR=1.00 AR=1.00) | LangGraph (more complete) |
| 3 | How many planets are in the solar system? | There are 8 planets in the solar system. | Our Solar System has 8 major planets. | 1 | 1 (CR=1.00 GR=1.00 AR=1.00) | Comparable |
| 4 | Tell me about that thing from class | [Clarify] Could you clarify: Which topic from class are you asking about? | *(rewrite triggered)* Rewrote to "What is the definition and explanation of precipitation." Returned a correct answer on iteration 2. (CR=1.00 GR=1.00 AR=1.00) | 1 | 2 | Depends on context — ReAct is faster but gives up; LangGraph self-corrects and delivers a real answer |
| 5 | Do the stuff | [Clarify] Could you clarify: What task would you like me to do? | *(rewrite triggered twice)* Reached MAX_ITERATIONS=3. CR stayed at 0.20 — context never matched the vague intent. Final answer: "context does not contain SOPs for rain gauges." | 1 | 3 (accept_maxed) | ReAct (faster for truly unanswerable vague queries) |

---

## 3. Self-Correction Demonstration (Actual Run Output)

### Happy Path — 1 iteration each, all scores 1.00

```
Question: What is a rainbow?
  [Generate] Iteration 1
  [Evaluate] CR: 1.00 | GR: 1.00 | AR: 1.00
  [Decision] All scores >= 0.7. Accepting.
  Route Decision: accept

Question: What causes rain and precipitation?
  [Generate] Iteration 1
  [Evaluate] CR: 1.00 | GR: 1.00 | AR: 1.00
  [Decision] All scores >= 0.7. Accepting.
  Route Decision: accept

Question: How many planets are in the solar system?
  [Generate] Iteration 1
  [Evaluate] CR: 1.00 | GR: 1.00 | AR: 1.00
  [Decision] All scores >= 0.7. Accepting.
  Route Decision: accept
```

### Self-Correction Triggered — "Tell me about that thing from class"

```
  [Generate] Iteration 1 | Q: Tell me about that thing from class
  [Evaluate] CR: 0.00 | GR: 1.00 | AR: 1.00
  [Decision] Min score 0.00 < 0.7. Rewriting...
  [Rewrite] → 'What is the definition and explanation of precipitation...'
  [Generate] Iteration 2
  [Evaluate] CR: 1.00 | GR: 1.00 | AR: 1.00
  [Decision] All scores >= 0.7. Accepting.
  Route Decision: accept
```

Rewrite turned a zero-CR query into a perfect-scoring one on the second try.

### MAX_ITERATIONS Reached — "Do the stuff"

```
  [Generate] Iteration 1 | Q: Do the stuff
  [Evaluate] CR: 0.00 | GR: 1.00 | AR: 1.00
  [Decision] Min score 0.00 < 0.7. Rewriting...
  [Rewrite] → 'What are the specific steps and procedures for measuring...'
  [Generate] Iteration 2
  [Evaluate] CR: 0.20 | GR: 1.00 | AR: 1.00
  [Decision] Min score 0.20 < 0.7. Rewriting...
  [Rewrite] → 'What are the standard operational procedures (SOPs)...'
  [Generate] Iteration 3
  [Evaluate] CR: 0.20 | GR: 1.00 | AR: 1.00
  [Decision] Max iterations (3) reached. Accepting.
  Route Decision: accept_maxed
```

Graph gracefully stopped. CR improved from 0.00 → 0.20 across rewrites but never reached threshold — the query had no recoverable intent in the knowledge base.

---

## 4. Failure Analysis

### "Tell me about that thing from class" — why the rewrite helped

Original query had CR=0.00 because "that thing from class" has no semantic overlap with any stored document chunk. The rewrite node used the retrieved chunk (which happened to mention precipitation) as a clue and produced a more specific query about precipitation. On iteration 2, Qdrant returned a highly relevant chunk and all three scores hit 1.00.

### "Do the stuff" — why rewrites did not fully recover

The query is too abstract — there is no recoverable semantic intent without user input. Each rewrite tried to guess (measuring precipitation, rain gauge SOPs), and while CR improved from 0.00 to 0.20, none of the guesses matched what the user actually meant. The graph handled this correctly: it tried its best for 3 iterations then accepted gracefully with `accept_maxed`. The ReAct loop would have routed this to `clarify` immediately, which is faster for this edge case.

### "What did the course say?" — unexpectedly passed in 1 iteration

This vague query passed because Qdrant returned a weather course transcript chunk that actually does describe course content. CR=1.00 because the chunk and question were a reasonable semantic match. LangGraph accepted it without rewriting.

---

## 5. Architecture Differences Summary

| Dimension | ReAct Loop (Week 5) | LangGraph (Week 6) |
|-----------|--------------------|--------------------|
| **State** | `WorkingMemory.history` — flat list of dicts | Typed `AgentState` TypedDict — every field declared |
| **Routing** | `if fc_part is not None` inside the loop | `should_retry()` conditional edge — one function |
| **Evaluation** | Post-loop wrapper (`run_with_evaluation`) | Built-in node (`evaluate_node`) — runs every iteration |
| **Self-correction** | Rewrites the *answer* once | Rewrites the *query*, re-retrieves, re-generates — up to MAX_ITERATIONS |
| **Safety** | `for turn in range(max_iterations)` | `MAX_ITERATIONS` check + `recursion_limit` in `app.invoke()` |
| **Traceability** | Print statements | Structured `log` list with per-iteration scores and reasons |
| **Tool selection** | LLM dynamically picks: search / calculate / clarify | Always `search_documents` — no dynamic dispatch |
| **route_decision** | Not tracked | Saved in `AgentState` — visible in final result |
