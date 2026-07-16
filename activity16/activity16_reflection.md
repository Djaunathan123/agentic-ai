# Activity 16 — Reflection

---

**1. What was the hardest part of converting your ReAct loop to LangGraph?**

The hardest part was separating routing logic from execution logic. In the ReAct loop, both lived in the same `for` loop — detect a function call, run it, append the result, repeat. In LangGraph, each node only mutates state and returns it, while all routing decisions live exclusively in `should_retry()`. The temptation was to keep writing `if/else` branches inside the nodes, but doing so would break the graph's single-source-of-truth design.

A concrete bug I hit: I originally set `state["route_decision"]` inside `should_retry()`. Since `should_retry` is a conditional edge function (not a node), LangGraph doesn't persist its state mutations — the function is just a router, not a state writer. `route_decision` was always empty in the final result. Moving that assignment into `evaluate_node` fixed it immediately.

---

**2. How did the `route_decision` field help with debugging? Would you add more fields?**

`route_decision` let me see at a glance whether the agent finished cleanly (`accept`), hit the iteration cap (`accept_maxed`), or kept rewriting. From the actual run:

- Questions 1–3 and "What did the course say?" all showed `accept` — healthy path.
- "Tell me about that thing from class" showed `accept` after 2 iterations — confirmed the rewrite worked.
- "Do the stuff" showed `accept_maxed` — confirmed the graph hit MAX_ITERATIONS gracefully.

Without this field, I'd have to infer the outcome by scanning the iteration count and final scores together.

I'd add a `rewrite_history: list` field to capture each rewritten query alongside the scores that triggered it. Right now the log records the question at each iteration, but it doesn't make the causal chain explicit — you have to visually line up entries to see that iteration 2's question was a rewrite of iteration 1's.

---

**3. Why are both `MAX_ITERATIONS` in `should_retry` and `recursion_limit` in `app.invoke()` necessary?**

They operate at different levels:

`MAX_ITERATIONS` in `should_retry` is the **application rule** — "after 3 tries, return whatever we have." It's intentional product behavior. It fired correctly for "Do the stuff": after 3 iterations with CR only reaching 0.20, the graph accepted with `accept_maxed` rather than running forever.

`recursion_limit` in `app.invoke()` is the **framework safety net** — it counts graph steps, not iterations. One full cycle (generate → evaluate → rewrite) consumes 3 steps. If `recursion_limit` were set to 3, LangGraph would cut the run off mid-cycle before `should_retry` even gets a chance to fire `accept_maxed`. In our run, `recursion_limit=10` gave the graph room to complete all 3 full cycles for "Do the stuff" (9 steps) with one step to spare.

If you set `recursion_limit=3`, the third question in a rewrite cycle would never reach the evaluate node and LangGraph would raise a `GraphRecursionError` instead of returning a graceful answer.

---

**4. Is running 3 judge calls per iteration (up to 9 per question) cost-justified?**

For the 6-question demo, 9 calls was the maximum and only "Do the stuff" hit it. Most questions used 3 calls (1 iteration). The quality gain was real — the judges correctly identified CR=0.00 for vague queries and triggered rewrites that improved scores on the next iteration.

For production, ways to reduce cost without losing quality:

- **Skip re-scoring unchanged legs.** If the same chunk is retrieved twice in a row (same query embedding lands on the same top result), GR won't change — skip re-evaluating it.
- **Only re-run failing judges.** After iteration 1, if only CR failed, only re-score CR on iteration 2. GR and AR from the previous pass are still valid for comparison.
- **Use a smaller judge model.** The judge prompts are short and structured — a lighter model handles them accurately at a fraction of the cost.

---

**5. How would you add `calculate` and `clarify` back into the graph?**

They need a **router node** at the entry point, before `generate`, that classifies the question:

```
User Question
     │
     ▼
┌──────────────┐
│    route     │  ← LLM classifies: "search" | "calculate" | "clarify"
└──────┬───────┘
       │
  ┌────┼─────────────┐
  ▼    ▼             ▼
search  calculate   clarify
  │       │            │
  ▼       ▼            ▼
evaluate  END          END
  │
  ▼
should_retry → ...
```

`calculate` and `clarify` would be leaf nodes that write directly to `state["answer"]` and route to `END`, bypassing the evaluate/rewrite cycle entirely — there is nothing to retrieve or score for math and clarifications.

This also would have helped "Do the stuff": a `clarify` path would have intercepted it at iteration 1 instead of burning all 3 iterations trying to guess the user's intent from a meaningless query.
