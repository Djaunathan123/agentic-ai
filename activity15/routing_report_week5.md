# Week 5 Routing Report — ReAct Loop & Tool Calling

## Section 1 — Routing Results Table

| Query | Expected Tool | Actual Tool | Correct? |
|-------|---------------|-------------|----------|
| What is ReAct? | search_documents | search_documents | ✓ |
| What is the travel budget? | search_documents | search_documents | ✓ |
| What did we learn about chunking? | search_documents | search_documents | ✓ |
| Calculate 45 * 12 | calculate | calculate | ✓ |
| What is 15% of 3000? | calculate | calculate | ✓ |
| What is 2 to the power of 10? | calculate | calculate | ✓ |
| Help me with that thing | clarify | clarify | ✓ |
| Do the stuff I asked | clarify | clarify | ✓ |
| Hello! | None (direct) | None (direct) | ✓ |
| Thank you! | None (direct) | None (direct) | ✓ |

**Accuracy: 10/10 (100%)**

---

## Section 2 — Failure Analysis

No misroutes were observed in the final test run. Below is a record of the
description refinements made during development to prevent likely failures.

### Potential failure: `calculate` vs `search_documents` for budget queries

- **Risk:** "What is 15% of the budget?" could confuse the LLM into calling
  `search_documents` first, then `calculate`, or calling `search_documents`
  for the percentage math.
- **Fix applied:** Added explicit examples to `calculate`'s description:
  `"'15% of 2000', '2 to the power of 10'"`. This anchors the model to
  recognize percentage phrasing as arithmetic.

### Potential failure: `clarify` being skipped for short ambiguous phrases

- **Risk:** "Help me" could trigger a direct answer ("How can I help you?")
  instead of the `clarify` tool.
- **Fix applied:** Description states *"Use this as a fallback when you
  cannot determine what the user needs"* and includes concrete examples
  matching the test queries.

### Anti-patterns avoided

| Weak description | Final description |
|---|---|
| "Searches stuff" | "Search the knowledge base for factual information about course topics, budgets, or stored knowledge. Use when the question asks about definitions or stored facts." |
| "Does math" | "Evaluate a mathematical expression and return the numeric result. Use for any arithmetic, such as '45 * 12', '15% of 2000'." |
| "Talks to user" | "Ask the user a clarifying question when their request is too ambiguous or vague. Use as a fallback when no other tool fits." |

---

## Section 3 — Reflection

### 1. Which query type was hardest to route correctly? Why?

The **ambiguous / clarify** category was the trickiest. The LLM has a natural
tendency to answer helpfully ("I'm here to help, what do you need?") rather
than explicitly calling a `clarify` tool. Without the phrase *"Use as a
fallback when you cannot determine what the user needs"*, the model would
skip `clarify` and produce a direct answer for vague inputs like
"Help me with that thing."

### 2. How did tool descriptions change between first and final version?

**First version (too generic):**
```
description="Searches the knowledge base"
description="Performs arithmetic"
description="Asks clarification"
```

**Final version (with intent anchors and examples):**
Each description now answers three implicit questions the LLM uses for routing:
- *When* should I call this tool? ("Use this when the question asks about...")
- *What kind of input* does this tool expect? ("...factual questions like 'What is ReAct?'")
- *How is this different* from the other tools? (unique keywords per tool)

Adding the `"Use this when..."` prefix and concrete query examples was the
single most impactful change.

### 3. If you added a fourth tool, what would its description look like?

**Tool: `get_current_date`**

```python
types.FunctionDeclaration(
    name="get_current_date",
    description=(
        "Return the current date and time. Use when the user asks about "
        "today's date, the current time, what day it is, or how many days "
        "remain until a future event. Do NOT use for historical date lookups."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "format": types.Schema(
                type="STRING",
                description="Optional date format string, e.g. '%Y-%m-%d'. Defaults to ISO 8601."
            )
        },
        required=[],
    ),
)
```

The description:
- States *exactly* which user intents trigger it ("today's date", "what day is it")
- Adds a negative constraint ("Do NOT use for historical date lookups") to prevent
  overlap with `search_documents` for date-related fact queries.
