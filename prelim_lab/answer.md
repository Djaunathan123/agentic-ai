# Part I: Code Debugging/Correction (10 Points)

### 1. The Stateless Loop (2 pts)

**Error:**  
The agent creates a new chat session on every loop iteration, so it loses all conversation history

**Fix:**  
chat = client.chats.create(model="gemini-3.1-flash-lite")

### 2. The Leaky Identity (2 pts)

**Error:**  
The system instruction allows full answers because it only says "be helpful" without enforcing the constraint to avoid giving the direct answer.

**Fix (System Instruction):**  
identity = "You are a math tutor. You must NOT give direct answers. Only provide hints or guided reasoning."


### 3. The Memory Bloat (2 pts)

**Error:**  
This line incorrectly keeps only the first message instead of the last 2 turns, and it can also break expected chat structure.

**Fix (Line B):**  
chat.history = chat.history[-2:]

### 4. The Perception Crash (2 pts)

**Error:**  
The model output may not always include price, so Item(**response.parsed) crashes due to missing required field.

**Fix (Pydantic Model):**  
from typing import Optional

class Item(BaseModel):
    name: str
    price: Optional[float]

### 5. The Infinite Backoff (2 pts)

**Error:**  
All exceptions are retried indefinitely, including permanent failures like invalid API keys.

**Fix (Else Block):**  
break # stop retrying on non-429 errors

---

# Part II: Schema Design & Evaluation (10 Points)

## Task 1: The Multi-Agent Router (5 Points)

### Pydantic Schema

```python
from pydantic import BaseModel, Field
from enum import Enum

class Department(str, Enum):
    PAYROLL = "PAYROLL"
    RECRUITING = "RECRUITING"
    LEAVE_REQUEST = "LEAVE_REQUEST"

class HRAgentRouter(BaseModel):
    department: Department
    reasoning: str = Field(..., description="Reasoning for classification")
    urgency_level: int = Field(..., ge=1, le=5)
```

---
## Task 2: Architecture Evaluation (5 Points)
Architecture B is more resilient and cost-effective. It uses a Pydantic schema, which enforces structured outputs and reduces malformed responses. The sliding window also limits token usage, making it cheaper and more scalable for 24/7 operation.
