# RAG Chat App - RAG Triad & Exponential Backoff Integration

## Overview

This project now includes two critical enhancements to the RAG (Retrieval-Augmented Generation) chat application:

1. **RAG Triad Evaluation Framework** - A quality assurance system that evaluates responses across three dimensions
2. **Exponential Backoff Retry Logic** - Graceful handling of API rate limits (HTTP 429 errors)

## Features

### 1. RAG Triad Evaluation Framework

The RAG Triad is a three-part quality checklist that evaluates whether a retrieval-augmented agent:
- ✅ Retrieved the right context
- ✅ Stayed grounded in it
- ✅ Actually answered the question

#### The Three Checks

| Check | Compares | Asks | Threshold |
|-------|----------|------|-----------|
| **Context Relevance** | Question ↔ Retrieved Chunk | "Is this chunk relevant to the question?" | 0.7+ |
| **Groundedness** | Retrieved Chunk ↔ Answer | "Is every claim in the answer supported by the chunk?" | 0.7+ |
| **Answer Relevance** | Question ↔ Answer | "Does the answer address the original question?" | 0.7+ |

#### Scoring Thresholds
- **0.8–1.0**: High confidence — the component is working well
- **0.6–0.8**: Acceptable but worth monitoring
- **Below 0.6**: Likely failure — investigate and correct

#### Routing Decisions

Based on which check fails, the system recommends:

- **RE-RETRIEVE**: Context was irrelevant → improve chunking or query
- **RE-GENERATE**: Context was good but answer hallucinated → constrain generation
- **RE-PROMPT**: Answer was grounded but off-topic → clarify user intent
- **ACCEPT**: All checks passed → response is acceptable

### 2. Exponential Backoff Retry Logic

Handles API rate limits gracefully using exponential backoff with jitter:

```
Attempt | Base Wait | With Jitter (example)
--------|-----------|---------------------
0       | 1s        | 1.3s
1       | 2s        | 2.7s
2       | 4s        | 4.2s
3       | 8s        | 8.9s
4       | 16s       | 16.4s
```

**Features:**
- Automatic retry on HTTP 429 (Too Many Requests) errors
- Random jitter to prevent thundering herd problem
- Configurable max retries and base delay
- Detailed logging of retry attempts
- Applied to all API calls (embeddings, generation, evaluation)

## API Endpoints

### 1. Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

# Request
curl -X POST -F "file=@document.txt" http://localhost:8000/upload

# Response
{"message": "Uploaded successfully"}
```

### 2. Chat (Basic)
```bash
GET /chat?question=What%20is%20Python%3F

# Response
{"answer": "Python is a high-level programming language..."}
```

### 3. Chat with Evaluation
```bash
GET /chat?question=What%20is%20Python%3F&evaluate=true

# Response
{
  "answer": "Python is a high-level programming language...",
  "evaluation": {
    "context_relevance": 0.95,
    "groundedness": 0.92,
    "answer_relevance": 0.88,
    "decision": "ACCEPT",
    "reason": "All three triad metrics passed quality checks!",
    "average_score": 0.917
  }
}
```

### 4. Manual Answer Evaluation
```bash
GET /evaluate?question=What%20is%20Python%3F&answer=Python%20is%20a%20snake

# Response
{
  "question": "What is Python?",
  "answer": "Python is a snake",
  "context": "Python is a high-level, interpreted programming language...",
  "evaluation": {
    "context_relevance": 0.95,
    "groundedness": 0.15,
    "answer_relevance": 0.20,
    "decision": "RE-GENERATE",
    "reason": "Good context retrieved but answer contains hallucinations.",
    "average_score": 0.433
  }
}
```

## File Structure

```
backend/
├── app.py                    # FastAPI application with endpoints
├── rag.py                    # RAG logic with exponential backoff
├── evaluation.py             # RAG Triad evaluation framework
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (GOOGLE_API_KEY)
└── uploads/                  # Uploaded documents
```

## Key Modules

### `rag.py`
- `with_exponential_backoff()` - Decorator for retry logic
- `embed()` - Generate embeddings with automatic retry
- `upload_text()` - Ingest documents with semantic chunking
- `search()` - Retrieve relevant context from Qdrant
- `ask()` - Generate answer from context with automatic retry

### `evaluation.py`
- `EvaluationResult` - Pydantic model for structured scores
- `with_exponential_backoff()` - Same retry decorator
- `evaluate_metric()` - Judge a single criterion
- `evaluate_rag_triad()` - Run full three-part evaluation

### `app.py`
- `POST /upload` - Upload documents
- `GET /chat` - Chat with optional evaluation
- `GET /evaluate` - Manually evaluate answers

## Usage Examples

### Example 1: Basic Chat
```python
import requests

response = requests.get(
    "http://localhost:8000/chat",
    params={"question": "What is RAG?"}
)
print(response.json())
```

### Example 2: Chat with Evaluation
```python
response = requests.get(
    "http://localhost:8000/chat",
    params={
        "question": "What is RAG?",
        "evaluate": True
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Quality Score: {result['evaluation']['average_score']:.2%}")
print(f"Decision: {result['evaluation']['decision']}")
```

### Example 3: Manual Evaluation
```python
response = requests.get(
    "http://localhost:8000/evaluate",
    params={
        "question": "What is RAG?",
        "answer": "RAG stands for Retrieval-Augmented Generation..."
    }
)
evaluation = response.json()['evaluation']
print(f"Context Relevance: {evaluation['context_relevance']}")
print(f"Groundedness: {evaluation['groundedness']}")
print(f"Answer Relevance: {evaluation['answer_relevance']}")
```

## Error Handling

### Rate Limiting
When API rate limits are hit:
```
⚠️  Rate limited. Retrying in 2.7s... (Attempt 1/4)
⚠️  Rate limited. Retrying in 5.1s... (Attempt 2/4)
⚠️  Rate limited. Retrying in 10.3s... (Attempt 3/4)
```

### Evaluation Failures
If evaluation encounters errors, they're logged with retry attempts:
- Automatic retries with exponential backoff
- Max 4 retries before failure
- Clear error messages for debugging

## Configuration

### Environment Variables
```bash
# .env
GOOGLE_API_KEY=your-gemini-api-key
```

### Customization Options

#### Change Retry Parameters
```python
# In rag.py or evaluation.py
@with_exponential_backoff(max_retries=6, base_delay=3.0)
def my_function():
    # Function will retry up to 6 times with 3s base delay
    pass
```

#### Adjust Evaluation Threshold
```python
# In app.py
evaluation = evaluate_rag_triad(
    question, 
    context, 
    answer,
    threshold=0.8  # Higher threshold = stricter evaluation
)
```

## Performance Tips

1. **Reduce API Calls**: Cache embeddings for frequent queries
2. **Batch Operations**: Process multiple documents at once
3. **Optimize Chunking**: Use semantic chunking for better relevance
4. **Monitor Scores**: Track evaluation metrics to identify bottlenecks
5. **Tune Threshold**: Adjust threshold based on your use case

## Debugging

Enable detailed logging for troubleshooting:

```python
# In app.py or rag.py
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

Common issues:
- **Low Context Relevance** → Check chunking strategy
- **Low Groundedness** → Add grounding instructions to prompt
- **Low Answer Relevance** → Clarify user intent in prompt

## Testing

Test the evaluation with different answer qualities:

```bash
# Good answer (should get high scores)
curl "http://localhost:8000/evaluate?question=What%20is%20Python%3F&answer=Python%20is%20a%20high-level%20interpreted%20programming%20language"

# Bad answer (hallucination - should get low groundedness)
curl "http://localhost:8000/evaluate?question=What%20is%20Python%3F&answer=Python%20is%20written%20only%20in%20C%2B%2B"

# Off-topic answer (should get low answer relevance)
curl "http://localhost:8000/evaluate?question=What%20is%20Python%3F&answer=Java%20is%20a%20programming%20language"
```

## Future Enhancements

- [ ] Database persistence for evaluation results
- [ ] Historical analysis of evaluation metrics
- [ ] Automated rewriting for failed checks
- [ ] A/B testing different prompt variations
- [ ] Multi-document context synthesis
- [ ] Custom evaluation criteria

---

**Last Updated**: 2025-07-03  
**Framework**: FastAPI + Gemini AI + Qdrant Vector DB  
**RAG Triad**: Based on research-backed quality metrics
