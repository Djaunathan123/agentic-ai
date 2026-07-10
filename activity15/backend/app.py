import os

os.makedirs("uploads", exist_ok=True)

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from rag import upload_text, ask, search
from evaluation import evaluate_rag_triad
from project_react_loop import react_loop
from evaluation_integration import run_with_evaluation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Existing endpoints (unchanged)
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile):
    path = "uploads/" + file.filename
    with open(path, "wb") as f:
        f.write(await file.read())
    upload_text(path)
    return {"message": "Uploaded successfully"}


@app.get("/chat")
def chat(question: str, evaluate: bool = False):
    answer = ask(question)

    if evaluate:
        results = search(question)
        if len(results) > 0:
            context = results[0].payload["text"]
            evaluation = evaluate_rag_triad(question, context, answer)
            return {"answer": answer, "evaluation": evaluation}

    return {"answer": answer}


@app.get("/evaluate")
def evaluate_answer(question: str, answer: str):
    """Manual evaluation endpoint — retrieves context and runs RAG Triad."""
    results = search(question)

    if len(results) == 0:
        return {"error": "No relevant context found", "evaluation": None}

    context = results[0].payload["text"]
    evaluation = evaluate_rag_triad(question, context, answer)

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "evaluation": evaluation,
    }


# ---------------------------------------------------------------------------
# NEW — Week 5 ReAct endpoint
# ---------------------------------------------------------------------------

@app.get("/react-chat")
def react_chat(question: str, evaluate: bool = False):
    """
    ReAct loop endpoint.
    - Runs the full Reason → Act → Observe cycle.
    - Returns the labeled transcript + final answer.
    - When evaluate=true, also runs RAG Triad scoring + self-correction.
    """
    if evaluate:
        result = run_with_evaluation(question, verbose=False)
        return {
            "answer": result["corrected_answer"] or result["answer"],
            "original_answer": result["answer"],
            "transcript": result["transcript"],
            "evaluation": {
                "context_relevance": result["context_relevance"],
                "groundedness": result["groundedness"],
                "answer_relevance": result["answer_relevance"],
                "average_score": result["average_score"],
                "decision": result["decision"],
                "passed": result["passed"],
                "was_corrected": result["was_corrected"],
            },
        }

    transcript = react_loop(question)

    # Extract the final answer from the transcript
    answer = ""
    for entry in transcript:
        if entry["phase"] == "ANSWER":
            answer = entry["content"]
            break

    return {"answer": answer, "transcript": transcript}
