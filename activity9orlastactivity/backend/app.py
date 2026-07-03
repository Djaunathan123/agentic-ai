import os

os.makedirs("uploads", exist_ok=True)

from fastapi import FastAPI
from fastapi import UploadFile

from fastapi.middleware.cors import CORSMiddleware

from rag import upload_text, ask, search
from evaluation import evaluate_rag_triad

app=FastAPI()

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"]

)

@app.post("/upload")

async def upload(file:UploadFile):

    path="uploads/"+file.filename

    with open(path,"wb") as f:

        f.write(await file.read())

    upload_text(path)

    return {"message":"Uploaded successfully"}

@app.get("/chat")

def chat(question:str, evaluate:bool=False):

    answer=ask(question)

    if evaluate:
        # Retrieve context for evaluation
        results = search(question)
        if len(results) > 0:
            context = results[0].payload["text"]
            # Run full RAG Triad evaluation
            evaluation = evaluate_rag_triad(question, context, answer)
            return {
                "answer": answer,
                "evaluation": evaluation
            }
    
    return {"answer":answer}


@app.get("/evaluate")

def evaluate_answer(question:str, answer:str):
    """
    Endpoint for manual evaluation of an answer.
    Retrieves context and runs full RAG Triad evaluation.
    """
    results = search(question)
    
    if len(results) == 0:
        return {
            "error": "No relevant context found",
            "evaluation": None
        }
    
    context = results[0].payload["text"]
    evaluation = evaluate_rag_triad(question, context, answer)
    
    return {
        "question": question,
        "answer": answer,
        "context": context,
        "evaluation": evaluation
    }