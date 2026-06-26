import os

os.makedirs("uploads", exist_ok=True)

from fastapi import FastAPI
from fastapi import UploadFile

from fastapi.middleware.cors import CORSMiddleware

from rag import upload_text
from rag import ask

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

def chat(question:str):

    answer=ask(question)

    return {"answer":answer}