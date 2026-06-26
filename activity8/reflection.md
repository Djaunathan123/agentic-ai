# Activity 8 Reflection

## Purpose of this code
The goal of this project is to build a simple RAG (retrieval-augmented generation) backend that uploads text files, stores them as vector embeddings in Qdrant, and answers user questions by retrieving the most relevant stored content.

The main purpose of adding LlamaIndex is to improve how the uploaded text is split and indexed before embedding. Instead of embedding the whole file as one large vector, the app can break the text into smaller, semantically meaningful chunks and store multiple points. This makes retrieval more accurate and helpful for longer documents or transcripts.

## What changed
- Added optional LlamaIndex imports in `activity8/backend/rag.py`:
  - `from llama_index.core import Document`
  - `from llama_index.core.node_parser import SemanticSplitterNodeParser`
  - `from llama_index.embeddings.google import GeminiEmbedding`
- Updated `upload_text()` to use semantic chunking when LlamaIndex is installed.
- Kept the original fallback path so the app still works without LlamaIndex.

## Why this is important
- `upload_text()` is the main ingestion function used by `app.py` when the user uploads a file.
- Better chunking means Qdrant can return a more relevant `text` segment for the passed question.
- The code still works if LlamaIndex is not installed, so the backend stays functional.

## Most important points
1. Purpose: convert uploaded text into vector data for Qdrant retrieval.
2. Improvement: use LlamaIndex semantic chunking to create richer retrieval points.
3. Compatibility: keep a working fallback if the `llama_index` library is unavailable.

## What to test
- Upload a text file through the backend endpoint.
- Ask a question at `/chat` and verify the answer is based on stored content.
- If you install `llama_index`, confirm the app stores multiple text chunks instead of one large chunk.

## Practical note
This reflection is based on the actual project flow:
- `activity8/backend/app.py` receives uploads and forwards them to `rag.upload_text()`.
- `activity8/backend/rag.py` creates embeddings and stores them in Qdrant.
- `activity8/backend/rag.py` also searches Qdrant and asks Gemini to answer from retrieved context.
