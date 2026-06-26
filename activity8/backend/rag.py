import uuid
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams

try:
    from llama_index.core import Document
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.google import GeminiEmbedding
except ImportError:
    Document = None
    SemanticSplitterNodeParser = None
    GeminiEmbedding = None

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

qdrant = QdrantClient(url="http://localhost:6333")

COLLECTION = "simple_rag"

if not qdrant.collection_exists(COLLECTION):

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=3072,
            distance=Distance.COSINE
        )
    )

def embed(text):

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values


def upload_text(filepath):

    with open(filepath,"r",encoding="utf-8") as f:
        text = f.read()

    if Document and SemanticSplitterNodeParser and GeminiEmbedding:
        # Use LlamaIndex semantic chunking for better spoken or long text handling
        splitter = SemanticSplitterNodeParser(
            buffer_size=3,
            breakpoint_percentile_threshold=95,
            embed_model=GeminiEmbedding(model_name="models/gemini-embedding-2")
        )
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        points = []
        for node in nodes:
            vector = embed(node.text)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": node.text
                    }
                )
            )
        if points:
            qdrant.upsert(
                collection_name=COLLECTION,
                points=points
            )
    else:
        vector = embed(text)
        qdrant.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": text
                    }
                )
            ]
        )

def search(query):

    vector = embed(query)

    results = qdrant.query_points(

        collection_name=COLLECTION,

        query=vector,

        limit=1,

        score_threshold=0.60

    )

    return results.points

def ask(question):

    results = search(question)

    if len(results)==0:

        return "I don't know."

    context = results[0].payload["text"]

    prompt=f"""

Answer ONLY using this information.

{context}

Question:

{question}

"""

    response = client.models.generate_content(

        model="gemini-3.1-flash-lite",

        contents=prompt

    )

    return response.text