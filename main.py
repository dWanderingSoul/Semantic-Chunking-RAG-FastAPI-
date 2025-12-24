import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
import google.generativeai as genai



load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST")
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "data")
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", 500))
PORT = int(os.getenv("PORT", 8000))

os.makedirs(RAG_DATA_DIR, exist_ok=True)


embedder = SentenceTransformer(EMBED_MODEL_NAME)

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel(LLM_MODEL_NAME)

chroma_client = chromadb.Client(
    Settings(persist_directory="./chroma", anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(
    name="rag_collection"
)


app = FastAPI()



def semantic_chunk(text: str, chunk_size: int):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def embed_and_store(chunks):
    embeddings = embedder.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )


def retrieve_context(query, k=4):
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return " ".join(results["documents"][0])




class ChatRequest(BaseModel):
    query: str



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(files: UploadFile = File(...)):
    content = await files.read()
    text = content.decode("utf-8")

    file_path = os.path.join(RAG_DATA_DIR, files.filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    chunks = semantic_chunk(text, CHUNK_LENGTH)
    embed_and_store(chunks)

    return {"message": "File uploaded and indexed successfully"}


@app.post("/chat")
def chat(request: ChatRequest):
    context = retrieve_context(request.query)

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{request.query}
"""

    response = llm.generate_content(prompt)

    return {"response": response.text}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
