import os
import re
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import logging

# Load environment variables
load_dotenv()

# Environment variables (exact names from requirements)
HF_API_KEY = os.getenv("HF_API_KEY", "")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-exp")
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "localhost")
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "./data")
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", "500"))
PORT = int(os.getenv("PORT", "8000"))

# Validate required variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env file")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

# Initialize FastAPI
app = FastAPI(title="Semantic-Chunking RAG System")

# Global components
embedding_model = None
chroma_client = None
collection = None
genai_model = None
current_chunk_length = CHUNK_LENGTH


def semantic_chunking(text: str, chunk_length: int) -> List[str]:
    """
    Semantic chunking that preserves sentence boundaries.
    Splits text at sentence endings (., !, ?) while respecting chunk_length.
    """
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence exceeds chunk_length
        if len(current_chunk) + len(sentence) + 1 > chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def initialize_components():
    """Initialize all RAG components."""
    global embedding_model, chroma_client, collection, genai_model
    
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    logger.info(f"Connecting to ChromaDB at: {CHROMA_DB_HOST}")
    # Use local persistent storage
    if CHROMA_DB_HOST == "localhost" or CHROMA_DB_HOST.startswith("127.0.0.1"):
        chroma_client = chromadb.Client(Settings(
            is_persistent=True,
            persist_directory="./chroma_db"
        ))
    else:
        # If host is specified, use HTTP client
        host_parts = CHROMA_DB_HOST.replace("http://", "").replace("https://", "").split(":")
        host = host_parts[0]
        port = int(host_parts[1]) if len(host_parts) > 1 else 8000
        chroma_client = chromadb.HttpClient(host=host, port=port)
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name="rag_documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info(f"Initializing Gemini model: {LLM_MODEL_NAME}")
    genai.configure(api_key=GEMINI_API_KEY)
    genai_model = genai.GenerativeModel(LLM_MODEL_NAME)
    
    logger.info("All components initialized successfully!")


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    initialize_components()
    
    # Create data directory
    Path(RAG_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load existing documents
    data_dir = Path(RAG_DATA_DIR)
    total_chunks = 0
    
    if data_dir.exists():
        for file_path in data_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunks = semantic_chunking(content, current_chunk_length)
                    
                    if not chunks:
                        continue
                    
                    # Generate embeddings
                    embeddings = embedding_model.encode(chunks).tolist()
                    
                    # Store in ChromaDB
                    ids = [f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": file_path.name, "chunk_id": i} for i in range(len(chunks))]
                    
                    collection.add(
                        documents=chunks,
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metadatas
                    )
                    total_chunks += len(chunks)
                    logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
    
    if total_chunks > 0:
        logger.info(f"Total chunks loaded on startup: {total_chunks}")


# ========== REQUIRED ENDPOINTS (from template) ==========

@app.post("/upload")
async def upload(file: UploadFile):
    """
    Upload and index a document with semantic chunking.
    This is the REQUIRED endpoint from the template.
    """
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty file content")
        
        # Perform semantic chunking
        chunks = semantic_chunking(text, current_chunk_length)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created")
        
        # Generate embeddings
        embeddings = embedding_model.encode(chunks).tolist()
        
        # Store in ChromaDB
        file_name = file.filename or "unknown.txt"
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name, "chunk_id": i} for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        # Save to data directory
        save_path = Path(RAG_DATA_DIR) / file_name
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return {
            "message": f"Document '{file_name}' uploaded and indexed successfully",
            "filename": file_name,
            "chunks_created": len(chunks),
            "chunk_length": current_chunk_length
        }
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/prompt")
async def prompt(payload: dict):
    """
    Query the RAG system with a prompt and get AI-generated response.
    This is the REQUIRED endpoint from the template.
    """
    try:
        # Extract query from payload
        query = payload.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="'query' field is required")
        
        top_k = payload.get("top_k", 3)
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Check if we have results
        if not results['documents'] or not results['documents'][0]:
            return {
                "answer": "I couldn't find any relevant information in the indexed documents.",
                "sources": [],
                "query": query
            }
        
        # Prepare context from retrieved chunks
        contexts = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        context_text = "\n\n".join([f"[Source {i+1}]: {doc}" for i, doc in enumerate(contexts)])
        
        # Generate response using Gemini
        prompt_text = f"""You are a helpful assistant answering questions based on the provided context.

Context:
{context_text}

Question: {query}

Instructions:
- Answer the question based ONLY on the information in the context above
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- Cite the source number when referencing information

Answer:"""
        
        response = genai_model.generate_content(prompt_text)
        answer = response.text
        
        # Prepare sources
        sources = [
            {
                "chunk": contexts[i],
                "source": metadatas[i].get("source", "unknown"),
                "chunk_id": metadatas[i].get("chunk_id", 0),
                "relevance_score": float(1 - distances[i])
            }
            for i in range(len(contexts))
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "num_sources": len(sources)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/rechunk")
async def rechunk(payload: dict):
    """
    Re-chunk all existing documents with a new chunk length.
    This is the REQUIRED endpoint from the template.
    """
    global current_chunk_length
    
    try:
        # Extract chunk_length from payload
        new_chunk_length = payload.get("chunk_length")
        if not new_chunk_length:
            raise HTTPException(status_code=400, detail="'chunk_length' field is required")
        
        if not isinstance(new_chunk_length, int) or new_chunk_length < 50:
            raise HTTPException(status_code=400, detail="chunk_length must be an integer >= 50")
        
        # Clear existing collection
        chroma_client.delete_collection("rag_documents")
        global collection
        collection = chroma_client.create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Update chunk length
        old_chunk_length = current_chunk_length
        current_chunk_length = new_chunk_length
        
        # Re-process all documents
        data_dir = Path(RAG_DATA_DIR)
        total_chunks = 0
        files_processed = 0
        
        if data_dir.exists():
            for file_path in data_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        chunks = semantic_chunking(content, new_chunk_length)
                        
                        if not chunks:
                            continue
                        
                        # Generate embeddings
                        embeddings = embedding_model.encode(chunks).tolist()
                        
                        # Store in ChromaDB
                        ids = [f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]
                        metadatas = [{"source": file_path.name, "chunk_id": i} for i in range(len(chunks))]
                        
                        collection.add(
                            documents=chunks,
                            embeddings=embeddings,
                            ids=ids,
                            metadatas=metadatas
                        )
                        total_chunks += len(chunks)
                        files_processed += 1
                        
                except Exception as e:
                    logger.error(f"Error re-chunking {file_path.name}: {e}")
        
        return {
            "message": "Documents re-chunked successfully",
            "old_chunk_length": old_chunk_length,
            "new_chunk_length": new_chunk_length,
            "files_processed": files_processed,
            "total_chunks": total_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error re-chunking: {str(e)}")


# ========== HELPER ENDPOINTS ==========

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Semantic-Chunking RAG System",
        "endpoints": {
            "POST /upload": "Upload documents for indexing",
            "POST /prompt": "Query the RAG system",
            "POST /rechunk": "Re-chunk documents with new length"
        },
        "config": {
            "embedding_model": EMBED_MODEL_NAME,
            "llm_model": LLM_MODEL_NAME,
            "chunk_length": current_chunk_length,
            "chroma_host": CHROMA_DB_HOST
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "embedding_model": embedding_model is not None,
            "chroma_client": chroma_client is not None,
            "genai_model": genai_model is not None,
            "collection": collection is not None
        },
        "stats": {
            "total_chunks": collection.count() if collection else 0,
            "chunk_length": current_chunk_length
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)