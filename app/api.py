"""
FastAPI Backend for the Domain Q&A RAG Assistant.

Endpoints:
    POST /upload         — Upload and index a PDF document
    POST /query          — Ask a question (blocking, returns full answer)
    POST /query/stream   — Ask a question (SSE, streams tokens live)

The /query/stream endpoint uses Server-Sent Events (SSE) to push
individual tokens to the frontend as they are generated. This gives
the user a "ChatGPT-like" typing effect instead of a loading spinner.

Professional Commit Message:
    feat(api): add /query/stream SSE endpoint with StreamingResponse
"""

import json
import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import cfg
from src.ingestion import load_pdf, detect_document_type, split_by_document_type
from src.indexing import Indexer
from src.rag import RAGPipeline

# ── App Initialization ──────────────────────────────────────────

app = FastAPI(
    title="VectorLoom: Domain-Specific Q/A RAG Assistant",
    description="A privacy-first RAG system for educational documents",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components using config
indexer = Indexer()
if os.path.exists(indexer.index_path):
    indexer.load_index()

rag = RAGPipeline(indexer)

# Pre-load model at startup for faster first query
print("Pre-loading LLM model for faster queries...")
rag._load_model()

# Store current document type for query responses
current_doc_type = "general"


# ── Request / Response Models ────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


# ── Endpoints ────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, chunk it adaptively, and index the chunks."""
    global current_doc_type
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        text = load_pdf(file_location)

        # Detect document type and use adaptive chunking
        current_doc_type = detect_document_type(text)
        chunks = split_by_document_type(text, current_doc_type)

        # Update RAG pipeline with document type
        rag.set_document_type(current_doc_type)

        indexer.create_vector_store(chunks)
        indexer.save_index()

        os.remove(file_location)

        return {
            "message": "PDF processed and indexed successfully",
            "chunks_count": len(chunks),
            "document_type": current_doc_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(request: QueryRequest):
    """Ask a question — returns the full answer at once (blocking)."""
    try:
        context_chunks = rag.retrieve(request.question)
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information in the documents.",
                "sources": [],
            }

        answer = rag.generate_response(
            request.question, context_chunks, current_doc_type
        )

        # Return both the answer and the source chunks with scores
        # (scores are Cross-Encoder relevance — higher = more relevant)
        sources = [
            {
                "text": chunk["text"],
                "parent_text": chunk["parent_text"],
                "score": round(score, 4),
            }
            for chunk, score in context_chunks
        ]
        return {"answer": answer, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Ask a question — streams tokens back via Server-Sent Events (SSE).

    The response uses the text/event-stream MIME type. Each token is
    sent as a JSON-encoded SSE `data:` line. The stream ends with a
    final event containing the retrieved source chunks and their
    distance scores for the X-Ray panel.

    SSE Format:
        data: {"token": "The"}
        data: {"token": " answer"}
        data: {"token": " is..."}
        data: {"done": true, "sources": [...]}
    """
    try:
        context_chunks = rag.retrieve(request.question)

        if not context_chunks:
            # Even for "no results", we use SSE format for consistency
            def no_results():
                msg = {
                    "token": "I couldn't find any relevant information in the documents."
                }
                yield f"data: {json.dumps(msg)}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"

            return StreamingResponse(
                no_results(), media_type="text/event-stream"
            )

        # Build the source metadata for the X-Ray panel (now including parent context)
        sources = [
            {
                "text": chunk["text"],
                "parent_text": chunk["parent_text"],
                "score": round(score, 4),
            }
            for chunk, score in context_chunks
        ]

        def token_generator():
            """Wraps rag.stream_response() into SSE-formatted lines."""
            for token in rag.stream_response(
                request.question, context_chunks, current_doc_type
            ):
                # Each token is a separate SSE event
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Final event: signal completion and attach source metadata
            yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

        return StreamingResponse(
            token_generator(), media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Main Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=cfg.server.api_host,
        port=cfg.server.api_port,
    )
