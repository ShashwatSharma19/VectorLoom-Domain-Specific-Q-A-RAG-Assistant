# VectorLoom: Domain-Specific Q/A RAG Assistant

> **Project Timeline**: Base Architecture completed **January 2026**. Cycle 2 (Advanced Features & Intelligence) integrated.

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about them using a multi-stage hybrid retrieval pipeline.

## Features

- **Adaptive Document Processing**: Auto-detects document types (Research Papers, Textbooks, Technical Docs) and applies domain-specific system prompts.
- **Intelligent Chunking**: Type-specific chunking strategies with sentence-boundary awareness and code block protection.
- **PDF Ingestion**: Parallel text extraction using `ThreadPoolExecutor` for multi-page PDFs.
- **Hybrid Retrieval Pipeline**:
  - Dense search via `FAISS` (semantic similarity)
  - Sparse search via `BM25` (exact keyword matching)
  - Result fusion using **Reciprocal Rank Fusion (RRF)**
  - Final reranking with a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `Qwen/Qwen2.5-1.5B-Instruct` (lazy-loaded, GPU-accelerated when available)
- **Web Interface**: Streamlit UI with real-time token streaming and an X-Ray panel showing retrieval scores.
- **Evaluation Framework**: Retrieval metrics (Recall@k, MRR) and an LLM Judge for answer faithfulness and relevance scoring.

## Setup

1. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

There are two ways to run the project: using the Web App or the Demo Notebook.

### Option 1: Web Application (Recommended)

You need to run the Backend and Frontend in separate terminals.

**Terminal 1: Start the Backend API**
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
This starts the FastAPI server at `http://localhost:8000`.

**Terminal 2: Start the Frontend UI**
```bash
streamlit run app/ui.py
```
This opens the web interface in your browser (usually at `http://localhost:8501`).

1. Upload a PDF using the sidebar.
2. Click "Process PDF".
3. Ask a question in the main text box.

### Option 2: Demo Notebook

Open `demo.ipynb` in VS Code or Jupyter Notebook/Lab to walk through the full pipeline step-by-step (Ingestion → Indexing → Retrieval → Generation → Evaluation).

## Directory Structure

- `app/`: Application servers.
  - `api.py`: FastAPI backend.
  - `ui.py`: Streamlit frontend.
- `src/`: Core logic.
  - `ingestion.py`: PDF parsing and adaptive chunking.
  - `indexing.py`: FAISS + BM25 index management.
  - `retriever.py`: Hybrid retrieval with RRF fusion and Cross-Encoder reranking.
  - `rag.py`: RAG pipeline (retrieval + generation).
  - `evaluation.py`: Metrics and LLM Judge.
  - `config.py`: Configuration loader.
- `tests/`: Basic tests.
- `requirements.txt`: Python dependencies.
