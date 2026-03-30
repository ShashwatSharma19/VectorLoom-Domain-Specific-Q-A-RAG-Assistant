# VectorLoom: Domain-Specific Q/A RAG Assistant

> **Project Timeline**: Initial development and Base Architecture completed **January 2026**.

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about them.

## Features
- **PDF Ingestion**: Upload and parse PDF documents.
- **RAG Pipeline**: 
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - Vector Store: `FAISS`
  - LLM: `Qwen/Qwen2.5-1.5B-Instruct` (Runs locally on CPU)
- **Web Interface**: User-friendly UI built with Streamlit.
- **Evaluation**: Tools for calculating retrieval metrics and LLM-based answer scoring.

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

Open `demo.ipynb` in VS Code or Jupyter Notebook/Lab to walk through the pipeline step-by-step (Ingestion -> Indexing -> Retrieval -> Generation -> Evaluation).

## Directory Structure

- `app/`: Contains the application servers.
  - `api.py`: FastAPI backend.
  - `ui.py`: Streamlit frontend.
- `src/`: Core logic.
  - `ingestion.py`: PDF parsing and chunking.
  - `indexing.py`: Vector store management (FAISS).
  - `rag.py`: RAG pipeline (Retrieval + Generation).
  - `evaluation.py`: Metrics and LLM Judge.
- `tests/`: Basic tests.
- `requirements.txt`: Python dependencies.
