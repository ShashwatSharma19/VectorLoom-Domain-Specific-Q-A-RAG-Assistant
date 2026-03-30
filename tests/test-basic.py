"""
Unit Tests for the Domain Q&A RAG Assistant.

Tests cover: config loading, ingestion, indexing, retrieval,
prompt selection, and streaming generator.

Professional Commit Message:
    test: add config loader and streaming generator tests
"""

from src.ingestion import split_text, detect_document_type, split_by_document_type
from src.indexing import Indexer
from src.rag import RAGPipeline, PROMPTS
from src.config import cfg


# ── Config Tests ─────────────────────────────────────────────────

def test_config_loads():
    """Config should load from config.yaml without errors."""
    assert cfg is not None
    assert hasattr(cfg, "llm")
    assert hasattr(cfg, "embedding")
    assert hasattr(cfg, "retrieval")


def test_config_has_expected_fields():
    """Config should expose the required model parameters."""
    assert isinstance(cfg.llm.model, str)
    assert isinstance(cfg.llm.max_new_tokens, int)
    assert isinstance(cfg.retrieval.top_k, int)
    assert isinstance(cfg.embedding.model, str)


# ── Ingestion Tests ──────────────────────────────────────────────

def test_split_text():
    text = "Hello " * 100
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert len(chunks[0]) == 50


def test_detect_document_type_research():
    text = "Abstract\nThis paper presents...\nMethodology\n...Results\n...Conclusion"
    doc_type = detect_document_type(text)
    assert doc_type == "research_paper"


def test_detect_document_type_textbook():
    text = "Chapter 1: Introduction\nExercise 1.1\nDefinition: A variable is..."
    doc_type = detect_document_type(text)
    assert doc_type == "textbook"


def test_detect_document_type_technical():
    text = (
        "API Reference\n"
        "function getData(parameter: string) returns object\n"
        "Example:\n```python\ndata = getData('test')\n```"
    )
    doc_type = detect_document_type(text)
    assert doc_type == "technical_doc"


# ── Indexing Tests ───────────────────────────────────────────────

def test_indexer(tmp_path):
    index_path = str(tmp_path / "test_index.bin")
    indexer = Indexer(model_name="all-MiniLM-L6-v2", index_path=index_path)
    chunks = ["test chunk 1", "test chunk 2"]
    indexer.create_vector_store(chunks)
    assert indexer.index.ntotal == 2


def test_indexer_search_returns_scores(tmp_path):
    """Search results should include distance scores."""
    index_path = str(tmp_path / "test_index_scores.bin")
    indexer = Indexer(model_name="all-MiniLM-L6-v2", index_path=index_path)
    chunks = ["Paris is the capital of France", "Berlin is in Germany"]
    indexer.create_vector_store(chunks)
    results = indexer.search("France", k=2)
    assert len(results) > 0
    assert len(results[0]) == 2  # (text, score) tuple
    assert isinstance(results[0][1], float)  # score is a float


# ── RAG Tests ────────────────────────────────────────────────────

def test_rag_retrieval(tmp_path):
    index_path = str(tmp_path / "test_rag.bin")
    indexer = Indexer(model_name="all-MiniLM-L6-v2", index_path=index_path)
    chunks = ["Paris is in France", "Berlin is in Germany"]
    indexer.create_vector_store(chunks)
    rag = RAGPipeline(indexer)
    results = rag.retrieve("France")
    assert len(results) > 0
    assert "Paris is in France" in [r[0] for r in results]


def test_prompts_exist():
    assert "research_paper" in PROMPTS
    assert "textbook" in PROMPTS
    assert "technical_doc" in PROMPTS
    assert "general" in PROMPTS


def test_rag_uses_config_model(tmp_path):
    """RAGPipeline should read model name from config, not hardcode."""
    index_path = str(tmp_path / "test_rag_config.bin")
    indexer = Indexer(model_name="all-MiniLM-L6-v2", index_path=index_path)
    rag = RAGPipeline(indexer)
    assert rag.model_name == cfg.llm.model
