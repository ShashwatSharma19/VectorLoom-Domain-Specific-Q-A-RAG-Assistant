"""
RAG Pipeline for the Domain Q&A RAG Assistant.

This module implements the Retrieval-Augmented Generation loop:
  1. Retrieve relevant chunks from the vector store
  2. Construct a domain-specific prompt with the retrieved context
  3. Generate an answer using a local LLM

Supports two generation modes:
  - Blocking:  generate_response()  → returns full string
  - Streaming: stream_response()    → yields tokens one-by-one (for SSE)

All model parameters are loaded from config.yaml.

Professional Commit Message:
    feat(rag): add SSE-compatible streaming via TextIteratorStreamer
"""

import time
import threading
from typing import List, Tuple, Generator

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from src.config import cfg
from src.indexing import Indexer
from src.retriever import HybridRetriever


# ── Domain-specific system prompts ──────────────────────────────
PROMPTS = {
    "research_paper": (
        "You are a research paper analyst. When answering:\n"
        "- Cite specific sections (e.g., 'In the Methods section...')\n"
        "- Distinguish between methodology and findings\n"
        "- Note any limitations mentioned in the paper\n"
        "- Use precise academic language"
    ),
    "journal_article": (
        "You are a journal article analyst. When answering:\n"
        "- Focus on key findings and supporting evidence\n"
        "- Reference the abstract for overview questions\n"
        "- Note statistical significance when mentioned\n"
        "- Distinguish between claims and evidence"
    ),
    "textbook": (
        "You are a textbook tutor. When answering:\n"
        "- Explain concepts step-by-step for clarity\n"
        "- Reference relevant chapters or sections when possible\n"
        "- Provide examples from the text when available\n"
        "- Define technical terms if they appear"
    ),
    "technical_doc": (
        "You are a technical documentation expert. When answering:\n"
        "- Be precise about function signatures and parameters\n"
        "- Include code examples when relevant\n"
        "- Note any version-specific information\n"
        "- Reference specific sections of the documentation"
    ),
    "general": (
        "You are a helpful document assistant. Answer questions using "
        "only the provided context. Be precise and cite relevant parts "
        "of the document."
    ),
}


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Usage:
        indexer = Indexer()
        rag = RAGPipeline(indexer)

        # Blocking
        answer = rag.generate_response(query, chunks)

        # Streaming (yields tokens)
        for token in rag.stream_response(query, chunks):
            print(token, end="", flush=True)
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.retriever = HybridRetriever(indexer)
        self.model_name = cfg.llm.model
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
        self.doc_type = "general"

    # ── Model Management ────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the LLM on first use (saves startup RAM)."""
        if self._model_loaded:
            return

        print(f"Loading model: {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Resolve torch dtype from config string
        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(cfg.llm.torch_dtype, torch.float16)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self._model_loaded = True
        print("Model loaded successfully.")

    def set_document_type(self, doc_type: str):
        """Set the document type for prompt selection."""
        self.doc_type = doc_type if doc_type in PROMPTS else "general"

    # ── Retrieval ───────────────────────────────────────────────

    def retrieve(self, query: str, k: int = None) -> List[Tuple[dict, float]]:
        """Retrieve top-k chunks using Hybrid Search and Reranking."""
        return self.retriever.search(query, final_k=k)

    # ── Prompt Construction ─────────────────────────────────────

    def _build_messages(
        self,
        query: str,
        context_chunks: List[Tuple[dict, float]],
        doc_type: str = None,
    ) -> list:
        """Build the chat messages list for the LLM using Parent Context."""
        # We feed the larger "parent_text" to the LLM for better reasoning context
        context = "\n".join([f"- {chunk['parent_text']}" for chunk, _ in context_chunks])
        current_doc_type = doc_type or self.doc_type
        system_prompt = PROMPTS.get(current_doc_type, PROMPTS["general"])

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

    # ── BLOCKING Generation (original behavior) ────────────────

    def generate_response(
        self,
        query: str,
        context_chunks: List[Tuple[dict, float]],
        doc_type: str = None,
    ) -> str:
        """Generate a complete answer (blocking). Used by /query endpoint."""
        start = time.time()
        self._load_model()
        print(f"DEBUG: Model check/load took {time.time() - start:.2f}s")

        messages = self._build_messages(query, context_chunks, doc_type)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        gen_start = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=cfg.llm.max_new_tokens,
            temperature=cfg.llm.temperature,
            do_sample=True,
            top_p=cfg.llm.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        print(f"DEBUG: Generation took {time.time() - gen_start:.2f}s")

        generated_ids = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, outputs)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return response.strip()

    # ── STREAMING Generation (new — for SSE) ────────────────────

    def stream_response(
        self,
        query: str,
        context_chunks: List[Tuple[dict, float]],
        doc_type: str = None,
    ) -> Generator[str, None, None]:
        """
        Yield answer tokens one-by-one as a Python generator.

        This is the core of the SSE streaming feature. FastAPI's
        StreamingResponse wraps this generator and pushes each
        token to the client as a Server-Sent Event.

        Under the hood we use HuggingFace's TextIteratorStreamer
        which runs model.generate() in a background thread and
        feeds tokens into a thread-safe queue that we iterate over.
        """
        self._load_model()

        messages = self._build_messages(query, context_chunks, doc_type)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # TextIteratorStreamer is a special HuggingFace class that
        # turns model.generate() into an iterable of decoded strings.
        # It uses a thread-safe queue internally.
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation must run in a background thread because
        # TextIteratorStreamer blocks until .generate() finishes,
        # but we want to yield tokens as they are produced.
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=cfg.llm.max_new_tokens,
            temperature=cfg.llm.temperature,
            do_sample=True,
            top_p=cfg.llm.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = threading.Thread(
            target=self.model.generate, kwargs=generation_kwargs
        )
        thread.start()

        # Yield tokens as they arrive from the streamer
        for token_text in streamer:
            if token_text:
                yield token_text

        thread.join()

    # ── Convenience method (backward compatible) ────────────────

    def query_rag(self, query: str) -> str:
        """Full RAG cycle: retrieve + generate (blocking)."""
        if self.indexer.index is None:
            self.indexer.load_index()

        context_chunks = self.retrieve(query)
        if not context_chunks:
            return "I couldn't find any relevant information in the documents."

        return self.generate_response(query, context_chunks)
