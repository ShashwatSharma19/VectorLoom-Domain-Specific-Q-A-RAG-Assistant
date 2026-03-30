"""
Streamlit Frontend for the Domain Q&A RAG Assistant.

Features (Cycle 1 Upgrades):
    - Chat-style interface using st.chat_message
    - Live token streaming via SSE consumption
    - X-Ray Expander showing retrieved chunks with distance scores
    - Document type badge after upload
    - Persistent chat history in session state

Professional Commit Message:
    feat(ui): add streaming chat UI with X-Ray retrieval expander
"""

import json
import streamlit as st
import requests

# ── Page Configuration ───────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="VectorLoom: Domain-Specific Q/A RAG Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Session State Initialization ─────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_type" not in st.session_state:
    st.session_state.doc_type = None
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

# ── Sidebar: Document Upload ────────────────────────────────────

st.sidebar.header("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.sidebar.button("Process PDF", use_container_width=True):
        with st.spinner("Parsing and indexing document..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            try:
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.doc_type = data.get("document_type", "general")
                    st.session_state.chunks_count = data.get("chunks_count", 0)
                    st.sidebar.success("PDF indexed successfully!")
                else:
                    st.sidebar.error(f"Error: {response.text}")
            except Exception as e:
                st.sidebar.error(f"Failed to connect to backend: {e}")

# Show document metadata badge if a document has been processed
if st.session_state.doc_type:
    doc_type_display = st.session_state.doc_type.replace("_", " ").title()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Document Type:** `{doc_type_display}`")
    st.sidebar.markdown(f"**Indexed Chunks:** `{st.session_state.chunks_count}`")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ── Main Area: Chat Interface ────────────────────────────────────

st.title("🔬 VectorLoom: Domain-Specific Q/A RAG Assistant")
st.caption("Upload a research paper, textbook, or technical doc — then ask anything about it.")

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # If this assistant message has sources, show the X-Ray expander
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_xray_expander(msg["sources"]) if callable(
                globals().get("_render_xray_expander")
            ) else None


def _render_xray_expander(sources: list):
    """
    The X-Ray Expander — shows exactly what the retrieval engine found.

    This is the "show your work" panel that hiring managers love.
    It displays each retrieved chunk alongside its L2 distance score
    so the user (or evaluator) can see WHY the LLM gave that answer.

    Lower L2 distance = closer semantic match = more relevant.
    """
    with st.expander("🔍 X-Ray: Retrieved Chunks & Relevance Scores", expanded=False):
        for i, source in enumerate(sources):
            score = source.get("score", 0)
            text = source.get("text", str(source))

            # Color-code relevance: green (very close) → yellow → red (far)
            if score < 0.5:
                relevance = "🟢 High"
            elif score < 1.0:
                relevance = "🟡 Medium"
            else:
                relevance = "🔴 Low"

            st.markdown(
                f"**Chunk {i + 1}** — L2 Distance: `{score:.4f}` | "
                f"Relevance: {relevance}"
            )
            st.code(text[:500] + ("..." if len(text) > 500 else ""), language=None)
            if i < len(sources) - 1:
                st.divider()


def _consume_sse_stream(question: str):
    """
    Connects to the /query/stream SSE endpoint and yields tokens
    as they arrive. Collects source metadata from the final event.

    Returns a generator that yields token strings (for st.write_stream)
    and stores sources in session state for the X-Ray panel.
    """
    try:
        response = requests.post(
            f"{API_URL}/query/stream",
            json={"question": question},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        sources = []

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            # Parse the JSON payload after "data: "
            payload = json.loads(line[6:])

            if payload.get("done"):
                # Final event — contains source metadata
                sources = payload.get("sources", [])
                st.session_state._last_sources = sources
                break

            token = payload.get("token", "")
            if token:
                yield token

    except requests.exceptions.ConnectionError:
        yield "❌ Could not connect to the backend. Is the API server running?"
    except Exception as e:
        yield f"❌ Error: {str(e)}"


# ── Chat Input & Streaming Response ─────────────────────────────

if question := st.chat_input("Ask a question about your document..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        # Initialize sources storage
        st.session_state._last_sources = []

        # st.write_stream consumes a generator and displays tokens
        # as they arrive — giving the live "typing" effect
        full_response = st.write_stream(_consume_sse_stream(question))

        # Retrieve the sources that were collected during streaming
        sources = st.session_state.get("_last_sources", [])

        # Render the X-Ray panel below the answer
        if sources:
            _render_xray_expander(sources)

    # Store in chat history (including sources for re-rendering)
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })
