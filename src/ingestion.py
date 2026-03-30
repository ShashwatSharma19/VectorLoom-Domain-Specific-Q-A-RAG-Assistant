import pypdf
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor


def _extract_page_text(page):
    """Extract text from a single page (for parallel processing)."""
    return page.extract_text() or ""


def load_pdf(filepath: str) -> str:
    """Reads a PDF and returns the text content using parallel extraction."""
    try:
        reader = pypdf.PdfReader(filepath)
        # Parallel extraction for multi-page PDFs
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(_extract_page_text, reader.pages))
        text = "\n".join(texts)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""
    return text


def detect_document_type(text: str) -> str:
    """Auto-detect document type from content patterns."""
    # Check first ~5000 chars for efficiency
    sample = text[:5000].lower()
    
    # Journal article indicators (DOI, journal refs, structured abstract)
    if 'doi:' in sample or 'doi.org' in sample:
        return 'journal_article'
    
    # Research paper indicators
    research_markers = ['abstract', 'methodology', 'results', 'conclusion', 'hypothesis']
    if sum(1 for m in research_markers if m in sample) >= 3:
        return 'research_paper'
    
    # Technical documentation indicators
    tech_markers = ['api', 'function', 'parameter', 'returns', 'example:', 'usage:', '```', 'def ', 'class ']
    if sum(1 for m in tech_markers if m in sample) >= 2:
        return 'technical_doc'
    
    # Textbook indicators
    textbook_markers = ['chapter', 'exercise', 'example', 'definition', 'theorem', 'proof', 'learning objectives']
    if sum(1 for m in textbook_markers if m in sample) >= 2:
        return 'textbook'
    
    return 'general'


def _chunk_with_sentences(text: str, max_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks respecting sentence boundaries."""
    if not text or not text.strip():
        return []
    
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap from end of previous
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence + " "
            else:
                current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_academic(text: str) -> List[str]:
    """Split research papers and journal articles by sections."""
    # Remove references section (noisy for retrieval)
    ref_pattern = r'\n(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n'
    ref_match = re.search(ref_pattern, text)
    if ref_match:
        text = text[:ref_match.start()]
    
    # Common academic section headers
    section_pattern = r'\n(?:(?:Abstract|ABSTRACT|Introduction|INTRODUCTION|Background|BACKGROUND|Methods?|METHODS?|Methodology|METHODOLOGY|Results?|RESULTS?|Discussion|DISCUSSION|Conclusion|CONCLUSION)s?)\s*\n'
    
    sections = re.split(section_pattern, text)
    
    chunks = []
    for section in sections:
        if section.strip():
            # Use larger chunks for academic content (1500 chars)
            chunks.extend(_chunk_with_sentences(section, max_size=1500, overlap=150))
    
    return chunks if chunks else _chunk_with_sentences(text, max_size=1500, overlap=150)


def _split_textbook(text: str) -> List[str]:
    """Split textbooks by chapters and sections."""
    # Chapter and section markers
    chapter_pattern = r'\n(?:Chapter\s+\d+|CHAPTER\s+\d+|\d+\.\s+[A-Z])'
    
    sections = re.split(chapter_pattern, text)
    
    chunks = []
    for section in sections:
        if section.strip():
            # Medium chunks for textbook content
            chunks.extend(_chunk_with_sentences(section, max_size=1200, overlap=120))
    
    return chunks if chunks else _chunk_with_sentences(text, max_size=1200, overlap=120)


def _split_technical(text: str) -> List[str]:
    """Split technical documentation, keeping code blocks intact."""
    # Extract and protect code blocks
    code_pattern = r'```[\s\S]*?```'
    
    # Find all code blocks
    code_blocks = re.findall(code_pattern, text)
    
    # Replace code blocks with placeholders
    placeholder_text = text
    placeholders = {}
    for i, block in enumerate(code_blocks):
        placeholder = f"__CODE_BLOCK_{i}__"
        placeholders[placeholder] = block
        placeholder_text = placeholder_text.replace(block, placeholder, 1)
    
    # Chunk the text (smaller chunks for precision)
    chunks = _chunk_with_sentences(placeholder_text, max_size=800, overlap=80)
    
    # Restore code blocks
    restored_chunks = []
    for chunk in chunks:
        for placeholder, block in placeholders.items():
            chunk = chunk.replace(placeholder, block)
        restored_chunks.append(chunk)
    
    return restored_chunks


def split_by_document_type(text: str, doc_type: str = None) -> List[str]:
    """Adaptive chunking based on document type."""
    if doc_type is None:
        doc_type = detect_document_type(text)
    
    print(f"Detected document type: {doc_type}")
    
    if doc_type in ('research_paper', 'journal_article'):
        return _split_academic(text)
    elif doc_type == 'textbook':
        return _split_textbook(text)
    elif doc_type == 'technical_doc':
        return _split_technical(text)
    else:
        return _chunk_with_sentences(text, max_size=1000, overlap=100)


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Legacy function - splits text into fixed chunks with overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
