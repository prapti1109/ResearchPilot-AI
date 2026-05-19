"""
rag/pdf_loader.py — PyMuPDF-based PDF text extractor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from loguru import logger


@dataclass
class PageContent:
    source: str        # file path or arxiv_id
    page_num: int
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


def _clean_text(raw: str) -> str:
    """Remove excessive whitespace and common PDF artefacts."""
    import re
    # Collapse runs of whitespace (but keep paragraph breaks)
    text = re.sub(r'[ \t]+', ' ', raw)
    # Remove hyphenation at line ends
    text = re.sub(r'-\n(\w)', r'\1', text)
    # Replace single newlines with space (keep double for paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_pdf(path: str | Path, source_id: str | None = None) -> List[PageContent]:
    """
    Extract text from a PDF file page by page.

    Args:
        path: Filesystem path to the PDF.
        source_id: Optional identifier (e.g. arxiv_id) attached to metadata.

    Returns:
        List of PageContent objects, one per page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

    path = Path(path)
    if not path.exists():
        logger.error(f"PDF not found: {path}")
        return []

    source = source_id or path.stem
    pages: List[PageContent] = []

    try:
        doc = fitz.open(str(path))
        logger.info(f"Loading PDF: {path.name} ({doc.page_count} pages)")

        for page_num in range(doc.page_count):
            page = doc[page_num]
            raw_text = page.get_text("text")
            cleaned = _clean_text(raw_text)

            if len(cleaned) < 50:
                # Skip near-empty pages (cover images, blank pages, etc.)
                continue

            pages.append(PageContent(
                source=source,
                page_num=page_num + 1,
                text=cleaned,
            ))

        doc.close()
        total_chars = sum(p.char_count for p in pages)
        logger.info(f"Extracted {len(pages)} pages, {total_chars:,} chars from {path.name}")
    except Exception as exc:
        logger.error(f"Failed to parse {path}: {exc}")

    return pages


def load_pdfs(paths: List[str | Path],
              source_ids: List[str] | None = None) -> List[PageContent]:
    """Load multiple PDFs; returns concatenated list of PageContent."""
    all_pages: List[PageContent] = []
    for i, path in enumerate(paths):
        sid = source_ids[i] if source_ids and i < len(source_ids) else None
        all_pages.extend(load_pdf(path, source_id=sid))
    return all_pages
