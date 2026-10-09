"""
PDF text extraction.

We keep page numbers so answers can cite "page 7". pypdf is pure Python and
good enough for text-based PDFs; scanned PDFs (images) would need OCR, which
is out of scope here.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    number: int   # 1-based, as humans count pages
    text: str


def extract_pages(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""   # extract_text() returns None for blank pages
        text = _normalise_whitespace(text)
        if text:
            pages.append(Page(number=i, text=text))
    return pages


def _normalise_whitespace(text: str) -> str:
    """PDFs are full of stray line breaks mid-sentence; collapse runs of whitespace."""
    return " ".join(text.split())
