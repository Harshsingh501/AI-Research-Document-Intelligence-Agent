from app.services.pdf import extract_pages
from tests.conftest import SAMPLE_PDF


def test_sample_paper_extracts_all_pages():
    pages = extract_pages(SAMPLE_PDF)
    assert len(pages) == 15
    assert pages[0].number == 1
    assert "Attention Is All You Need" in pages[0].text


def test_whitespace_is_normalised():
    pages = extract_pages(SAMPLE_PDF)
    assert all("\n" not in p.text and "  " not in p.text for p in pages)
