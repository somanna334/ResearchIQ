import pytest

from app.ingestion.pdf import PdfExtractionError, extract_pdf


def test_extracts_text_with_page_numbers(make_pdf):
    path = make_pdf(["Alpha content on page one", "Beta content on page two"])
    result = extract_pdf(path, max_pages=10)

    assert [p.page_number for p in result.pages] == [1, 2]
    assert "Alpha content" in result.pages[0].text
    assert "Beta content" in result.pages[1].text


def test_blank_pages_have_no_text(make_pdf):
    result = extract_pdf(make_pdf(["", ""]), max_pages=10)

    assert result.page_count == 2
    assert result.pages_with_text == 0


def test_mixed_pdf_counts_pages_with_text(make_pdf):
    result = extract_pdf(make_pdf(["Some real text here", ""]), max_pages=10)

    assert result.page_count == 2
    assert result.pages_with_text == 1


def test_corrupted_pdf_raises(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not really a pdf")

    with pytest.raises(PdfExtractionError):
        extract_pdf(path, max_pages=10)


def test_too_many_pages_raises(make_pdf):
    path = make_pdf(["a"] * 6)

    with pytest.raises(PdfExtractionError):
        extract_pdf(path, max_pages=5)