def test_upload_list_read_delete_lifecycle(client, make_pdf, upload, tmp_path):
    pdf = make_pdf(["Alpha content on page one", "Beta content on page two"])
    response = upload("paper.pdf", pdf.read_bytes())

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["filename"] == "paper.pdf"
    assert body["page_count"] == 2
    assert body["pages_with_text"] == 2
    document_id = body["id"]

    assert [d["id"] for d in client.get("/documents").json()] == [document_id]

    page = client.get(f"/documents/{document_id}/pages/2")
    assert page.status_code == 200
    assert "Beta content" in page.json()["text"]

    assert client.delete(f"/documents/{document_id}").status_code == 204
    assert client.get(f"/documents/{document_id}").status_code == 404
    assert list((tmp_path / "uploads").glob("*.pdf")) == []


def test_rejects_wrong_extension(upload):
    assert upload("notes.txt", b"%PDF-1.4 pretending").status_code == 400


def test_rejects_fake_pdf_content(upload):
    assert upload("notes.pdf", b"just some plain text").status_code == 400


def test_rejects_empty_file(upload):
    assert upload("empty.pdf", b"").status_code == 400


def test_rejects_oversized_file(upload, tmp_path):
    # The test settings allow 1 MB.
    response = upload("big.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024))

    assert response.status_code == 413
    assert list((tmp_path / "uploads").iterdir()) == []


def test_corrupted_pdf_is_recorded_as_failed(upload, tmp_path):
    response = upload("bad.pdf", b"%PDF-1.4\nthis is not really a pdf")

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_message"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_scanned_pdf_is_marked_no_text(make_pdf, upload):
    response = upload("scan.pdf", make_pdf(["", ""]).read_bytes())

    body = response.json()
    assert response.status_code == 201
    assert body["status"] == "no_text"
    assert body["page_count"] == 2
    assert body["pages_with_text"] == 0


def test_unknown_document_and_page_return_404(client, make_pdf, upload):
    assert client.get("/documents/does-not-exist").status_code == 404

    document_id = upload("a.pdf", make_pdf(["Some real text here"]).read_bytes()).json()["id"]
    assert client.get(f"/documents/{document_id}/pages/99").status_code == 404