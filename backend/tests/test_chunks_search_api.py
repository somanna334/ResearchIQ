def _upload_text_pdf(upload, make_pdf, name, pages):
    return upload(name, make_pdf(pages, name=name).read_bytes()).json()["id"]


def test_upload_creates_default_chunks_with_page_numbers(client, make_pdf, upload):
    doc_id = _upload_text_pdf(
        upload, make_pdf, "p.pdf", ["Alpha content on page one", "Beta content on page two"]
    )
    chunks = client.get(f"/documents/{doc_id}/chunks").json()

    assert [c["page_number"] for c in chunks] == [1, 2]
    assert chunks[0]["chunk_config"] == "fixed-800-150"


def test_rechunk_with_custom_config_keeps_the_default_set(client, make_pdf, upload):
    doc_id = _upload_text_pdf(
        upload, make_pdf, "p.pdf", ["Alpha content on page one", "Beta content on page two"]
    )
    response = client.post(
        f"/documents/{doc_id}/chunks",
        json={"strategy": "sentence", "chunk_size": 300, "overlap": 50},
    )

    assert response.status_code == 200
    assert response.json()["chunk_config"] == "sentence-300-50"
    assert response.json()["chunk_count"] == 2
    assert len(client.get(f"/documents/{doc_id}/chunks").json()) == 2
    custom = client.get(
        f"/documents/{doc_id}/chunks", params={"chunk_config": "sentence-300-50"}
    ).json()
    assert len(custom) == 2


def test_invalid_chunk_config_returns_422(client, make_pdf, upload):
    doc_id = _upload_text_pdf(upload, make_pdf, "p.pdf", ["Alpha content on page one"])
    response = client.post(
        f"/documents/{doc_id}/chunks", json={"chunk_size": 300, "overlap": 200}
    )

    assert response.status_code == 422


def test_rechunk_unknown_document_returns_404(client):
    assert client.post("/documents/nope/chunks").status_code == 404


def test_rechunk_document_without_text_returns_409(client, make_pdf, upload):
    doc_id = upload("scan.pdf", make_pdf(["", ""]).read_bytes()).json()["id"]

    assert client.post(f"/documents/{doc_id}/chunks").status_code == 409


def test_search_returns_the_right_page(client, make_pdf, upload):
    _upload_text_pdf(
        upload,
        make_pdf,
        "bio.pdf",
        ["Mitochondria produce energy for the cell", "Photosynthesis happens inside chloroplasts"],
    )
    hits = client.get("/search", params={"q": "chloroplasts photosynthesis"}).json()

    assert hits[0]["page_number"] == 2
    assert hits[0]["filename"] == "bio.pdf"
    assert "chloroplasts" in hits[0]["text"]


def test_search_with_no_match_returns_empty_list(client, make_pdf, upload):
    _upload_text_pdf(upload, make_pdf, "bio.pdf", ["Mitochondria produce energy for the cell"])

    assert client.get("/search", params={"q": "zzzqqq"}).json() == []


def test_search_can_be_limited_to_one_document(client, make_pdf, upload):
    _upload_text_pdf(upload, make_pdf, "one.pdf", ["Gradient descent in document one"])
    two_id = _upload_text_pdf(upload, make_pdf, "two.pdf", ["Gradient descent in document two"])
    hits = client.get(
        "/search", params={"q": "gradient descent", "document_id": two_id}
    ).json()

    assert len(hits) == 1
    assert hits[0]["document_id"] == two_id


def test_deleting_a_document_removes_its_chunks_from_search(client, make_pdf, upload):
    doc_id = _upload_text_pdf(upload, make_pdf, "bio.pdf", ["Photosynthesis happens inside chloroplasts"])
    assert client.get("/search", params={"q": "chloroplasts"}).json()

    client.delete(f"/documents/{doc_id}")

    assert client.get("/search", params={"q": "chloroplasts"}).json() == []