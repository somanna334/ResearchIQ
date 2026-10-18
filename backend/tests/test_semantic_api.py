def _upload(upload, make_pdf, name, pages):
    return upload(name, make_pdf(pages, name=name).read_bytes()).json()["id"]


BIO = [
    "Mitochondria produce energy for the cell",
    "Photosynthesis happens inside chloroplasts",
]


def test_embed_then_semantic_search_finds_the_right_page(client, make_pdf, upload):
    doc_id = _upload(upload, make_pdf, "bio.pdf", BIO)

    response = client.post(f"/documents/{doc_id}/embeddings")
    assert response.status_code == 200
    assert response.json()["total_chunks"] == 2
    assert response.json()["newly_embedded"] == 2

    hits = client.get("/search/semantic", params={"q": "chloroplasts photosynthesis"}).json()
    assert hits[0]["page_number"] == 2
    assert hits[0]["score"] > hits[1]["score"]


def test_embedding_twice_only_embeds_new_chunks(client, make_pdf, upload):
    doc_id = _upload(upload, make_pdf, "bio.pdf", BIO)
    client.post(f"/documents/{doc_id}/embeddings")

    second = client.post(f"/documents/{doc_id}/embeddings").json()

    assert second["total_chunks"] == 2
    assert second["newly_embedded"] == 0


def test_semantic_search_before_embedding_returns_empty(client, make_pdf, upload):
    _upload(upload, make_pdf, "bio.pdf", BIO)

    assert client.get("/search/semantic", params={"q": "chloroplasts"}).json() == []


def test_rechunking_discards_stale_embeddings(client, make_pdf, upload):
    doc_id = _upload(upload, make_pdf, "bio.pdf", BIO)
    client.post(f"/documents/{doc_id}/embeddings")
    assert client.get("/search/semantic", params={"q": "chloroplasts"}).json()

    client.post(f"/documents/{doc_id}/chunks")  # rebuilds the default chunk set

    assert client.get("/search/semantic", params={"q": "chloroplasts"}).json() == []


def test_deleting_a_document_removes_its_vectors_from_search(client, make_pdf, upload):
    doc_id = _upload(upload, make_pdf, "bio.pdf", BIO)
    client.post(f"/documents/{doc_id}/embeddings")

    client.delete(f"/documents/{doc_id}")

    assert client.get("/search/semantic", params={"q": "chloroplasts"}).json() == []


def test_semantic_search_can_be_limited_to_one_document(client, make_pdf, upload):
    _upload(upload, make_pdf, "one.pdf", ["Gradient descent in document one"])
    two = _upload(upload, make_pdf, "two.pdf", ["Gradient descent in document two"])
    for doc in client.get("/documents").json():
        client.post(f"/documents/{doc['id']}/embeddings")

    hits = client.get(
        "/search/semantic", params={"q": "gradient descent", "document_id": two}
    ).json()

    assert len(hits) == 1
    assert hits[0]["document_id"] == two


def test_embedding_unknown_document_or_chunk_set_returns_404(client, make_pdf, upload):
    assert client.post("/documents/nope/embeddings").status_code == 404

    doc_id = _upload(upload, make_pdf, "bio.pdf", BIO)
    response = client.post(
        f"/documents/{doc_id}/embeddings", params={"chunk_config": "sentence-999-1"}
    )
    assert response.status_code == 404