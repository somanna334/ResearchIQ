def upload(who, make_pdf, n: int):
    path = make_pdf([f"Page about topic {n}"], name=f"d{n}.pdf")
    return who.post(
        "/documents", files={"file": (f"d{n}.pdf", path.read_bytes(), "application/pdf")}
    )


def test_document_count_quota(client, make_client, make_pdf, tune):
    bob = make_client("bob@example.com")
    tune(max_documents_per_user=2)

    first = upload(client, make_pdf, 1)
    assert first.status_code == 201
    assert upload(client, make_pdf, 2).status_code == 201
    blocked = upload(client, make_pdf, 3)

    assert blocked.status_code == 403
    assert "limit" in blocked.json()["detail"].lower()
    assert upload(bob, make_pdf, 4).status_code == 201  # quotas are per user
    # Deleting a document frees a slot.
    assert client.delete(f"/documents/{first.json()['id']}").status_code == 204
    assert upload(client, make_pdf, 5).status_code == 201


def test_storage_quota_rejects_the_upload_and_leaves_no_file(
    client, make_client, make_pdf, tune
):
    bob = make_client("bob@example.com")
    first = upload(client, make_pdf, 1)
    size = first.json()["size_bytes"]
    settings = tune(max_storage_mb_per_user=size * 1.5 / (1024 * 1024))

    blocked = upload(client, make_pdf, 2)

    assert blocked.status_code == 403
    assert "storage" in blocked.json()["detail"].lower()
    assert len(list(settings.upload_dir.glob("*.pdf"))) == 1  # rejected file removed
    assert upload(bob, make_pdf, 3).status_code == 201  # bob has his own allowance
