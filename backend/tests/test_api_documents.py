import io

from sqlalchemy import func, select

from app.db.models import Chunk
from app.db.session import AsyncSessionLocal
from tests.conftest import upload_sample


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_rejects_non_pdf(client):
    r = await client.post("/documents", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})
    assert r.status_code == 415


async def test_upload_ingests_and_chunks(client):
    doc = await upload_sample(client)
    assert doc["page_count"] == 15
    assert doc["size_bytes"] > 1_000_000
    assert doc["error_message"] is None

    async with AsyncSessionLocal() as db:
        n = await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc["id"]))
        pages = await db.scalar(
            select(func.count(func.distinct(Chunk.page_number))).where(Chunk.document_id == doc["id"])
        )
    assert n > 50
    assert pages == 15


async def test_unreadable_pdf_is_marked_failed(client):
    bogus = io.BytesIO(b"%PDF-1.4 definitely not a real pdf")
    r = await client.post("/documents", files={"file": ("bad.pdf", bogus, "application/pdf")})
    assert r.status_code == 202
    r = await client.get(f"/documents/{r.json()['id']}")
    assert r.json()["status"] == "failed"
    assert r.json()["error_message"]


async def test_list_and_delete(client):
    doc = await upload_sample(client)
    r = await client.get("/documents")
    assert r.json()["total"] == 1 and r.json()["items"][0]["id"] == doc["id"]

    assert (await client.delete(f"/documents/{doc['id']}")).status_code == 204
    assert (await client.get(f"/documents/{doc['id']}")).status_code == 404

    async with AsyncSessionLocal() as db:  # chunks cascaded away
        assert await db.scalar(select(func.count()).select_from(Chunk)) == 0
