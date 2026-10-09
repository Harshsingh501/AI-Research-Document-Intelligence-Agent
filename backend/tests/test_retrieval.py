from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.services.retrieval import retrieve
from tests.conftest import upload_sample


async def test_hybrid_surfaces_exact_numeric_token(client):
    """
    The fake embedding ignores numbers entirely, so only the keyword leg can
    know about "28.4". Hybrid search must still put that chunk first.
    """
    await upload_sample(client)
    async with AsyncSessionLocal() as db:
        chunks = await retrieve(db, "28.4", top_k=3)
    assert chunks, "expected results"
    assert "28.4" in chunks[0].content


async def test_vector_only_when_hybrid_disabled(client, monkeypatch):
    await upload_sample(client)
    monkeypatch.setattr(settings, "HYBRID_SEARCH", False)
    async with AsyncSessionLocal() as db:
        chunks = await retrieve(db, "multi-head attention parallel heads", top_k=4)
    assert len(chunks) == 4
    assert chunks[0].score >= chunks[-1].score  # pure cosine order


async def test_results_are_scoped_by_document_and_capped(client):
    doc = await upload_sample(client)
    async with AsyncSessionLocal() as db:
        chunks = await retrieve(db, "attention heads", top_k=2, document_id=doc["id"])
    assert len(chunks) == 2
    assert all(str(c.document_id) == doc["id"] for c in chunks)
    assert len({c.chunk_id for c in chunks}) == 2  # fusion never duplicates


async def test_embeddings_use_task_prefixes(fake_llm, monkeypatch):
    import app.services.embeddings as emb

    seen: list[str] = []

    class Spy:
        async def embed(self, model, input):
            seen.extend(input)
            return {"embeddings": [[0.0] * 768 for _ in input]}

    monkeypatch.setattr(emb, "_client", Spy())
    await emb.embed_texts(["passage one", "passage two"])
    await emb.embed_query("a question")
    assert seen == ["search_document: passage one", "search_document: passage two", "search_query: a question"]
