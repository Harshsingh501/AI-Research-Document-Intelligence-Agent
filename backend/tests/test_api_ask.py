import json

from tests.conftest import upload_sample


async def test_ask_without_documents_short_circuits(client, fake_llm):
    r = await client.post("/ask", json={"question": "What is attention?"})
    assert r.status_code == 200
    body = r.json()
    assert body["short_circuited"] is True and body["sources"] == []
    assert fake_llm.calls == []


async def test_ask_validates_input(client):
    assert (await client.post("/ask", json={"question": "hi"})).status_code == 422
    assert (await client.post("/ask", json={"question": "long enough", "top_k": 0})).status_code == 422


async def test_ask_returns_grounded_structured_answer(client, fake_llm):
    await upload_sample(client)
    fake_llm.queue(json.dumps({"answer": "Eight heads [1]; see also [2, 9].", "citations": [1, 2, 9], "confidence": "high"}))

    r = await client.post("/ask", json={"question": "multi-head attention parallel heads", "top_k": 4})
    body = r.json()
    assert r.status_code == 200
    assert body["short_circuited"] is False
    assert body["answer"] == "Eight heads [1]; see also [2]."  # [9] does not exist → stripped
    assert body["citations"] == [1, 2]
    assert body["confidence"] == "high"
    assert len(body["sources"]) == 4
    assert body["sources"][0]["page_number"] is not None
    assert body["sources"][0]["score"] >= body["sources"][-1]["score"]  # best first

    call = fake_llm.calls[-1]
    assert call["format"]["properties"].keys() == {"answer", "citations", "confidence"}
    assert "[1]" in call["messages"][-1]["content"]  # numbered context was sent


async def test_ask_off_topic_skips_llm(client, fake_llm):
    await upload_sample(client)
    r = await client.post("/ask", json={"question": "zebra giraffe hippopotamus savanna"})
    body = r.json()
    assert body["short_circuited"] is True
    assert body["confidence"] == "none"
    assert body["sources"]  # weak matches still returned for debugging
    assert fake_llm.calls == []


async def test_ask_uncited_answer_cannot_be_high_confidence(client, fake_llm):
    await upload_sample(client)
    fake_llm.queue(json.dumps({"answer": "Probably eight.", "citations": [], "confidence": "high"}))
    r = await client.post("/ask", json={"question": "multi-head attention parallel heads"})
    assert r.json()["confidence"] == "low"


async def test_ask_document_filter_with_unknown_id(client):
    r = await client.post(
        "/ask", json={"question": "attention heads", "document_id": "00000000-0000-0000-0000-000000000000"}
    )
    assert r.json()["short_circuited"] is True
    assert "not ready" in r.json()["answer"] or "No indexed" in r.json()["answer"]


async def test_scoped_question_bypasses_threshold(client, fake_llm):
    """Vague prompt + explicit document → the model answers, no "not covered"."""
    doc = await upload_sample(client)
    r = await client.post("/ask", json={"question": "give summary", "document_id": doc["id"], "top_k": 3})
    body = r.json()
    assert body["short_circuited"] is False
    assert body["answer"] == "Grounded answer [1]."
    assert len(fake_llm.calls) == 1


async def test_unscoped_vague_question_still_short_circuits(client, fake_llm):
    await upload_sample(client)
    r = await client.post("/ask", json={"question": "zebra giraffe hippopotamus savanna"})
    assert r.json()["short_circuited"] is True
