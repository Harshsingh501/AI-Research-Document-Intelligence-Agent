import json

from tests.conftest import upload_sample


async def test_agent_lists_documents(client, fake_llm):
    doc = await upload_sample(client)
    fake_llm.queue([("list_documents", {})], "You have one document: the Transformer paper.")

    r = await client.post("/agent", json={"question": "Which documents do I have?"})
    body = r.json()
    assert r.status_code == 200
    assert [s["tool"] for s in body["steps"]] == ["list_documents"]
    assert doc["filename"] in body["steps"][0]["result_preview"]
    assert body["fallback_used"] is False
    assert body["confidence"] == "medium"  # metadata only, nothing to cite
    assert body["sources"] == []
    # The tool result was fed back to the model as a `tool` message
    roles = [m["role"] if isinstance(m, dict) else m.role for m in fake_llm.calls[1]["messages"]]
    assert roles[-1] == "tool"


async def test_agent_searches_and_cites(client, fake_llm):
    await upload_sample(client)
    fake_llm.queue(
        [("search_documents", {"query": "multi-head attention parallel heads"})],
        "The model uses eight heads [1].",
    )
    r = await client.post("/agent", json={"question": "How many heads?"})
    body = r.json()
    assert body["steps"][0]["tool"] == "search_documents"
    assert body["citations"] == [1] and body["confidence"] == "high"
    assert len(body["sources"]) == 5


async def test_agent_read_page(client, fake_llm):
    doc = await upload_sample(client)
    fake_llm.queue([("read_page", {"document_id": doc["id"], "page_number": 1})], "Page 1 is the abstract.")
    r = await client.post("/agent", json={"question": "What is on page 1?"})
    body = r.json()
    assert body["steps"][0]["tool"] == "read_page"
    assert body["steps"][0]["result_preview"].startswith(f"{doc['filename']}, page 1:")
    assert all(s["page_number"] == 1 for s in body["sources"])


async def test_agent_recovers_from_unknown_tool(client, fake_llm):
    await upload_sample(client)
    fake_llm.queue([("fetch_wikipedia", {"q": "x"})], [("list_documents", {})], "Done.")
    r = await client.post("/agent", json={"question": "Which documents do I have?"})
    body = r.json()
    assert [s["tool"] for s in body["steps"]] == ["fetch_wikipedia", "list_documents"]
    assert "unknown tool" in body["steps"][0]["result_preview"]
    assert body["fallback_used"] is False


async def test_agent_falls_back_when_model_skips_tools(client, fake_llm):
    await upload_sample(client)
    fake_llm.queue("France won the 2018 World Cup.")  # answered from memory, no tool
    fake_llm.queue(json.dumps({"answer": "Grounded [1].", "citations": [1], "confidence": "medium"}))
    r = await client.post("/agent", json={"question": "multi-head attention parallel heads"})
    body = r.json()
    assert body["fallback_used"] is True
    assert "without using any tool" in body["fallback_reason"]
    assert body["answer"] == "Grounded [1]."  # the ungrounded answer was discarded


async def test_agent_falls_back_after_max_steps(client, fake_llm):
    await upload_sample(client)
    fake_llm.queue(*([[("list_documents", {})]] * 4))  # never stops calling tools
    r = await client.post("/agent", json={"question": "zebra giraffe"})
    body = r.json()
    assert body["fallback_used"] is True
    assert "after 4 steps" in body["fallback_reason"]
    assert len(body["steps"]) == 4
