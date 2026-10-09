import json

from tests.conftest import upload_sample


async def test_create_list_get_delete(client):
    r = await client.post("/conversations", json={})
    assert r.status_code == 201
    cid = r.json()["id"]
    assert r.json()["title"] is None

    assert (await client.get("/conversations")).json()["total"] == 1
    assert (await client.get(f"/conversations/{cid}")).json()["messages"] == []
    assert (await client.delete(f"/conversations/{cid}")).status_code == 204
    assert (await client.get(f"/conversations/{cid}")).status_code == 404
    assert (await client.post(f"/conversations/{cid}/messages", json={"question": "hello?"})).status_code == 404


async def test_first_message_sets_title_and_stores_sources(client, fake_llm):
    await upload_sample(client)
    cid = (await client.post("/conversations", json={})).json()["id"]

    r = await client.post(
        f"/conversations/{cid}/messages", json={"question": "multi-head attention parallel heads", "top_k": 3}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["standalone_question"] == "multi-head attention parallel heads"  # no history → unchanged
    a = body["assistant_message"]
    assert a["role"] == "assistant"
    assert a["meta"]["mode"] == "ask" and a["meta"]["confidence"] == "high"
    assert a["meta"]["standalone_question"] == "multi-head attention parallel heads"
    assert len(a["sources"]) == 3 and a["sources"][0]["index"] == 1

    detail = (await client.get(f"/conversations/{cid}")).json()
    assert detail["title"] == "multi-head attention parallel heads"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


async def test_follow_up_is_condensed_and_history_sent(client, fake_llm):
    await upload_sample(client)
    cid = (await client.post("/conversations", json={})).json()["id"]
    await client.post(f"/conversations/{cid}/messages", json={"question": "multi-head attention parallel heads"})
    fake_llm.calls.clear()

    fake_llm.queue("How many parallel attention heads does multi-head attention use?")  # condense
    fake_llm.queue(json.dumps({"answer": "Eight [1].", "citations": [1], "confidence": "high"}))  # answer
    r = await client.post(f"/conversations/{cid}/messages", json={"question": "How many heads?"})
    body = r.json()
    assert body["standalone_question"] == "How many parallel attention heads does multi-head attention use?"

    condense_call, answer_call = fake_llm.calls
    assert condense_call["format"] is None
    assert answer_call["format"] is not None
    roles = [m["role"] for m in answer_call["messages"]]
    # system, then the two earlier turns, then the new question with context
    assert roles == ["system", "user", "assistant", "user"]
    assert answer_call["messages"][-1]["content"].endswith(body["standalone_question"])

    detail = (await client.get(f"/conversations/{cid}")).json()
    assert len(detail["messages"]) == 4


async def test_agent_mode_records_steps(client, fake_llm):
    await upload_sample(client)
    cid = (await client.post("/conversations", json={})).json()["id"]
    fake_llm.queue([("list_documents", {})], "One document.")
    r = await client.post(f"/conversations/{cid}/messages", json={"question": "Which documents?", "mode": "agent"})
    meta = r.json()["assistant_message"]["meta"]
    assert meta["mode"] == "agent"
    assert meta["steps"] == [{"tool": "list_documents", "arguments": {}}]
    assert meta["fallback_used"] is False


async def test_deleting_conversation_removes_messages(client, fake_llm):
    cid = (await client.post("/conversations", json={})).json()["id"]
    await client.post(f"/conversations/{cid}/messages", json={"question": "anything at all"})
    assert (await client.delete(f"/conversations/{cid}")).status_code == 204
    assert (await client.get("/conversations")).json()["total"] == 0


# --- Streaming ----------------------------------------------------------------------

from tests.conftest import parse_sse  # noqa: E402


async def test_stream_emits_tokens_then_done_and_persists(client, fake_llm):
    await upload_sample(client)
    cid = (await client.post("/conversations", json={})).json()["id"]
    fake_llm.queue(json.dumps({"answer": "Eight parallel heads [1].", "citations": [1], "confidence": "high"}))

    r = await client.post(
        f"/conversations/{cid}/messages/stream", json={"question": "multi-head attention parallel heads", "top_k": 3}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e for e, _ in events]

    assert names[0] == "status" and events[0][1]["stage"] == "retrieving"
    assert names[1] == "sources" and len(events[1][1]) == 3
    assert names[2] == "status" and events[2][1]["stage"] == "generating"
    assert names[-1] == "done"
    tokens = "".join(d["text"] for e, d in events if e == "token")
    assert tokens == "Eight parallel heads [1]."

    done = events[-1][1]
    assert done["assistant_message"]["content"] == "Eight parallel heads [1]."
    assert done["assistant_message"]["meta"]["confidence"] == "high"
    detail = (await client.get(f"/conversations/{cid}")).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


async def test_stream_off_topic_short_circuits_without_tokens(client, fake_llm):
    await upload_sample(client)
    cid = (await client.post("/conversations", json={})).json()["id"]
    r = await client.post(f"/conversations/{cid}/messages/stream", json={"question": "zebra giraffe hippopotamus"})
    events = parse_sse(r.text)
    assert "token" not in [e for e, _ in events]
    assert events[-1][0] == "done"
    assert events[-1][1]["assistant_message"]["meta"]["short_circuited"] is True
    assert fake_llm.calls == []


async def test_stream_reports_errors_as_event_and_rolls_back(client, fake_llm, monkeypatch):
    await upload_sample(client)
    cid = (await client.post("/conversations", json={})).json()["id"]

    async def boom(*a, **k):
        raise RuntimeError("model exploded")
        yield  # pragma: no cover — makes this an async generator

    import app.services.rag as rag
    monkeypatch.setattr(rag, "chat_stream", boom)

    r = await client.post(f"/conversations/{cid}/messages/stream", json={"question": "multi-head attention parallel heads"})
    events = parse_sse(r.text)
    assert events[-1][0] == "error"
    assert "model exploded" in events[-1][1]["detail"]
    # Nothing was persisted: the user turn was rolled back with the failure
    assert (await client.get(f"/conversations/{cid}")).json()["messages"] == []


async def test_stream_unknown_conversation_is_404(client):
    r = await client.post("/conversations/00000000-0000-0000-0000-000000000000/messages/stream", json={"question": "hello?"})
    assert r.status_code == 404
