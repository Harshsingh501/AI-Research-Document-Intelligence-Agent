"""
Shared test fixtures.

Design choices worth understanding:

* A SEPARATE DATABASE (`docintel_test`). Tests truncate tables between runs,
  which must never touch your real documents. The fixture creates the DB
  and runs the Alembic migrations on it the first time.

* OLLAMA IS FAKED. The LLM takes a minute per call on CPU and its output is
  not deterministic — neither is acceptable in a test. We replace the two
  Ollama clients with fakes:
    - embeddings: a bag-of-words hash into 768 dims. Texts sharing words get
      similar vectors, so retrieval behaves sensibly without a model.
    - chat: returns scripted replies and records every call for assertions.

* Settings are read at import time, so the environment is set BEFORE the app
  is imported (that's why the os.environ lines are at the very top).
"""

import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

BACKEND_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PDF = BACKEND_DIR.parent / "samples" / "attention_is_all_you_need.pdf"

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://docintel:docintel@localhost:5433/docintel_test"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="docintel_test_uploads_")
os.environ["RELEVANCE_THRESHOLD"] = "0.3"  # fake embeddings are coarser than real ones

# Only now is it safe to import the app
import httpx  # noqa: E402
import ollama  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402
from app.main import app  # noqa: E402

# --- Test database --------------------------------------------------------------


def _dsn_parts(url: str) -> dict:
    m = re.match(r"postgresql\+asyncpg://(.+?):(.+?)@(.+?):(\d+)/(.+)", url)
    assert m, f"unexpected DATABASE_URL format: {url}"
    user, password, host, port, db = m.groups()
    return dict(user=user, password=password, host=host, port=int(port), database=db)


async def _ensure_test_db() -> None:
    p = _dsn_parts(TEST_DATABASE_URL)
    admin = await asyncpg.connect(**{**p, "database": "postgres"})
    try:
        exists = await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", p["database"])
        if not exists:
            await admin.execute(f'CREATE DATABASE "{p["database"]}"')
    finally:
        await admin.close()
    conn = await asyncpg.connect(**p)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()


@pytest.fixture(scope="session", autouse=True)
def test_database():
    """Create the test DB and bring its schema to head. Runs once per session."""
    asyncio.run(_ensure_test_db())
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env={**os.environ, "DATABASE_URL": TEST_DATABASE_URL},
        check=True,
        capture_output=True,
    )


@pytest_asyncio.fixture(autouse=True)
async def clean_tables():
    """Empty every table before each test, and release pooled connections after."""
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE messages, conversations, chunks, documents CASCADE"))
    yield
    # asyncpg connections are bound to the event loop they were created on.
    # Disposing the pool after each test avoids "attached to a different loop"
    # errors if the loop changes between tests.
    await engine.dispose()


# --- Fake Ollama ---------------------------------------------------------------------

DIMS = 768


def fake_embedding(text_: str) -> list[float]:
    """Bag-of-words hashed into a fixed-size vector, L2-normalised."""
    # The real model needs the task prefixes; the fake must not let them
    # inject shared tokens that make every query look like every passage.
    for prefix in ("search_query: ", "search_document: "):
        text_ = text_.replace(prefix, "")
    vec = [0.0] * DIMS
    for word in re.findall(r"[a-z]{3,}", text_.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % DIMS] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeEmbedClient:
    async def embed(self, model: str, input: list[str]) -> dict:
        return {"embeddings": [fake_embedding(t) for t in input]}


class FakeChatClient:
    """
    Scripted LLM. Push replies with `.queue(...)`; when the queue is empty a
    sensible default is produced based on what kind of call it is.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._scripted: list = []

    def queue(self, *replies) -> None:
        """Each reply is a str (plain text) or a list of (tool_name, args) tool calls."""
        self._scripted.extend(replies)

    async def chat(self, model, messages, tools=None, options=None, format=None, stream=False):
        # Copy: the agent keeps appending to the same list after this call returns
        self.calls.append({"messages": list(messages), "tools": tools, "format": format, "options": options})
        reply = self._scripted.pop(0) if self._scripted else self._default(messages, format)
        if stream:
            assert isinstance(reply, str), "streaming fakes must be text"
            return self._stream(reply)
        if isinstance(reply, list):
            tool_calls = [
                ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name=n, arguments=a))
                for n, a in reply
            ]
            msg = ollama.Message(role="assistant", content="", tool_calls=tool_calls)
        else:
            msg = ollama.Message(role="assistant", content=reply)
        return ollama.ChatResponse(message=msg)

    @staticmethod
    async def _stream(text: str, size: int = 7):
        """Yield the reply in small fragments, like a real token stream."""
        for i in range(0, len(text), size):
            yield ollama.ChatResponse(message=ollama.Message(role="assistant", content=text[i : i + size]))

    @staticmethod
    def _default(messages, format):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if format is not None:
            return json.dumps({"answer": "Grounded answer [1].", "citations": [1], "confidence": "high"})
        if "rewrite follow-up" in system.lower():
            latest = messages[-1]["content"].rsplit("Latest message:", 1)[-1].strip()
            return f"{latest.rstrip('?')} in the Transformer?"
        return "Plain reply."


@pytest.fixture
def fake_llm(monkeypatch) -> FakeChatClient:
    """Swap both Ollama clients for fakes. Returns the chat fake for scripting/inspection."""
    import app.services.embeddings as emb
    import app.services.llm as llm

    chat = FakeChatClient()
    monkeypatch.setattr(emb, "_client", FakeEmbedClient())
    monkeypatch.setattr(llm, "_client", chat)
    return chat


# --- HTTP client ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(fake_llm) -> httpx.AsyncClient:
    """
    In-process client. ASGITransport calls the app directly — no network, no
    uvicorn — and waits for BackgroundTasks to finish before returning, which
    makes the ingestion pipeline synchronous from the test's point of view.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def upload_sample(client: httpx.AsyncClient) -> dict:
    """Upload the bundled paper and return the document JSON once ingested."""
    with SAMPLE_PDF.open("rb") as f:
        r = await client.post("/documents", files={"file": (SAMPLE_PDF.name, f, "application/pdf")})
    assert r.status_code == 202, r.text
    doc_id = r.json()["id"]
    r = await client.get(f"/documents/{doc_id}")
    assert r.json()["status"] == "ready", r.json()
    return r.json()


def parse_sse(body: str) -> list[tuple[str, object]]:
    """Turn a raw text/event-stream body into [(event, parsed_json), ...]."""
    events = []
    for block in body.strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event is not None:
            events.append((event, data))
    return events
