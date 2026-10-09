# AI Research & Document Intelligence Agent

Upload PDFs, ask questions about them, and get answers with page-level citations. Everything runs locally: a FastAPI backend, PostgreSQL with the pgvector extension for similarity search, [Ollama](https://ollama.com) for both embeddings and the chat model, and a React frontend. No API keys, no cloud costs.

This is a from-scratch learning build. Source files carry explanatory comments on *why* each piece exists, not just what it does.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Environment (Python, Ollama, Postgres + pgvector) | Done |
| 2 | FastAPI skeleton, `/health` | Done |
| 3 | Database models + Alembic migration | Done |
| 4 | Ingestion pipeline: upload → extract → chunk → embed → store | Done |
| 5 | Basic RAG: `POST /ask` with retrieved sources | Done |
| 6 | Agent with tool calling: `POST /agent` | Done |
| 7 | Structured output, relevance threshold, timeouts | Done |
| 8 | Conversation history: `/conversations` | Done |
| 9 | React frontend | Done |
| 10 | Tests, streaming, hybrid retrieval | Done |

## Architecture

```
            ┌──────────────┐   POST /documents   ┌────────────────────────┐
  client ──▶│   FastAPI    │────────────────────▶│  uploads/<uuid>.pdf    │
            │  (uvicorn)   │  202 Accepted        └───────────┬────────────┘
            └──────┬───────┘                                  │ background task
                   │                                          ▼
                   │                      pypdf ──▶ chunker ──▶ Ollama embed
                   │                                          │ (nomic-embed-text, 768-d)
                   ▼                                          ▼
            ┌────────────────────────────────────────────────────────────┐
            │  PostgreSQL 16 + pgvector                                  │
            │  documents ─< chunks(embedding vector(768), HNSW index)    │
            │  conversations ─< messages(sources JSONB)                  │
            └────────────────────────────────────────────────────────────┘
```

**Ingestion flow.** The upload endpoint does only the fast work: validate the content type, stream the file to disk in 1 MB pieces, insert a `documents` row, and return `202 Accepted`. A FastAPI `BackgroundTask` then runs [ingestion.py](backend/app/services/ingestion.py), which moves the document through `pending → processing → ready` (or `failed` with an error message). Clients poll `GET /documents/{id}` for the status.

**Chunking.** [chunker.py](backend/app/services/chunker.py) splits each page on sentence boundaries, targeting 800 characters per chunk with 150 characters of overlap so no sentence is cut in half. Every chunk keeps its page number for citations.

**Embeddings.** [embeddings.py](backend/app/services/embeddings.py) is a thin async wrapper over Ollama's embed endpoint, batching 32 texts per call. Swapping providers later means editing one file.

**Storage.** [models.py](backend/app/db/models.py) defines four tables with UUID primary keys and database-side timestamps. Chunks hold a `vector(768)` column with an HNSW index tuned for cosine distance. Deleting a document cascades to its chunks.

## Project layout

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, /health
│   │   ├── api/
│   │   │   ├── documents.py     # /documents CRUD + upload
│   │   │   ├── ask.py           # POST /ask (RAG)
│   │   │   ├── agent.py         # POST /agent (tool-calling agent)
│   │   │   └── conversations.py # multi-turn chat with stored history
│   │   ├── core/config.py       # typed Settings loaded from .env
│   │   ├── db/
│   │   │   ├── session.py       # async engine, session factory, get_db dependency
│   │   │   └── models.py        # Document, Chunk, Conversation, Message
│   │   ├── schemas/             # Pydantic request/response models
│   │   └── services/
│   │       ├── pdf.py           # pypdf text extraction, page-aware
│   │       ├── chunker.py       # sentence-aware overlapping chunks
│   │       ├── embeddings.py    # Ollama embedding client
│   │       ├── ingestion.py     # background orchestrator
│   │       ├── retrieval.py     # hybrid search: pgvector + full-text, rank fusion
│   │       ├── llm.py           # Ollama chat client
│   │       ├── rag.py           # retrieve → threshold → structured answer with citations
│   │       ├── structured.py    # JSON answer schema, citation validation
│   │       ├── history.py       # question condensing, recent-turn window
│   │       ├── tools.py         # agent tools: search_documents, list_documents, read_page
│   │       └── agent.py         # tool-calling loop with RAG fallback
│   ├── alembic/                 # migrations (URL read from Settings, not alembic.ini)
│   ├── tests/                   # pytest suite, runs without Ollama
│   ├── .env.example
│   └── requirements.txt
├── frontend/                    # Vite + React + TypeScript client
│   └── src/
│       ├── App.tsx              # owns server state, three-column layout
│       ├── api.ts               # typed fetch wrapper for every endpoint
│       ├── types.ts             # mirrors the backend Pydantic schemas
│       └── components/
│           ├── DocumentsPanel.tsx   # upload, status polling, delete
│           ├── ConversationList.tsx # chats sidebar
│           ├── ChatView.tsx         # messages, citation chips, composer
│           └── SourcesPanel.tsx     # passages, scores, agent steps
├── tools/
│   └── pg.ps1                   # portable PostgreSQL manager (Windows, no admin needed)
├── samples/
│   └── attention_is_all_you_need.pdf
└── docker-compose.yml           # alternative: Postgres + pgvector in Docker
```

## Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer (for the frontend)
- [Ollama](https://ollama.com/download) running locally
- PostgreSQL 16 with the pgvector extension, via **either** Docker **or** the portable setup below

Pull the two models once:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

On a CPU-only machine with 16 GB RAM, expect the first chat response to take 15 to 30 seconds. Embedding is faster.

## Setup

### 1. Database

**Option A: Docker** (simplest if you have Docker Desktop)

```powershell
docker compose up -d
```

This listens on port 5432 with user, password, and database all set to `docintel`.

**Option B: Portable PostgreSQL** (no Docker, no admin rights)

Download the PostgreSQL 16 Windows binaries zip from EDB and extract it to `tools\pgsql`. Download a Windows build of pgvector for PG16 and copy its `lib`, `share`, and `include` contents over the matching folders in `tools\pgsql`. Then:

```powershell
.\tools\pg.ps1 init      # once: create cluster, database, enable pgvector
.\tools\pg.ps1 start     # every session
.\tools\pg.ps1 status
.\tools\pg.ps1 psql      # SQL shell
.\tools\pg.ps1 stop
```

The script uses port **5433** because an existing PostgreSQL service on the development machine already holds 5432. Edit `$DbPort` in [pg.ps1](tools/pg.ps1) if you want a different port, and keep `DATABASE_URL` in sync.

### 2. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and check `DATABASE_URL` matches your database option. The example file points at port 5433 for the portable setup. For Docker, change it to 5432.

Apply migrations and start the server:

```powershell
alembic upgrade head
uvicorn app.main:app --reload
```

Interactive API docs are at http://localhost:8000/docs.

### 3. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server expects the API on port 8000. Set `VITE_API_URL` if yours differs. `npm run build` type-checks and bundles into `frontend/dist`.

## Configuration

All settings live in `backend/.env` and are validated at startup by [config.py](backend/app/core/config.py).

| Variable | Default | Notes |
|----------|---------|-------|
| `DATABASE_URL` | required | `postgresql+asyncpg://user:pass@host:port/db` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `LLM_MODEL` | `qwen2.5:7b` | chat model for answers and tool calling |
| `EMBEDDING_MODEL` | `nomic-embed-text` | |
| `EMBEDDING_DIMENSIONS` | `768` | must match the embedding model; changing it requires a new migration and re-embedding |
| `EMBEDDING_DOCUMENT_PREFIX` | `search_document: ` | task prefix nomic-embed-text expects on passages |
| `EMBEDDING_QUERY_PREFIX` | `search_query: ` | task prefix on questions. Change either, then run the re-embed script |
| `UPLOAD_DIR` | `./uploads` | relative to `backend/` |
| `CHUNK_SIZE` | `800` | characters per chunk |
| `CHUNK_OVERLAP` | `150` | characters shared between adjacent chunks |
| `HYBRID_SEARCH` | `true` | fuse vector and keyword search; set false for pure vector |
| `RELEVANCE_THRESHOLD` | `0.55` | if the best retrieved chunk scores below this, answer "not covered" without calling the LLM. Ignored when the question is scoped to one document |
| `LLM_TIMEOUT_SECONDS` | `180` | how long to wait for Ollama before returning 504 |

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check, reports configured model names |
| `POST` | `/documents` | Upload a PDF (max 25 MB). Returns `202` and starts ingestion |
| `GET` | `/documents` | List documents, newest first |
| `GET` | `/documents/{id}` | Fetch one document, including `status` and `page_count` |
| `DELETE` | `/documents/{id}` | Remove document, its chunks, and the file on disk |
| `POST` | `/ask` | Ask a question over indexed documents. Returns the answer with numbered sources |
| `POST` | `/agent` | Same, but the model chooses tools (search, list documents, read a page) and may take several steps |
| `POST` | `/conversations` | Create a conversation (title optional, defaults to the first question) |
| `GET` | `/conversations` | List conversations, most recently active first |
| `GET` | `/conversations/{id}` | Conversation with all messages, their sources, and diagnostics |
| `DELETE` | `/conversations/{id}` | Delete a conversation and its messages |
| `POST` | `/conversations/{id}/messages` | Send a question in context. Body takes `question`, optional `document_id`, `mode` (`ask` or `agent`), `top_k` |
| `POST` | `/conversations/{id}/messages/stream` | Same, but responds with server-sent events: `status`, `sources`, `token` (many), then `done` or `error` |

Example:

```powershell
curl -F "file=@samples/attention_is_all_you_need.pdf" http://localhost:8000/documents
# {"id":"...","status":"pending",...}

curl http://localhost:8000/documents/<id>
# {"status":"ready","page_count":15,...}
```

The sample paper produces 15 pages and roughly 71 chunks.

Ask a question once the document is `ready`:

```powershell
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" `
  -d '{"question":"What is multi-head attention?","top_k":4}'
```

The request body accepts `question`, `top_k` (1 to 20, default 5), and an optional `document_id` to scope the search to one file. The response contains the answer with inline `[n]` citations, a `sources` list whose `index` matches those markers, and each source's filename, page number, similarity score, and a 300-character excerpt. It also carries `citations` (the validated passage numbers the answer relies on), `confidence` (high, medium, low, or none), and `short_circuited` (true when the LLM was never called). On a CPU-only machine a full answer takes 40 to 80 seconds. A question the documents do not cover returns in under 2 seconds.

**Question answering flow.** [retrieval.py](backend/app/services/retrieval.py) runs two searches and merges them. The vector leg embeds the question with the same model used for chunks and queries the HNSW index by cosine distance. The keyword leg runs Postgres full-text search over the chunk text with the question's terms OR-ed together, backed by a GIN index. Reciprocal rank fusion combines the two ranked lists, so a passage both searches like rises to the top. Keyword search is what catches exact numbers and acronyms that embeddings blur. In testing, the question about the number of attention heads and their dimension was unanswerable with vector search alone and answered correctly once the keyword leg surfaced the chunk containing the exact figures. [rag.py](backend/app/services/rag.py) numbers the retrieved passages, instructs the model to answer only from them and to cite by number, and tells it to say so when the context does not cover the question. Relevant passages typically score above 0.7; off-topic questions score around 0.5, which is why the model abstains on them.

**Agent flow.** `POST /agent` accepts the same `question` and optional `document_id`. Instead of always retrieving first, the model is given three tools defined in [tools.py](backend/app/services/tools.py) and decides which to call. [agent.py](backend/app/services/agent.py) runs the loop: call the model, execute any requested tools, append the results, repeat until the model writes a final answer or hits the four-step limit. The response includes a `steps` list showing every tool call made, plus `fallback_used` and `fallback_reason`.

Guard rails, because a 7B model's tool calling is imperfect:

- Unknown tool names or bad arguments are returned to the model as error text so it can recover, rather than failing the request.
- If the model answers without calling any tool, the answer is discarded and plain RAG runs instead. This prevents answering from training data.
- If the step limit is reached, plain RAG runs instead.

Observed on CPU: a list-documents question takes about 40 seconds, a search question about 65 seconds, and a read-page summary about 90 seconds.

**Answer quality controls.** Three mechanisms in [rag.py](backend/app/services/rag.py) and [structured.py](backend/app/services/structured.py) keep answers honest and fast:

- **Relevance threshold.** If the best retrieved chunk's cosine similarity is below `RELEVANCE_THRESHOLD`, the documents almost certainly do not cover the question. The system says so immediately without calling the LLM. The agent's search tool returns the same signal to the model so it abstains. In testing this cut off-topic answers from over 70 seconds to under 2. The check is skipped when a question is scoped to a single document, because the user has already said where the answer lives and vague prompts like "summarise this" legitimately score low.
- **Structured output.** The LLM is asked for JSON with `answer`, `citations`, and `confidence`, and Ollama's `format` parameter constrains generation to that schema. The schema is generated from a Pydantic model so the parser cannot drift from it.
- **Citation validation.** Both `[2]` and grouped `[1, 3]` markers are parsed and checked against the number of passages that exist. Impossible citations are stripped from the text, and an answer with no valid citations cannot report high confidence.

**Conversation flow.** Posting to a conversation loads the earlier turns, stores the user's message, and then does two things with the history. First, [history.py](backend/app/services/history.py) asks the model to rewrite the new question as a standalone one, so "how many heads does it use?" becomes "how many heads does multi-head attention use?" before it is embedded. Without this step, follow-up questions retrieve junk because the embedding has no idea what "it" refers to. Second, the last six messages are placed in the prompt so the model answers in context. The assistant's reply is stored with its sources and a `meta` object holding confidence, citations, mode, and agent steps. The response also returns `standalone_question` so you can see what was actually searched.

**Frontend.** A single page with three columns. The left sidebar lists documents with a live status dot that polls every three seconds while ingestion runs, and the conversations list. The centre is the chat: user and assistant turns, inline citation chips you can click, a confidence badge, and a composer with a mode switch (ask or agent) and a document scope selector. A running timer shows during the long CPU inference. The right panel shows the selected answer's sources with similarity bars and excerpts, highlights the ones actually cited, and surfaces diagnostics such as the condensed query, agent steps, and whether a fallback or fast path was used. The open conversation is kept in the URL hash so a refresh lands on the same chat. The backend allows the Vite origin via CORS middleware.

**Streaming.** The stream endpoint emits progress as server-sent events. The browser sees the stage change from retrieving to generating, the sources arrive as soon as retrieval finishes, and the answer text appears word by word with a blinking caret. The model still produces constrained JSON. A small incremental extractor in [structured.py](backend/app/services/structured.py) peels the answer field out of the JSON as fragments arrive, handling escapes and fragment boundaries, and the full JSON is validated normally at the end. Honest caveat for CPU inference: prompt processing happens before the first token, so on this machine the first word arrives after roughly a minute and the remaining 30 seconds stream visibly.

**Error handling.** Ollama failures map to specific status codes: 503 when the server is unreachable, 504 when a response exceeds `LLM_TIMEOUT_SECONDS`, and 502 for other Ollama errors such as a model that has not been pulled.

## Working with migrations

```powershell
cd backend
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
alembic downgrade -1
```

Alembic reads the database URL from `Settings`, so there is a single source of truth in `.env`.

## Troubleshooting

- **`initdb failed` or `pg_ctl start failed`.** Check `tools\pgdata\server.log`. The most common cause is the port already being in use.
- **`Failed to enable pgvector`.** The pgvector `vector.dll` is missing from `tools\pgsql\lib`, or the build does not match PostgreSQL 16.
- **Ingestion ends in `failed` with "No extractable text".** The PDF is scanned images. OCR is out of scope for now.
- **Ingestion ends in `failed` with a connection error.** Ollama is not running, or the embedding model is not pulled.
- **Dimension mismatch errors on insert.** `EMBEDDING_DIMENSIONS` no longer matches the model. Fix the value, write a migration to alter the column, and re-upload documents.
- **Every question says "not covered" for a document.** Scope the question to that document with the Scope dropdown or `document_id`, which bypasses the relevance threshold. Short or generic prompts such as "give summary" score low against any passage even when the document is exactly what you want. If you changed the embedding model or prefixes, re-embed:

  ```powershell
  cd backend
  .\.venv\Scripts\python.exe -m scripts.reembed
  ```

## Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

The suite runs in under a minute and needs Postgres but not Ollama. The conftest creates a separate `docintel_test` database, migrates it to head, and truncates the tables before every test. Both Ollama clients are replaced: embeddings become a bag-of-words hash into 768 dimensions so retrieval still behaves sensibly, and chat becomes a scripted fake that records every call. Coverage includes the chunker, citation validation, the streaming extractor, question condensing, PDF extraction, hybrid retrieval, and every endpoint including agent fallbacks and SSE error handling.

## Design notes

- **UUID primary keys** so IDs are safe to expose in URLs and clients cannot enumerate records.
- **Async everywhere.** SQLAlchemy 2.0 with asyncpg, async file I/O via aiofiles, and `async def` handlers so slow LLM calls do not block other requests.
- **Pydantic schemas are separate from ORM models.** The database can change without breaking the public JSON contract.
- **Files are stored by UUID**, never by the original filename, which avoids collisions and path traversal.
- **Background tasks open their own session.** The request session is already closed by the time ingestion runs.
