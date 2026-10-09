"""
FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload

Then open http://localhost:8000/docs for the auto-generated interactive API docs.
"""

import logging

import httpx
import ollama
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.agent import router as agent_router
from app.api.agent import router as agent_router
from app.api.ask import router as ask_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.core.config import settings

# Uvicorn configures its own loggers only; this makes our `app.*` loggers visible too.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")

app = FastAPI(
    title="AI Research & Document Intelligence Agent",
    description="Ask questions over your uploaded documents using RAG + a local LLM.",
    version="0.1.0",
)

# --- CORS -------------------------------------------------------------------
# Browsers block JavaScript on http://localhost:5173 (the Vite dev server) from
# calling http://localhost:8000 unless the API says it is allowed. Different
# port = different origin. In production the frontend is usually served from
# the same origin as the API and this list shrinks or disappears.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(ask_router)
app.include_router(agent_router)
app.include_router(conversations_router)


# --- Error handling ---------------------------------------------------------
# Without these, an unreachable or slow Ollama surfaces as a bare 500 with a
# stack trace in the server log and nothing useful for the client. Mapping
# each failure mode to a specific status code and message makes the problem
# obvious from the response alone.


@app.exception_handler(httpx.ConnectError)
async def ollama_unreachable(_: Request, exc: httpx.ConnectError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": f"Cannot reach Ollama at {settings.OLLAMA_BASE_URL}. Is 'ollama serve' running?"},
    )


@app.exception_handler(httpx.TimeoutException)
async def ollama_timeout(_: Request, exc: httpx.TimeoutException) -> JSONResponse:
    return JSONResponse(
        status_code=504,
        content={
            "detail": f"The model did not respond within {settings.LLM_TIMEOUT_SECONDS:.0f}s. "
            "Try a shorter question or raise LLM_TIMEOUT_SECONDS."
        },
    )


@app.exception_handler(ollama.ResponseError)
async def ollama_error(_: Request, exc: ollama.ResponseError) -> JSONResponse:
    # e.g. model not pulled: "model 'qwen2.5:7b' not found"
    return JSONResponse(status_code=502, content={"detail": f"Ollama error: {exc.error}"})
app.include_router(agent_router)


# --- Response schemas -------------------------------------------------------
# Pydantic models define the exact JSON shape an endpoint returns. FastAPI uses
# them to (1) serialize the response, (2) validate it, and (3) document it in
# /docs. If the code returns something that doesn't match, you get a clear error.


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embedding_model: str


# --- Endpoints --------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness check. `async def` means this handler runs on the event loop:
    while one request is waiting on I/O (DB, LLM), the server can serve others.
    """
    return HealthResponse(
        status="ok",
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
    )
