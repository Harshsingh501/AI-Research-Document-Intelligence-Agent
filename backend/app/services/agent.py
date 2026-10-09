"""
The agent loop: let the model decide which tools to call, run them, feed the
results back, repeat until it writes a final answer.

    user question
        │
        ▼
    ┌─ LLM (with tool specs) ──────────────────────────────┐
    │   did it ask for tools?                              │
    │     yes → run each tool, append results, loop  ──────┘
    │     no  → its text is the final answer
    └───────────────────────────────────────────────────────

Why a loop and not one shot?  The model can search, read the result, realise
it needs a different query or a specific page, and try again. That is what
makes it an agent rather than a pipeline.

Why so many guard rails?  qwen2.5:7b on CPU is a small model. It sometimes:
  - calls a tool that does not exist, or with malformed arguments,
  - answers from memory without searching at all,
  - keeps calling tools forever.
Each case has a fallback below. The key principle: the user must ALWAYS get a
grounded answer, so when the agent misbehaves we drop back to plain RAG
(Phase 5), which is slower to reason but cannot go off the rails.
"""

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm import chat_with_tools
from app.services.rag import answer_question
from app.services.retrieval import RetrievedChunk
from app.services.structured import Confidence, validate_citations
from app.services.tools import TOOL_IMPLS, TOOL_NAMES, TOOL_SPECS, ToolContext

logger = logging.getLogger(__name__)

MAX_STEPS = 4  # LLM round-trips. Each is ~30-60s on CPU, so keep it tight.

SYSTEM_PROMPT = """\
You are a research assistant with access to the user's uploaded documents
through tools. You know NOTHING about the documents until you use a tool.

Rules:
- For any question about document content, call search_documents first.
  Never answer from memory.
- If the user asks which documents exist, call list_documents.
- If the user mentions a specific page, call read_page.
- When you have enough information, write the final answer in plain text and
  cite passages by their numbers, e.g. "... 8 heads [2].".
- If the tools return nothing relevant, say that the documents do not cover
  the question. Do not guess."""


@dataclass
class AgentStep:
    """One tool call, recorded so the client can show the agent's reasoning trail."""

    tool: str
    arguments: dict
    result_preview: str


@dataclass
class AgentResult:
    answer: str
    sources: list[RetrievedChunk]
    steps: list[AgentStep] = field(default_factory=list)
    confidence: Confidence = "none"
    citations: list[int] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str | None = None


async def run_agent(
    db: AsyncSession,
    question: str,
    document_id: uuid.UUID | None = None,
    history: list[dict] | None = None,
) -> AgentResult:
    ctx = ToolContext(db=db)
    steps: list[AgentStep] = []

    user_content = question
    if document_id is not None:
        # Hand the model the id so it does not have to call list_documents first
        user_content += f"\n\n(Restrict your search to document_id={document_id}.)"

    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []  # earlier turns, so follow-ups make sense
    messages.append({"role": "user", "content": user_content})

    for step in range(1, MAX_STEPS + 1):
        message = await chat_with_tools(messages, TOOL_SPECS)
        messages.append(message)  # the model's own turn must stay in the history

        tool_calls = message.tool_calls or []
        if not tool_calls:
            # Model wrote a final answer.
            if not steps:
                # ...without ever looking at a document. For content questions
                # that is a hallucination risk, so ground it with plain RAG.
                return await _fallback(
                    db, question, document_id, steps, "model answered without using any tool", history
                )
            return _finalise(message.content, ctx.sources, steps)

        for call in tool_calls:
            name = call.function.name
            args = dict(call.function.arguments or {})
            if name not in TOOL_NAMES:
                result = f"Error: unknown tool '{name}'. Available tools: {sorted(TOOL_NAMES)}."
            else:
                try:
                    result = await TOOL_IMPLS[name](ctx, **args)
                except Exception as exc:  # noqa: BLE001 — surface to the model, don't crash
                    logger.exception("agent: tool %s failed", name)
                    result = f"Error: {type(exc).__name__}: {exc}"

            logger.info("agent: step %d → %s(%s)", step, name, args)
            steps.append(AgentStep(tool=name, arguments=args, result_preview=result[:200]))
            messages.append({"role": "tool", "content": result})

    # Ran out of steps while still calling tools.
    return await _fallback(db, question, document_id, steps, f"no final answer after {MAX_STEPS} steps", history)


def _finalise(text: str, sources: list[RetrievedChunk], steps: list[AgentStep]) -> AgentResult:
    """
    Validate the agent's citations and derive a confidence level.

    The agent's final turn is free text (tools + JSON-schema output do not mix
    well in small models), so unlike /ask we infer confidence from evidence:
      - cited at least one real passage              -> high
      - saw passages but cited none                  -> low   (weak support)
      - searched/read but found nothing relevant     -> none  (an abstention)
      - used only metadata tools (list_documents)    -> medium (nothing to cite)
    """
    text, valid, _ = validate_citations(text.strip(), [], len(sources))
    searched = any(s.tool in ("search_documents", "read_page") for s in steps)
    if valid:
        confidence: Confidence = "high"
    elif sources:
        confidence = "low"
    elif searched:
        confidence = "none"
    else:
        confidence = "medium"
    return AgentResult(answer=text, sources=sources, steps=steps, confidence=confidence, citations=valid)


async def _fallback(
    db: AsyncSession,
    question: str,
    document_id: uuid.UUID | None,
    steps: list[AgentStep],
    reason: str,
    history: list[dict] | None = None,
) -> AgentResult:
    logger.warning("agent: falling back to plain RAG (%s)", reason)
    rag = await answer_question(db, question, document_id=document_id, history=history)
    return AgentResult(
        answer=rag.answer,
        sources=rag.sources,
        steps=steps,
        confidence=rag.confidence,
        citations=rag.citations,
        fallback_used=True,
        fallback_reason=reason,
    )
