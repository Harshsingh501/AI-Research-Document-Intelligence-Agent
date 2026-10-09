"""
Conversation history helpers.

Two problems appear the moment a chat has more than one turn:

1. RETRIEVAL. The user asks "What is multi-head attention?" then "How many
   heads?". The second question, embedded on its own, is nearly meaningless:
   heads of what? Vector search returns junk. The fix is QUESTION CONDENSING:
   ask the LLM to rewrite the follow-up as a standalone question using the
   conversation so far ("How many heads does the Transformer's multi-head
   attention use?"), then embed THAT. It costs one extra short LLM call.

2. GENERATION. The model also needs the earlier turns to answer coherently
   ("as mentioned above..."). We pass the last few messages in the chat
   prompt. "Last few" matters: every token of history is a token the CPU has
   to process on every call, and old turns rarely help.
"""

import logging

from app.db.models import Message
from app.services.llm import chat

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 6  # 3 user/assistant pairs. Beyond that, cost outweighs benefit on CPU.

CONDENSE_PROMPT = """\
You rewrite follow-up questions so they can be understood on their own.
Given the conversation so far and the user's latest message, output ONE
standalone question that includes any topic, name or reference from earlier
turns that the latest message depends on.
- Output only the rewritten question, nothing else.
- If the latest message is already self-contained, output it unchanged.
- Never answer the question."""


def to_chat_messages(messages: list[Message]) -> list[dict]:
    """ORM messages → the role/content dicts Ollama expects, most recent N only."""
    recent = messages[-MAX_HISTORY_MESSAGES:]
    return [{"role": m.role.value, "content": m.content} for m in recent]


async def condense_question(history: list[dict], question: str) -> str:
    """Return a standalone version of `question`, or `question` itself if there is no history."""
    if not history:
        return question

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    raw = await chat(
        [
            {"role": "system", "content": CONDENSE_PROMPT},
            {"role": "user", "content": f"Conversation so far:\n{transcript}\n\nLatest message: {question}"},
        ],
        temperature=0.0,
        max_tokens=96,  # a question, not an essay
    )
    standalone = raw.strip().strip('"').strip()

    # Guard rails: a small model occasionally answers instead of rewriting,
    # or produces nothing. In both cases the original question is safer.
    if not standalone or len(standalone) > 400 or "\n" in standalone:
        logger.warning("history: condense produced unusable output, using original question")
        return question

    logger.info("history: condensed %r → %r", question, standalone)
    return standalone
