from app.db.models import Message, MessageRole
from app.services.history import MAX_HISTORY_MESSAGES, condense_question, to_chat_messages


def _msg(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content)


def test_to_chat_messages_keeps_only_recent_turns():
    msgs = [_msg(MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT, f"m{i}") for i in range(20)]
    out = to_chat_messages(msgs)
    assert len(out) == MAX_HISTORY_MESSAGES
    assert out[-1] == {"role": "assistant", "content": "m19"}
    assert out[0]["content"] == f"m{20 - MAX_HISTORY_MESSAGES}"


async def test_condense_without_history_is_identity(fake_llm):
    assert await condense_question([], "How many heads?") == "How many heads?"
    assert fake_llm.calls == []  # no LLM call wasted


async def test_condense_rewrites_follow_up(fake_llm):
    history = [{"role": "user", "content": "What is multi-head attention?"}]
    fake_llm.queue('"How many heads does multi-head attention use?"')
    out = await condense_question(history, "How many heads?")
    assert out == "How many heads does multi-head attention use?"  # quotes stripped
    # The LLM saw both the history and the latest message
    prompt = fake_llm.calls[0]["messages"][-1]["content"]
    assert "multi-head attention" in prompt and "How many heads?" in prompt
    assert fake_llm.calls[0]["options"]["num_predict"] == 96


async def test_condense_falls_back_on_unusable_output(fake_llm):
    history = [{"role": "user", "content": "x"}]
    fake_llm.queue("Sure! Here is the answer:\nIt uses 8 heads.")  # answered instead of rewriting
    assert await condense_question(history, "How many heads?") == "How many heads?"
    fake_llm.queue("")
    assert await condense_question(history, "How many heads?") == "How many heads?"
