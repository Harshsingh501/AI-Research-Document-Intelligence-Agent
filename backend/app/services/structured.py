"""
Structured answers: make the model return JSON we can validate.

Free text is fine for humans but useless for code. If the model returns
    {"answer": "...", "citations": [2, 4], "confidence": "high"}
we can check the citations exist, surface confidence in the UI, and refuse to
show a "high confidence" answer that cites nothing.

How it works: Ollama's `format` parameter takes a JSON schema and constrains
token sampling so the output MUST match it (grammar-constrained decoding).
The model cannot emit a stray word outside the JSON. We derive that schema
from a Pydantic model so the schema and the parser can never drift apart.

Why validate citations at all?  Small models invent them: "[7]" when only
five passages were given. We strip impossible numbers rather than show a
citation that points nowhere.
"""

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

Confidence = Literal["high", "medium", "low", "none"]


class StructuredAnswer(BaseModel):
    answer: str = Field(description="The answer in plain prose, citing passages inline as [n].")
    citations: list[int] = Field(
        default_factory=list, description="Numbers of the passages actually used."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="high = directly stated in the passages; medium = inferred; low = weakly supported."
    )


# Ollama reads this to constrain generation. Pydantic emits standard JSON Schema.
ANSWER_SCHEMA = StructuredAnswer.model_json_schema()

# Matches "[2]" and also grouped forms like "[1, 3]" or "[2,4]", which models
# often produce even when told to cite one number per bracket.
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _numbers(group: str) -> list[int]:
    return [int(n) for n in re.split(r"\s*,\s*", group.strip())]


def parse_structured(raw: str) -> StructuredAnswer | None:
    """Parse the model's JSON. Returns None if it is malformed — caller decides what to do."""
    try:
        return StructuredAnswer.model_validate_json(raw)
    except ValidationError as exc:
        logger.warning("structured: model output failed validation: %s", exc.errors()[:3])
        return None


def validate_citations(text: str, claimed: list[int], n_sources: int) -> tuple[str, list[int], list[int]]:
    """
    Reconcile the [n] markers in the text with the `citations` list and the
    number of passages that actually exist.

    Returns (cleaned_text, valid_citations, invalid_citations). Invalid
    markers are removed from the text so the reader is never pointed at a
    passage that does not exist.
    """
    in_text = {n for g in _CITE.findall(text) for n in _numbers(g)}
    all_claimed = in_text | set(claimed)
    valid = sorted(n for n in all_claimed if 1 <= n <= n_sources)
    invalid = sorted(all_claimed - set(valid))

    if invalid:
        logger.warning("structured: dropping invalid citations %s (only %d sources)", invalid, n_sources)

        def _rewrite(m: re.Match) -> str:
            keep = [n for n in _numbers(m.group(1)) if n not in invalid]
            return f"[{', '.join(map(str, keep))}]" if keep else ""

        text = _CITE.sub(_rewrite, text)
        text = re.sub(r" {2,}", " ", text)          # collapse doubled spaces left behind
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # "word ." → "word."
        text = text.strip()

    return text, valid, invalid


class AnswerStreamExtractor:
    """
    Pull the value of the "answer" field out of a JSON document as it streams in.

    The model emits something like  {"answer": "Eight heads [1].", "citations": ...
    one fragment at a time. We want to show the user the prose, not the braces.
    "answer" is the first property in the schema, so we wait for the opening
    `"answer": "`, then decode the string body (handling JSON escapes) until
    the closing quote. Everything after that is ignored; the full JSON is
    parsed properly once the stream ends.

    Usage: call `feed(fragment)` for every fragment; it returns whatever new
    answer text became available (often "" while still inside the prefix).
    """

    _MARK = re.compile(r'"answer"\s*:\s*"')
    _ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}

    def __init__(self) -> None:
        self._buf = ""
        self._started = False
        self._escape = False
        self.done = False

    def feed(self, fragment: str) -> str:
        if self.done:
            return ""
        self._buf += fragment
        if not self._started:
            m = self._MARK.search(self._buf)
            if not m:
                return ""
            self._started = True
            self._buf = self._buf[m.end():]

        out: list[str] = []
        s = self._buf
        i = 0
        while i < len(s):
            ch = s[i]
            if self._escape:
                if ch == "u":
                    if len(s) - i < 5:
                        break  # \uXXXX split across fragments; wait for the rest
                    out.append(chr(int(s[i + 1 : i + 5], 16)))
                    i += 5
                else:
                    out.append(self._ESCAPES.get(ch, ch))  # \" \\ \/ map to themselves
                    i += 1
                self._escape = False
                continue
            if ch == "\\":
                self._escape = True
                i += 1
                continue
            if ch == '"':
                self.done = True
                i += 1
                break
            out.append(ch)
            i += 1
        self._buf = s[i:]
        return "".join(out)
