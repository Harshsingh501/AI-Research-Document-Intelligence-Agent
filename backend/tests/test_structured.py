from app.services.structured import ANSWER_SCHEMA, parse_structured, validate_citations


def test_schema_has_required_fields():
    assert set(ANSWER_SCHEMA["properties"]) == {"answer", "citations", "confidence"}
    assert ANSWER_SCHEMA["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


def test_parse_valid_json():
    out = parse_structured('{"answer": "x [1]", "citations": [1], "confidence": "high"}')
    assert out is not None
    assert out.citations == [1] and out.confidence == "high"


def test_parse_rejects_bad_confidence_and_non_json():
    assert parse_structured('{"answer": "x", "citations": [], "confidence": "certain"}') is None
    assert parse_structured("not json at all") is None


def test_single_citations_are_kept():
    text, valid, invalid = validate_citations("Uses 8 heads [2]. See [1].", [], 5)
    assert text == "Uses 8 heads [2]. See [1]."
    assert valid == [1, 2] and invalid == []


def test_grouped_citations_are_parsed():
    text, valid, invalid = validate_citations("Not stated [1, 2, 3].", [], 5)
    assert valid == [1, 2, 3] and invalid == []
    assert text == "Not stated [1, 2, 3]."


def test_invalid_citations_are_stripped_from_text():
    text, valid, invalid = validate_citations("Heads [2, 7] and dims [9] .", [2], 5)
    assert text == "Heads [2] and dims."
    assert valid == [2]
    assert invalid == [7, 9]


def test_claimed_list_is_merged_with_text_markers():
    _, valid, _ = validate_citations("No markers here.", [3, 4], 5)
    assert valid == [3, 4]


def test_zero_is_not_a_valid_citation():
    text, valid, invalid = validate_citations("Thing [0].", [], 5)
    assert valid == [] and invalid == [0]
    assert text == "Thing."


# --- AnswerStreamExtractor --------------------------------------------------------

from app.services.structured import AnswerStreamExtractor  # noqa: E402


def _drain(fragments: list[str]) -> str:
    ex = AnswerStreamExtractor()
    return "".join(ex.feed(f) for f in fragments)


def test_extractor_pulls_answer_from_whole_json():
    assert _drain(['{"answer": "Eight heads [1].", "citations": [1], "confidence": "high"}']) == "Eight heads [1]."


def test_extractor_handles_arbitrary_fragment_boundaries():
    doc = '{"answer": "Eight heads [1], see p. 5.", "citations": [1], "confidence": "high"}'
    for size in (1, 2, 3, 5, 11):
        frags = [doc[i : i + size] for i in range(0, len(doc), size)]
        assert _drain(frags) == "Eight heads [1], see p. 5.", size


def test_extractor_decodes_escapes_even_when_split():
    doc = '{"answer": "Line one\\nquote \\"q\\" caf\\u00e9 back\\\\slash", "citations": []}'
    expected = 'Line one\nquote "q" caf\u00e9 back\\slash'
    assert _drain([doc]) == expected
    assert _drain(list(doc)) == expected  # one character at a time


def test_extractor_ignores_text_after_the_answer():
    ex = AnswerStreamExtractor()
    out = ex.feed('{"answer": "Done.", "citations": [1, 2], "confidence": "medium"}')
    assert out == "Done." and ex.done
    assert ex.feed("more stuff") == ""
