import pytest

from src.utils.parser import (
    extract_xml_tags,
    parse_citation_review,
    parse_citations,
    parse_lemmas,
    parse_protocol_blocks,
    parse_verified_lemmas,
)


def test_parser_contract_multi_tags() -> None:
    text = (
        "<solution>demo</solution>\n"
        "<lemma>lemma-A</lemma>\n"
        "<lemma>lemma-B</lemma>\n"
        "<verified_lemmas>verified-A</verified_lemmas>\n"
        "<verified_lemmas>verified-B</verified_lemmas>"
    )

    assert extract_xml_tags(text, "lemma") == ["lemma-A", "lemma-B"]
    assert parse_lemmas(text) == ["lemma-A", "lemma-B"]
    assert parse_verified_lemmas(text) == ["verified-A", "verified-B"]



def test_parser_contract_citation_and_review_parse() -> None:
    text = (
        "A claim [cite:artifact/papers/arXiv_123.md] and another [cite:artifact/lemmas/001.md].\n"
        "<citation_review>2 checked, 0 failed</citation_review>"
    )

    assert parse_citations(text) == [
        "artifact/papers/arXiv_123.md",
        "artifact/lemmas/001.md",
    ]
    assert parse_citation_review(text) == "2 checked, 0 failed"

    parsed = parse_protocol_blocks(text)
    assert parsed["citations"] == [
        "artifact/papers/arXiv_123.md",
        "artifact/lemmas/001.md",
    ]
    assert parsed["citation_review"] == "2 checked, 0 failed"



def test_parser_contract_bad_citation_block_diagnostic() -> None:
    with pytest.raises(ValueError, match="Malformed citation block"):
        parse_citations("broken cite [cite:]")
