import json
from pathlib import Path

import pytest

from src.tools.artifact_reader import read_artifact_layer
from src.utils.parsing.parser import (
    extract_xml_tags,
    parse_citation_review,
    parse_citations,
    parse_citations_with_claim_spans,
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


def test_parser_contract_extracts_sentence_level_claim_spans() -> None:
    text = "<solution>Claim alpha [cite:artifact/papers/a.md]\nNext line.</solution>"
    cites, spans = parse_citations_with_claim_spans(text)

    assert cites == ["artifact/papers/a.md"]
    assert spans == ["Claim alpha"]


def test_layer_reader_reads_target_layers(tmp_path: Path) -> None:
    artifact_file = tmp_path / "runs" / "p-layer" / "artifact" / "lemmas" / "001.md"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text(
        "---\n"
        "summary: demo lemma\n"
        "conclusion: demo\n"
        "---\n\n"
        "## Layer2-Proof\n"
        "This is detailed proof text.\n\n"
        "## Layer3-Source\n"
        "title: Demo Paper\n",
        encoding="utf-8",
    )

    layer1 = read_artifact_layer(str(artifact_file), 1)
    layer2 = read_artifact_layer(str(artifact_file), 2)
    layer3 = read_artifact_layer(str(artifact_file), 3)

    layer1_payload = json.loads(layer1)
    layer2_payload = json.loads(layer2)
    layer3_payload = json.loads(layer3)

    assert layer1_payload["status"] == "OK"
    assert layer2_payload["status"] == "OK"
    assert layer3_payload["status"] == "OK"

    assert "summary: demo lemma" in layer1_payload["data"]["content"]
    assert "detailed proof text" in layer2_payload["data"]["content"]
    assert "title: Demo Paper" in layer3_payload["data"]["content"]


def test_layer_reader_rejects_path_outside_runs_artifact(tmp_path: Path) -> None:
    outside_file = tmp_path / "outside.md"
    outside_file.write_text("demo", encoding="utf-8")

    output = read_artifact_layer(str(outside_file), 1)
    payload = json.loads(output)
    assert payload["status"] == "ERROR"
    assert payload["error"]["error_code"] == "PATH_NOT_ALLOWED"


def test_parser_contract_empty_input_defaults() -> None:
    parsed = parse_protocol_blocks("")
    assert parsed["lemmas"] == []
    assert parsed["verified_lemmas"] == []
    assert parsed["citations"] == []
    assert parsed["citation_review"] == ""
