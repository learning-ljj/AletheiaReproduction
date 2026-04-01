from pathlib import Path

from src.core.finalizer import build_final_output
from src.memory.problem_memory import ProblemMemory
from src.utils.reference_builder import build_references



def test_reference_builder_numbering_and_missing_warning(tmp_path: Path) -> None:
    memory = ProblemMemory(problem_id="p-ref", runs_root=tmp_path / "runs")
    paper_path = memory.add_paper(
        "---\n"
        "title: Demo\n"
        "---\n\n"
        "## Layer2-Extracted\n"
        "body\n\n"
        "## Layer3-Source\n"
        "title: Demo Paper\n"
        "authors: Alice; Bob\n"
        "url: https://example.org/demo\n",
        filename="demo.md",
    )
    assert paper_path.exists()

    solution = (
        "Paragraph A [cite:artifact/papers/demo.md]. "
        "Paragraph B [cite:artifact/papers/demo.md]. "
        "Paragraph C [cite:artifact/papers/missing.md]."
    )

    converted, references, warnings = build_references(solution, memory)

    assert converted.count("[1]") == 2
    assert "[2]" in converted
    assert len(references) == 1
    assert references[0].startswith("[1] Alice; Bob. Demo Paper.")
    assert len(warnings) == 1
    assert "missing citation path" in warnings[0]


def test_final_output_template_includes_references_and_warnings() -> None:
    final_output = build_final_output(
        success=True,
        solution_text="Solved with citation [1].",
        failure_reason=None,
        references=["[1] Alice; Bob. Demo Paper. https://example.org/demo."],
        warning_summary="- [2] missing citation path: artifact/papers/missing.md",
    )

    assert "Solved with citation [1]." in final_output
    assert "## References" in final_output
    assert "[1] Alice; Bob. Demo Paper." in final_output
    assert "## Citation Warnings" in final_output
    assert "missing citation path" in final_output
