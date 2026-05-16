import json

import pytest

# These imports will fail until main.py is updated — that's expected.
from interview_retro.main import _build_enrichment, _check_quality


class TestCheckQuality:
    def test_extraction_empty_routes_to_retry_extraction(self) -> None:
        result: dict[str, object] = {"rated_qa": [], "overall_score": None}
        issues, route = _check_quality(result, qa_pairs_extracted=0)
        assert "extraction_empty" in issues
        assert route == "retry_extraction"

    def test_thin_output_routes_to_retry_judge(self) -> None:
        result: dict[str, object] = {
            "rated_qa": [{"score": 8.0}],
            "overall_score": 8.0,
        }
        issues, route = _check_quality(result, qa_pairs_extracted=3)
        assert any(i.startswith("thin_output") for i in issues)
        assert route == "retry_judge"

    def test_parse_failure_routes_to_retry_judge(self) -> None:
        result: dict[str, object] = {}
        issues, route = _check_quality(result, qa_pairs_extracted=2)
        assert "parse_failure" in issues
        assert route == "retry_judge"

    def test_low_score_routes_to_retry_judge(self) -> None:
        result: dict[str, object] = {
            "rated_qa": [{"score": 5.0}, {"score": 4.0}],
            "overall_score": 4.5,
        }
        issues, route = _check_quality(result, qa_pairs_extracted=2)
        assert any(i.startswith("low_score") for i in issues)
        assert route == "retry_judge"

    def test_passing_quality_returns_no_issues(self) -> None:
        result: dict[str, object] = {
            "rated_qa": [{"score": 8.0}, {"score": 9.0}],
            "overall_score": 8.5,
        }
        issues, route = _check_quality(result, qa_pairs_extracted=2)
        assert issues == []
        assert route == ""

    def test_score_exactly_7_passes(self) -> None:
        result: dict[str, object] = {
            "rated_qa": [{"score": 7.0}],
            "overall_score": 7.0,
        }
        issues, route = _check_quality(result, qa_pairs_extracted=1)
        assert issues == []
        assert route == ""

    def test_extraction_empty_takes_priority_over_parse_failure(self) -> None:
        result: dict[str, object] = {}
        issues, route = _check_quality(result, qa_pairs_extracted=0)
        assert route == "retry_extraction"


class TestBuildEnrichment:
    def test_contains_retry_header(self) -> None:
        result: dict[str, object] = {"rated_qa": [], "overall_score": 3.0}
        text = _build_enrichment(result, ["low_score: 3.0"], retry_n=2, qa_pairs_extracted=1)
        assert "[QUALITY RETRY 2/5]" in text

    def test_contains_issues(self) -> None:
        result: dict[str, object] = {"rated_qa": [], "overall_score": 3.0}
        text = _build_enrichment(
            result, ["low_score: 3.0", "thin_output: 0/2"], retry_n=1, qa_pairs_extracted=2
        )
        assert "low_score: 3.0" in text
        assert "thin_output: 0/2" in text

    def test_previous_output_truncated_at_2000_chars(self) -> None:
        big_result: dict[str, object] = {"rated_qa": [{"data": "x" * 5000}], "overall_score": 2.0}
        text = _build_enrichment(big_result, ["low_score: 2.0"], retry_n=1, qa_pairs_extracted=1)
        snippet_start = text.find("---\n") + 4
        snippet_end = text.find("\n---\n", snippet_start)
        snippet = text[snippet_start:snippet_end]
        assert len(snippet) <= 2100

    def test_ends_with_instruction(self) -> None:
        result: dict[str, object] = {"rated_qa": [], "overall_score": 2.0}
        text = _build_enrichment(result, ["low_score: 2.0"], retry_n=1, qa_pairs_extracted=0)
        assert "Address the issues above" in text
