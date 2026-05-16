import json
from typing import Any

from crewai.flow.flow import Flow, FlowState, listen, or_, router, start
from pydantic import Field

from interview_retro.analysis import InterviewAnalysisCrew


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RetroState(FlowState):
    transcript: str = ""

    retry_count: int = 0
    quality_issues: list[str] = Field(default_factory=list)
    last_failure_reason: str = ""

    checkpoint_qa_pairs: list[dict[str, Any]] = []  # type: ignore[assignment]
    checkpoint_advocacy: list[dict[str, Any]] = []  # type: ignore[assignment]
    checkpoint_criticism: list[dict[str, Any]] = []  # type: ignore[assignment]

    last_result: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers (testable without a running crew)
# ---------------------------------------------------------------------------

def _check_quality(
    result: dict[str, Any], qa_pairs_extracted: int
) -> tuple[list[str], str]:
    """Return (issues, route). Route is 'retry_extraction', 'retry_judge', or ''."""
    if qa_pairs_extracted == 0:
        return ["extraction_empty"], "retry_extraction"

    issues: list[str] = []
    rated: list[Any] = result.get("rated_qa") or []
    if len(rated) < qa_pairs_extracted:
        issues.append(f"thin_output: {len(rated)}/{qa_pairs_extracted} pairs rated")
    if not result.get("rated_qa") or result.get("overall_score") is None:
        issues.append("parse_failure")
    elif float(result.get("overall_score", 0)) < 7.0:  # type: ignore[arg-type]
        issues.append(f"low_score: {result.get('overall_score')}")

    return issues, ("retry_judge" if issues else "")


def _build_enrichment(
    last_result: dict[str, Any],
    issues: list[str],
    retry_n: int,
    qa_pairs_extracted: int,
) -> str:
    score = last_result.get("overall_score", "N/A")
    rated_count = len(last_result.get("rated_qa") or [])
    issues_str = "\n".join(f"  - {i}" for i in issues)
    prev_output = json.dumps(last_result)[:2000]
    return (
        f"[QUALITY RETRY {retry_n}/5]\n"
        f"Issues:\n{issues_str}\n"
        f"Previous score: {score}/10 | Pairs rated: {rated_count}/{qa_pairs_extracted}\n"
        f"---\n{prev_output}\n---\n"
        f"Address the issues above and improve upon the previous output.\n\n"
    )


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

class InterviewRetroFlow(Flow[RetroState]):

    # Retry steps defined before evaluate_quality so or_() can reference them.

    @listen("retry_judge")
    def retry_from_judge(self) -> None:
        enrichment = _build_enrichment(
            self.state.last_result,
            self.state.quality_issues,
            retry_n=self.state.retry_count,
            qa_pairs_extracted=len(self.state.checkpoint_qa_pairs),
        )
        result = InterviewAnalysisCrew().run_from_judge(
            qa_pairs=self.state.checkpoint_qa_pairs,
            advocacy=self.state.checkpoint_advocacy,
            criticism=self.state.checkpoint_criticism,
            enrichment=enrichment,
        )
        self.state.last_result = result

    @listen("retry_extraction")
    def retry_from_extraction(self) -> None:
        enrichment = _build_enrichment(
            self.state.last_result,
            self.state.quality_issues,
            retry_n=self.state.retry_count,
            qa_pairs_extracted=len(self.state.checkpoint_qa_pairs),
        )
        result = InterviewAnalysisCrew().run(
            self.state.transcript,
            enrichment=enrichment,
        )
        checkpoints: dict[str, Any] = result.pop("_checkpoints", {})  # type: ignore[union-attr]
        self.state.last_result = result
        self.state.checkpoint_qa_pairs = checkpoints.get("qa_pairs", [])
        self.state.checkpoint_advocacy = checkpoints.get("advocacy", [])
        self.state.checkpoint_criticism = checkpoints.get("criticism", [])

    @start()
    def run_initial_crew(self) -> None:
        result = InterviewAnalysisCrew().run(
            self.state.transcript,
        )
        checkpoints: dict[str, Any] = result.pop("_checkpoints", {})  # type: ignore[union-attr]
        self.state.last_result = result
        self.state.checkpoint_qa_pairs = checkpoints.get("qa_pairs", [])
        self.state.checkpoint_advocacy = checkpoints.get("advocacy", [])
        self.state.checkpoint_criticism = checkpoints.get("criticism", [])

    @listen(or_(run_initial_crew, retry_from_judge, retry_from_extraction))
    def evaluate_quality(self) -> None:
        qa_count: int = self.state.last_result.get("qa_pairs_extracted", 0)  # type: ignore[assignment]
        issues, _ = _check_quality(self.state.last_result, qa_count)
        self.state.quality_issues = issues
        self.state.last_failure_reason = "; ".join(issues)

    @router(evaluate_quality)
    def quality_gate(self) -> str:
        if self.state.retry_count >= 5 or not self.state.quality_issues:
            return "complete"
        self.state.retry_count += 1
        for issue in self.state.quality_issues:
            if issue == "extraction_empty":
                return "retry_extraction"
        return "retry_judge"

    @listen("complete")
    def finalize(self) -> dict[str, Any]:
        return self.state.last_result


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def kickoff(
    transcript: str = "",
) -> None:
    InterviewRetroFlow().kickoff(inputs=dict(
        transcript=transcript,
    ))


def plot() -> None:
    InterviewRetroFlow().plot()


if __name__ == "__main__":
    kickoff()
