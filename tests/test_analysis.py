import json
from unittest.mock import MagicMock, patch

from interview_retro.analysis import InterviewAnalysisCrew


SAMPLE_RESULT = json.dumps({
    "rated_qa": [
        {"score": 8.0, "feedback": "Good answer.", "suggested_answer": None}
    ],
    "overall_score": 8.0,
    "strengths": ["Clear communication"],
    "weaknesses": [],
    "summary": "Strong performance.",
    "analysis_status": "complete",
    "analysis_error": None,
})

SAMPLE_QA = json.dumps({"qa_pairs": [
    {"question": "Tell me about yourself.", "answer": "I have 5 years...",
     "category": "behavioral", "timestamp_seconds": 0}
]})

SAMPLE_ADVOCACY = json.dumps({"advocacy": [
    {"strengths": ["concise"], "advocate_summary": "strong"}
]})

SAMPLE_CRITICISM = json.dumps({"criticism": [
    {"weaknesses": ["vague"], "rebuttal_of_advocate": "not so fast", "critic_summary": "weak"}
]})


def _make_crew_mock(task_raws: list[str], final_output: str) -> MagicMock:
    """Return a patched Crew class whose kickoff sets task.output.raw on each task."""
    mock_crew_cls = MagicMock()
    mock_crew = MagicMock()
    mock_crew_cls.return_value = mock_crew

    def _kickoff_side_effect() -> MagicMock:
        tasks = mock_crew_cls.call_args.kwargs["tasks"]
        for task, raw in zip(tasks, task_raws):
            task.output = MagicMock(raw=raw)
        result = MagicMock()
        result.__str__ = lambda self: final_output
        return result

    mock_crew.kickoff.side_effect = _kickoff_side_effect
    return mock_crew_cls


@patch("interview_retro.analysis.Crew")
def test_run_returns_qa_pairs_extracted(mock_crew_cls: MagicMock) -> None:
    mock_crew = MagicMock()
    final = MagicMock()
    final.__str__ = lambda self: SAMPLE_RESULT
    mock_crew.kickoff.return_value = final
    mock_crew_cls.return_value = mock_crew

    crew = InterviewAnalysisCrew()
    result = crew.run("some transcript text", "Acme", "Engineer", "onsite")

    assert "qa_pairs_extracted" in result
    assert "_checkpoints" in result


@patch("interview_retro.analysis.Crew")
def test_run_from_judge_creates_one_task(mock_crew_cls: MagicMock) -> None:
    mock_crew = MagicMock()
    final = MagicMock()
    final.__str__ = lambda self: SAMPLE_RESULT
    mock_crew.kickoff.return_value = final
    mock_crew_cls.return_value = mock_crew

    crew = InterviewAnalysisCrew()
    crew.run_from_judge(
        qa_pairs=[{"question": "Q?", "answer": "A.", "category": "behavioral", "timestamp_seconds": 0}],
        advocacy=[{"strengths": ["good"], "advocate_summary": "strong"}],
        criticism=[{"weaknesses": ["bad"], "rebuttal_of_advocate": "nope", "critic_summary": "weak"}],
        company_name="Acme",
        role="Engineer",
        stage="onsite",
    )

    tasks = mock_crew_cls.call_args.kwargs["tasks"]
    assert len(tasks) == 1, "run_from_judge must build exactly one task (judge only)"


@patch("interview_retro.analysis.Crew")
def test_run_from_judge_embeds_qa_in_description(mock_crew_cls: MagicMock) -> None:
    mock_crew = MagicMock()
    final = MagicMock()
    final.__str__ = lambda self: SAMPLE_RESULT
    mock_crew.kickoff.return_value = final
    mock_crew_cls.return_value = mock_crew

    qa = [{"question": "Why here?", "answer": "Growth.", "category": "motivation", "timestamp_seconds": 5}]
    crew = InterviewAnalysisCrew()
    crew.run_from_judge(
        qa_pairs=qa,
        advocacy=[],
        criticism=[],
        company_name="Acme",
        role="Engineer",
        stage="onsite",
    )

    task = mock_crew_cls.call_args.kwargs["tasks"][0]
    assert "Why here?" in task.description
    assert "Growth." in task.description


@patch("interview_retro.analysis.Crew")
def test_run_from_judge_prepends_enrichment(mock_crew_cls: MagicMock) -> None:
    mock_crew = MagicMock()
    final = MagicMock()
    final.__str__ = lambda self: SAMPLE_RESULT
    mock_crew.kickoff.return_value = final
    mock_crew_cls.return_value = mock_crew

    crew = InterviewAnalysisCrew()
    crew.run_from_judge(
        qa_pairs=[],
        advocacy=[],
        criticism=[],
        company_name="Acme",
        role="Engineer",
        stage="onsite",
        enrichment="[QUALITY RETRY 1/5]\nIssues: low_score\n",
    )

    task = mock_crew_cls.call_args.kwargs["tasks"][0]
    assert task.description.startswith("[QUALITY RETRY 1/5]")


@patch("interview_retro.analysis.Crew")
def test_run_from_debate_creates_three_tasks(mock_crew_cls: MagicMock) -> None:
    mock_crew = MagicMock()
    final = MagicMock()
    final.__str__ = lambda self: SAMPLE_RESULT
    mock_crew_cls.return_value = mock_crew

    def _side() -> MagicMock:
        tasks = mock_crew_cls.call_args.kwargs["tasks"]
        for t in tasks:
            t.output = MagicMock(raw='{"advocacy": [], "criticism": []}')
        return final

    mock_crew.kickoff.side_effect = _side

    crew = InterviewAnalysisCrew()
    crew.run_from_debate(
        qa_pairs=[{"question": "Q?", "answer": "A.", "category": "behavioral", "timestamp_seconds": 0}],
        company_name="Acme",
        role="Engineer",
        stage="onsite",
    )

    tasks = mock_crew_cls.call_args.kwargs["tasks"]
    assert len(tasks) == 3, "run_from_debate must build advocate + critic + judge tasks"


@patch("interview_retro.analysis.Crew")
def test_run_from_debate_embeds_qa_in_advocate_description(mock_crew_cls: MagicMock) -> None:
    mock_crew = MagicMock()
    final = MagicMock()
    final.__str__ = lambda self: SAMPLE_RESULT
    mock_crew_cls.return_value = mock_crew

    def _side() -> MagicMock:
        tasks = mock_crew_cls.call_args.kwargs["tasks"]
        for t in tasks:
            t.output = MagicMock(raw="{}")
        return final

    mock_crew.kickoff.side_effect = _side

    qa = [{"question": "Describe a challenge.", "answer": "I led a migration.",
           "category": "behavioral", "timestamp_seconds": 10}]
    crew = InterviewAnalysisCrew()
    crew.run_from_debate(
        qa_pairs=qa,
        company_name="Acme",
        role="Engineer",
        stage="onsite",
    )

    advocate_task = mock_crew_cls.call_args.kwargs["tasks"][0]
    assert "Describe a challenge." in advocate_task.description
