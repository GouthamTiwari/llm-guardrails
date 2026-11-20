"""The numbers rule, enforced by CI (docs/SPEC.md 13.7; CLAUDE.md "Numbers rule").

Every number in the README results block must come from the committed report files:
summary.md is regenerated here from reports/*.json and compared with the committed copy,
and the README block must equal it exactly. A hand-edited number anywhere fails the build.
"""

from llm_guardrails.eval import report
from llm_guardrails.eval.reports import REPORTS_DIR


def test_summary_is_generated_from_the_report_files() -> None:
    committed = (REPORTS_DIR / "summary.md").read_text(encoding="utf-8")
    assert report.summary_markdown(REPORTS_DIR) == committed


def test_readme_block_equals_the_generated_summary() -> None:
    summary = report.summary_markdown(REPORTS_DIR)
    assert report.current_block() == report.readme_block(summary)


def test_every_report_the_summary_links_to_exists() -> None:
    summary = report.summary_markdown(REPORTS_DIR)
    for name in ("injection", "pii", "schema", "grounding", "e2e"):
        assert f"reports/{name}.md" in summary
        assert (REPORTS_DIR / f"{name}.md").is_file()
