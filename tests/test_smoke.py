import tomllib
from pathlib import Path

import llm_guardrails


def test_package_imports_and_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text("utf-8"))
    assert llm_guardrails.__version__ == pyproject["project"]["version"] == "1.0.0"
