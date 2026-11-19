import llm_guardrails


def test_package_imports_and_has_version() -> None:
    assert llm_guardrails.__version__ == "0.1.0"
