import pytest

from llm_guardrails.settings import load_settings, read_env_file
from llm_guardrails.types import ConfigError

FAKE_TOKEN = "fake-bearer-token-for-tests-only"


def test_defaults_without_env(tmp_path) -> None:
    s = load_settings(tmp_path / "none.env", environ={})
    assert s.aws_region == "ap-south-1"
    assert s.chat_model_id == "global.amazon.nova-2-lite-v1:0"
    assert s.embed_dimensions == 1024
    assert s.llm_mode == "replay"
    assert s.live_budget_usd == 5.0
    assert s.auth_method == "none"


def test_env_file_parsed_and_environment_wins(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nAWS_REGION=us-east-1\nLIVE_BUDGET_USD='2.5'\nEMBED_DIMENSIONS=512\n\n",
        encoding="utf-8",
    )
    assert read_env_file(env)["LIVE_BUDGET_USD"] == "2.5"
    s = load_settings(env, environ={"AWS_REGION": "ap-south-1"})
    assert s.aws_region == "ap-south-1"
    assert s.live_budget_usd == 2.5
    assert s.embed_dimensions == 512


def test_secrets_hidden_from_repr(tmp_path) -> None:
    s = load_settings(
        tmp_path / "x", environ={"AWS_BEARER_TOKEN_BEDROCK": FAKE_TOKEN, "LOG_HASH_SALT": "salty"}
    )
    assert s.auth_method == "bearer"
    assert FAKE_TOKEN not in repr(s) and "salty" not in repr(s)


def test_both_auth_options_is_an_error(tmp_path) -> None:
    with pytest.raises(ConfigError) as err:
        load_settings(
            tmp_path / "x",
            environ={
                "AWS_BEARER_TOKEN_BEDROCK": FAKE_TOKEN,
                "AWS_ACCESS_KEY_ID": "AKIDEXAMPLE",
                "AWS_SECRET_ACCESS_KEY": "secret",
            },
        )
    assert FAKE_TOKEN not in str(err.value)


def test_half_iam_keys_is_an_error(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "x", environ={"AWS_ACCESS_KEY_ID": "AKIDEXAMPLE"})


def test_bad_values(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "x", environ={"GUARDRAILS_LLM_MODE": "yolo"})
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "x", environ={"LIVE_BUDGET_USD": "lots"})
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "x", environ={"LIVE_BUDGET_USD": "-1"})
