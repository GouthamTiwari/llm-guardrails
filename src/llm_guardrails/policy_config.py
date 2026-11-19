"""Policy file schema and loader (docs/SPEC.md section 9.1).

A policy is a YAML file. Every section is validated with `extra="forbid"`, so a typo such
as `max_input_char` fails loudly at startup instead of silently using a default.
"""

from __future__ import annotations

import logging
import re
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from llm_guardrails.types import BUILTIN_ENTITY_TYPES, GUARD_NAMES, PolicyError

logger = logging.getLogger("llm_guardrails.policy")

PiiAction = Literal["allow", "redact", "block"]

# Seconds. Generous on purpose: timeouts exist to bound a request, not to tune it (spec 14).
DEFAULT_TIMEOUTS_S: dict[str, float] = {
    "normalize": 0.5,
    "policy": 0.5,
    "pii": 1.0,
    "injection": 10.0,
    "injection_rules": 0.2,
    "injection_embed": 3.0,
    "injection_llm": 8.0,
    "context": 15.0,
    "model": 30.0,
    "schema": 45.0,
    "grounding": 20.0,
    "grounding_entailment": 15.0,
    "leak": 1.0,
    "request": 90.0,
}

# Every guard is a security guard, so all default to fail-closed (spec 4.3). The grounding
# entailment layer's behaviour is set by `grounding.entailment` instead.
SECURITY_GUARDS: frozenset[str] = frozenset(GUARD_NAMES)


def known_entity_types() -> frozenset[str]:
    """Entity types a policy may name: whatever the detector registry holds."""
    from llm_guardrails.detectors.entities import registered_types  # avoid import at load

    return registered_types() | BUILTIN_ENTITY_TYPES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Limits(_Strict):
    max_input_chars: int = Field(8000, gt=0)
    max_context_docs: int = Field(20, ge=0)
    max_doc_chars: int = Field(20000, gt=0)
    max_output_tokens: int = Field(1024, gt=0)


class Topic(_Strict):
    name: str
    keywords: list[str] = Field(min_length=1)


class Topics(_Strict):
    mode: Literal["allow_all", "allow_list"] = "allow_all"
    allowed: list[Topic] = []
    min_keyword_hits: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _allow_list_needs_topics(self) -> Topics:
        if self.mode == "allow_list" and not self.allowed:
            raise ValueError("topics.mode 'allow_list' needs at least one allowed topic")
        return self


class Tools(_Strict):
    risky: list[str] = []


class PiiEntityPolicy(_Strict):
    input: PiiAction
    output: PiiAction


class PiiPolicy(_Strict):
    min_confidence: float = Field(0.6, ge=0.0, le=1.0)
    entities: dict[str, PiiEntityPolicy]
    restore: list[str] = []

    @model_validator(mode="after")
    def _known_types(self) -> PiiPolicy:
        known = known_entity_types()
        unknown = sorted(set(self.entities) - known)
        if unknown:
            raise ValueError(f"unknown PII entity type(s): {unknown}")
        for etype in self.restore:
            if etype not in self.entities:
                raise ValueError(f"restore names {etype}, which is not in pii.entities")
            if self.entities[etype].input != "redact":
                # Only a redacted value has a placeholder to restore.
                raise ValueError(f"restore names {etype}, but its input action is not redact")
        return self


class InjectionPolicy(_Strict):
    use_embeddings: bool = True
    use_llm: bool = True
    # Optional overrides; None = use the values from reports/operating_point.json.
    t_block: float | None = Field(None, ge=0.0, le=1.0)
    t_allow: float | None = Field(None, ge=0.0, le=1.0)
    t_llm: float | None = Field(None, ge=0.0, le=1.0)
    benign_lambda: float | None = Field(None, ge=0.0)


class ContextPolicy(_Strict):
    on_injection: Literal["strip_line", "drop_doc", "block_request"] = "strip_line"
    llm_classifier: bool = False


class SchemaPolicy(_Strict):
    max_retries: int = Field(2, ge=0, le=5)


class GroundingPolicy(_Strict):
    enabled: bool = True
    mode: Literal["reject", "strip"] = "reject"
    entailment: Literal["off", "optional", "required"] = "off"
    min_overlap: float = Field(0.5, ge=0.0, le=1.0)


class LeakPolicy(_Strict):
    allowed_domains: list[str] = []
    max_query_chars: int = Field(30, ge=0)

    @field_validator("allowed_domains")
    @classmethod
    def _lowercase(cls, value: list[str]) -> list[str]:
        return [d.strip().lower().rstrip(".") for d in value if d.strip()]


class Policy(_Strict):
    name: str
    version: int = 1
    limits: Limits = Limits()
    topics: Topics = Topics()
    denied_actions: list[str] = []
    tools: Tools = Tools()
    pii: PiiPolicy
    injection: InjectionPolicy = InjectionPolicy()
    context: ContextPolicy = ContextPolicy()
    schema_: SchemaPolicy = Field(SchemaPolicy(), alias="schema")
    grounding: GroundingPolicy = GroundingPolicy()
    leak: LeakPolicy = LeakPolicy()
    timeouts_s: dict[str, float] = {}
    fail_mode: dict[str, Literal["closed", "open"]] = {}

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("denied_actions")
    @classmethod
    def _regexes_compile(cls, value: list[str]) -> list[str]:
        for i, pattern in enumerate(value):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"denied_actions[{i}] is not a valid regex: {exc}") from exc
        return value

    @field_validator("timeouts_s")
    @classmethod
    def _timeouts_known(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - set(DEFAULT_TIMEOUTS_S))
        if unknown:
            raise ValueError(f"unknown timeout key(s): {unknown}")
        if any(v <= 0 for v in value.values()):
            raise ValueError("timeouts must be positive")
        return value

    @field_validator("fail_mode")
    @classmethod
    def _fail_mode_known(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(value) - set(GUARD_NAMES))
        if unknown:
            raise ValueError(f"unknown guard name(s) in fail_mode: {unknown}")
        return value

    def timeout_for(self, key: str) -> float:
        return self.timeouts_s.get(key, DEFAULT_TIMEOUTS_S[key])

    def fail_mode_for(self, guard: str) -> Literal["closed", "open"]:
        default = "closed" if guard in SECURITY_GUARDS else "open"
        return self.fail_mode.get(guard, default)

    def compiled_denied_actions(self) -> list[re.Pattern[str]]:
        return [re.compile(p, re.IGNORECASE) for p in self.denied_actions]


def parse_policy(data: object) -> Policy:
    """Validate a loaded YAML mapping. Warns when a security guard is set to fail open."""
    try:
        policy = Policy.model_validate(data)
    except ValidationError as exc:
        raise PolicyError(f"invalid policy: {exc}") from exc
    opened = sorted(g for g, m in policy.fail_mode.items() if m == "open" and g in SECURITY_GUARDS)
    if opened:
        logger.warning(
            "policy %s sets security guard(s) %s to fail OPEN: errors will let requests through",
            policy.name,
            opened,
        )
    return policy


def load_policy(name_or_path: str | Path) -> Policy:
    """Load a bundled policy by name ('default', 'banking_tax') or a YAML file by path."""
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} or path.is_file():
        if not path.is_file():
            raise PolicyError(f"policy file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        name = str(name_or_path)
        if not re.fullmatch(r"[a-z0-9_]{1,40}", name):
            raise PolicyError(f"invalid policy name: {name!r}")
        resource = resources.files("llm_guardrails.policies").joinpath(f"{name}.yaml")
        if not resource.is_file():
            raise PolicyError(f"no bundled policy named {name!r}")
        text = resource.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(f"policy is not valid YAML: {exc}") from exc
    return parse_policy(data)


def bundled_policy_names() -> list[str]:
    folder = resources.files("llm_guardrails.policies")
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".yaml"))
