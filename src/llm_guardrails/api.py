"""HTTP API (docs/SPEC.md 15.1). Binds 127.0.0.1 by default: put it behind your own gateway
for auth and rate limits, which are deliberately out of scope.

Status codes: a block is a normal outcome (200 with "decision": "block"). A malformed request
is 422, and the error lists only WHICH fields are wrong, never their values (FastAPI's
default 422 body echoes the input back, which may contain personal data). Anything
unexpected is 500 with a request id and no details, since error text can contain user data.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_guardrails import __version__
from llm_guardrails.policy_config import bundled_policy_names
from llm_guardrails.runtime import Pipelines
from llm_guardrails.schemas import SCHEMAS
from llm_guardrails.settings import Settings, load_settings
from llm_guardrails.types import ConfigError, SourceDoc

logger = logging.getLogger("llm_guardrails.api")

# Schemas a client may ask the answering model for. The judge schemas are internal.
CLIENT_SCHEMAS = frozenset({"TicketSummary"})


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    policy: str = "default"

    @field_validator("policy")
    @classmethod
    def _known_policy(cls, value: str) -> str:
        if value not in bundled_policy_names():
            raise ValueError("unknown policy")
        return value


def _check_schema(value: str | None) -> str | None:
    if value is not None and (value not in SCHEMAS or value not in CLIENT_SCHEMAS):
        raise ValueError("unknown schema")
    return value


class ChatBody(_Body):
    question: str
    context: list[SourceDoc] = []
    schema_: str | None = Field(None, alias="schema")

    @field_validator("schema_")
    @classmethod
    def _schema_ok(cls, value: str | None) -> str | None:
        return _check_schema(value)


class CheckInputBody(_Body):
    text: str
    context: list[SourceDoc] = []


class CheckOutputBody(_Body):
    output: str
    sources: list[SourceDoc] = []
    schema_: str | None = Field(None, alias="schema")
    system_prompt: str | None = None
    canary: str | None = Field(None, pattern=r"^cnry-[0-9a-f]{24}$")

    @field_validator("schema_")
    @classmethod
    def _schema_ok(cls, value: str | None) -> str | None:
        return _check_schema(value)


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json") if model is not None else None


def create_app(settings: Settings | None = None, pipelines: Pipelines | None = None) -> FastAPI:
    settings = settings or load_settings()
    pipelines = pipelines or Pipelines(settings)
    app = FastAPI(title="LLM Guardrails", version=__version__)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()})
        return JSONResponse(status_code=422, content={"error": "invalid_request",
                                                      "fields": fields})  # fmt: skip

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception) -> JSONResponse:
        request_id = uuid.uuid4().hex
        logger.error("request %s failed with %s", request_id, type(exc).__name__)
        body = {"request_id": request_id, "error": "internal_error"}
        return JSONResponse(status_code=500, content=body)

    @app.post("/v1/guarded-chat")
    async def guarded_chat(body: ChatBody) -> dict[str, Any]:
        result = await pipelines.get(body.policy, body.schema_).run(body.question, body.context)
        return {
            "request_id": result.request_id,
            "decision": result.decision,
            "answer": result.answer,
            "data": _dump(result.data),
            "user_message": result.user_message,
            "report": _dump(result.report),
        }

    @app.post("/v1/check-input")
    async def check_input(body: CheckInputBody) -> dict[str, Any]:
        result = await pipelines.get(body.policy).check_input(body.text, body.context)
        return {
            "request_id": result.request_id,
            "decision": result.decision,
            "sanitized_text": result.sanitized_text,
            "wrapped_context": result.wrapped_context,
            "report": _dump(result.report),
        }

    @app.post("/v1/check-output")
    async def check_output(body: CheckOutputBody) -> dict[str, Any]:
        result = await pipelines.get(body.policy).check_output(
            body.output, sources=body.sources, schema_name=body.schema_,
            canary=body.canary, system_prompt=body.system_prompt,
        )  # fmt: skip
        return {
            "request_id": result.request_id,
            "decision": result.decision,
            "output": result.output,
            "data": _dump(result.data),
            "report": _dump(result.report),
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "policy_names": bundled_policy_names(),
            "llm_mode": settings.llm_mode,
        }

    return app


def app_from_env() -> FastAPI:
    """Factory for `uvicorn --factory llm_guardrails.api:app_from_env`."""
    try:
        return create_app()
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from None
