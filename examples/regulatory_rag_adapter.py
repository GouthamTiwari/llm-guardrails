"""How a RAG app (like the Regulatory RAG project) plugs in, without depending on it.

Run:  uv run python examples/regulatory_rag_adapter.py      (fake model: no AWS)

1. Its retriever returns chunks shaped {chunk_id, text}. Chunk ids such as
   "rbi/kyc-2024#3" are not valid source ids here (letters, digits, _ and - only), so each
   chunk becomes S1, S2, ... and a map keeps the way back to the original id.
2. Its answer function is wrapped with @guarded: PII is redacted before it runs, the
   documents are scanned for planted instructions, and the answer is grounding-checked.
3. The same grounding guard can also be called directly on answers it already has.
"""

import os

os.environ.setdefault("GUARDRAILS_LLM_MODE", "fake")  # safe default: never calls AWS

from llm_guardrails.adapters import guarded  # noqa: E402
from llm_guardrails.guards.grounding import check_grounding  # noqa: E402
from llm_guardrails.types import SourceDoc  # noqa: E402

# A stand-in for the Regulatory RAG retriever's output (fictional circulars).
CHUNKS = [
    {"chunk_id": "rbi/kyc-2024#3",
     "text": "Example Bank must complete re-KYC every 10 years for low-risk customers."},
    {"chunk_id": "rbi/kyc-2024#4",
     "text": "High-risk customers need re-KYC every 2 years at Example Bank."},
]  # fmt: skip


def to_source_docs(chunks: list[dict]) -> tuple[list[SourceDoc], dict[str, str]]:
    docs, back = [], {}
    for i, chunk in enumerate(chunks, 1):
        sid = f"S{i}"
        docs.append(SourceDoc(id=sid, text=chunk["text"]))
        back[sid] = chunk["chunk_id"]
    return docs, back


def rag_answer(question: str, sources: list[SourceDoc]) -> str:
    """Stand-in for the project's generator: answers from the first source, with citation."""
    return f"{sources[0].text.rstrip('.')} [{sources[0].id}]."


@guarded(policy="banking_tax")
def guarded_rag(question: str, sources: list[SourceDoc]) -> str:
    return rag_answer(question, sources)


@guarded(policy="banking_tax")
def careless_rag(question: str, sources: list[SourceDoc]) -> str:
    return "Re-KYC is needed every 5 years for everyone."  # uncited and wrong


def main() -> None:
    docs, back = to_source_docs(CHUNKS)

    good = guarded_rag("How often is re-KYC needed for a low-risk account?", docs)
    print(f"guarded answer: {good.decision} | {good.answer}")
    print("  cites:", [back[s] for s in back if f"[{s}]" in (good.answer or "")])

    bad = careless_rag("How often is re-KYC needed for my account?", docs)
    reasons = [r for v in bad.report.verdicts for r in v.reasons if r.startswith("grounding")]
    print(f"careless answer: {bad.decision} | {reasons}")

    pii = guarded_rag("My PAN is ABCPE1234F. How often is re-KYC needed for my account?", docs)
    print(f"question with a PAN: {pii.decision} | the RAG saw: "
          f"{pii.input_result.sanitized_text!r}")  # fmt: skip

    # The grounding guard on its own, for answers the RAG already produced:
    direct = check_grounding("High-risk customers need re-KYC every 3 years [S2].", docs)
    print(f"direct grounding check: {direct.action.value} | {direct.reasons}")


if __name__ == "__main__":
    main()
