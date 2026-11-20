"""Optional baseline: Amazon Bedrock Guardrails on our test sets (docs/SPEC.md 13.4 item 6).

`guardrails baseline bedrock --live`:
1. creates ONE guardrail: the prompt-attack content filter at HIGH on input, plus the
   managed PII types that overlap ours (EMAIL, PHONE, CREDIT_DEBIT_CARD_NUMBER);
2. waits until it is READY, then calls ApplyGuardrail(source="INPUT") on every injection
   and PII test-split text, recording each response;
3. deletes the guardrail in a `finally` block - it runs even if step 2 crashes - and checks
   with ListGuardrails that nothing with our name is left.

Without --live the report is rebuilt from the recorded responses (free, offline). If the
API key may not create guardrails, or the region has no Guardrails API, the report says
so and the task still closes.

The managed service has no Indian ID types, so Aadhaar, PAN, GSTIN, UPI ID and IFSC are
listed as "no managed equivalent" instead of being scored as misses.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from llm_guardrails.eval.datasets import DATA_DIR, FILES, load_suite
from llm_guardrails.eval.metrics import Outcome, pct, precision_recall_f1, rate, recall_and_fpr
from llm_guardrails.eval.reports import meta, write_json, write_text
from llm_guardrails.eval.synth import ascii_digits
from llm_guardrails.pricing import GUARDRAILS_PRICE_PER_1K_UNITS

RECORDINGS = DATA_DIR / "recordings" / "baseline_bedrock.jsonl"
NAME_PREFIX = "llm-guardrails-baseline-"
TAG = {"key": "project", "value": "llm-guardrails-baseline"}
# Ours -> the managed type with the same meaning. Everything else of ours has no equivalent.
MANAGED_EQUIVALENT = {"EMAIL": "EMAIL", "PHONE_IN": "PHONE",
                      "CREDIT_CARD": "CREDIT_DEBIT_CARD_NUMBER"}  # fmt: skip
NO_EQUIVALENT = ("AADHAAR", "PAN", "GSTIN", "UPI_ID", "IFSC")
READY_TIMEOUT_S = 120
DELETE_TIMEOUT_S = 120


class BaselineUnavailable(Exception):
    """The account or region cannot run the baseline (permission, region, quota)."""


# ----------------------------------------------------------------------------- AWS calls


def guardrail_config(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Temporary baseline for the llm-guardrails evaluation; deleted at the end "
        "of the same run.",
        "contentPolicyConfig": {"filtersConfig": [
            {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
        ]},
        # The input-side fields must be explicit: with only the legacy `action`, the first
        # live run's PII filter was never evaluated on input (0 sensitive-information
        # units billed), so it detected nothing (SPEC decision 15).
        "sensitiveInformationPolicyConfig": {"piiEntitiesConfig": [
            {"type": t, "action": "ANONYMIZE", "inputAction": "ANONYMIZE",
             "inputEnabled": True, "outputEnabled": False}
            for t in MANAGED_EQUIVALENT.values()
        ]},
        "blockedInputMessaging": "Blocked by the baseline guardrail.",
        "blockedOutputsMessaging": "Blocked by the baseline guardrail.",
        "tags": [TAG],
    }  # fmt: skip


def _wait(control: Any, gid: str, *, want: str, timeout_s: float, sleep=time.sleep) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        status = control.get_guardrail(guardrailIdentifier=gid)["status"]
        if status == want:
            return
        if status == "FAILED" or time.monotonic() > deadline:
            raise BaselineUnavailable(f"guardrail status {status} (wanted {want})")
        sleep(2)


def leftover_guardrails(control: Any) -> list[str]:
    """Ids of guardrails whose name starts with our prefix (should be none after a run)."""
    found, token = [], None
    while True:
        kwargs = {"nextToken": token} if token else {}
        page = control.list_guardrails(**kwargs)
        found += [g["id"] for g in page.get("guardrails", [])
                  if g.get("name", "").startswith(NAME_PREFIX)]  # fmt: skip
        token = page.get("nextToken")
        if not token:
            return found


def run_live(control: Any, runtime: Any, texts: dict[str, str], *, sleep=time.sleep) -> dict:
    """Create -> apply to every text -> delete (always) -> verify. Returns raw results."""
    from botocore.exceptions import ClientError

    name = NAME_PREFIX + secrets.token_hex(4)
    try:
        created = control.create_guardrail(**guardrail_config(name))
    except ClientError as exc:
        raise BaselineUnavailable(exc.response.get("Error", {}).get("Code", "ClientError")) from exc
    gid = created["guardrailId"]
    responses: dict[str, dict] = {}
    deleted = False
    try:
        _wait(control, gid, want="READY", timeout_s=READY_TIMEOUT_S, sleep=sleep)
        for key, text in texts.items():
            started = time.perf_counter()
            resp = runtime.apply_guardrail(
                guardrailIdentifier=gid, guardrailVersion="DRAFT", source="INPUT",
                content=[{"text": {"text": text}}],
            )  # fmt: skip
            resp = scrub({k: v for k, v in resp.items() if k != "ResponseMetadata"})
            resp["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            responses[key] = resp
    finally:
        # Runs whether the loop finished, raised, or was interrupted: never leave the
        # guardrail behind in the account.
        control.delete_guardrail(guardrailIdentifier=gid)
        deadline = time.monotonic() + DELETE_TIMEOUT_S
        while time.monotonic() < deadline:
            if gid not in leftover_guardrails(control):
                deleted = True
                break
            sleep(2)
    return {"guardrail_name": name, "responses": responses, "deleted_and_verified": deleted}


# ----------------------------------------------------------------------------- recording


def scrub(resp: dict) -> dict:
    """Drop what identifies the AWS account before anything is saved: each assessment's
    `appliedGuardrailDetails` carries the guardrail ARN, which contains the account id.
    Scoring never reads it."""
    out = dict(resp)
    out["assessments"] = [{k: v for k, v in a.items() if k != "appliedGuardrailDetails"}
                          for a in resp.get("assessments", [])]  # fmt: skip
    return out


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_recordings(result: dict, path: Path | None = None) -> dict:
    """Add one live run to the recordings. A later run's response for the same text
    replaces the earlier one; every run's header (name, texts, cost, deletion check) is
    kept, so the report can show what each run did and what it cost."""
    path = path or RECORDINGS  # looked up at call time (tests point it elsewhere)
    previous = load_recordings(path) or {"runs": [], "responses": {}}
    run_header = {
        "key": "_run", "index": len(previous["runs"]) + 1,
        "guardrail_name": result["guardrail_name"], "texts": len(result["responses"]),
        "deleted_and_verified": result["deleted_and_verified"],
        "pii_input_enabled": result.get("pii_input_enabled", True),
        "usd": round(sum(units_cost(x) for x in result["responses"].values()), 6),
    }  # fmt: skip
    runs = previous["runs"] + [run_header]
    responses = previous["responses"] | result["responses"]
    rows = [{"key": k, **{f: v for f, v in r.items() if f != "key"}}
            for k, r in sorted(responses.items())]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"key": "_run", **{k: v for k, v in h.items() if k != "key"}})
             for h in runs]  # fmt: skip
    lines += [json.dumps(r, ensure_ascii=False, default=str) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return load_recordings(path)  # type: ignore[return-value]


def load_recordings(path: Path | None = None) -> dict | None:
    path = path or RECORDINGS
    if not path.is_file():
        return None
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    runs = []
    for i, r in enumerate([r for r in rows if r["key"] == "_run"], 1):
        # The first live run predates these fields: its PII filter was not input-enabled.
        runs.append({"index": r.get("index", i), "guardrail_name": r["guardrail_name"],
                     "texts": r.get("texts"), "deleted_and_verified": r["deleted_and_verified"],
                     "pii_input_enabled": r.get("pii_input_enabled", False),
                     "usd": r.get("usd")})  # fmt: skip
    responses = {r["key"]: r for r in rows if r["key"] != "_run"}
    if runs and runs[0]["usd"] is None:  # the first run's header predates these fields
        runs[0]["usd"] = round(sum(units_cost(x) for x in responses.values()), 6)
        runs[0]["texts"] = len(responses)
    return {"runs": runs, "responses": responses,
            "deleted_and_verified": all(r["deleted_and_verified"] for r in runs)}  # fmt: skip


# ----------------------------------------------------------------------------- scoring


def prompt_attack_detected(resp: dict) -> bool:
    for a in resp.get("assessments", []):
        for f in a.get("contentPolicy", {}).get("filters", []):
            if f.get("type") == "PROMPT_ATTACK" and f.get("detected", f.get("action") == "BLOCKED"):
                return True
    return False


def pii_found(resp: dict) -> Counter:
    found: Counter = Counter()
    for a in resp.get("assessments", []):
        for e in a.get("sensitiveInformationPolicy", {}).get("piiEntities", []):
            found[(e.get("type"), _norm(e.get("type"), e.get("match", "")))] += 1
    return found


def _norm(managed_type: str | None, value: str) -> str:
    v = ascii_digits(value).strip().lower()
    if managed_type in ("PHONE", "CREDIT_DEBIT_CARD_NUMBER"):
        digits = "".join(c for c in v if c.isdigit())
        return digits[-10:] if managed_type == "PHONE" else digits
    return v


def units_cost(resp: dict) -> float:
    u = resp.get("usage", {})
    content = u.get("contentPolicyUnits", 0)
    pii = max(0, u.get("sensitiveInformationPolicyUnits", 0)
              - u.get("sensitiveInformationPolicyFreeUnits", 0))  # fmt: skip
    return (content * GUARDRAILS_PRICE_PER_1K_UNITS["content_filter"]
            + pii * GUARDRAILS_PRICE_PER_1K_UNITS["sensitive_information"]) / 1000  # fmt: skip


def score(responses: dict[str, dict]) -> dict[str, Any]:
    inj = [r for r in load_suite("injection") if r["split"] == "test"]
    pii = [r for r in load_suite("pii") if r["split"] == "test"]
    outs = [Outcome(r["label"], prompt_attack_detected(responses[text_key(r["text"])]),
                    r["category"], r["language"]) for r in inj]  # fmt: skip
    hard = [o for o in outs if o.category == "benign_hard"]
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    flagged_without_pii = 0
    for r in pii:
        resp = responses[text_key(r["text"])]
        gold = Counter((MANAGED_EQUIVALENT[e["type"]], _norm(MANAGED_EQUIVALENT[e["type"]],
                                                             e["value"]))
                       for e in r["entities"] if e["type"] in MANAGED_EQUIVALENT)  # fmt: skip
        pred = pii_found(resp)
        for (t, _), n in (gold & pred).items():
            tp[t] += n
        for (t, _), n in (pred - gold).items():
            fp[t] += n
        for (t, _), n in (gold - pred).items():
            fn[t] += n
        if not r["entities"] and pred:
            flagged_without_pii += 1
    every = list(responses.values())
    latencies = [x["latency_ms"] for x in every if "latency_ms" in x]
    return {
        "injection": {**recall_and_fpr(outs), "hard_benign_fpr": rate(
            sum(o.blocked for o in hard), len(hard))},
        "pii_overlapping_types": {t: precision_recall_f1(tp[t], fp[t], fn[t])
                                  for t in MANAGED_EQUIVALENT.values()},
        "pii_rows_without_pii_flagged": rate(flagged_without_pii,
                                             sum(not r["entities"] for r in pii)),
        "no_managed_equivalent": list(NO_EQUIVALENT),
        "calls": len(every),
        "usd": round(sum(units_cost(x) for x in every), 6),
        "latency_ms": {"p50": round(statistics.median(latencies), 1) if latencies else None,
                       "p95": round(statistics.quantiles(latencies, n=100,
                                                         method="inclusive")[94], 1)
                       if len(latencies) > 1 else None},
    }  # fmt: skip


def eval_texts(suites: tuple[str, ...] = ("injection", "pii")) -> dict[str, str]:
    texts = [r["text"] for s in suites for r in load_suite(s) if r["split"] == "test"]
    return {text_key(t): t for t in texts}


# ----------------------------------------------------------------------------- run


def _ours(reports_dir: Path) -> dict[str, Any]:
    inj = json.loads((reports_dir / "injection.json").read_text(encoding="utf-8"))
    pii = json.loads((reports_dir / "pii.json").read_text(encoding="utf-8"))
    per_type = pii["test"]["shipped"]["per_type"]
    return {
        "injection": inj["test"]["layers"]["all"] | {"hard_benign_fpr": inj["test"][
            "hard_benign_fpr"]},
        "pii": {ours: per_type.get(ours) for ours in MANAGED_EQUIVALENT},
        "pii_rows_without_pii_flagged": pii["test"]["shipped"][
            "rows_without_pii_with_a_detection"],
    }  # fmt: skip


def run(*, live: bool, settings: Any, reports_dir: Path, clients: tuple | None = None,
        suites: tuple[str, ...] = ("injection", "pii")) -> dict:  # fmt: skip
    status: dict[str, Any] = {"available": True, "reason": None}
    if live:
        from llm_guardrails.spend import SpendLedger

        texts = eval_texts(suites)
        ledger = SpendLedger(limit_usd=settings.live_budget_usd)
        estimate = len(texts) * sum(GUARDRAILS_PRICE_PER_1K_UNITS.values()) / 1000
        ledger.check(estimate)
        control, runtime = clients or make_clients(settings)
        try:
            fresh = run_live(control, runtime, texts)
        except BaselineUnavailable as exc:
            status = {"available": False, "reason": str(exc)}
        else:
            save_recordings(fresh)
            usd = round(sum(units_cost(x) for x in fresh["responses"].values()), 6)
            ledger.record(purpose="guardrails_baseline", model_id="bedrock-guardrails",
                          tokens_in=0, tokens_out=0, usd=usd)  # fmt: skip
    result = load_recordings()
    if result is None and status["available"]:
        status = {"available": False, "reason": "not run (no recordings)"}
    report: dict[str, Any] = {"status": status}
    if result is not None and status["available"]:
        report |= {
            "guardrail": {"runs": result["runs"],
                          "deleted_and_verified": result["deleted_and_verified"],
                          "prompt_attack_strength": "HIGH",
                          "managed_pii_types": list(MANAGED_EQUIVALENT.values())},
            "bedrock": score(result["responses"]) | {
                "usd": round(sum(r["usd"] for r in result["runs"]), 6)},
            "ours": _ours(reports_dir),
        }  # fmt: skip
        b, o = report["bedrock"]["injection"], report["ours"]["injection"]
        report["summary"] = (f"prompt attacks blocked {pct(b['attack_recall'])} vs ours "
                             f"{pct(o['attack_recall'])}; normal messages blocked "
                             f"{pct(b['benign_fpr'])} vs ours {pct(o['benign_fpr'])}; "
                             "PII look-alike rows flagged "
                             f"{pct(report['bedrock']['pii_rows_without_pii_flagged'])} vs ours "
                             f"{pct(report['ours']['pii_rows_without_pii_flagged'])}")  # fmt: skip
    spend = report.get("bedrock", {}).get("usd", 0.0)
    full = meta(paths=[DATA_DIR / FILES["injection"], DATA_DIR / FILES["pii"]],
                mode="live" if live else "replay", spend_usd=spend,
                operating_point=None) | report  # fmt: skip
    write_json("baseline_bedrock_guardrails.json", full, reports_dir)
    write_text("baseline_bedrock_guardrails.md", baseline_markdown(full), reports_dir)
    return full


def make_clients(settings: Any) -> tuple[Any, Any]:
    """`bedrock` (control plane) and `bedrock-runtime` clients, same auth as the model calls."""
    import boto3

    from llm_guardrails.llm import make_runtime_client

    runtime = make_runtime_client(settings)  # also exports the API key for boto3
    if settings.auth_method == "iam":
        control = boto3.client("bedrock", region_name=settings.aws_region,
                               aws_access_key_id=settings.access_key_id,
                               aws_secret_access_key=settings.secret_access_key)  # fmt: skip
    else:
        control = boto3.client("bedrock", region_name=settings.aws_region)
    return control, runtime


# ----------------------------------------------------------------------------- markdown


def baseline_markdown(r: dict[str, Any]) -> str:
    lines = ["# Baseline: Amazon Bedrock Guardrails vs this project", ""]
    if not r["status"]["available"]:
        return "\n".join(lines + [f"Not available: {r['status']['reason']}. The comparison "
                                  "row is omitted.", ""])  # fmt: skip
    b, o, g = r["bedrock"], r["ours"], r["guardrail"]
    lines += [
        f"Mode: {r['mode']}. Temporary guardrails (prompt-attack filter "
        f"{g['prompt_attack_strength']} on input; managed PII types "
        f"{', '.join(g['managed_pii_types'])}) applied to the {b['calls']} injection and PII "
        "test-split texts with ApplyGuardrail; each was deleted at the end of its run "
        f"(deletion verified with ListGuardrails: {'yes' if g['deleted_and_verified'] else 'NO'}).",
        "",
        "| Run | Texts | PII filter evaluated on input | USD | Deleted and verified |",
        "|---|---|---|---|---|",
        *[f"| {x['index']} | {x['texts']} | {'yes' if x['pii_input_enabled'] else 'no'} | "
          f"{x['usd']:.4f} | {'yes' if x['deleted_and_verified'] else 'NO'} |"
          for x in g["runs"]],
        "",
        "Run 1 set only the legacy PII `action` field, so the PII filter was never evaluated on "
        "input (0 sensitive-information units billed). Run 2 repeated the PII rows with the "
        "input fields set; its responses replace run 1's for those rows. Injection results "
        "come from run 1, whose prompt-attack filter was configured correctly.",
        "",
        "## Prompt injection (test split)",
        "",
        "| System | Attacks blocked | Normal messages blocked | Tricky-but-harmless blocked |",
        "|---|---|---|---|",
        f"| Bedrock Guardrails, prompt-attack HIGH | {pct(b['injection']['attack_recall'])} | "
        f"{pct(b['injection']['benign_fpr'])} | {pct(b['injection']['hard_benign_fpr'])} |",
        f"| This project (shipped) | {pct(o['injection']['attack_recall'])} | "
        f"{pct(o['injection']['benign_fpr'])} | {pct(o['injection']['hard_benign_fpr'])} |",
        "",
        "## PII, types both systems support (test split)",
        "",
        "| Type | Bedrock precision / recall | Ours precision / recall |",
        "|---|---|---|",
    ]  # fmt: skip
    for ours, managed in MANAGED_EQUIVALENT.items():
        bm, om = b["pii_overlapping_types"][managed], o["pii"][ours] or {}
        fmt = lambda m: "–" if m.get("precision") is None else f"{100 * m['precision']:.1f}%"  # noqa: E731
        fmr = lambda m: "–" if m.get("recall") is None else f"{100 * m['recall']:.1f}%"  # noqa: E731
        lines.append(f"| {ours} / {managed} | {fmt(bm)} / {fmr(bm)} | {fmt(om)} / {fmr(om)} |")
    lines += [
        "",
        f"Rows without PII that Bedrock flagged: {pct(b['pii_rows_without_pii_flagged'])}.",
        f"No managed equivalent (not scored for Bedrock): {', '.join(b['no_managed_equivalent'])}.",
        "",
        f"Cost of both runs: USD {b['usd']:.4f}; ApplyGuardrail latency "
        f"p50 {b['latency_ms']['p50']} ms, p95 {b['latency_ms']['p95']} ms (live, one run).",
        "",
    ]  # fmt: skip
    return "\n".join(lines)
