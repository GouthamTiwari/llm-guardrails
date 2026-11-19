import io
import json
import time

import boto3
import numpy as np
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from llm_guardrails.embeddings import (
    FakeEmbedder,
    Pacer,
    RecordedEmbedder,
    TitanEmbedder,
    VectorBundle,
)
from llm_guardrails.settings import Settings
from llm_guardrails.spend import SpendLedger
from llm_guardrails.types import BudgetExceeded, ReplayMissing

TITAN = "amazon.titan-embed-text-v2:0"


def titan_response(vector, tokens=4):
    data = json.dumps({"embedding": vector, "inputTextTokenCount": tokens}).encode()
    return {"body": StreamingBody(io.BytesIO(data), len(data)), "contentType": "application/json"}


def make_titan(tmp_path, *, limit=1.0, memo=True):
    client = boto3.client(
        "bedrock-runtime", region_name="ap-south-1",
        aws_access_key_id="testing", aws_secret_access_key="testing",
    )  # fmt: skip
    settings = Settings(embed_dimensions=4, embed_requests_per_second=1000)
    emb = TitanEmbedder(
        settings,
        ledger=SpendLedger(tmp_path / "l.jsonl", limit),
        client=client,
        memo_dir=tmp_path / "memo" if memo else None,
    )
    return emb, Stubber(client)


async def test_titan_request_normalised_vector_cost_and_memo(tmp_path) -> None:
    emb, stub = make_titan(tmp_path)
    stub.add_response(
        "invoke_model",
        titan_response([3.0, 4.0, 0.0, 0.0]),
        expected_params={
            "modelId": TITAN,
            "body": json.dumps({"inputText": "hello", "dimensions": 4, "normalize": True}),
            "contentType": "application/json",
            "accept": "application/json",
        },
    )
    with stub:
        first = await emb.embed(["hello"], purpose="unit")
        again = await emb.embed(["hello"], purpose="unit")  # memo: no second AWS call
    np.testing.assert_allclose(first[0], [0.6, 0.8, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_array_equal(first, again)
    assert first.dtype == np.float32 and emb.api_calls == 1
    assert emb.ledger.total() == pytest.approx(4 * 0.024 / 1e6)


async def test_titan_budget_checked_before_call(tmp_path) -> None:
    emb, stub = make_titan(tmp_path, limit=0.0, memo=False)
    with stub, pytest.raises(BudgetExceeded):
        await emb.embed(["hello"], purpose="unit")


async def test_empty_input_returns_empty_matrix(tmp_path) -> None:
    emb, _ = make_titan(tmp_path)
    assert (await emb.embed([], purpose="unit")).shape == (0, 4)


def test_pacer_spaces_calls() -> None:
    pacer = Pacer(rate=20)  # one call every 50 ms
    start = time.monotonic()
    for _ in range(4):
        pacer.wait()
    assert time.monotonic() - start >= 0.14


async def test_fake_embedder_is_deterministic_and_unit_length() -> None:
    fake = FakeEmbedder(dims=16)
    a = await fake.embed(["x", "y"], purpose="t")
    b = await fake.embed(["x"], purpose="t")
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.allclose(a[0], a[1])
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)


async def test_fake_embedder_table_and_failure() -> None:
    fake = FakeEmbedder(dims=2, table={"attack": [1, 0]})
    np.testing.assert_allclose((await fake.embed(["attack"], purpose="t"))[0], [1, 0])
    broken = FakeEmbedder(fail=TimeoutError())
    with pytest.raises(TimeoutError):
        await broken.embed(["x"], purpose="t")


async def test_bundle_save_load_and_recorded_embedder(tmp_path) -> None:
    prefix = tmp_path / "vectors" / "suite"
    bundle = VectorBundle(prefix, "fake-embed", 8)
    recorder = RecordedEmbedder(bundle, inner=FakeEmbedder(dims=8))
    vecs = await recorder.embed(["a", "b"], purpose="t")
    bundle.save()

    loaded = VectorBundle(prefix, "fake-embed", 8)
    assert len(loaded) == 2
    replay = RecordedEmbedder(loaded)
    np.testing.assert_allclose(await replay.embed(["b", "a"], purpose="t"), vecs[::-1])
    with pytest.raises(ReplayMissing):
        await replay.embed(["never seen"], purpose="t")
    # A bundle for a different model or size is ignored, never mixed in.
    assert len(VectorBundle(prefix, "other-model", 8)) == 0
    assert len(VectorBundle(prefix, "fake-embed", 16)) == 0
