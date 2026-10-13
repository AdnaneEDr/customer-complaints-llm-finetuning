import os

import pytest

from complaints.core import ROOT, fingerprint, read


@pytest.mark.skipif(not (ROOT / "data/train.json").exists(), reason="real data absent in lightweight CI")
def test_partitions():
    splits = [read(f"data/{s}.json") for s in ["train", "validation", "test"]]
    sets = [{fingerprint(r["text"]) for r in split} for split in splits]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


@pytest.mark.skipif(not (ROOT / "artifacts/classic.joblib").exists(), reason="baseline absent")
def test_real_classic():
    from complaints.models import Engine

    engine = Engine("classic")
    assert (
        engine.predict("My mortgage payment was incorrectly charged.")["output"]["category"] in engine.labels
    )
    engine.close()


@pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1", reason="opt-in actual adapter reload")
@pytest.mark.parametrize("adapter", ["adapter_smoke", "adapter_full"])
def test_adapter_reload(adapter):
    from complaints.models import Engine, encode

    engine = Engine("lora", adapter=adapter)
    assert any(
        "lora_B" in name and parameter.detach().abs().sum().item() > 0
        for name, parameter in engine.model.named_parameters()
    )
    result = engine.predict("My credit card charge was unauthorized.")
    assert isinstance(result["raw"], str) and result["seconds"] > 0
    item = encode(engine, {"text": "My credit card charge was unauthorized.", "category": "Credit card"})
    prefix = engine.prompt_tokens("My credit card charge was unauthorized.")
    assert item["labels"][: len(prefix)] == [-100] * len(prefix)
    assert item["labels"][len(prefix) :] == item["input_ids"][len(prefix) :]
    engine.close()
