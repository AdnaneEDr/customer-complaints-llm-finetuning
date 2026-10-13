import json

from complaints.core import clean, fingerprint, parse
from complaints.data import deduplicate
from complaints.evaluate import score


def test_clean_and_identity():
    assert clean("  a\n b ") == "a b"
    assert fingerprint("HELLO!") == fingerprint("hello")


def test_parser():
    assert parse("bad", ["Mortgage"])[1] == "json"
    assert parse("{}", ["Mortgage"])[1] == "schema"
    assert (
        parse(json.dumps({"category": "Other", "summary": None, "justification": None}), ["Mortgage"])[1]
        == "category"
    )
    assert (
        parse(json.dumps({"category": "Mortgage", "summary": None, "justification": None}), ["Mortgage"])[1]
        is None
    )


def test_dedup_across_labels():
    rows = [
        {"text": "This bank charged me twice for my mortgage payment.", "category": c} for c in ["a", "b"]
    ]
    assert len(deduplicate(rows)) == 1


def test_invalid_predictions_count():
    rows = [{"category": "a"}, {"category": "b"}]
    predictions = [
        {"output": {"category": "a"}, "error": None, "seconds": 1},
        {"output": None, "error": "json", "seconds": 1},
    ]
    assert score(rows, predictions, ["a", "b"])["accuracy"] == 0.5


def test_near_duplicates():
    text = "The company keeps contacting me about a debt that I do not owe. " * 8
    rows = [
        {"text": text + tail, "category": label}
        for tail, label in [("Please help.", "a"), ("Please assist.", "b")]
    ]
    assert len(deduplicate(rows)) == 1
