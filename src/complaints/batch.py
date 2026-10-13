"""Shared CSV import/export used by UI and integration tests."""

import json

import pandas as pd


def load_csv(source):
    frame = pd.read_csv(source)
    if "text" not in frame or not 1 <= len(frame) <= 20 or frame["text"].isna().any():
        raise ValueError("CSV : colonne text, 1 à 20 lignes non vides requises.")
    texts = frame["text"].astype(str).tolist()
    if any(not text.strip() for text in texts):
        raise ValueError("CSV : réclamation vide.")
    return texts


def export_csv(results):
    return (
        pd.DataFrame([{**r, "output": json.dumps(r["output"], ensure_ascii=False)} for r in results])
        .to_csv(index=False)
        .encode("utf-8")
    )
