"""Bounded reservoir sample from official historical CFPB archive."""

import csv
import hashlib
import io
import random
import zipfile
from collections import Counter

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from .core import CFG, ROOT, clean, fingerprint, save

URL = "https://files.consumerfinance.gov/f/documents/CCDB_Export_1_December_2011_through_April_2018.zip"
LABELS = ["Mortgage", "Debt collection", "Credit card"]


def acquire():
    path = ROOT / "data/archive.zip"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".partial")
        with requests.get(URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        temporary.replace(path)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt archive")
    save(
        ROOT / "data/source.json",
        {
            "url": URL,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "language": "en",
            "source": "CFPB historical public narratives",
        },
    )
    return path


def deduplicate(rows, threshold=0.90):
    unique, seen = [], set()
    for row in rows:
        key = fingerprint(row["text"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    if len(unique) < 2:
        return unique
    x = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000).fit_transform(
        [r["text"].lower() for r in unique]
    )
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(x)
    neighbors = nn.radius_neighbors(x, radius=1 - threshold, return_distance=False)
    parent = list(range(len(unique)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, nearby in enumerate(neighbors):
        for j in nearby:
            a, b = find(i), find(int(j))
            parent[max(a, b)] = min(a, b)
    return [row for i, row in enumerate(unique) if find(i) == i]


def prepare():
    path = acquire()
    rng = random.Random(CFG["seed"])
    counts = Counter()
    samples = {label: [] for label in LABELS}
    excluded = Counter()
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("Official archive contains no CSV")
        for name in names:
            with archive.open(name) as handle:
                reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig"))
                required = {"Product", "Consumer complaint narrative", "Complaint ID"}
                if not required <= set(reader.fieldnames or []):
                    raise ValueError(f"Unexpected source schema: {reader.fieldnames}")
                for raw in reader:
                    label = raw["Product"]
                    if label not in LABELS:
                        excluded["other_product"] += 1
                        continue
                    text = clean(raw["Consumer complaint narrative"])
                    if len(text) < 80:
                        excluded["missing_or_short"] += 1
                        continue
                    row = {"id": str(raw["Complaint ID"]), "text": text, "category": label}
                    counts[label] += 1
                    target = samples[label]
                    cap = CFG["per_class"]
                    if len(target) < cap:
                        target.append(row)
                    else:
                        j = rng.randrange(counts[label])
                        if j < cap:
                            target[j] = row
    rows = [row for label in LABELS for row in samples[label]]
    original = len(rows)
    rows = deduplicate(rows)
    if any(sum(r["category"] == label for r in rows) < 10 for label in LABELS):
        raise ValueError("Insufficient real narratives per class")
    train, held = train_test_split(
        rows, test_size=0.30, random_state=CFG["seed"], stratify=[r["category"] for r in rows]
    )
    val, test = train_test_split(
        held, test_size=0.5, random_state=CFG["seed"], stratify=[r["category"] for r in held]
    )
    splits = [("train", train), ("validation", val), ("test", test)]
    for name, split in splits:
        save(ROOT / f"data/{name}.json", split)
    save(
        ROOT / "data/manifest.json",
        {
            "seed": CFG["seed"],
            "labels": LABELS,
            "eligible_source_counts": dict(counts),
            "exclusions": dict(excluded),
            "sample_before_dedup": original,
            "removed_duplicates": original - len(rows),
            "near_duplicate_rule": "char_wb TF-IDF cosine >=0.90; connected components before split",
            "word_length_percentiles": np.percentile(
                [len(r["text"].split()) for r in rows], [0, 50, 90, 100]
            ).tolist(),
            "splits": {
                n: {
                    "n": len(s),
                    "classes": dict(Counter(r["category"] for r in s)),
                    "sha256": hashlib.sha256((ROOT / f"data/{n}.json").read_bytes()).hexdigest(),
                }
                for n, s in splits
            },
        },
    )
    return len(rows)
