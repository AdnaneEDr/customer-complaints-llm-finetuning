"""Invalid generations count as classification failures."""

import csv
import json
import platform
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .core import ROOT, read, save


def score(rows, predictions, labels):
    true = [r["category"] for r in rows]
    pred = [p["output"]["category"] if p["output"] else "__INVALID__" for p in predictions]
    errors = Counter(p["error"] for p in predictions if p["error"])
    invalid_categories = 0
    for prediction in predictions:
        try:
            decoded = json.loads(prediction["raw"]) if "raw" in prediction else prediction["output"]
        except (ValueError, TypeError):
            continue
        if isinstance(decoded, dict) and "category" in decoded and decoded["category"] not in labels:
            invalid_categories += 1
    all_labels = labels + ["__INVALID__"]
    return {
        "n": len(rows),
        "accuracy": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, labels=labels, average="macro", zero_division=0),
        "per_class": classification_report(true, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_labels": all_labels,
        "confusion_matrix": confusion_matrix(true, pred, labels=all_labels).tolist(),
        "valid_schema_rate": sum(p["output"] is not None for p in predictions) / len(rows),
        "json_parse_rate": sum(p["error"] != "json" for p in predictions) / len(rows),
        "invalid_category_rate": invalid_categories / len(rows),
        "missing_response_rate": sum(not p.get("raw", "").strip() for p in predictions) / len(rows),
        "missing_or_uninterpretable_rate": sum(p["output"] is None for p in predictions) / len(rows),
        "errors": dict(errors),
        "latency_median_seconds": float(np.median([p["seconds"] for p in predictions])),
        "latency_p95_seconds": float(np.percentile([p["seconds"] for p in predictions], 95)),
    }


def select_prompt():
    from .models import Engine

    rows = read("data/validation.json")[:6]
    engine = Engine("base")
    candidates = []
    for variant in [0, 1]:
        engine.variant = variant
        predictions = [engine.predict(r["text"]) for r in rows]
        candidates.append({"variant": variant, "metrics": score(rows, predictions, engine.labels)})
    best = max(candidates, key=lambda c: c["metrics"]["macro_f1"])
    save(
        ROOT / "artifacts/prompt.json",
        {
            "variant": best["variant"],
            "validation_n": len(rows),
            "candidates": candidates,
            "tie_break": "first candidate",
        },
    )
    engine.close()


def evaluate(kinds=None, limit=None, adapter="adapter_smoke"):
    from .models import Engine

    rows = read("data/test.json")
    if limit:
        labels = read("data/manifest.json")["labels"]
        rows = [r for label in labels for r in [x for x in rows if x["category"] == label][:limit]]
    report = {
        "hardware": platform.platform(),
        "device": "CPU float32",
        "batch_size": 1,
        "method": "perf_counter; warmup excluded; model loading excluded; sequential",
        "test_ids": [r["id"] for r in rows],
        "adapter": adapter,
        "models": {},
    }
    for kind in kinds or ["classic", "base", "lora"]:
        engine = Engine(kind, adapter=adapter)
        engine.predict(read("data/validation.json")[0]["text"])
        predictions = [engine.predict(r["text"]) for r in rows]
        report["models"][kind] = score(rows, predictions, engine.labels)
        save(ROOT / f"reports/predictions_{kind}.json", [{**r, **p} for r, p in zip(rows, predictions)])
        engine.close()
        save(ROOT / "reports/comparison.json", report)
        import matplotlib.pyplot as plt

        matrix = np.array(report["models"][kind]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(matrix, cmap="Blues")
        ticks = report["models"][kind]["confusion_labels"]
        ax.set_xticks(range(len(ticks)), ticks, rotation=30, ha="right")
        ax.set_yticks(range(len(ticks)), ticks)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Référence")
        ax.set_title(kind)
        for i in range(len(ticks)):
            for j in range(len(ticks)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(ROOT / f"reports/confusion_{kind}.png")
        plt.close(fig)
    lines = [
        "# Rapport comparatif mesuré",
        "",
        f"Évaluation commune : {len(rows)} récits anglais, CPU, batch 1.",
        f"Adaptateur : {adapter}. Smoke valide le fonctionnement, pas une amélioration de performance.",
        "",
        "| Modèle | Accuracy | Macro-F1 | JSON parsable | Schéma valide | Latence médiane (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in report["models"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['json_parse_rate']:.3f} | "
            f"{m['valid_schema_rate']:.3f} | {m['latency_median_seconds']:.3f} |"
        )
    lines += [
        "",
        "Les sorties invalides sont des échecs. Aucun score automatique de résumé/justification :",
        "pas de références dans les données ; champs null pendant l'entraînement.",
        "Voir les JSON pour les métriques par classe, matrices et erreurs sur les exemples réels.",
    ]
    (ROOT / "reports/COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    errors = [
        "# Exemples d’erreurs réels — analyse qualitative",
        "",
        "Diagnostic automatique de conformité et comparaison à l’étiquette CFPB ; aucune note humaine inventée.",
    ]
    for kind in report["models"]:
        predictions = read(f"reports/predictions_{kind}.json")
        failures = [p for p in predictions if not p["output"] or p["output"]["category"] != p["category"]]
        errors += ["", f"## {kind} : {len(failures)} échecs / {len(predictions)}"]
        for p in failures[:3]:
            diagnosis = (
                f"Sortie rejetée : {p['error']}." if p["error"] else "JSON conforme, catégorie incorrecte."
            )
            errors += [
                "",
                f"ID {p['id']} ; référence : {p['category']} ; {diagnosis}",
                "Récit (extrait, pas résumé de référence) :",
                p["text"][:500],
                "Sortie brute :",
                "```text",
                p["raw"][:500],
                "```",
            ]
    (ROOT / "reports/ERROR_EXAMPLES.md").write_text("\n".join(errors), encoding="utf-8")
    with (ROOT / "reports/human_rubric.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "model",
                "field",
                "faithfulness_1_5",
                "relevance_1_5",
                "no_hallucination_1_5",
                "conciseness_1_5",
                "available",
                "notes",
            ],
        )
        writer.writeheader()
        for kind in report["models"]:
            if kind == "classic":
                continue
            for prediction in read(f"reports/predictions_{kind}.json"):
                for field in ["summary", "justification"]:
                    available = bool(prediction["output"] and prediction["output"][field])
                    writer.writerow(
                        {
                            "id": prediction["id"],
                            "model": kind,
                            "field": field,
                            "available": available,
                            "notes": "" if available else "Indisponible ; ne pas noter",
                        }
                    )
    return report
