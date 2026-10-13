"""Plot measured train and validation losses from saved Trainer histories."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
for name in ["adapter_smoke", "adapter_full"]:
    path = root / f"artifacts/{name}/run.json"
    if not path.exists():
        continue
    history = json.loads(path.read_text(encoding="utf-8"))["history"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for field, label in [("loss", "Entraînement"), ("eval_loss", "Validation")]:
        rows = [r for r in history if field in r]
        ax.plot(
            [r["step"] for r in rows],
            [r[field] for r in rows],
            label=label,
            marker="o" if field == "eval_loss" else None,
        )
    ax.set_xlabel("Étape optimiseur")
    ax.set_ylabel("Cross-entropy, réponse seulement")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / f"reports/training_losses_{name}.png")
    plt.close(fig)
