"""Build the portfolio report exclusively from recorded measurements."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((root / name).read_text(encoding="utf-8"))


report = read("reports/comparison.json")
full = read("reports/full_training.json")
env = read("reports/environment.json")
manifest = read("reports/data_manifest.json")
lines = [
    "# Rapport comparatif mesuré — 13 octobre 2026",
    "",
    "300 récits CFPB réels en anglais ; trois catégories équilibrées. Train 210, validation 45, test 45.",
    "Le prompt est choisi sur six exemples de validation, jamais sur le test. La comparaison finale utilise",
    f"les mêmes {len(report['test_ids'])} récits, le même prompt et la même génération pour les deux LLM.",
    "",
    f"Matériel : Intel i7-13620H, Intel UHD sans CUDA, {env['ram_bytes'] / 2**30:.1f} Gio RAM, CPU float32, quatre threads.",
    f"Python {env['python']}, PyTorch {env['torch']}. Batch 1, perf_counter, chauffe sur validation exclue ; chargement exclu.",
    "",
    "| Approche | Accuracy | Macro-F1 | JSON parsable | Schéma conforme | Latence médiane (s) | p95 (s) |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for name, m in report["models"].items():
    lines.append(
        f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['json_parse_rate']:.3f} | "
        f"{m['valid_schema_rate']:.3f} | {m['latency_median_seconds']:.4f} | {m['latency_p95_seconds']:.4f} |"
    )
lines += [
    "",
    "La baseline classique reste meilleure sur ce test. Le LLM de base échoue au contrat JSON :",
    "ses 45 sorties invalides comptent comme échecs. LoRA rétablit le format mais commet neuf erreurs",
    "de catégorie. Un score nul du pipeline de base ne mesure pas séparément sa compréhension sémantique.",
    "",
    "Les textes sont tronqués à 160 tokens pour les deux LLM ; la baseline classique lit tout le récit.",
    "172 récits sur 300 sont concernés par cette troncature. La latence dépend aussi de la longueur",
    "effectivement générée et de l’arrêt EOS ; elle ne prouve pas une accélération intrinsèque de LoRA.",
    "",
    "## Résultats par classe",
    "",
    "| Modèle | Classe | Précision | Rappel | F1 | Support |",
    "|---|---|---:|---:|---:|---:|",
]
for name, m in report["models"].items():
    for label in manifest["labels"]:
        c = m["per_class"][label]
        lines.append(
            f"| {name} | {label} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1-score']:.3f} | {int(c['support'])} |"
        )
lines += [
    "",
    "## Conformité et matrices",
    "",
    "Les matrices incluent une colonne INVALID. Les sorties rejetées ne sont jamais retirées du dénominateur.",
    "",
]
for name, m in report["models"].items():
    lines += [
        f"### {name}",
        "",
        f"Catégorie non autorisée dans un JSON parsable : {m['invalid_category_rate']:.1%} ; réponse vide : {m['missing_response_rate']:.1%} ; sortie inexploitable : {m['missing_or_uninterpretable_rate']:.1%}.",
        "Une réponse non JSON ne permet pas de compter une catégorie extraite ; elle reste un échec global.",
        "",
        f"![Matrice {name}](confusion_{name}.png)",
        "",
    ]
lines += [
    "## Apprentissage et ressources",
    "",
    f"LoRA complet effectivement exécuté : trois epochs, 315 étapes, {full['seconds']:.1f} s ({full['seconds'] / 60:.1f} min).",
    f"{full['trainable']:,} paramètres entraînables sur {full['total']:,} paramètres au total ; q_proj/v_proj, r=8, alpha=16, dropout=0,05.",
    "LR 3e-4, batch 1, accumulation 2, loss sur la complétion seulement. Les cibles sont la catégorie et deux champs null, sans références inventées.",
    "L’adaptateur smoke exécute trois étapes seulement ; il ne constitue pas une preuve de performance.",
    "",
    "![Losses complètes](training_losses_adapter_full.png)",
    "",
    "Loss de validation : 0,04995 à epoch 1 ; 0,04158 à epoch 2 ; 0,03942 à epoch 3.",
    "Pic RSS échantillonné : environ 1,45 Gio ; échantillonnage cinq secondes commencé en cours de run, donc borne inférieure du pic réel. Pas de VRAM CUDA utilisée.",
    "",
    "## Erreurs et qualité des textes",
    "",
    "[Analyse qualitative](ANALYSIS.md) : quatre erreurs classiques et neuf LoRA, avec identifiants réels et hypothèses explicitement distinguées des faits.",
    "Les extraits et sorties brutes sont dans ERROR_EXAMPLES.md et predictions_*.json, conservés localement et exclus de l’archive source.",
    "La grille human_rubric.csv contient une ligne séparée pour chaque résumé et justification. Aucune note humaine attribuée ; les champs null ou sorties invalides sont indisponibles.",
    "Il n’y a aucune validation de fidélité, pertinence ou absence d’hallucinations des résumés et justifications. Aucun score ROUGE inventé.",
    "",
    "## Vérifications et limites",
    "",
    "13 tests pytest locaux réussis, sans skip : données, parsing, échecs, CSV, baseline, rechargement des deux adaptateurs entraînés et comparaison Streamlit. Ruff passe.",
    "Répétition de CI sans PyTorch ni dataset : six tests passent, sept tests d’intégration sont ignorés. GitHub Actions est préparé, pas exécuté sur un compte distant.",
    "Le serveur Streamlit local répond HTTP 200. Le parcours CSV teste les fonctions réellement partagées avec l’interface ; aucune automatisation du clic sur le sélecteur de fichier du navigateur n’est prétendue.",
    "Limites : test de 45 exemples, trois produits financiers historiques, labels consommateur bruités, échantillon équilibré artificiellement, troncature et aucune validation française. Pas de conclusion statistique générale ni de comparaison avec les grands LLM.",
    "Colab préparé, non exécuté ; aucune ressource payante créée. Aucun dépôt, dataset ou modèle publié sur un compte externe.",
]
(root / "reports/COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
