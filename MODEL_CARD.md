# Fiche modèle

Base : HuggingFaceTB/SmolLM2-135M-Instruct, Apache-2.0 ; révision exacte dans artifacts/model_revision.json. Adaptateurs LoRA distincts smoke et full, lorsque leurs adapter_config.json existent. Les poids de base ne sont pas inclus dans Git et ne doivent pas être confondus avec les poids d’adaptateur.

Usage : expérimentation locale, classification de récits CFPB anglais en Mortgage, Debt collection, Credit card. Hors périmètre : validation française, automatisation de décisions financières, juridique ou détection de fraude. Rien dans ce projet n’établit la véracité des réclamations.

Données : archive officielle décembre 2011–avril 2018 ; reservoir sample 300 récits, splits 210/45/45, provenance et SHA dans data/. Le CFPB décrit les récits antérieurement publiés comme domaine public pour FOIA ; conserver la provenance et vérifier avant toute redistribution. Informations sensibles résiduelles possibles. Pas de données synthétiques d’entraînement ou de références de justification inventées.

Adaptation : LoRA q_proj/v_proj, r=8, alpha=16, dropout=0,05, AdamW via Trainer, LR 3e-4, batch 1 accumulation 2, seed 42, CPU float32. Loss sur la complétion JSON, catégorie supervisée et champs null. Les runs enregistrent étapes, losses, durée, RSS finale et nombre de paramètres entraînables. Smoke = trois étapes seulement. Full = trois epochs, si exécuté. Aucun entraînement Colab prétendu réalisé.

Performances mesurées le 13 octobre 2026 sur 45 récits de test : baseline classique accuracy 0,911 et macro-F1 0,910 ; modèle instruction de base accuracy et macro-F1 0, aucun JSON valide ; adaptateur complet accuracy 0,800, macro-F1 0,796, 100 % de JSON conformes. LoRA complet effectivement exécuté : trois epochs, 315 étapes, 2108,5 secondes sur CPU ; rechargement vérifié par une génération réelle. Voir reports/COMPARISON.md et comparison.json pour les matrices, métriques par classe et latences. Grille humaine préparée, non annotée ; aucune qualité de résumé ou justification validée.

Limites : modèle 135M peu robuste au JSON, vocabulaire métier et longues réclamations ; troncature à 160 tokens ; petit test ; classes équilibrées artificiellement par échantillonnage, différentes des fréquences naturelles ; aucune validation en français ; labels consommateur bruités. Le système affiche les sorties invalides et ne les remplace pas par celles de la baseline.
