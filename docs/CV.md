# Trois puces pour le CV

- Conception d’un pipeline reproductible sur 300 réclamations CFPB réelles en anglais : acquisition officielle, empreintes SHA-256, détection de doublons et partitions stratifiées 210/45/45 ; baseline TF-IDF et régression logistique atteignant 0,910 de macro-F1 sur 45 exemples de test.
- Adaptation de SmolLM2-135M-Instruct par LoRA sur CPU, avec 460 800 paramètres entraînables, loss sur la réponse seule et trois epochs exécutées ; adaptateur sauvegardé et rechargé, 80 % d’accuracy et 100 % de JSON conformes sur le test commun, sans dépasser la baseline classique.
- Développement d’une démonstration Streamlit en français avec comparaison des modèles, import/export CSV et contrôles JSON ; tests pytest, lint Ruff et configuration GitHub Actions légère, sans publication externe.
