# Comprendre et défendre le projet

Une réclamation est un texte libre. La cible supervisée est le produit financier choisi par le consommateur, pas son sentiment ni une décision juridique. L’étiquette peut être bruitée. Les catégories historiques ne doivent pas être fusionnées sans décision métier.

La baseline transforme les mots et couples de mots en poids TF-IDF : un terme fréquent dans un document, rare ailleurs, reçoit davantage de poids. La régression logistique apprend des coefficients par catégorie. Sur peu de données et un vocabulaire spécialisé, elle peut surpasser un petit LLM. Elle ne produit ni résumé ni justification ; ses champs sont null.

Un modèle instruction reçoit le chat template natif du tokenizer. L’ajout de tokens spéciaux deux fois peut dégrader le résultat ; ici apply_chat_template tokenise directement. La génération est déterministe, sans sampling. Le prompt est choisi sur validation puis identique pour base et LoRA.

LoRA gèle les poids W et apprend une petite correction BA : W' = W + (alpha/r) BA. B et A ont un rang r bien plus petit que les dimensions de W. Cela réduit les paramètres et les états optimiseur entraînables ; le modèle de base doit toujours tenir en mémoire. Ici r=8, alpha=16, dropout=0,05, projections attention q_proj et v_proj. Pas de quantification, donc pas de QLoRA. Voir le nombre mesuré de paramètres dans le run.

La loss cross-entropy compare chaque token cible au suivant prédit. Les tokens du prompt ont label -100 et sont ignorés. Le dataset ne fournit pas de résumés ou explications de référence : les cibles sont catégorie + champs null. On apprend la classification et la syntaxe JSON ; on n’apprend pas une compétence de résumé supervisée. Une baisse de loss n’est pas une preuve de meilleure classification.

Pourquoi trois partitions ? Train ajuste les poids ; validation choisit le prompt et permet de suivre la loss ; test mesure le résultat final sans guider les réglages. Les doublons proches peuvent donner une illusion de généralisation, d’où leur détection avant split. Les empreintes permettent de contrôler les mêmes données, mais pas une identité numérique parfaite entre CPU et GPU.

Accuracy = proportion correcte. Macro-F1 = moyenne des F1 des catégories métier, donc chaque classe a le même poids. Precision = proportion correcte parmi les prédictions de la classe ; recall = proportion retrouvée parmi les vrais exemples. F1 combine les deux. Une sortie invalide prend une catégorie d’échec et réduit le rappel de sa vraie classe. Elle n’est jamais éliminée.

Deux taux distincts : du JSON peut être syntaxiquement valide mais violer le schéma ou choisir une catégorie absente. Une matrice avec colonne INVALID rend ces échecs visibles. Le temps par exemple exclut le chargement, mais celui-ci est ressenti dans la démo. Les summaries/rationales doivent être évalués par humains séparément : leur absence de référence interdit un ROUGE ou une exactitude inventés.

Points à expliquer en entretien : périmètre financier anglais, petits effectifs, class labels historiques, troncature du récit, prompt fixé avant test, même génération pour les deux LLM, latence CPU, et résultats négatifs éventuels. Améliorations futures : plus de train réel, meilleure validation du prompt, petite étude d’ablation de troncature, modèle 360M, annotations humaines vérifiées pour le résumé ; ces travaux ne sont pas prétendus réalisés.
