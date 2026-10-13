# Analyse qualitative des erreurs

Ces observations sont une lecture technique des récits et prédictions conservés localement, pas une annotation humaine indépendante de résumés ou justifications.

La baseline classique commet quatre erreurs sur 45 récits. Les identifiants renvoient aux exemples réels dans les fichiers predictions, exclus de l’archive source.

- 2134946, Mortgage → Credit card : le récit mentionne explicitement un crédit immobilier remboursé, puis un remboursement de compte séquestre non reçu. La mauvaise catégorie montre que la simple présence du mot « mortgage » ne garantit pas la décision du classifieur. Une confusion avec le vocabulaire de remboursement est une hypothèse, non une explication causale mesurée.
- 2697991, Mortgage → Credit card : court récit de paiements après faillite, avec mention explicite de mortgage. La baseline échoue même sur un texte dont le produit est nommé.
- 2435427, Credit card → Mortgage : le texte parle de retards sur le rapport de crédit et de relevés mensuels sans nommer clairement le produit. L’étiquette CFPB contient une information qui peut être difficile à retrouver dans le seul récit.
- 1531825, Credit card → Debt collection : le récit décrit un compte frauduleux et une usurpation d’identité sans préciser le produit. Plusieurs catégories financières peuvent partager ce vocabulaire.

Implication : l’évaluation mesure la reproduction du produit sélectionné par le consommateur, pas la vérité juridique de sa plainte. Les étiquettes et les récits peuvent être ambigus. Le classifieur ne doit pas être présenté comme un système qui vérifie les allégations.

Les erreurs des deux LLM, leur conformité JSON et leurs catégories sont détaillées après exécution dans COMPARISON.md, comparison.json et ERROR_EXAMPLES.md. Une sortie invalide ne fournit pas une classification exploitable et compte comme un échec, même si elle contient des mots pertinents.

Le LLM de base ne produit aucun JSON interprétable sur les 45 exemples. Le score nul décrit donc l’échec du pipeline au format demandé ; il ne mesure pas séparément sa compréhension sémantique. Par exemple, 1587013 donne une prose répétitive sur une hypothèque, et 1695125 reformule un problème de service client au lieu de retourner les trois champs. Les sorties brutes sont conservées pour vérifier ces observations. Aucune stratégie de réparation n’a été choisie après lecture du test.

La latence peut être influencée par la longueur réellement générée et l’arrêt EOS, même quand le maximum de tokens et le décodage sont identiques. Ne pas interpréter une réponse JSON courte comme une accélération intrinsèque de LoRA.

L’adaptateur complet produit 45 JSON conformes, mais neuf catégories incorrectes. Le rappel Credit card est de 0,60 ; la précision Mortgage est de 0,70, avec 20 prédictions Mortgage pour 15 références de cette classe. La conformité syntaxique ne garantit donc pas une bonne catégorie.

- 1695125, Credit card → Mortgage : le récit commence par une carte de fidélité de magasin et un problème de frais de retard. Le produit est nommé, mais le modèle adapté choisit Mortgage.
- 2034642, Credit card → Debt collection : le consommateur mentionne explicitement les cartes de crédit et demande l’annulation d’une dette. La prédiction peut confondre le sujet « dette » avec le produit de recouvrement ; c’est une hypothèse de confusion lexicale, pas une causalité démontrée.
- 1531825 est mal classé en Debt collection par les deux modèles supervisés : le récit d’usurpation d’identité ne nomme pas clairement le produit.

LoRA corrige trois des quatre erreurs de la baseline classique, mais ajoute huit autres erreurs : neuf au total contre quatre. Il apprend à fournir le format demandé et obtient 80 % d’accuracy, sans battre la baseline à 91,1 % sur ce petit test. L’échantillon de 45 récits ne permet pas une conclusion générale sur tous les modèles ou domaines. Aucune réoptimisation n’a été faite après consultation de ces erreurs.
