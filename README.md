# 🌾 Segmentation des graines de blé — Apprentissage non supervisé (DBSCAN)

Projet de **clustering de graines de blé** à partir de leurs caractéristiques morphologiques, réalisé dans le cadre du module *Unsupervised Learning* du **Sen HUB**. Trois algorithmes de clustering (KMeans, CAH, DBSCAN) sont comparés — avec et sans normalisation — le meilleur modèle est sélectionné puis **déployé sous forme d'application web interactive** (Streamlit) permettant d'attribuer une classe à une nouvelle graine.

## 🎯 Objectif

Regrouper des graines de blé en fonction de leur morphologie (surface, périmètre, compacité, longueur/largeur du grain, etc.), sans étiquette préalable, afin de :

- identifier des familles de graines aux caractéristiques similaires,
- comparer objectivement plusieurs algorithmes de clustering pour choisir le plus performant,
- détecter les graines au profil atypique (anomalies),
- fournir un outil simple permettant de classer instantanément une nouvelle graine.

## 📁 Structure du dépôt

```
.
├── Kmeans_On_Wheat_Seeds_Dataset.ipynb   # Notebook complet : EDA, PCA, comparaison des modèles, déploiement
├── wheat_seeds_dataset.csv               # Jeu de données brut
├── app.py                                # Application Streamlit de prédiction (modèle final : DBSCAN)
├── modele_dbscan.joblib                  # Artefacts du modèle DBSCAN entraîné (points cœur, eps, colonnes, etc.)
├── requirements.txt                      # Dépendances Python de l'application
└── README.md
```

## 🗂️ Jeu de données

Le jeu de données contient **199 graines de blé**, chacune décrite par **7 variables morphologiques** issues de l'analyse d'image des grains :

| Variable | Description |
|---|---|
| `area A` | Surface du grain |
| `perimeter` | Périmètre du grain |
| `compactness` | Compacité (forme, proche de 1 = grain rond) |
| `length of kernel` | Longueur du grain |
| `width of kernel` | Largeur du grain |
| `asymmetry coefficient` | Coefficient d'asymétrie du grain |
| `length of kernel groove` | Longueur du sillon du grain |

## 🧪 Méthodologie

Le notebook original ne testait que KMeans. Pour identifier un véritable **meilleur modèle**, la même démarche comparative que pour le projet de segmentation clients a été reproduite : **trois algorithmes**, chacun testé **avec et sans normalisation** des variables (`sklearn.preprocessing.normalize`), évalués par **score de silhouette**.

1. **Réduction de dimension** : ACP à 2 composantes pour la visualisation — 99,3 % de variance expliquée conservée.
2. **KMeans** : recherche du k optimal par score de silhouette (k=2 à 8).
3. **CAH** (`linkage='average'`) : recherche du nombre de classes optimal par score de silhouette.
4. **DBSCAN** : recherche de `eps` et `min_samples` par grille, en ne retenant que les combinaisons produisant un taux de bruit raisonnable.

### Résultats comparatifs

| Modèle | Normalisation | Paramètres | Clusters | Bruit | Silhouette |
|---|---|---|---|---|---|
| KMeans | Non | k=2 | 2 | – | 0.530 |
| KMeans | Oui | k=2 | 2 | – | 0.499 |
| CAH (average) | Non | k=2 | 2 | – | 0.506 |
| CAH (average) | Oui | k=2 | 2 | – | 0.484 |
| DBSCAN | Non | eps=1.25, min_samples=14 | 2 | 9.0 % | 0.538 |
| **DBSCAN** | **Oui** | **eps=0.026, min_samples=9** | **2** | **18.6 %** | **0.549** |

➡️ **DBSCAN obtient le meilleur score de silhouette**, avec ou sans normalisation — il devance systématiquement KMeans et la CAH.

Entre les deux variantes de DBSCAN, l'écart de performance est faible (0.549 contre 0.538), mais la version **sans normalisation** classe beaucoup moins de graines comme anomalies (9,0 % contre 18,6 %). C'est donc **DBSCAN sans normalisation** (`eps=1.25`, `min_samples=14`) qui a été retenu pour le déploiement : performance quasi identique, pour une application plus utile en pratique (moins de graines jugées « atypiques »).

> 💡 Fait notable : sur ce jeu de données, les trois algorithmes convergent naturellement vers **2 groupes** plutôt que vers un plus grand nombre — la structure des données semble dominée par un seul grand axe de séparation (probablement lié à la taille des graines).

## 📈 Résultats

Le clustering DBSCAN final identifie :

| Classe | Effectif | Interprétation |
|---|---|---|
| Classe 0 | 127 graines | **Petites graines** — surface et périmètre plus faibles |
| Classe 1 | 54 graines | **Grandes graines** — surface et périmètre plus élevés |
| -1 (bruit) | 18 graines | **Anomalie** — profil atypique, ne correspond à aucun groupe |

## 🚀 Déploiement

DBSCAN ne possède pas de méthode `predict()` native : la même stratégie que pour le projet de segmentation clients est réutilisée, basée sur les **points cœur** (*core samples*) mémorisés par le modèle :

1. La nouvelle graine est utilisée telle quelle (le modèle final n'utilise **pas** de normalisation).
2. On calcule sa distance euclidienne à tous les points cœur du modèle.
3. Si le point cœur le plus proche est à une distance ≤ `eps`, la graine hérite de la classe de ce point cœur.
4. Sinon, elle est classée comme **anomalie** (`-1`).

Cette logique est implémentée dans `app.py`.

## ⚙️ Installation et utilisation

### Prérequis
- Python 3.9+

### Étapes

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd Kmeans_On_Wheat_Seeds_Dataset

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre dans le navigateur. Il suffit de choisir un préremplissage (valeurs médianes ou exemple), d'ajuster les 7 caractéristiques de la graine, puis de cliquer sur **« Prédire la classe »** pour obtenir son groupe.

## 🛠️ Technologies utilisées

- **Python** — pandas, numpy
- **Scikit-learn** — `PCA`, `KMeans`, `AgglomerativeClustering`, `DBSCAN`, `normalize`, `silhouette_score`, `NearestNeighbors`
- **Yellowbrick** — `KElbowVisualizer`
- **Plotly** — visualisations interactives (notebook)
- **Streamlit** — interface web de l'application déployée
- **Joblib** — sérialisation du modèle

## 📌 Limites et pistes d'amélioration

- Les scores de silhouette obtenus (0.48–0.55) restent modérés : les groupes se recoupent partiellement, et 2 clusters ne rendent sans doute pas compte de toute la diversité morphologique réelle des graines.
- Le seuil de bruit toléré (25 %) lors de la recherche des paramètres DBSCAN est un choix arbitraire ; un seuil différent pourrait faire ressortir une autre combinaison `eps`/`min_samples`.
- Les noms des classes (« petites »/« grandes » graines) sont dérivés automatiquement du profil moyen (variable `area A`) ; ils restent une interprétation et pourraient être affinés avec une expertise agronomique.

## 👤 Contexte

Projet réalisé dans le cadre du programme **Sen HUB** — module Apprentissage non supervisé / Clustering.
