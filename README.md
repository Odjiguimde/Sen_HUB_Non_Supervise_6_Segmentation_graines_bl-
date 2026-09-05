# 🌾 Segmentation des graines de blé — Apprentissage non supervisé (KMeans)

Projet de **clustering de graines de blé** à partir de leurs caractéristiques morphologiques, réalisé dans le cadre du module *Unsupervised Learning* du **Sen HUB**. Le modèle **KMeans** est entraîné, évalué, puis **déployé sous forme d'application web interactive** (Streamlit) permettant d'attribuer un groupe à une nouvelle graine à partir de ses mesures.

## 🎯 Objectif

Regrouper des graines de blé en fonction de leur morphologie (surface, périmètre, compacité, longueur/largeur du grain, etc.), sans étiquette préalable, afin de :

- identifier des familles de graines aux caractéristiques similaires,
- comprendre les variables qui différencient le plus les groupes,
- fournir un outil simple permettant de classer instantanément une nouvelle graine.

## 📁 Structure du dépôt

```
.
├── Kmeans_On_Wheat_Seeds_Dataset.ipynb   # Notebook complet : EDA, PCA, KMeans, évaluation, déploiement
├── wheat_seeds_dataset.csv               # Jeu de données brut
├── app.py                                # Application Streamlit de prédiction (modèle final : KMeans)
├── modele_kmeans.joblib                  # Modèle KMeans entraîné + métadonnées de déploiement
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

1. **Pas de normalisation** des variables : le clustering est effectué directement sur les mesures brutes (choix fait dans le notebook source).
2. **Réduction de dimension** : ACP (PCA) à 2 composantes pour la visualisation — **99,3 %** de la variance expliquée conservée.
3. **Recherche du nombre optimal de clusters** via la méthode du coude (courbe d'inertie + `KElbowVisualizer`) : **k = 5** retenu.
4. **Entraînement du modèle final** : `KMeans(n_clusters=5, init='random', max_iter=500)`.
5. **Évaluation** : score de silhouette ≈ **0.37**.

> Contrairement au projet de segmentation clients (DBSCAN), un seul algorithme (KMeans) est étudié ici — c'est donc naturellement le modèle déployé. KMeans a l'avantage de posséder une méthode `predict()` native (chaque cluster a un centre), ce qui simplifie grandement le déploiement par rapport à DBSCAN.

## 📈 Résultats

La partition finale répartit les 199 graines en **5 groupes**, nommés automatiquement selon leur taille moyenne (surface `area A`) :

| Groupe | Effectif | Profil |
|---|---|---|
| Très petites graines | 53 | Surface et périmètre les plus faibles |
| Petites graines | 30 | Surface faible |
| Graines de taille moyenne | 44 | Profil intermédiaire |
| Grandes graines | 25 | Surface élevée |
| Très grandes graines | 47 | Surface et périmètre les plus élevés |

*(effectifs approximatifs, peuvent varier légèrement selon l'initialisation aléatoire du KMeans)*

## 🚀 Déploiement

Le déploiement est plus direct que pour un modèle DBSCAN : KMeans calcule un **centre par cluster**, donc classer une nouvelle graine revient simplement à appeler `modele.predict(nouvelle_graine)`, qui renvoie le cluster dont le centre est le plus proche.

Étapes suivies :
1. Calcul du **profil moyen** de chaque cluster (`df.groupby(partition).mean()`).
2. **Nommage automatique** des classes en les triant par surface moyenne croissante (« Très petites graines » → « Très grandes graines »).
3. Sauvegarde d'un seul fichier `modele_kmeans.joblib` contenant : le modèle entraîné, la liste des colonnes, les valeurs médianes (préremplissage), 3 exemples, et les noms des classes.
4. `app.py` charge ces artefacts et propose un formulaire Streamlit pour tester une nouvelle graine.

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
- **Scikit-learn** — `PCA`, `KMeans`, `silhouette_score`
- **Yellowbrick** — `KElbowVisualizer` (choix du nombre de clusters)
- **Plotly** — visualisations interactives (notebook)
- **Streamlit** — interface web de l'application déployée
- **Joblib** — sérialisation du modèle

## 📌 Limites et pistes d'amélioration

- Le score de silhouette (≈0.37) est modéré : une normalisation des variables (comme dans le projet de segmentation clients) pourrait améliorer la séparation des clusters, les variables n'étant pas à la même échelle (`area A` ~10–21 vs `compactness` ~0.8–0.92).
- Un seul algorithme a été testé ici ; comparer KMeans à une CAH ou un DBSCAN (comme dans le projet clients) permettrait de confirmer si k=5 est réellement le meilleur choix.
- Les noms des classes (« petites »/« grandes » graines) sont dérivés automatiquement du profil moyen et restent une interprétation ; ils pourraient être affinés avec une expertise agronomique.

## 👤 Contexte

Projet réalisé dans le cadre du programme **Sen HUB** — module Apprentissage non supervisé / Clustering.
