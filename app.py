import numpy as np
import joblib
import streamlit as st

# Charger le modèle sauvegardé
artefacts = joblib.load("modele_dbscan.joblib")
points_coeur = artefacts["points_coeur"]
labels_coeur = artefacts["labels_coeur"]
eps = artefacts["eps"]
colonnes = artefacts["colonnes"]
valeurs_defaut = artefacts["valeurs_defaut"]
exemples = artefacts["exemples"]
noms_classes = artefacts["noms_classes"]


def predire_classe(nouvelle_graine):
    graine = np.array(nouvelle_graine, dtype=float).reshape(1, -1)
    distances = np.linalg.norm(points_coeur - graine, axis=1)
    plus_proche = distances.argmin()
    if distances[plus_proche] <= eps:
        return labels_coeur[plus_proche]
    else:
        return -1


# Configuration de la page
st.set_page_config(page_title="Segmentation des graines de blé", page_icon="🌾")

st.title("Segmentation des graines de blé - modèle DBSCAN")
st.write("Saisissez les caractéristiques morphologiques d'une graine pour connaître sa classe.")

# Choix d'un exemple pour préremplir le formulaire
choix = st.selectbox(
    "Pré-remplir le formulaire",
    ["Valeurs médianes", "Exemple 1", "Exemple 2", "Exemple 3"],
)

if choix == "Valeurs médianes":
    defauts = valeurs_defaut
else:
    defauts = exemples[int(choix[-1]) - 1]

# Un champ de saisie par variable, sur deux colonnes
valeurs = []
colonne_gauche, colonne_droite = st.columns(2)

for i, (col, val) in enumerate(zip(colonnes, defauts)):
    zone = colonne_gauche if i % 2 == 0 else colonne_droite
    valeurs.append(zone.number_input(col, value=float(val)))

# Bouton de prédiction
if st.button("Prédire la classe"):
    classe = predire_classe(valeurs)

    if classe == -1:
        st.warning("Graine atypique (anomalie) : elle ne ressemble à aucune classe connue.")
    else:
        st.success(f"Cette graine appartient à la classe : **{noms_classes[classe]}**")
