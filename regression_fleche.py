"""Ce fichier contient le code réalisant la régression linéaire
permettant de déterminer la constante EI pour la branche de l'arc droit."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import (
    LinearRegression,
)  # Permet de réaliser la régression linéaire
from sklearn.metrics import r2_score  # Permet de calculer le coefficient de corrélation

g = 9.812  # Valeur approchée de la constante de pesanteur g à Paris

charges = np.linspace(0.5, 5.0, 10)  # Liste des charges appliquées, en kilogramme
fleches = np.array(
    [
        1.5,
        2.4,
        3.9,
        4.8,
        6.1,
        7.1,
        8.3,
        9.6,
        10.2,
        11.6,
    ]
)  # Liste des flèches correspondantes en centimètres

# On convertit les charges en Newtons, et les flèches en mètres
charges *= g
fleches *= 10 ** (-2)

# On effectue la régression linéaire :
modele_regression = LinearRegression(
    fit_intercept=False
)  # Initialisation du modèle de régression linéaire (on prend 0 comme ordonnée à l'origine)

modele_regression.fit(
    charges.reshape(-1, 1), fleches.reshape(-1, 1)
)  # Entraînement du modèle
prediction_fleches = modele_regression.predict(
    charges.reshape(-1, 1)
)  # Calcul de la régression
r2 = r2_score(fleches, prediction_fleches)  # Calcul du coefficient de corrélation

# On affiche le résultat de la régression
print("Pente :", modele_regression.coef_[0, 0])
print("Ordonnée à l'origine :", modele_regression.intercept_)
print("r^2 :", r2)

# Affichage :
plt.plot(
    charges,
    fleches,
    marker="+",
    linestyle="None",
    color="k",
    label="Données expérimentales",
)
plt.plot(
    charges,
    prediction_fleches,
    linestyle="--",
    color="#9F5941",
    label="Régression linéaire",
)

# Légende pour les axes
plt.xlabel("Charge appliquée (en Newtons)")
plt.ylabel("Flèche (en mètres)")

# Affichage de la légende
plt.legend()

plt.show()
