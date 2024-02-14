"""Ce fichier contient le script permettant de résoudre numériquement
les équations du modèle de déformation statique de l'arc."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Données décrivant l'arc droit
L = 0.648  # Longueur en mètres d'une branche de l'arc droit (moitié de la longueur totale)
l = 1.18  # Longueur en mètre de la corde de l'arc droit

theta_0 = lambda _: 0  # Sans corde, l'arc droit est une poutre plate, non déformée
W = lambda _: 18.9  # Rigidité flexionnelle supposée constante pour l'arc droit


def calcul_deformation(K, alpha, b):
    """Fonction calculant la déformation de l'arc qui correspondrait à un couple
    (force, angle, allonge) valant <K>, <alpha> et <b> respectivement."""

    # Fonction décrivant le membre de droite du système différentiel
    # L'équation différentielle porte sur le vecteur Y = (phi, x, y)
    F = lambda s, Y: np.array(
        [
            K / W(s) * ((b - Y[1]) * np.cos(alpha) - Y[2] * np.sin(alpha)),
            np.sin(Y[0] + theta_0(s)),
            np.cos(Y[0] + theta_0(s)),
        ]
    )

    # "Événement" permettant d'interrompre l'intégration en cours de route
    # Si phi(s) + alpha + theta_0(s) = 0, on a trouvé s_w et on peut s'arrêter
    parallele = lambda s, Y: Y[0] + alpha + theta_0(s)

    # Condition initiale : phi = x = y = 0
    Y_0 = np.array([0, 0, 0])

    # On intègre les équations pour 0 <= s <= 1 (s est une grandeur non dimensionnée)
    res = solve_ivp(F, [0, 1], Y_0, events=parallele, dense_output=True)

    # On retourne la solution <sol>
    # La variable <success> vaut True si l'intégration s'est déroulée jusqu'au bout de l'intervalle indiqué
    return res.sol, res.success
