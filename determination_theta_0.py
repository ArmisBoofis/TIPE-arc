"""
 Ce fichier comprend les scripts permettant :
 - de donner une équation cartésienne de la courbure de l'arc en utilisant
   l'interpolation <scipy.interpolate.CubicSpline> 
 - en déduire une équation paramétrée de la courbure de l'arc par son abscisse curviligne
 - en déduire la fonction theta_0 en fonction de l'abscisse curviligne
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

# ÉQUATION CARTÉSIENNE

# Dimensions de l'image source
hauteur, largeur = 3024, 4032

# Liste de points obtenus à partir d'une image (pointage manuel)
points = np.array(
    [
        [47, 1341],
        [168, 1421],
        [318, 1471],
        [455, 1469],
        [653, 1415],
        [917, 1294],
        [1141, 1178],
        [1303, 1105],
        [1430, 1055],
        [1633, 980],
        [1816, 935],
        [1930, 914],
        [2053, 903],
    ]
)

min_abs, max_abs = np.min(points[:, 0]), np.max(points[:, 0])
plage_x = np.arange(min_abs, max_abs, 0.1)

# On retourne les points et on recale la courbe le long des axes
points[:, 1] = hauteur - points[:, 1]

min_ord = np.min(points[:, 1])

points[:, 0] -= min_abs
points[:, 1] -= min_ord

max_abs -= min_abs

# # Interpolation
Y_cartesien = CubicSpline(points[:, 0], points[:, 1])

# ÉQUATION PARAMÉTRÉE

dist = lambda x: np.sqrt(x**2 + Y_cartesien(x, 1) ** 2)  # Fonction distance

# On calcule <Y_curviligne>, <X_curviligne> et <S> sur un ensemble restreint de points, puis on va interpoler
Y_curviligne, X_curviligne, S = (
    np.empty((0, 2)),
    np.empty((0, 2)),
    np.empty((0, 2)),
)

for x in np.arange(min_abs, max_abs, 50):
    s = quad(dist, 0, x)[0]

    S = np.vstack((S, np.array([[x, s]])))
    X_curviligne = np.vstack((X_curviligne, np.array([[s, x]])))
    Y_curviligne = np.vstack((Y_curviligne, np.array([[s, Y_cartesien(x)]])))

S = CubicSpline(S[:, 0], S[:, 1])
X_curviligne = CubicSpline(X_curviligne[:, 0], X_curviligne[:, 1])
Y_curviligne = CubicSpline(Y_curviligne[:, 0], Y_curviligne[:, 1])

plt.plot(plage_x, Y_cartesien(plage_x, 1))

plt.show()
