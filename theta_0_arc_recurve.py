"""Ce fichier a pour but d'exploiter le fichier <courbure_arc_recurve.svg>
afin de déterminer la fonction <theta_0> pour l'arc recurve."""

import json

import matplotlib.pyplot as plt
import numpy as np
from svgpathtools import svg2paths

plt.rcParams["text.usetex"] = True

# On récupère la courbure de l'arc à partir du fichier
courbure_path = svg2paths("courbure_arc_recurve.svg")[0][0]

# Affichage de la courbure
N, L_v, L_r = 1000, courbure_path.length(), 0.863
T, theta_0, S = np.linspace(0.0, 1.0, N), [], []
deg = 10  # Degré du polynôme qui réalise l'approximation finale

for t in T:
    # On récupère le paramètre correspondant à l'abscisse curviligne t * L_v
    t_geo = courbure_path.ilength(t * L_v)
    S.append(t * L_r)

    # Calcul de l'angle correspondant
    u = courbure_path.unit_tangent(t_geo)
    theta_0.append(np.arctan2(u.imag, u.real))

# On exclut le dernier point, qui est incorrect
theta_0, S = np.array(theta_0[:-1]), np.array(S[:-1])

# La solution obtenue présente des discontinuités en raison de la modélisation par morceaux de l'arc
# On l'approxime donc par un polynôme pour obtenir quelque chose de plus régulier
theta_0_p = np.poly1d(np.polyfit(S, theta_0, deg))

X = np.linspace(0.0, L_r, 1000)
Y = theta_0_p(X)

plt.plot(X, Y, color="brown", label="Approximation", linestyle="--")
plt.plot(S, theta_0, color="black", label="Solution exacte", linewidth=1)

plt.gca().set(xlabel="$s$ (cm)", ylabel="$\\theta_{0,r}(s)$ (rad)")

plt.legend()
plt.show()

# # Enfin, on enregistre les coefficients du polynôme dans un fichier pour un usage ultérieur
# with open("theta_0_arc_recurve.json", "w") as f:
#     json.dump(theta_0_p.coef.tolist(), f)
