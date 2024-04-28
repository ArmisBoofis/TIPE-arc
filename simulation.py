"""Ce fichier contient les fonctions permettant de tracer différents graphes :
 - force en fonction de l'allonge
 - déformation de l'arc
 - courbes f_1 = 0 et f_2 = 0 dans le plan (alpha, K)
"""

import json
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g

from outils_resolution import deformation_arc, dichotomie_2D

# Caractéristiques retenues pour l'arc droit
arc_droit = {
    "b_max": 0.685,  # Allonge maximale
    "F_max": 7.5 * g,  # Force à l'allonge maximale
    "L": 0.55,  # Demie-longueur de l'arc
    "l": 0.52,  # Demie-longueur de la corde
    "theta_0": lambda _: 0,  # Déformation initiale (sans corde) de l'arc
    "W": lambda _: 18.9,  # Rigidité flexionnelle de l'arc
}

# Caractéristiques retenues pour l'arc recurve
L0 = 0.28  # Partie de la branche de l'arc recurve qui est particulièrement rigide
L = 0.863  # Demie-longueur de l'arc recurve

# On récupère la fonction theta_0 pour l'arc recurve depuis le fichier correspondant
with open("theta_0_arc_recurve.json", "r") as f:
    coefs = json.load(f)  # On récupère les coefficients du polynôme
    theta_0_recurve = np.poly1d(coefs)


def W_recurve(s):
    """Distribution de rigidité pour la branche de l'arc recurve"""

    # On fait une disjonction de cas selon que l'on se trouve sur la partie rigide de la branche ou non
    if s <= L0:
        return 10e3

    else:
        return max(25 * ((L - s) / (L - L0)) ** 2 + 25, 1)


arc_recurve = {
    "b_max": 0.875,  # Allonge maximale
    "F_max": 13.5 * g,  # Force à l'allonge maximale
    "L": L,  # Demie-longueur de l'arc
    "l": 0.80,  # 0.76,  # Demie-longueur de la corde
    "theta_0": lambda s: theta_0_recurve(s),
    "W": W_recurve,  # Rigidité flexionnelle de l'arc
}


def adimensionne(arc: Mapping[str, Any]) -> Mapping[str, Any]:
    """Prend en paramètre un dictionnaire contenant les caractéristiques d'un
    arc, et renvoyant un dictionnaire semblable contenant les valeurs adimensionnées
    correspondantes (afin de faire ensuite des calculs dessus)."""

    return {
        "b_max": arc["b_max"],
        "F_max": arc["F_max"],
        "L": arc["L"] / arc["b_max"],
        "l": arc["l"] / arc["b_max"],
        "theta_0": arc["theta_0"],
        "W": lambda s: arc["W"](s) / (arc["F_max"] * arc["b_max"] ** 2),
    }


# Fonctions f_1 et f_2, utilisée pour le calcul des équilibres
def f_1(params: tuple[float, float], b: float, arc: Mapping[str, Any]) -> float:
    """Fonction calculant la quantité f_1 pour un jeu de paramètres donné:
    - <params[0]> : angle alpha formé par la corde et la verticale au niveau de l'encoche
    - <params[1]> : force K de tension dans la corde
    - <b> : allonge
    - <arc> : caractéristiques de l'arc
    """

    alpha, K = params  # On nomme les paramètres
    _, sol_sw = deformation_arc(alpha, K, b, arc)  # Calcul de la déformation associée

    # On retourne la quantité voulue
    return (sol_sw[1] - b) * np.cos(alpha) + sol_sw[2] * np.sin(alpha)


def f_2(params: tuple[float, float], b: float, arc: Mapping[str, Any]) -> float:
    """Fonction calculant la quantité f_2 pour un jeu de paramètres donné:
    - <params[0]> : angle alpha formé par la corde et la verticale au niveau de l'encoche
    - <params[1]> : force K de tension dans la corde
    - <b> : allonge
    - <arc> : caractéristiques de l'arc
    """

    alpha, K = params  # On nomme les paramètres
    sw, sol_sw = deformation_arc(alpha, K, b, arc)  # Calcul de la déformation associée

    # On retourne la quantité voulue
    return sol_sw[2] - (arc["l"] - arc["L"] + sw) * np.cos(alpha)


def affichage_surfaces_3D(
    f: Callable[[float, float], float],
    g: Callable[[float, float], float],
    bg: tuple[float, float],  # Points en bas à gauche du domaine
    hd: tuple[float, float],
    nb_points_1=20,  # Nombre de subdivisions selon le premier axe
    nb_points_2=20,  # Nombre de subdivisions selon le second axe
    label1="",  # Nom du premier axe
    label2="",  # Nom du second axe
    labelf="",  # Légende pour le graphe de f
    labelg="",  # Légende pour le graphe de g
):
    """Fonction affichant les graphes superposés en 3 dimensions de deux fonctions f et g (de deux variables),
    sur un domaine de R^2 spécifié par l'utilisateur. Les lieux des points f = 0 et g = 0 sont également représentés.
    """

    X = np.linspace(bg[0], hd[0], nb_points_1)  # Subdivision du premier axe
    Y = np.linspace(bg[1], hd[1], nb_points_2)  # Subdivision du second axe

    Zf, Zg = [], []  # Liste contenant les cotes des points des surfaces pour f et g

    # Calcul de Zf et Zg
    for y in Y:
        for x in X:
            Zf.append(f((x, y)))
            Zg.append(g((x, y)))

    # On donne la bonne forme à X, Y, Zf, et Zg
    X, Y = np.meshgrid(X, Y)

    Zf = np.array(Zf).reshape((nb_points_2, nb_points_1))
    Zg = np.array(Zg).reshape((nb_points_2, nb_points_1))

    # On crée une zone d'affichage 3D
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot_surface(X, Y, Zf, lw=0.5, alpha=0.5, color="blue", label=labelf)  # Surface pour <f_1>
    ax.plot_surface(X, Y, Zg, lw=0.5, alpha=0.5, color="brown", label=labelg)  # Surface pour <f_2>

    # Affichage de l'intersection des surfaces avec le plan z = 0
    ax.contour(X, Y, Zf, zdir="z", offset=-1.5, levels=[0], colors=["blue"])
    ax.contour(X, Y, Zf, levels=[0], colors=["blue"])

    ax.contour(X, Y, Zg, zdir="z", offset=-1.5, levels=[0], colors=["brown"])
    ax.contour(X, Y, Zg, levels=[0], colors=["brown"])

    # On affiche les différentes légendes
    ax.set(xlabel=label1, ylabel=label2)
    ax.set_zlim(zmin=-1.5)

    if labelf != "" or labelg != "":  # On ne veut pas voir de légende dans le cas où rien n'a été spécifié
        ax.legend()

    plt.show()


def affichage_deformation(arc: Mapping[str, Any], nb_pos=5, nb_points_branche=200) -> None:
    """Fonction permettant d'afficher la déformation d'un arc pour un
    certain nombre d'allonges différentes.
     - <arc> : caractéristiques de l'arc
     - <nb_pos> (optionnel) : nombre de positions à afficher
    """

    # On commence par calculer le band de l'arc
    sol = dichotomie_2D(
        lambda p: f_1((0.0, p[1]), p[0], arc),
        lambda p: f_2((0.0, p[1]), p[0], arc),
        (0.15, 0.0),
        (1.0, 10),
        sol_unique=True,
    )

    if sol is not None:
        band, _ = sol

    else:
        return None  # Si on n'arrive pas à déterminer le band, on s'arrête là

    # On génère les différentes valeurs de b pour lesquelles on va calculer la déformation
    B = np.linspace(band + 0.01, 1.0, nb_pos)

    for b in B:
        print("Calcul de la solution pour b =", b)

        # On commence par calculer les valeurs de alpha et K correspondantes
        sol = dichotomie_2D(
            lambda p: f_1(p, b, arc), lambda p: f_2(p, b, arc), (0.0, 0.0), (np.pi / 2, 10.0), sol_unique=True
        )

        if sol is not None:
            alpha, K = sol

            # On calcule la déformation
            deformation, sw, sol_sw = deformation_arc(alpha, K, b, arc, sol_complete=True)

            # Ensemble des abscisses curvilignes
            S = np.linspace(0.0, arc["L"], nb_points_branche)
            X, Y = [], []  # Points formant la branche
            Xc, Yc = [], []  # Points formant la partie de la corde collée à la branche

            for s in S:
                point = deformation(s)

                X.append(point[1] * arc["b_max"])
                Y.append(point[2] * arc["b_max"])

                if s >= sw:
                    Xc.append(point[1] * arc["b_max"])
                    Yc.append(point[2] * arc["b_max"])

            # On affiche la branche et la corde
            plt.plot(X, Y, linestyle="-", color="brown")
            plt.plot(Xc, Yc, linestyle="-", color="black", linewidth=0.5)
            plt.plot(
                [sol_sw[1] * arc["b_max"], b * arc["b_max"]],
                [sol_sw[2] * arc["b_max"], 0.0],
                linestyle="-",
                color="black",
                linewidth=0.5,
            )

        else:
            print("Échec pour b=", b)

    plt.gca().set_aspect("equal")  # Repère orthonormé
    plt.show()


def affiche_courbe_force_allonge(
    arc: Mapping[str, Any], donnees_exp: list[list[float], list[float]], nb_points=10
) -> None:
    """Fonction permettant d'afficher la courbe force / allonge théorique d'un arc, et
    à la superposer à la courbe expérimentale."""

    # On commence par calculer le band de l'arc
    sol = dichotomie_2D(
        lambda p: f_1((0.0, p[1]), p[0], arc),
        lambda p: f_2((0.0, p[1]), p[0], arc),
        (0.15, 0.0),
        (1.0, 10),
        sol_unique=True,
    )

    if sol is not None:
        band, _ = sol

    else:
        return None  # Si on n'arrive pas à déterminer le band, on s'arrête là

    # On génère les différentes valeurs de b pour lesquelles on va calculer la force associée
    B = np.linspace(band + 0.01, 1.0, nb_points)
    X, F = [], []  # Liste qui contiendra les valeurs des forces correspondantes

    for b in B:
        print("Calcul de la solution pour b =", b)

        # On calcule la valeur de K correspondante
        sol = dichotomie_2D(
            lambda p: f_1(p, b, arc), lambda p: f_2(p, b, arc), (0.0, 0.0), (np.pi / 2, 10.0), sol_unique=True
        )

        if sol is not None:
            alpha, K = sol

            # Calcul de l'allonge réelle en comptant à partir du band
            X.append((b - band) * arc["b_max"])

            # On déduit F de K à partir de la condition d'équilibre de l'encoche
            # Il ne faut pas oublier non plus de redimensionner la variable
            F.append(np.sin(alpha) * K * arc["F_max"])

        else:
            print("Échec pour b =", b)

    # On affiche les points calculés
    plt.plot(X, F, linestyle="--", color="#9F5941", label="Courbe théorique")
    plt.plot(
        donnees_exp[0],
        np.array(donnees_exp[1]) * g,
        linestyle="None",
        marker="+",
        color="black",
        label="Mesures expérimentales",
    )

    plt.gca().set(xlabel="Allonge (cm)", ylabel="Force (N)")

    plt.legend()
    plt.show()


arc_droit_ad = adimensionne(arc_droit)
arc_recurve_ad = adimensionne(arc_recurve)

# On récupère les données expérimentales stockées dans le fichier
with open("donnees_experiences.json", "r") as f:
    donnees_exp = json.load(f)

# Affichage des résultats pour l'arc droit
# affiche_courbe_force_allonge(arc_droit_ad, donnees_exp["arc_droit"], nb_points=40)
affiche_courbe_force_allonge(arc_recurve_ad, donnees_exp["arc_recurve"])
