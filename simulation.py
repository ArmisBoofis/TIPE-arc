"""Ce fichier contient les fonctions permettant de tracer différents graphes :
 - force en fonction de l'allonge
 - déformation de l'arc
 - courbes f_1 = 0 et f_2 = 0 dans le plan (alpha, K)
"""

import json
from typing import Any, Callable, Mapping

import colorama
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g
from scipy.integrate import simpson

from outils_resolution import deformation_arc, solveur_2D
from progress_bar import progress_bar

# Initialisation de colorama
colorama.init()

# Paramètres pour l'affichage du texte dans les graphiques
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.size": 20})

# Caractéristiques retenues pour l'arc droit
arc_droit = {
    "b_max": 0.685,  # Allonge maximale
    "F_max": 7.5 * g,  # Force à l'allonge maximale
    "L": 0.55,  # Demie-longueur de l'arc
    "l": 0.52,  # Demie-longueur de la corde
    "theta_0": lambda _: 0,  # Déformation initiale (sans corde) de l'arc
    "W": lambda _: 18.9,  # Rigidité flexionnelle de l'arc
    "zone_recherche": [(0.0, 0.0), (1.0, 3.0)],
}

# Caractéristiques retenues pour l'arc recurve
L0 = 0.28  # Partie de la branche de l'arc recurve qui est particulièrement rigide
L = 0.863  # Demie-longueur de l'arc recurve
W_m, W_p = 100.0, 45.0  # Rigidité pour le manche, pour les branches

# On récupère la fonction theta_0 pour l'arc recurve depuis le fichier correspondant
with open("theta_0_arc_recurve.json", "r") as f:
    coefs = json.load(f)  # On récupère les coefficients du polynôme
    theta_0_recurve = np.poly1d(coefs)


def W_recurve(s):
    """Distribution de rigidité pour la branche de l'arc recurve"""
    # On fait une disjonction de cas selon que l'on se trouve sur la partie rigide de la branche ou non
    return W_m if s <= L0 else W_p


arc_recurve = {
    "b_max": 0.875,  # Allonge maximale
    "F_max": 13.5 * g,  # Force à l'allonge maximale
    "L": L,  # Demie-longueur de l'arc
    "l": 0.813,  # 0.76,  # Demie-longueur de la corde
    "theta_0": lambda s: theta_0_recurve(s),
    "W": W_recurve,  # Rigidité flexionnelle de l'arc
    "zone_recherche": [(0.0, 0.0), (0.75, 3.0)],
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
        "zone_recherche": arc["zone_recherche"],
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

    # On crée une zone d'affichage 3D
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Calcul de Zf et Zg
    for y in Y:
        for x in X:
            # Évaluation de f et g au point de coordonnées (x,y)
            zf, zg = f((x, y)), g((x, y))

            # On marque le point où les fonctions ont été calculées
            ax.scatter(x, y, zf, marker=".", color="blue", s=1.0, linewidth=0.5)
            ax.scatter(x, y, zg, marker=".", color="brown", s=1.0, linewidth=0.5)

            # On stocke les valeurs calculées dans les tableaux
            Zf.append(zf)
            Zg.append(zg)

    # On donne la bonne forme à X, Y, Zf, et Zg
    X, Y = np.meshgrid(X, Y)

    Zf = np.array(Zf).reshape((nb_points_2, nb_points_1))
    Zg = np.array(Zg).reshape((nb_points_2, nb_points_1))

    ax.plot_surface(X, Y, Zf, lw=0.5, alpha=0.2, color="blue", label=labelf)  # Surface pour <f_1>
    ax.plot_surface(X, Y, Zg, lw=0.5, alpha=0.2, color="brown", label=labelg)  # Surface pour <f_2>

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
    sol = solveur_2D(
        lambda p: f_1((0.0, p[1]), p[0], arc),
        lambda p: f_2((0.0, p[1]), p[0], arc),
        [(0.15, arc["zone_recherche"][0][1]), (1.0, arc["zone_recherche"][1][1])],
    )

    if sol is None:
        print("Le calcul du band a échoué.")
        return None

    # On génère les différentes valeurs de b pour lesquelles on va calculer la déformation
    B = np.linspace(sol[0] + 0.01, 1.0, nb_pos)

    for k in range(len(B)):
        # Affichage d'une barre de chargement (le calcul peut prendre du temps si <nb_pos> est grand)
        progress_bar(k, nb_pos, prefix=colorama.Fore.RESET + f"Calcul de la position n°{k}... ")

        # On commence par calculer les valeurs de alpha et K correspondantes
        sol = solveur_2D(lambda p: f_1(p, B[k], arc), lambda p: f_2(p, B[k], arc), arc["zone_recherche"])

        if sol is not None:
            alpha, K = sol  # On récupère les valeurs calculées pour alpha et K

            # On calcule la déformation
            deformation, sw, sol_sw = deformation_arc(alpha, K, B[k], arc, sol_complete=True)

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
                [sol_sw[1] * arc["b_max"], B[k] * arc["b_max"]],
                [sol_sw[2] * arc["b_max"], 0.0],
                linestyle="-",
                color="black",
                linewidth=0.5,
            )

    plt.gca().set_aspect("equal")  # Repère orthonormé
    plt.show()


def affiche_courbe_force_allonge(
    arc_1: Mapping[str, Any],  # Caractéristiques du premier arc
    donnees_exp_1: list[list[float], list[float]],  # Données expérimentales pour le premier arc
    arc_2: Mapping[str, Any],  # Second arc
    donnees_exp_2: list[list[float], list[float]],  # Second jeu de données expérimentales
    params_affichage_1=(
        True,
        True,
    ),  # Le premier élément commande l'affichage de la courbe expérimentale, le second celui de la modélisation
    params_affichage_2=(
        True,
        True,
    ),  # Paramètres d'affichage pour le second arc (même signification que pour le premier)
    nb_points=10,  # Nombre de points à calculer par la modélisation
    label_exp_1="Courbe expérimentale pour l'arc 1",  # Légende pour la courbe expérimentale de l'arc 1
    label_mod_1="Courbe issue de la modélisation pour l'arc 1",  # Légende pour la courbe modélisée de l'arc 1
    label_exp_2="Courbe expérimentale pour l'arc 2",  # Idem pour l'arc 2
    label_mod_2="Courbe issue de la modélisation pour l'arc 2",  # Idem pour l'arc 2
) -> None:
    """Fonction permettant d'afficher les courbes force - allonge issues de la modélisation pour deux arcs donnés,
    et de les superposer aux courbes expérimentales"""

    # On commence par calculer le band des arcs
    if params_affichage_1[1]:
        sol_band_1 = solveur_2D(
            lambda p: f_1((0.0, p[1]), p[0], arc_1),
            lambda p: f_2((0.0, p[1]), p[0], arc_1),
            [(0.15, arc_1["zone_recherche"][0][1]), (1.0, arc_1["zone_recherche"][1][1])],
        )

        if sol_band_1 is None:
            print("Le calcul du band pour l'arc 1 a échoué.")
            return None

        band_1 = sol_band_1[0]

    else:
        band_1 = 0.0

    if params_affichage_2[1]:
        sol_band_2 = solveur_2D(
            lambda p: f_1((0.0, p[1]), p[0], arc_2),
            lambda p: f_2((0.0, p[1]), p[0], arc_2),
            [(0.15, arc_2["zone_recherche"][0][1]), (1.0, arc_2["zone_recherche"][1][1])],
        )

        if sol_band_2 is None:
            print("Le calcul du band pour l'arc 2 a échoué.")
            return None

        band_2 = sol_band_2[0]

    else:
        band_2 = 0.0

    # On génère les différentes valeurs de b pour lesquelles on va calculer la force associée
    band = max(band_1, band_2)

    B = np.linspace(band + 0.01, 1.0, nb_points)
    X_1, F_1 = [], []  # Liste qui contiendront les résultats de la modélisation pour l'arc 1
    X_2, F_2 = [], []  # Idem pour l'arc 2

    for k in range(len(B)):
        # Affichage d'une barre de chargement (le calcul peut prendre du temps si <nb_pos> est grand)
        progress_bar(k, nb_points, prefix=colorama.Fore.RESET + f"Calcul du point {k}/{nb_points}... ")

        if params_affichage_1[1]:
            # On calcule la valeur de K correspondante
            sol_1 = solveur_2D(lambda p: f_1(p, B[k], arc_1), lambda p: f_2(p, B[k], arc_1), arc_1["zone_recherche"])

            if sol_1 is not None:
                alpha_1, K_1 = sol_1  # On récupère les valeurs calculées pour alpha et K

                # Calcul de l'allonge réelle en comptant à partir du band
                X_1.append((B[k] - band) * arc_1["b_max"])

                # On déduit F de K à partir de la condition d'équilibre de l'encoche
                # On redimensionne également la variable
                F_1.append(np.sin(alpha_1) * K_1 * arc_1["F_max"])

        # On recommence pour le second arc
        if params_affichage_2[1]:
            sol_2 = solveur_2D(lambda p: f_1(p, B[k], arc_2), lambda p: f_2(p, B[k], arc_2), arc_2["zone_recherche"])

            if sol_2 is not None:
                alpha_2, K_2 = sol_2
                X_2.append((B[k] - band) * arc_2["b_max"])
                F_2.append(np.sin(alpha_2) * K_2 * arc_2["F_max"])

    # On affiche les données qui le doivent
    if params_affichage_1[0]:
        #

        plt.plot(
            np.array(donnees_exp_1[0]) - donnees_exp_1[0][0],  # On compte à partir du band
            np.array(donnees_exp_1[1]) * g,
            linestyle="None",
            marker="+",
            color="#FF0015",
            label=label_exp_1,
        )

    if params_affichage_1[1]:
        plt.plot(X_1, F_1, linestyle="--", color="#7A000A", label=label_mod_1)

    if params_affichage_2[0]:
        plt.plot(
            np.array(donnees_exp_2[0]) - donnees_exp_2[0][0],  # On compte à partir du band
            np.array(donnees_exp_2[1]) * g,
            linestyle="None",
            marker="+",
            color="#121517",
            label=label_exp_2,
        )

    if params_affichage_2[1]:
        plt.plot(X_2, F_2, linestyle="--", color="#47555C", label=label_mod_2)

    plt.gca().set(xlabel="Allonge (m)", ylabel="Force (N)")

    plt.legend()
    plt.show()


def affiche_courbe_energie(
    arc_1: Mapping[str, Any],  # Caractéristiques du premier arc
    donnees_exp_1: list[list[float], list[float]],  # Données expérimentales pour le premier arc
    arc_2: Mapping[str, Any],  # Second arc
    donnees_exp_2: list[list[float], list[float]],  # Second jeu de données expérimentales
    params_affichage_1=(
        True,
        True,
    ),  # Le premier élément commande l'affichage de la courbe expérimentale, le second celui de la modélisation
    params_affichage_2=(
        True,
        True,
    ),  # Paramètres d'affichage pour le second arc (même signification que pour le premier)
    nb_points=10,  # Nombre de points à calculer par la modélisation
    label_exp_1="Courbe expérimentale pour l'arc 1",  # Légende pour la courbe expérimentale de l'arc 1
    label_mod_1="Courbe issue de la modélisation pour l'arc 1",  # Légende pour la courbe modélisée de l'arc 1
    label_exp_2="Courbe expérimentale pour l'arc 2",  # Idem pour l'arc 2
    label_mod_2="Courbe issue de la modélisation pour l'arc 2",  # Idem pour l'arc 2
) -> None:
    """Fonction permettant d'afficher les courbes énergie - allonge issues de la modélisation
    pour deux arcs donnés, et de les superposer aux courbes expérimentales"""

    # On commence par calculer le band des arcs
    if params_affichage_1[1]:
        sol_band_1 = solveur_2D(
            lambda p: f_1((0.0, p[1]), p[0], arc_1),
            lambda p: f_2((0.0, p[1]), p[0], arc_1),
            [(0.15, arc_1["zone_recherche"][0][1]), (1.0, arc_1["zone_recherche"][1][1])],
        )

        if sol_band_1 is None:
            print("Le calcul du band pour l'arc 1 a échoué.")
            return None

        band_1 = sol_band_1[0]

    else:
        band_1 = 0.0

    if params_affichage_2[1]:
        sol_band_2 = solveur_2D(
            lambda p: f_1((0.0, p[1]), p[0], arc_2),
            lambda p: f_2((0.0, p[1]), p[0], arc_2),
            [(0.15, arc_2["zone_recherche"][0][1]), (1.0, arc_2["zone_recherche"][1][1])],
        )

        if sol_band_2 is None:
            print("Le calcul du band pour l'arc 2 a échoué.")
            return None

        band_2 = sol_band_2[0]

    else:
        band_2 = 0.0

    # On génère les différentes valeurs de b pour lesquelles on va calculer la force associée
    band = max(band_1, band_2)

    B = np.linspace(band + 0.01, 1.0, nb_points)
    X_1, F_1 = [], []  # Liste qui contiendront les résultats de la modélisation pour l'arc 1
    X_2, F_2 = [], []  # Idem pour l'arc 2

    for k in range(len(B)):
        # Affichage d'une barre de chargement (le calcul peut prendre du temps si <nb_pos> est grand)
        progress_bar(k, nb_points, prefix=colorama.Fore.RESET + f"Calcul du point {k}/{nb_points}... ")

        if params_affichage_1[1]:
            # On calcule la valeur de K correspondante
            sol_1 = solveur_2D(lambda p: f_1(p, B[k], arc_1), lambda p: f_2(p, B[k], arc_1), arc_1["zone_recherche"])

            if sol_1 is not None:
                alpha_1, K_1 = sol_1  # On récupère les valeurs calculées pour alpha et K

                # Calcul de l'allonge réelle en comptant à partir du band
                X_1.append((B[k] - band) * arc_1["b_max"])

                # On déduit F de K à partir de la condition d'équilibre de l'encoche
                # On redimensionne également la variable
                F_1.append(np.sin(alpha_1) * K_1 * arc_1["F_max"])

        # On recommence pour le second arc
        if params_affichage_2[1]:
            sol_2 = solveur_2D(lambda p: f_1(p, B[k], arc_2), lambda p: f_2(p, B[k], arc_2), arc_2["zone_recherche"])

            if sol_2 is not None:
                alpha_2, K_2 = sol_2
                X_2.append((B[k] - band) * arc_2["b_max"])
                F_2.append(np.sin(alpha_2) * K_2 * arc_2["F_max"])

    # On affiche les données qui le doivent
    if params_affichage_1[0]:
        plt.plot(
            np.array(donnees_exp_1[0])[:-1] - donnees_exp_1[0][0],  # On compte à partir du band
            [
                simpson(np.array(donnees_exp_1[1])[:k] * g, x=(np.array(donnees_exp_1[0]) - donnees_exp_1[0][0])[:k])
                for k in range(1, len(donnees_exp_1[0]))
            ],  # On intègre l'échantillon de courbe expérimentale à l'aide de la fonction <simpson>
            linestyle="None",
            marker="+",
            color="#FF0015",
            label=label_exp_1,
        )

    if params_affichage_1[1]:
        plt.plot(
            X_1[:-1],
            [simpson(F_1[:k], x=X_1[:k]) for k in range(1, len(X_1))],
            linestyle="--",
            color="#7A000A",
            label=label_mod_1,
        )

    if params_affichage_2[0]:
        plt.plot(
            np.array(donnees_exp_2[0])[:-1] - donnees_exp_2[0][0],  # On compte à partir du band
            [
                simpson(np.array(donnees_exp_2[1])[:k] * g, x=(np.array(donnees_exp_2[0]) - donnees_exp_2[0][0])[:k])
                for k in range(1, len(donnees_exp_2[0]))
            ],
            linestyle="None",
            marker="+",
            color="#121517",
            label=label_exp_2,
        )

    if params_affichage_2[1]:
        plt.plot(
            X_2[:-1],
            [simpson(F_2[:k], x=X_2[:k]) for k in range(1, len(X_2))],
            linestyle="--",
            color="#47555C",
            label=label_mod_2,
        )

    plt.gca().set(xlabel="Allonge (m)", ylabel="Énergie stockée (J)")

    plt.legend()
    plt.show()


arc_droit_ad = adimensionne(arc_droit)
arc_recurve_ad = adimensionne(arc_recurve)

with open("donnees_experiences.json", "r") as f:
    donnees_exp = json.load(f)

# affiche_courbe_force_allonge(
#     arc_droit_ad,
#     donnees_exp["arc_droit"],
#     arc_recurve_ad,
#     donnees_exp["arc_recurve"],
#     params_affichage_1=(True, True),
#     params_affichage_2=(True, True),
#     nb_points=50,
#     label_exp_1="Courbe expérimentale pour l'arc droit",
#     label_exp_2="Courbe expérimentale pour l'arc recurve",
#     label_mod_1="Courbe issue de la modélisation pour l'arc droit",
#     label_mod_2="Courbe issue de la modélisation pour l'arc recurve",
# )

# affiche_courbe_energie(
#     arc_droit_ad,
#     donnees_exp["arc_droit"],
#     arc_recurve_ad,
#     donnees_exp["arc_recurve"],
#     params_affichage_1=(True, True),
#     params_affichage_2=(True, True),
#     nb_points=100,
#     label_exp_1="Courbe expérimentale pour l'arc droit",
#     label_exp_2="Courbe expérimentale pour l'arc recurve",
#     label_mod_1="Courbe issue de la modélisation pour l'arc droit",
#     label_mod_2="Courbe issue de la modélisation pour l'arc recurve",
# )

b = 0.80

affichage_surfaces_3D(
    lambda p: f_1(p, b, arc_recurve_ad),
    lambda p: f_2(p, b, arc_recurve_ad),
    (0.0, 0.0),
    (1.0, 4.0),
    nb_points_1=30,
    nb_points_2=30,
    label1="$\\alpha$",
    label2="$K$",
    labelf="$f_1$",
    labelg="$f_2$",
)
