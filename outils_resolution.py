"""Ce fichier contient les fonctions permettant de résoudre numériquement
les équations du modèle de l'arc."""

from typing import Any, Callable, Mapping

import numpy as np
from scipy import integrate, optimize


def deformation_arc(
    alpha: float, K: float, b: float, arc: Mapping[str, Any], sol_complete=False
) -> tuple[integrate.OdeSolution, float, list[float]] | tuple[float, list[float]]:
    """Fonction calculant la déformation de l'arc pour un jeu de paramètres donnés :
    - <alpha> : angle formé par la corde et la verticale au niveau de l'encoche
    - <K> : force de tension dans la corde
    - <b> : allonge
    - <arc> : paramètres de l'arc
    """
    # On récupère les paramètres intéressant de l'arc pour le calcul de la déformation
    W, theta_0, L = arc["W"], arc["theta_0"], arc["L"]

    # Fonction décrivant le membre de droite du système différentiel
    # L'équation différentielle porte sur le vecteur Y = (phi, x, y)
    F = lambda s, Y: np.array(
        [
            (K / W(s)) * ((b - Y[1]) * np.cos(alpha) - Y[2] * np.sin(alpha)),
            np.sin(Y[0] + theta_0(s)),
            np.cos(Y[0] + theta_0(s)),
        ]
    )

    # Condition initiale : phi = x = y = 0
    Y_0 = np.array([0, 0, 0])

    # "événement" permettant de récupérer l'abscisse curviligne de contact
    event_sw = lambda s, Y: Y[0] + alpha + theta_0(s)
    event_sw.terminal = False  # On n'arrête pas l'intégration si l'événement se produit

    # le paramètre <sol_complete> vaut False si on ne souhaite pas obtenir la déformation complète
    if sol_complete:
        # On intègre les équations pour 0 <= s <= L
        res = integrate.solve_ivp(F, [0, L], Y_0, dense_output=True, events=event_sw)
        sw, sol_sw, sol = L, res.sol(L), res.sol  # Assignation temporaire

        contact_corde = False  # Indique si s_w < L

        # On détermine si l'événement s'est produit (pour une abscisse sw suffisamment élevée)
        if len(res.t_events) > 0 and len(res.t_events[0] > 0):
            for k in range(len(res.t_events[0])):
                if res.t_events[0][k] >= L / 2:
                    contact_corde = True

                    sw = res.t_events[0][k]
                    sol_sw = res.y_events[0][k]

                    break

        # Si s_w < L, on doit reprendre l'intégration entre s_w et L en prenant phi = phi(s_w)
        if contact_corde:
            F = lambda s, Y: np.array(
                [
                    0,
                    np.sin(Y[0] + theta_0(s)),
                    np.cos(Y[0] + theta_0(s)),
                ]
            )

            res2 = integrate.solve_ivp(F, [sw, L], sol_sw, dense_output=True)  # Deuxième solution
            sol = lambda s: (res.sol(s) if s <= sw else res2.sol(s))  # On recolle les morceaux

        # On retourne la solution <sol>, l'abscisse curviligne de contact et la solution évaluée en l'abscisse curviligne de contact
        return sol, sw, sol_sw

    else:
        # On intègre comme avant, mais on ne s'intéresse qu'à sol(sw)
        res = integrate.solve_ivp(F, [0, L], Y_0, t_eval=[L], events=event_sw)
        sw, sol_sw = res.t[0], np.array(res.y).reshape(1, 3)[0]  # Assignation temporaire

        # On détermine si l'événement s'est produit (pour une abscisse sw suffisamment élevée)
        if len(res.t_events) > 0 and len(res.t_events[0] > 0):
            for k in range(len(res.t_events[0])):
                if res.t_events[0][k] >= L / 2:
                    sw = res.t_events[0][k]
                    sol_sw = res.y_events[0][k]

                    break

        # On retourne l'abscisse curviligne de contact et sol(L)
        return sw, sol_sw


def dichotomie_2D(
    f1: Callable[[float, float], float],
    f2: Callable[[float, float], float],
    bg: tuple[float, float],
    hd: tuple[float, float],
    evals1: tuple[float, float, float, float] = None,  # Dans le sens horaire : (f1(bg), f1(hg), f1(hd), f1(bd))
    evals2: tuple[float, float, float, float] = None,  # Idem pour f2
    epsilon=1e-11,  # Précision recherchée pour le maximum de la hauteur / largeur de la zone de recherche
    decoupage_initial=10,  # Ampleur du découpage initial
    sol_unique=False,  # Ne renvoie qu'une seule solution si True
) -> list[float] | None:
    """Fonction prenant en paramètre deux fonctions f1(x,y) et f2(x, y) et renvoyant
    un point (x_0, y_0) contenu dans la zone rectangulaire délimitée par les points bg
    (en bas à gauche) et hd (en haut à droite) tel que f1(x_0, y_0) = f2(x_0, y_0) = 0.
    On procède par dichotomie, en excluant les zones sans solutions."""

    # Si <evals_1> ou <evals_2> vaut None, il s'agit de l'appel initial
    # On découpe la grande zone de recherche en une centaine de plus petites zones
    if evals1 is None or evals2 is None:
        solutions = []  # Ensemble des solutions trouvées
        A = np.linspace(bg[0], hd[0], decoupage_initial)  # Plage de alpha
        K = np.linspace(bg[1], hd[1], decoupage_initial)  # Plage de K

        for i in range(decoupage_initial - 1):
            for j in range(decoupage_initial - 1):
                bg, hd = (A[i], K[j]), (A[i + 1], K[j + 1])
                hg, bd = (bg[0], hd[1]), (hd[0], bg[1])

                evals1 = (f1(bg), f1(hg), f1(hd), f1(bd))
                evals2 = (f2(bg), f2(hg), f2(hd), f2(bd))

                sol = dichotomie_2D(f1, f2, bg, hd, evals1, evals2)

                if sol is not None:
                    solutions.append(sol)

        if len(solutions) > 0:
            return solutions[0] if sol_unique else solutions

        else:
            return None

    # Pile des zones à traiter
    a_traiter = [(bg, hd, evals1, evals2)]

    # On continue tant qu'il y a des zones à traiter (et qu'on n'a pas trouver de solution satisfaisante)
    while len(a_traiter) > 0:
        # On récupère les paramètres de la zone à traiter
        bg, hd, evals1, evals2 = a_traiter.pop()

        # Largeur et hauteur de la zone de recherche
        largeur, hauteur = hd[0] - bg[0], hd[1] - bg[1]

        # Si l'on atteint la précision recherchée, on s'arrête là
        if max(largeur, hauteur) < epsilon:
            # On retourne le centre de la zone de recherche
            return (bg[0] + largeur / 2.0, bg[1] + hauteur / 2.0)

        else:
            # Points en haut à gauche et en bas à droite
            hg, bd = (bg[0], hd[1]), (hd[0], bg[1])

            # On découpe la zone en deux, dans le sens permettant d'équilibrer la largeur et la hauteur au mieux
            if largeur > hauteur:
                # Calcul des points médians
                mh, mb = ((bg[0] + hd[0]) / 2.0, hd[1]), ((bg[0] + hd[0]) / 2.0, bg[1])

                bg_1, hd_1 = bg, mh  # Demie-zone à gauche
                bg_2, hd_2 = mb, hd  # Demie-zone à droite

                # On évalue f1 et f2 en les points médians
                evals1_m, evals2_m = (f1(mb), f1(mh)), (f2(mb), f2(mh))

                evals1_1 = (evals1[0], evals1[1], evals1_m[1], evals1_m[0])  # Zone de gauche, f1
                evals2_1 = (evals2[0], evals2[1], evals2_m[1], evals2_m[0])  # Zone de gauche, f2

                evals1_2 = (evals1_m[0], evals1_m[1], evals1[2], evals1[3])  # Zone de droite, f1
                evals2_2 = (evals2_m[0], evals2_m[1], evals2[2], evals2[3])  # Zone de droite, f2

            else:
                # Calcul des points médians
                mg, md = (bg[0], (bg[1] + hg[1]) / 2.0), (hd[0], (bg[1] + hg[1]) / 2.0)

                bg_1, hd_1 = bg, md  # Demie-zone en bas
                bg_2, hd_2 = mg, hd  # Demie-zone en haut

                # On évalue f1 et f2 en les points médians
                evals1_m, evals2_m = (f1(mg), f1(md)), (f2(mg), f2(md))

                evals1_1 = (evals1[0], evals1_m[0], evals1_m[1], evals1[3])  # Zone du bas, f1
                evals2_1 = (evals2[0], evals2_m[0], evals2_m[1], evals2[3])  # Zone du bas, f2

                evals1_2 = (evals1_m[0], evals1[1], evals1[2], evals1_m[1])  # Zone du haut, f1
                evals2_2 = (evals2_m[0], evals2[1], evals2[2], evals2_m[1])  # Zone du haut, f2

            # On teste si les zones croisent les courbes d'équation f1 = 0 ou f2 = 0
            testeur = lambda som: (som[0] * som[1] <= 0) or (som[1] * som[2] <= 0) or (som[2] * som[3] <= 0)

            test1_1, test2_1 = testeur(evals1_1), testeur(evals2_1)  # Tests pour la première zone
            test1_2, test2_2 = testeur(evals1_2), testeur(evals2_2)  # Tests pour la deuxième zone

            # On répète le processus seulement pour les zones où un croisement est possible
            if test1_1 and test2_1:
                a_traiter.append((bg_1, hd_1, evals1_1, evals2_1))

            if test1_2 and test2_2:
                a_traiter.append((bg_2, hd_2, evals1_2, evals2_2))

    # Si on a épuisé les zones de recherche, c'est qu'il n'existe pas de solutions
    return None


def solveur_2D(
    f1: Callable[[float, float], float],  # Première fonction
    f2: Callable[[float, float], float],  # Deuxième fonction
    zone_recherche: list[tuple[float, float]],  # Zone de recherche pour la solution
    tol=1e-10,  # Précision du résultat
) -> tuple[float, float] | None:
    """Fonction permettant de résoudre un système de deux équations à deux inconnues.
    On utilise en premier lieu la fonction <root> du module <scipy.optimize>.
    En cas de non convergence ou résultat erroné, on passe à la méthode dichotomique, plus stable."""

    # On essaye dans un premier temps la méthode Newtonienne avec <scipy.optimize>
    milieu = [(zone_recherche[0][0] + zone_recherche[1][0]) / 2.0, (zone_recherche[0][1] + zone_recherche[1][1]) / 2.0]
    sol = optimize.root(
        lambda p: [f1(p), f2(p)], milieu, tol=tol
    )  # On prend comme point de départ le milieu de la zone de recherche

    if (
        sol.success
        and zone_recherche[0][0] <= sol.x[0] <= zone_recherche[1][0]
        and zone_recherche[0][1] <= sol.x[1] <= zone_recherche[1][1]
    ):
        return sol.x

    else:
        # La méthode Newtonienne a échoué, on passe à la méthode dichotomique
        # Si la méthode dichotomique échoue, None est renvoyé (le cas est prévu à l'intérieur de <dichotomie_2D>)
        return dichotomie_2D(f1, f2, zone_recherche[0], zone_recherche[1], epsilon=tol, sol_unique=True)
