# TP4 — Rapport

## Partie 0 — Validation et performances en série

### Ce que demandent les fichiers de départ

Les commentaires des fichiers fournis indiquent une version de référence à boucles et suggèrent une optimisation vectorisée. Une version intermédiaire de la version vectorisée ne suivait pas strictement les règles discrètes de Conway. Le travail consiste donc à avoir une version vectorisée correcte, à comparer les temps de calcul par itération entre boucles et vectorisé, et à vérifier que quelques patterns classiques se comportent comme attendu.

### Vérifications

Deux types de vérifications ont été faits.

La première compare directement la grille produite par [TP4/game_of_life.py](TP4/game_of_life.py) et [TP4/game_of_life_vect.py](TP4/game_of_life_vect.py) à chaque pas sur un état initial aléatoire. Le benchmark série inclut un contrôle sur 10 pas en 30×30 avec égalité à chaque pas.

La seconde exécute quelques patterns du `dico_patterns` et vérifie des comportements connus :

| Pattern | Propriété attendue | Résultat |
|---|---|---|
| boat | still life | OK |
| blinker | période 2 | OK |
| toad | période 2 | OK |
| beacon | période 2 | OK |
| pulsar | période 3 | OK |
| glider | translation (1,1) en 4 pas | OK |

Ces checks sont exécutés par [TP4/pattern_sanity.py](TP4/pattern_sanity.py) et la comparaison boucles vs vectorisé est aussi effectuée sur plusieurs pas pour blinker, toad, beacon, boat et glider.

### Résultats de performances (série)

Mesures de temps de calcul par itération, sans affichage. Le speedup est $T_{\mathrm{boucles}} / T_{\mathrm{vect}}$.

| Taille | Boucles ms par itération | Vectorisé ms par itération | Speedup |
|---:|---:|---:|---:|
| 50×50 | 26.037 | 0.152 | 171.60 |
| 100×100 | 81.647 | 0.445 | 183.59 |
| 200×200 | 303.835 | 2.233 | 136.09 |
| 400×400 | 1209.683 | 9.889 | 122.32 |

On observe un gain très important en vectorisant la mise à jour, ce qui est cohérent avec le coût élevé des boucles Python sur des stencils 8-voisines.

### Reproduction (série)

Depuis le dossier TP4, les commandes utilisées sont :

```bash
../.venv/bin/python benchmark_gol.py > bench_serial.json
../.venv/bin/python pattern_sanity.py | tee pattern_sanity.txt
```

Les valeurs du tableau ci-dessus sont extraites de [TP4/bench_serial.json](TP4/bench_serial.json). Les checks de patterns et leurs statuts sont dans [TP4/pattern_sanity.txt](TP4/pattern_sanity.txt).

Dépendances (série) : `numpy` et `scipy` pour la version vectorisée, `pygame` pour l’affichage.

## Partie 1 — Mesure de performances avec MPI

### Objectif
Mesurer le temps d’une itération du Jeu de la Vie en parallèle avec MPI et observer l’effet du nombre de rangs. Les codes de départ sont [TP4/game_of_life.py](TP4/game_of_life.py) pour la version à boucles et [TP4/game_of_life_vect.py](TP4/game_of_life_vect.py) pour la version vectorisée. On mesure uniquement l’étape de calcul, sans affichage.

### Principe de décomposition

Le domaine global ny × nx est découpé en bandes horizontales. Chaque rang possède un bloc contigu de lignes. À chaque itération, une ligne de halo est échangée en haut et en bas, puis la règle locale est appliquée sur le sous-domaine.

### Noyaux de calcul

Deux noyaux sont considérés au niveau local. Le premier compte explicitement les 8 voisines par cellule avec des boucles Python. Le second calcule le stencil en NumPy, avec des décalages `np.roll` en x et les halos en y.

### Correction de la version vectorisée (Conway)

Dans [TP4/game_of_life_vect.py](TP4/game_of_life_vect.py), `compute_next_iteration()` applique les règles discrètes de Conway à partir du nombre de voisines calculé par convolution 3×3 (centre nul, tore via `boundary='wrap'`). Une cellule naît si `voisins == 3` et une cellule vivante survit si `voisins` vaut 2 ou 3. Sur une grille aléatoire, la version vectorisée reproduit les mêmes états que la version à boucles sur plusieurs itérations.

### Méthode de mesure

La mesure MPI est faite avec un script dédié, [TP4/benchmark_gol_mpi.py](TP4/benchmark_gol_mpi.py). L’état initial est une grille aléatoire binaire en `uint8`, avec seed fixe, diffusée par `Scatterv`. Après quelques itérations de warmup, on mesure `steps` itérations. Le temps retenu est le maximum sur les rangs. Le script sépare aussi deux composantes, `comm` pour les échanges de halos et `comp` pour le calcul local.

Environnement : Python 3.12.3, Open MPI 4.1.6, mpi4py 4.1.1, NumPy 2.4.1, Linux 6.6.87.2 WSL2.

Données brutes : sorties JSON du script MPI. Une vérification est faite sur une petite grille en comparant la version MPI vectorisée à une référence série, avec les mêmes résultats après plusieurs itérations.

### Résultats

Mesures sur une grille 400×400, 50 itérations. On note $T_p$ le temps moyen par itération avec $p$ rangs, $S_p = T_1 / T_p$ et $E_p = S_p / p$.

| Rangs | Total ms par itération | Comm ms par itération | Calcul ms par itération | Speedup $S_p$ | Efficacité $E_p$ |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.707 | 0.040 | 1.665 | 1.00 | 1.00 |
| 2 | 1.223 | 0.215 | 1.014 | 1.40 | 0.70 |
| 4 | 0.412 | 0.101 | 0.363 | 4.14 | 1.04 |
| 8 | 0.438 | 0.170 | 0.358 | 3.90 | 0.49 |

À 8 rangs, l’exécution est faite avec sur-allocation via `--oversubscribe` sur la machine de test. Le cas à 4 rangs est légèrement superlinéaire, ce qui peut arriver sur de petits problèmes quand le découpage améliore l’utilisation des caches ou quand la mesure est bruitée.

### Démonstration

Animation, pattern « glider_gun » (Gosper), vue zoomée :

![Animation Jeu de la Vie](gol_demo.gif)

### Exécution interactive

Pour afficher le Jeu de la Vie dans une fenêtre Pygame, lancer depuis le dossier TP4 :

```bash
../.venv/bin/python game_of_life.py glider 800 800
../.venv/bin/python game_of_life_vect.py pulsar 800 800
```

### Découplage calcul/affichage

Pour éviter de mélanger calcul et rendu, une version « split » utilise deux processus.

Le worker maintient la grille et calcule chaque itération. Il produit une liste `diff_cells` contenant les indices plats `i*nx + j` des cellules qui changent d’état, puis envoie uniquement cette liste via une `multiprocessing.Queue` de petite taille. Le processus principal garde la fenêtre Pygame et l’event loop. Il maintient sa propre copie locale de `cells`, applique un flip binaire sur les indices reçus, et met à jour l’écran avec un redraw incrémental : seuls les rectangles modifiés sont redessinés, puis `pg.display.update(rects)` est appelé.

Script : [TP4/game_of_life_split.py](TP4/game_of_life_split.py).

Exemples :

```bash
../.venv/bin/python game_of_life_split.py glider 800 800 --kernel vector
../.venv/bin/python game_of_life_split.py glider 800 800 --kernel loops
```

Sanity-check (sans affichage) :

```bash
../.venv/bin/python game_of_life_split.py glider --sanity 50 --kernel vector
```

Patterns disponibles : `blinker`, `toad`, `acorn`, `beacon`, `boat`, `glider`, `glider_gun`, `space_ship`, `die_hard`, `pulsar`, `floraison`, `block_switch_engine`, `u`, `flat`. Sous WSL, l’affichage nécessite WSLg ou un serveur X.

### Analyse

Le passage de 1 à 4 rangs réduit nettement le temps par itération. Au-delà, le coût des communications et l’oversubscription limitent le gain, ce qui explique que 8 rangs ne fasse pas mieux que 4 dans ces conditions. Le découpage en bandes ne nécessite que deux échanges de halos par itération, mais le halo contient nx éléments, donc la part `comm` devient visible quand nx est grand.

### Reproduction

Depuis le dossier TP4 :

```bash
mpirun -np 4 ../.venv/bin/python benchmark_gol_mpi.py --ny 400 --nx 400 --steps 50 --kernel vector
mpirun --oversubscribe -np 8 ../.venv/bin/python benchmark_gol_mpi.py --ny 400 --nx 400 --steps 50 --kernel vector
```

Dépendances minimales dans le venv : `numpy`, `mpi4py`.
