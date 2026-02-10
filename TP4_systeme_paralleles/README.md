# TP4 — Jeu de la Vie (boucles, vectorisé, MPI)

## Fichiers

- [game_of_life.py](game_of_life.py) : version à boucles (référence), affichage Pygame.
- [game_of_life_vect.py](game_of_life_vect.py) : version vectorisée (NumPy/SciPy), affichage Pygame.
- [benchmark_gol.py](benchmark_gol.py) : benchmark **série** boucles vs vectorisé (temps de calcul par itération).
- [benchmark_gol_mpi.py](benchmark_gol_mpi.py) : benchmark **MPI** (décomposition 1D en bandes, échanges de halos).
- [pattern_sanity.py](pattern_sanity.py) : vérifications rapides sur quelques patterns du `dico_patterns`.
- [game_of_life_split.py](game_of_life_split.py) : version avec **découplage calcul/affichage** (worker + pygame) via `multiprocessing.Queue`.
- [relatorio.md](relatorio.md) : rapport (FR).

## Exécution interactive (Pygame)

Depuis le dossier TP4 :

```bash
../.venv/bin/python game_of_life.py glider 800 800
../.venv/bin/python game_of_life_vect.py pulsar 800 800
```

## Mode split (calcul/affichage séparés)

Le worker calcule les itérations et envoie uniquement `diff_cells` (indices plats `i*nx+j`). Le processus principal garde la fenêtre Pygame et fait un redraw incrémental.

```bash
../.venv/bin/python game_of_life_split.py glider 800 800 --kernel vector
../.venv/bin/python game_of_life_split.py glider 800 800 --kernel loops
```

Sanity-check (headless) :

```bash
../.venv/bin/python game_of_life_split.py glider --sanity 50 --kernel vector
```

## Benchmark série (boucles vs vectorisé)

```bash
../.venv/bin/python benchmark_gol.py
```

Le script affiche un JSON contenant les temps moyens par itération et le speedup.

## Benchmark MPI

```bash
mpirun -np 4 ../.venv/bin/python benchmark_gol_mpi.py --ny 400 --nx 400 --steps 50 --kernel vector
mpirun --oversubscribe -np 8 ../.venv/bin/python benchmark_gol_mpi.py --ny 400 --nx 400 --steps 50 --kernel vector
```

## Sanity-check patterns

```bash
../.venv/bin/python pattern_sanity.py
```

Les checks incluent : still-life (boat), oscillateurs (blinker/toad/beacon/pulsar), et glider (translation en 4 pas), plus une comparaison boucles vs vectorisé sur quelques itérations.

## Dépendances

Les scripts utilisent `numpy`, `scipy` (convolution) et `pygame` (affichage), et `mpi4py` pour la partie MPI.
