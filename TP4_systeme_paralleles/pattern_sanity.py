#!/usr/bin/env python3
import os

# Keep imports headless when running under CI/terminal.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import dataclass

import numpy as np

import game_of_life
import game_of_life_vect


# Subset copied from dico_patterns in the original scripts.
DICO_PATTERNS: dict[str, tuple[tuple[int, int], list[tuple[int, int]]]] = {
    "blinker": ((5, 5), [(2, 1), (2, 2), (2, 3)]),
    "toad": ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
    "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
    "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
    "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
    "pulsar": (
        (17, 17),
        [
            (2, 4),
            (2, 5),
            (2, 6),
            (7, 4),
            (7, 5),
            (7, 6),
            (9, 4),
            (9, 5),
            (9, 6),
            (14, 4),
            (14, 5),
            (14, 6),
            (2, 10),
            (2, 11),
            (2, 12),
            (7, 10),
            (7, 11),
            (7, 12),
            (9, 10),
            (9, 11),
            (9, 12),
            (14, 10),
            (14, 11),
            (14, 12),
            (4, 2),
            (5, 2),
            (6, 2),
            (4, 7),
            (5, 7),
            (6, 7),
            (4, 9),
            (5, 9),
            (6, 9),
            (4, 14),
            (5, 14),
            (6, 14),
            (10, 2),
            (11, 2),
            (12, 2),
            (10, 7),
            (11, 7),
            (12, 7),
            (10, 9),
            (11, 9),
            (12, 9),
            (10, 14),
            (11, 14),
            (12, 14),
        ],
    ),
}


@dataclass(frozen=True)
class PatternCheck:
    name: str
    dims: tuple[int, int]
    steps: int
    ok: bool
    details: str


def _grid_from_pattern(cls, dims: tuple[int, int], coords: list[tuple[int, int]]):
    grid = cls(dims, init_pattern=coords)
    return grid


def _coords_shift(coords: list[tuple[int, int]], dims: tuple[int, int], di: int, dj: int) -> set[tuple[int, int]]:
    ny, nx = dims
    return {((i + di) % ny, (j + dj) % nx) for i, j in coords}


def _cells_to_coords(cells: np.ndarray) -> set[tuple[int, int]]:
    ii, jj = np.nonzero(cells)
    return set(zip(ii.tolist(), jj.tolist()))


def _run_and_compare(name: str, dims: tuple[int, int], coords: list[tuple[int, int]], steps: int) -> PatternCheck:
    slow = _grid_from_pattern(game_of_life.Grille, dims, coords)
    vect = _grid_from_pattern(game_of_life_vect.Grille, dims, coords)

    for s in range(steps):
        slow.compute_next_iteration()
        vect.compute_next_iteration()
        if not np.array_equal(slow.cells, vect.cells):
            return PatternCheck(
                name=name,
                dims=dims,
                steps=steps,
                ok=False,
                details=f"Mismatch at step {s + 1}",
            )

    return PatternCheck(name=name, dims=dims, steps=steps, ok=True, details="loops == vectorisé")


def _check_period(name: str, period: int) -> PatternCheck:
    dims, coords = DICO_PATTERNS[name]
    grid = _grid_from_pattern(game_of_life_vect.Grille, dims, coords)
    start = grid.cells.copy()
    for _ in range(period):
        grid.compute_next_iteration()
    ok = bool(np.array_equal(start, grid.cells))
    return PatternCheck(
        name=name,
        dims=dims,
        steps=period,
        ok=ok,
        details=f"période {period}" if ok else f"n'est pas revenu à l'état initial après {period} pas",
    )


def _check_still_life(name: str, steps: int = 1) -> PatternCheck:
    dims, coords = DICO_PATTERNS[name]
    grid = _grid_from_pattern(game_of_life_vect.Grille, dims, coords)
    start = grid.cells.copy()
    for _ in range(steps):
        grid.compute_next_iteration()
    ok = bool(np.array_equal(start, grid.cells))
    return PatternCheck(
        name=name,
        dims=dims,
        steps=steps,
        ok=ok,
        details="still life" if ok else "a changé (attendu stable)",
    )


def _check_glider_translation() -> PatternCheck:
    dims, coords = DICO_PATTERNS["glider"]
    grid = _grid_from_pattern(game_of_life_vect.Grille, dims, coords)

    for _ in range(4):
        grid.compute_next_iteration()

    expected = _coords_shift(coords, dims, di=1, dj=1)
    observed = _cells_to_coords(grid.cells)
    ok = expected == observed
    return PatternCheck(
        name="glider",
        dims=dims,
        steps=4,
        ok=ok,
        details="translation (1,1) en 4 pas" if ok else "translation attendue non observée",
    )


def main() -> int:
    checks: list[PatternCheck] = []

    # Cross-check loops vs vectorised on a few steps.
    for name, (dims, coords) in [
        ("blinker", DICO_PATTERNS["blinker"]),
        ("toad", DICO_PATTERNS["toad"]),
        ("beacon", DICO_PATTERNS["beacon"]),
        ("boat", DICO_PATTERNS["boat"]),
        ("glider", DICO_PATTERNS["glider"]),
    ]:
        checks.append(_run_and_compare(name, dims, coords, steps=10))

    # Behavior checks (vectorised kernel only).
    checks.append(_check_still_life("boat", steps=1))
    checks.append(_check_period("blinker", period=2))
    checks.append(_check_period("toad", period=2))
    checks.append(_check_period("beacon", period=2))
    checks.append(_check_period("pulsar", period=3))
    checks.append(_check_glider_translation())

    all_ok = all(c.ok for c in checks)

    print("Pattern sanity-checks")
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        print(f"- {status:4s} {c.name:8s} dims={c.dims} steps={c.steps}: {c.details}")

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
