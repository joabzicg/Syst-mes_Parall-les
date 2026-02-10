#!/usr/bin/env python3
import os

# Avoid opening windows / noisy banner when importing pygame.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import platform
import sys
import time
from dataclasses import dataclass

import numpy as np

import game_of_life
import game_of_life_vect


def _maybe_run_mpi() -> None:
    """If launched under mpirun, delegate to the MPI benchmark."""
    try:
        from mpi4py import MPI  # type: ignore

        if MPI.COMM_WORLD.Get_size() > 1:
            import benchmark_gol_mpi

            raise SystemExit(benchmark_gol_mpi.main())
    except ImportError:
        return


@dataclass(frozen=True)
class BenchResult:
    dims: tuple[int, int]
    steps: int
    slow_s_per_iter: float
    vect_s_per_iter: float

    @property
    def speedup(self) -> float:
        return self.slow_s_per_iter / self.vect_s_per_iter if self.vect_s_per_iter else float("inf")


def _time_iterations(grid, steps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(steps):
        grid.compute_next_iteration()
    t1 = time.perf_counter()
    return (t1 - t0) / steps


def run_benchmark(dims: tuple[int, int], steps: int, warmup: int, seed: int) -> BenchResult:
    rng = np.random.default_rng(seed)
    init_cells = rng.integers(0, 2, size=dims, dtype=np.uint8)

    slow_grid = game_of_life.Grille(dims)
    slow_grid.cells = init_cells.copy()

    vect_grid = game_of_life_vect.Grille(dims)
    vect_grid.cells = init_cells.copy()

    for _ in range(warmup):
        slow_grid.compute_next_iteration()
        vect_grid.compute_next_iteration()

    slow_s = _time_iterations(slow_grid, steps)
    vect_s = _time_iterations(vect_grid, steps)

    return BenchResult(dims=dims, steps=steps, slow_s_per_iter=slow_s, vect_s_per_iter=vect_s)


def quick_correctness_check(dims: tuple[int, int] = (30, 30), steps: int = 10, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    init_cells = rng.integers(0, 2, size=dims, dtype=np.uint8)

    slow_grid = game_of_life.Grille(dims)
    slow_grid.cells = init_cells.copy()

    vect_grid = game_of_life_vect.Grille(dims)
    vect_grid.cells = init_cells.copy()

    equal_each_step = []
    for _ in range(steps):
        slow_grid.compute_next_iteration()
        vect_grid.compute_next_iteration()
        equal_each_step.append(bool(np.array_equal(slow_grid.cells, vect_grid.cells)))

    return {
        "dims": dims,
        "steps": steps,
        "all_equal": all(equal_each_step),
        "equal_each_step": equal_each_step,
    }


def main() -> int:
    _maybe_run_mpi()
    sizes = [(50, 50), (100, 100), (200, 200), (400, 400)]
    # Keep runtime reasonable: more steps for small grids, fewer for large.
    steps_for = {
        (50, 50): 200,
        (100, 100): 80,
        (200, 200): 30,
        (400, 400): 10,
    }
    warmup = 5
    seed = 42

    results = [run_benchmark(sz, steps_for[sz], warmup, seed) for sz in sizes]

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "pygame": __import__("pygame").version.ver,
        "correctness": quick_correctness_check(),
    }

    payload = {
        "meta": meta,
        "results": [
            {
                "dims": list(r.dims),
                "steps": r.steps,
                "slow_s_per_iter": r.slow_s_per_iter,
                "vect_s_per_iter": r.vect_s_per_iter,
                "speedup": r.speedup,
            }
            for r in results
        ],
    }

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
