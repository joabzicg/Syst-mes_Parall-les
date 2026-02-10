#!/usr/bin/env python3
"""Split Game of Life: decouple compute (worker) and display (pygame).

- Worker process maintains the grid, computes next iteration and emits diff_cells
  as flat indices (i*nx + j) for cells that changed.
- Main process owns pygame window/event loop, keeps a local copy of cells,
  applies XOR/flip on received indices, and redraws only changed rectangles.

This keeps UI responsive and avoids sending the full grid each frame.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass

import numpy as np


# Patterns copied from the original scripts.
DICO_PATTERNS: dict[str, tuple[tuple[int, int], list[tuple[int, int]]]] = {
    "blinker": ((5, 5), [(2, 1), (2, 2), (2, 3)]),
    "toad": ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
    "acorn": ((100, 100), [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)]),
    "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
    "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
    "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
    "glider_gun": (
        (400, 400),
        [
            (51, 76),
            (52, 74),
            (52, 76),
            (53, 64),
            (53, 65),
            (53, 72),
            (53, 73),
            (53, 86),
            (53, 87),
            (54, 63),
            (54, 67),
            (54, 72),
            (54, 73),
            (54, 86),
            (54, 87),
            (55, 52),
            (55, 53),
            (55, 62),
            (55, 68),
            (55, 72),
            (55, 73),
            (56, 52),
            (56, 53),
            (56, 62),
            (56, 66),
            (56, 68),
            (56, 69),
            (56, 74),
            (56, 76),
            (57, 62),
            (57, 68),
            (57, 76),
            (58, 63),
            (58, 67),
            (59, 64),
            (59, 65),
        ],
    ),
    "space_ship": (
        (25, 25),
        [(11, 13), (11, 14), (12, 11), (12, 12), (12, 14), (12, 15), (13, 11), (13, 12), (13, 13), (13, 14), (14, 12), (14, 13)],
    ),
    "die_hard": ((100, 100), [(51, 57), (52, 51), (52, 52), (53, 52), (53, 56), (53, 57), (53, 58)]),
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
    "floraison": ((40, 40), [(19, 18), (19, 19), (19, 20), (20, 17), (20, 19), (20, 21), (21, 18), (21, 19), (21, 20)]),
    "block_switch_engine": (
        (400, 400),
        [(201, 202), (201, 203), (202, 202), (202, 203), (211, 203), (212, 204), (212, 202), (214, 204), (214, 201), (215, 201), (215, 202), (216, 201)],
    ),
    "u": (
        (200, 200),
        [(101, 101), (102, 102), (103, 102), (103, 101), (104, 103), (105, 103), (105, 102), (105, 101), (105, 105), (103, 105), (102, 105), (101, 105), (101, 104)],
    ),
    "flat": (
        (200, 400),
        [
            (80, 200),
            (81, 200),
            (82, 200),
            (83, 200),
            (84, 200),
            (85, 200),
            (86, 200),
            (87, 200),
            (89, 200),
            (90, 200),
            (91, 200),
            (92, 200),
            (93, 200),
            (97, 200),
            (98, 200),
            (99, 200),
            (106, 200),
            (107, 200),
            (108, 200),
            (109, 200),
            (110, 200),
            (111, 200),
            (112, 200),
            (114, 200),
            (115, 200),
            (116, 200),
            (117, 200),
            (118, 200),
        ],
    ),
}


def build_initial_cells(pattern: str, ny: int | None, nx: int | None, seed: int) -> tuple[np.ndarray, tuple[int, int]]:
    if pattern == "random":
        if ny is None or nx is None:
            raise ValueError("pattern=random requires --ny and --nx")
        rng = np.random.default_rng(seed)
        cells = rng.integers(0, 2, size=(ny, nx), dtype=np.uint8)
        return cells, (ny, nx)

    try:
        dims, coords = DICO_PATTERNS[pattern]
    except KeyError as exc:
        raise ValueError(f"Unknown pattern '{pattern}'") from exc

    cells = np.zeros(dims, dtype=np.uint8)
    ii = [i for i, _ in coords]
    jj = [j for _, j in coords]
    cells[ii, jj] = 1
    return cells, dims


def step_vector(cells: np.ndarray) -> tuple[np.ndarray, list[int]]:
    # Toroidal boundary via rolls.
    n = (
        np.roll(cells, 1, axis=0)
        + np.roll(cells, -1, axis=0)
        + np.roll(cells, 1, axis=1)
        + np.roll(cells, -1, axis=1)
        + np.roll(np.roll(cells, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(cells, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(cells, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(cells, -1, axis=0), -1, axis=1)
    ).astype(np.uint8)

    alive = cells == 1
    next_cells = ((n == 3) | (alive & (n == 2))).astype(np.uint8)
    diff = np.flatnonzero(next_cells != cells).astype(int).tolist()
    return next_cells, diff


def step_loops(cells: np.ndarray) -> tuple[np.ndarray, list[int]]:
    ny, nx = cells.shape
    next_cells = np.empty_like(cells)
    diff: list[int] = []

    for i in range(ny):
        i_above = (i + ny - 1) % ny
        i_below = (i + 1) % ny
        for j in range(nx):
            j_left = (j - 1 + nx) % nx
            j_right = (j + 1) % nx

            nb = (
                cells[i_above, j_left]
                + cells[i_above, j]
                + cells[i_above, j_right]
                + cells[i, j_left]
                + cells[i, j_right]
                + cells[i_below, j_left]
                + cells[i_below, j]
                + cells[i_below, j_right]
            )

            if cells[i, j] == 1:
                if nb < 2 or nb > 3:
                    next_cells[i, j] = 0
                    diff.append(i * nx + j)
                else:
                    next_cells[i, j] = 1
            else:
                if nb == 3:
                    next_cells[i, j] = 1
                    diff.append(i * nx + j)
                else:
                    next_cells[i, j] = 0

    return next_cells, diff


def _get_stepper(kernel: str):
    if kernel == "vector":
        return step_vector
    if kernel == "loops":
        return step_loops
    raise ValueError("kernel must be 'vector' or 'loops'")


def compute_worker(
    init_cells: np.ndarray,
    kernel: str,
    out_q: mp.Queue,
    stop_event: mp.Event,
    sleep_s: float,
) -> None:
    cells = init_cells.copy()
    stepper = _get_stepper(kernel)

    while not stop_event.is_set():
        next_cells, diff = stepper(cells)
        cells = next_cells

        # Keep ordering: never drop diffs.
        while not stop_event.is_set():
            try:
                out_q.put(diff, timeout=0.05)
                break
            except queue.Full:
                continue

        if sleep_s > 0:
            time.sleep(sleep_s)


@dataclass
class DisplayConfig:
    resx: int
    resy: int


class Display:
    def __init__(self, cfg: DisplayConfig, cells: np.ndarray):
        import pygame as pg

        self.pg = pg
        self.cells = cells
        self.ny, self.nx = cells.shape

        self.size_x = cfg.resx // self.nx
        self.size_y = cfg.resy // self.ny
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color("lightgrey")
        else:
            self.draw_color = None

        self.width = self.nx * self.size_x
        self.height = self.ny * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))

        self.col_life = pg.Color("black")
        self.col_dead = pg.Color("white")

    def compute_rectangle(self, i: int, j: int):
        # Same geometry convention as the original scripts.
        return (self.size_x * j, self.height - self.size_y * (i + 1), self.size_x, self.size_y)

    def _cell_color(self, i: int, j: int):
        return self.col_life if self.cells[i, j] else self.col_dead

    def draw_full(self) -> None:
        for i in range(self.ny):
            for j in range(self.nx):
                self.screen.fill(self._cell_color(i, j), self.compute_rectangle(i, j))

        if self.draw_color is not None:
            for i in range(self.ny):
                self.pg.draw.line(self.screen, self.draw_color, (0, i * self.size_y), (self.width, i * self.size_y))
            for j in range(self.nx):
                self.pg.draw.line(self.screen, self.draw_color, (j * self.size_x, 0), (j * self.size_x, self.height))

        self.pg.display.update()

    def draw_diff(self, diff_cells: list[int]) -> None:
        rects = []
        for idx in diff_cells:
            i = idx // self.nx
            j = idx % self.nx
            self.cells[i, j] ^= 1
            rect = self.compute_rectangle(i, j)
            self.screen.fill(self._cell_color(i, j), rect)
            if self.draw_color is not None:
                self.pg.draw.rect(self.screen, self.draw_color, rect, width=1)
            rects.append(rect)

        if rects:
            self.pg.display.update(rects)


def sanity_check(pattern: str, kernel: str, ny: int | None, nx: int | None, seed: int, steps: int) -> bool:
    init, _dims = build_initial_cells(pattern, ny, nx, seed)
    stepper = _get_stepper(kernel)

    # Single-process reference.
    ref = init.copy()
    for _ in range(steps):
        ref, _diff = stepper(ref)

    # Split simulation: apply diffs via XOR to a display copy.
    worker_cells = init.copy()
    display_cells = init.copy()
    for _ in range(steps):
        worker_next, diff = stepper(worker_cells)
        worker_cells = worker_next
        for idx in diff:
            display_cells.flat[idx] ^= 1

    return bool(np.array_equal(ref, display_cells))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Game of Life with compute/display split using multiprocessing")
    p.add_argument("pattern", help="pattern name (from dico_patterns) or 'random'")
    p.add_argument("resx", nargs="?", type=int, default=800)
    p.add_argument("resy", nargs="?", type=int, default=800)
    p.add_argument("--kernel", choices=["loops", "vector"], default="vector")
    p.add_argument("--mode", choices=["split", "single"], default="split")
    p.add_argument("--queue-size", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.0, help="sleep seconds per iteration in worker")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--sanity", type=int, default=0, metavar="N", help="run headless sanity check for N iterations and exit")
    return p.parse_args(argv)


def run_single_process(pattern: str, kernel: str, resx: int, resy: int, ny: int | None, nx: int | None, seed: int) -> int:
    import pygame as pg

    init, _dims = build_initial_cells(pattern, ny, nx, seed)
    cells = init.copy()
    disp = Display(DisplayConfig(resx=resx, resy=resy), cells)

    stepper = _get_stepper(kernel)

    pg.init()
    disp.draw_full()

    must_continue = True
    while must_continue:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                must_continue = False

        next_cells, diff = stepper(cells)
        cells[:, :] = next_cells
        disp.draw_diff(diff)

    pg.quit()
    return 0


def run_split(pattern: str, kernel: str, resx: int, resy: int, ny: int | None, nx: int | None, seed: int, queue_size: int, sleep_s: float) -> int:
    import pygame as pg

    init, _dims = build_initial_cells(pattern, ny, nx, seed)

    # Display maintains its own local copy (requirement #3).
    display_cells = init.copy()

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=max(1, queue_size))
    stop_event = ctx.Event()

    worker = ctx.Process(
        target=compute_worker,
        args=(init, kernel, q, stop_event, sleep_s),
        daemon=True,
    )

    pg.init()
    disp = Display(DisplayConfig(resx=resx, resy=resy), display_cells)
    disp.draw_full()

    worker.start()

    must_continue = True
    try:
        while must_continue:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    must_continue = False
                    stop_event.set()

            try:
                diff = q.get(timeout=0.02)
            except queue.Empty:
                continue

            disp.draw_diff(diff)

    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
        pg.quit()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.sanity > 0:
        ok = sanity_check(args.pattern, args.kernel, args.ny, args.nx, args.seed, args.sanity)
        print("OK" if ok else "FAIL")
        return 0 if ok else 2

    if args.mode == "single":
        return run_single_process(args.pattern, args.kernel, args.resx, args.resy, args.ny, args.nx, args.seed)

    return run_split(args.pattern, args.kernel, args.resx, args.resy, args.ny, args.nx, args.seed, args.queue_size, args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
