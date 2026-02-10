#!/usr/bin/env python3
"""MPI benchmark for Game of Life (toroidal grid).

Domain decomposition: 1D split along rows.
Each rank owns a contiguous block of rows of the global grid and exchanges 1-row halos
(top/bottom) with neighboring ranks each iteration.

We benchmark two kernels on the local subdomain:
- 'loops': explicit Python loops (slow)
- 'vector': pure NumPy vectorized stencil using halos

This script is meant for performance measurement (no display).

Usage examples:
  mpirun -np 4 python benchmark_gol_mpi.py --ny 400 --nx 400 --steps 50 --kernel vector
  mpirun -np 4 python benchmark_gol_mpi.py --ny 200 --nx 200 --steps 20 --kernel both

It outputs a JSON payload on rank 0.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI


@dataclass
class Decomp:
    counts: np.ndarray  # rows per rank
    displs: np.ndarray  # starting row per rank


def decompose_rows(ny: int, size: int) -> Decomp:
    counts = np.full(size, ny // size, dtype=np.int32)
    counts[: ny % size] += 1
    displs = np.zeros(size, dtype=np.int32)
    displs[1:] = np.cumsum(counts)[:-1]
    return Decomp(counts=counts, displs=displs)


def scatter_initial_grid(comm: MPI.Comm, ny: int, nx: int, seed: int) -> tuple[np.ndarray, Decomp]:
    rank = comm.Get_rank()
    size = comm.Get_size()

    decomp = decompose_rows(ny, size)
    local_ny = int(decomp.counts[rank])

    if rank == 0:
        rng = np.random.default_rng(seed)
        global_cells = rng.integers(0, 2, size=(ny, nx), dtype=np.uint8)
        sendbuf = global_cells
    else:
        sendbuf = None

    recvbuf = np.empty((local_ny, nx), dtype=np.uint8)

    counts_elems = (decomp.counts * nx).astype(np.int32)
    displs_elems = (decomp.displs * nx).astype(np.int32)

    comm.Scatterv([sendbuf, counts_elems, displs_elems, MPI.UNSIGNED_CHAR], recvbuf, root=0)
    return recvbuf, decomp


def exchange_halos(comm: MPI.Comm, local: np.ndarray, halo_top: np.ndarray, halo_bottom: np.ndarray) -> None:
    """Exchange 1-row halos with torus wrap in the vertical (rank) direction."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    up = (rank - 1) % size
    down = (rank + 1) % size

    # Send first row up, receive bottom halo from down (their first row wraps to our bottom halo)
    comm.Sendrecv(sendbuf=local[0, :], dest=up, sendtag=0,
                  recvbuf=halo_bottom, source=down, recvtag=0)

    # Send last row down, receive top halo from up
    comm.Sendrecv(sendbuf=local[-1, :], dest=down, sendtag=1,
                  recvbuf=halo_top, source=up, recvtag=1)


def step_vector(local: np.ndarray, halo_top: np.ndarray, halo_bottom: np.ndarray) -> np.ndarray:
    """One iteration using vectorized stencil with halos (toroidal in x)."""
    ext = np.vstack([halo_top[None, :], local, halo_bottom[None, :]]).astype(np.uint8)

    up = ext[:-2, :]
    mid = ext[1:-1, :]
    down = ext[2:, :]

    # neighbor count (8-neighborhood)
    n = (
        np.roll(up, 1, axis=1) + up + np.roll(up, -1, axis=1)
        + np.roll(mid, 1, axis=1) + np.roll(mid, -1, axis=1)
        + np.roll(down, 1, axis=1) + down + np.roll(down, -1, axis=1)
    ).astype(np.uint8)

    alive = mid == 1
    born = (~alive) & (n == 3)
    survive = alive & ((n == 2) | (n == 3))

    nxt = np.zeros_like(local, dtype=np.uint8)
    nxt[born | survive] = 1
    return nxt


def step_loops(local: np.ndarray, halo_top: np.ndarray, halo_bottom: np.ndarray) -> np.ndarray:
    """One iteration using explicit loops on the local domain, with halos."""
    ny, nx = local.shape
    ext = np.vstack([halo_top[None, :], local, halo_bottom[None, :]]).astype(np.uint8)
    nxt = np.empty_like(local, dtype=np.uint8)

    for i in range(1, ny + 1):
        i_above = i - 1
        i_below = i + 1
        for j in range(nx):
            j_left = (j - 1) % nx
            j_right = (j + 1) % nx
            nb = (
                int(ext[i_above, j_left]) + int(ext[i_above, j]) + int(ext[i_above, j_right])
                + int(ext[i, j_left]) + int(ext[i, j_right])
                + int(ext[i_below, j_left]) + int(ext[i_below, j]) + int(ext[i_below, j_right])
            )
            if ext[i, j] == 1:
                nxt[i - 1, j] = 1 if (nb == 2 or nb == 3) else 0
            else:
                nxt[i - 1, j] = 1 if (nb == 3) else 0

    return nxt


def time_kernel(comm: MPI.Comm, local0: np.ndarray, steps: int, warmup: int, kernel: str) -> dict:
    local = local0.copy()
    nx = local.shape[1]

    halo_top = np.empty((nx,), dtype=np.uint8)
    halo_bottom = np.empty((nx,), dtype=np.uint8)

    def step(local_arr: np.ndarray) -> np.ndarray:
        exchange_halos(comm, local_arr, halo_top, halo_bottom)
        if kernel == "vector":
            return step_vector(local_arr, halo_top, halo_bottom)
        if kernel == "loops":
            return step_loops(local_arr, halo_top, halo_bottom)
        raise ValueError("unknown kernel")

    # warmup
    for _ in range(warmup):
        local = step(local)

    comm.Barrier()
    t0 = MPI.Wtime()
    comm_time = 0.0
    comp_time = 0.0

    for _ in range(steps):
        # Split comm / comp to understand scaling
        tc0 = MPI.Wtime()
        exchange_halos(comm, local, halo_top, halo_bottom)
        tc1 = MPI.Wtime()

        if kernel == "vector":
            local = step_vector(local, halo_top, halo_bottom)
        else:
            local = step_loops(local, halo_top, halo_bottom)

        tc2 = MPI.Wtime()
        comm_time += (tc1 - tc0)
        comp_time += (tc2 - tc1)

    comm.Barrier()
    t1 = MPI.Wtime()

    # The relevant metric is typically the max across ranks.
    total = t1 - t0
    total_max = comm.allreduce(total, op=MPI.MAX)
    comm_max = comm.allreduce(comm_time, op=MPI.MAX)
    comp_max = comm.allreduce(comp_time, op=MPI.MAX)

    return {
        "kernel": kernel,
        "steps": steps,
        "warmup": warmup,
        "total_s": total_max,
        "comm_s": comm_max,
        "comp_s": comp_max,
        "total_s_per_iter": total_max / steps,
        "comm_s_per_iter": comm_max / steps,
        "comp_s_per_iter": comp_max / steps,
    }


def gather_grid(comm: MPI.Comm, local: np.ndarray, decomp: Decomp, ny: int, nx: int) -> np.ndarray | None:
    rank = comm.Get_rank()
    counts_elems = (decomp.counts * nx).astype(np.int32)
    displs_elems = (decomp.displs * nx).astype(np.int32)

    if rank == 0:
        global_cells = np.empty((ny, nx), dtype=np.uint8)
    else:
        global_cells = None

    comm.Gatherv(local, [global_cells, counts_elems, displs_elems, MPI.UNSIGNED_CHAR], root=0)
    return global_cells


def serial_reference(ny: int, nx: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 2, size=(ny, nx), dtype=np.uint8)
    for _ in range(steps):
        up = np.roll(cells, 1, axis=0)
        down = np.roll(cells, -1, axis=0)
        left = np.roll(cells, 1, axis=1)
        right = np.roll(cells, -1, axis=1)

        n = (
            np.roll(up, 1, axis=1) + up + np.roll(up, -1, axis=1)
            + left + right
            + np.roll(down, 1, axis=1) + down + np.roll(down, -1, axis=1)
        ).astype(np.uint8)

        alive = cells == 1
        born = (~alive) & (n == 3)
        survive = alive & ((n == 2) | (n == 3))
        nxt = np.zeros_like(cells)
        nxt[born | survive] = 1
        cells = nxt
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel", choices=["vector", "loops", "both"], default="vector")
    parser.add_argument("--check", action="store_true", help="check final grid vs serial reference (small sizes recommended)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local0, decomp = scatter_initial_grid(comm, args.ny, args.nx, args.seed)

    kernels = [args.kernel] if args.kernel != "both" else ["vector", "loops"]
    timings = []
    for k in kernels:
        timings.append(time_kernel(comm, local0, args.steps, args.warmup, k))

    correctness = None
    if args.check:
        # Re-run the chosen kernel to obtain final grid for comparison.
        # (Keep simple: only validate 'vector' kernel.)
        if args.kernel == "vector" or args.kernel == "both":
            local = local0.copy()
            halo_top = np.empty((args.nx,), dtype=np.uint8)
            halo_bottom = np.empty((args.nx,), dtype=np.uint8)
            for _ in range(args.steps):
                exchange_halos(comm, local, halo_top, halo_bottom)
                local = step_vector(local, halo_top, halo_bottom)
            global_cells = gather_grid(comm, local, decomp, args.ny, args.nx)
            if rank == 0:
                ref = serial_reference(args.ny, args.nx, args.steps, args.seed)
                correctness = {
                    "checked_kernel": "vector",
                    "ok": bool(np.array_equal(global_cells, ref)),
                }

    if rank == 0:
        meta = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mpi": MPI.Get_library_version().strip(),
            "ranks": size,
            "ny": args.ny,
            "nx": args.nx,
        }
        payload = {
            "meta": meta,
            "timings": timings,
            "correctness": correctness,
        }
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
