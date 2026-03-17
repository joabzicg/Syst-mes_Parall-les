import sys

import numpy as np
from mpi4py import MPI
from numba import njit, prange


G = 1.560339e-13


def decompose_axis(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.full(size, n // size, dtype=np.int64)
    counts[: n % size] += 1
    displs = np.zeros(size, dtype=np.int64)
    displs[1:] = np.cumsum(counts)[:-1]
    return counts, displs


@njit
def compute_cell_data(positions: np.ndarray, grid_min: np.ndarray, cell_size: np.ndarray, n_cells: np.ndarray):
    n_bodies = positions.shape[0]
    cell_coords = np.empty((n_bodies, 3), dtype=np.int64)
    morse_indices = np.empty(n_bodies, dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        cell_coords[ibody, :] = cell_idx
        morse_indices[ibody] = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
    return cell_coords, morse_indices


@njit
def accumulate_owned_cells(owned_indices: np.ndarray, morse_indices: np.ndarray,
                           masses: np.ndarray, positions: np.ndarray,
                           n_total_cells: int):
    local_cell_masses = np.zeros(n_total_cells, dtype=np.float32)
    local_weighted_positions = np.zeros((n_total_cells, 3), dtype=np.float32)
    for iloc in range(owned_indices.shape[0]):
        ibody = owned_indices[iloc]
        morse_idx = morse_indices[ibody]
        mass = masses[ibody]
        local_cell_masses[morse_idx] += mass
        local_weighted_positions[morse_idx, :] += positions[ibody, :] * mass
    return local_cell_masses, local_weighted_positions


@njit(parallel=True)
def compute_acceleration_owned(owned_indices: np.ndarray,
                               positions: np.ndarray,
                               masses: np.ndarray,
                               cell_coords: np.ndarray,
                               cell_start_indices: np.ndarray,
                               body_indices: np.ndarray,
                               cell_masses: np.ndarray,
                               cell_com_positions: np.ndarray,
                               n_cells: np.ndarray):
    accelerations = np.zeros((owned_indices.shape[0], 3), dtype=np.float32)
    for iloc in prange(owned_indices.shape[0]):
        ibody = owned_indices[iloc]
        pos = positions[ibody]
        cell_idx = cell_coords[ibody]
        acc = np.zeros(3, dtype=np.float32)
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    if (abs(ix - cell_idx[0]) > 2) or (abs(iy - cell_idx[1]) > 2) or (abs(iz - cell_idx[2]) > 2):
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com_positions[morse_idx] - pos
                            distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                acc += G * direction * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx + 1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    acc += G * direction * inv_dist3 * masses[jbody]
        accelerations[iloc, :] = acc
    return accelerations


def load_system(filename: str):
    positions = []
    velocities = []
    masses = []
    box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)
    with open(filename, "r", encoding="utf-8") as stream:
        for line in stream:
            data = line.split()
            masses.append(float(data[0]))
            positions.append([float(data[1]), float(data[2]), float(data[3])])
            velocities.append([float(data[4]), float(data[5]), float(data[6])])
            for i in range(3):
                box[0][i] = min(box[0][i], positions[-1][i] - 1.0e-6)
                box[1][i] = max(box[1][i], positions[-1][i] + 1.0e-6)
    return (
        np.array(positions, dtype=np.float32),
        np.array(velocities, dtype=np.float32),
        np.array(masses, dtype=np.float32),
        box,
    )


def build_subset_lists(morse_indices: np.ndarray, subset_mask: np.ndarray, n_total_cells: int):
    subset_indices = np.flatnonzero(subset_mask).astype(np.int64)
    subset_morse = morse_indices[subset_indices]
    order = np.argsort(subset_morse, kind="stable")
    body_indices = subset_indices[order]
    sorted_morse = subset_morse[order]
    counts = np.bincount(sorted_morse, minlength=n_total_cells)
    cell_start_indices = np.empty(n_total_cells + 1, dtype=np.int64)
    cell_start_indices[0] = 0
    cell_start_indices[1:] = np.cumsum(counts)
    return subset_indices, cell_start_indices, body_indices


def synchronize_owned_array(comm: MPI.Comm, local_values: np.ndarray, owned_indices: np.ndarray, global_shape: tuple[int, ...]):
    sendbuf = np.zeros(global_shape, dtype=np.float32)
    if owned_indices.size > 0:
        sendbuf[owned_indices] = local_values
    recvbuf = np.empty(global_shape, dtype=np.float32)
    comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
    return recvbuf


def distributed_step(comm: MPI.Comm,
                     positions: np.ndarray,
                     velocities: np.ndarray,
                     masses: np.ndarray,
                     grid_min: np.ndarray,
                     cell_size: np.ndarray,
                     n_cells: np.ndarray,
                     ix_start: int,
                     ix_end: int,
                     dt: float):
    rank = comm.Get_rank()
    n_total_cells = int(np.prod(n_cells))

    comp_time = 0.0
    comm_time = 0.0

    t0 = MPI.Wtime()
    cell_coords, morse_indices = compute_cell_data(positions, grid_min, cell_size, n_cells)
    owned_mask = (cell_coords[:, 0] >= ix_start) & (cell_coords[:, 0] < ix_end)
    ghost_mask = (cell_coords[:, 0] >= max(0, ix_start - 2)) & (cell_coords[:, 0] < min(int(n_cells[0]), ix_end + 2))
    owned_indices = np.flatnonzero(owned_mask).astype(np.int64)
    _, cell_start_indices, body_indices = build_subset_lists(morse_indices, ghost_mask, n_total_cells)
    local_cell_masses, local_weighted_positions = accumulate_owned_cells(owned_indices, morse_indices, masses, positions, n_total_cells)
    t1 = MPI.Wtime()
    comp_time += t1 - t0

    t0 = MPI.Wtime()
    global_cell_masses = np.empty_like(local_cell_masses)
    global_weighted_positions = np.empty_like(local_weighted_positions)
    comm.Allreduce(local_cell_masses, global_cell_masses, op=MPI.SUM)
    comm.Allreduce(local_weighted_positions, global_weighted_positions, op=MPI.SUM)
    t1 = MPI.Wtime()
    comm_time += t1 - t0

    t0 = MPI.Wtime()
    global_cell_com_positions = np.zeros_like(global_weighted_positions)
    non_zero = global_cell_masses > 0.0
    global_cell_com_positions[non_zero] = global_weighted_positions[non_zero] / global_cell_masses[non_zero, None]
    accelerations = compute_acceleration_owned(
        owned_indices,
        positions,
        masses,
        cell_coords,
        cell_start_indices,
        body_indices,
        global_cell_masses,
        global_cell_com_positions,
        n_cells,
    )
    local_positions = positions[owned_indices] + velocities[owned_indices] * dt + 0.5 * accelerations * dt * dt
    t1 = MPI.Wtime()
    comp_time += t1 - t0

    t0 = MPI.Wtime()
    positions = synchronize_owned_array(comm, local_positions, owned_indices, positions.shape)
    t1 = MPI.Wtime()
    comm_time += t1 - t0

    t0 = MPI.Wtime()
    new_cell_coords, new_morse_indices = compute_cell_data(positions, grid_min, cell_size, n_cells)
    ghost_mask = (new_cell_coords[:, 0] >= max(0, ix_start - 2)) & (new_cell_coords[:, 0] < min(int(n_cells[0]), ix_end + 2))
    _, new_cell_start_indices, new_body_indices = build_subset_lists(new_morse_indices, ghost_mask, n_total_cells)
    local_cell_masses, local_weighted_positions = accumulate_owned_cells(owned_indices, new_morse_indices, masses, positions, n_total_cells)
    t1 = MPI.Wtime()
    comp_time += t1 - t0

    t0 = MPI.Wtime()
    comm.Allreduce(local_cell_masses, global_cell_masses, op=MPI.SUM)
    comm.Allreduce(local_weighted_positions, global_weighted_positions, op=MPI.SUM)
    t1 = MPI.Wtime()
    comm_time += t1 - t0

    t0 = MPI.Wtime()
    global_cell_com_positions.fill(0.0)
    non_zero = global_cell_masses > 0.0
    global_cell_com_positions[non_zero] = global_weighted_positions[non_zero] / global_cell_masses[non_zero, None]
    new_accelerations = compute_acceleration_owned(
        owned_indices,
        positions,
        masses,
        new_cell_coords,
        new_cell_start_indices,
        new_body_indices,
        global_cell_masses,
        global_cell_com_positions,
        n_cells,
    )
    local_velocities = velocities[owned_indices] + 0.5 * (accelerations + new_accelerations) * dt
    t1 = MPI.Wtime()
    comp_time += t1 - t0

    t0 = MPI.Wtime()
    velocities = synchronize_owned_array(comm, local_velocities, owned_indices, velocities.shape)
    t1 = MPI.Wtime()
    comm_time += t1 - t0

    return positions, velocities, owned_indices.shape[0], comp_time, comm_time


def parse_args():
    filename = "data/galaxy_5000"
    dt = 0.001
    n_cells_per_dir = (20, 20, 1)
    steps = 5
    warmup = 1
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        n_cells_per_dir = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    if len(sys.argv) > 6:
        steps = int(sys.argv[6])
    if len(sys.argv) > 7:
        warmup = int(sys.argv[7])
    return filename, dt, n_cells_per_dir, steps, warmup


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename, dt, n_cells_per_dir, steps, warmup = parse_args()
    positions, velocities, masses, box = load_system(filename)

    n_cells = np.array(n_cells_per_dir, dtype=np.int64)
    grid_min = box[0].astype(np.float32)
    grid_max = box[1].astype(np.float32)
    cell_size = (grid_max - grid_min) / n_cells
    counts, displs = decompose_axis(int(n_cells[0]), size)
    ix_start = int(displs[rank])
    ix_end = int(displs[rank] + counts[rank])

    if rank == 0:
        print(
            f"Simulation MPI distribuée de {filename} avec dt = {dt}, grille {n_cells_per_dir}, "
            f"processus={size}, warmup={warmup}, steps={steps}"
        )

    owned_counts = []
    for _ in range(warmup):
        positions, velocities, owned_count, _, _ = distributed_step(
            comm, positions, velocities, masses, grid_min, cell_size, n_cells, ix_start, ix_end, dt
        )
        owned_counts.append(owned_count)

    comm.Barrier()
    total_t0 = MPI.Wtime()
    comp_time = 0.0
    comm_time = 0.0
    for _ in range(steps):
        positions, velocities, owned_count, comp_dt, comm_dt = distributed_step(
            comm, positions, velocities, masses, grid_min, cell_size, n_cells, ix_start, ix_end, dt
        )
        owned_counts.append(owned_count)
        comp_time += comp_dt
        comm_time += comm_dt
    comm.Barrier()
    total_t1 = MPI.Wtime()

    total_max = comm.allreduce(total_t1 - total_t0, op=MPI.MAX)
    comp_max = comm.allreduce(comp_time, op=MPI.MAX)
    comm_max = comm.allreduce(comm_time, op=MPI.MAX)
    min_owned = comm.reduce(min(owned_counts) if owned_counts else 0, op=MPI.MIN, root=0)
    max_owned = comm.reduce(max(owned_counts) if owned_counts else 0, op=MPI.MAX, root=0)

    if rank == 0:
        print(
            f"RESULT p={size} total_ms={1000.0 * total_max / steps:.2f} "
            f"comp_ms={1000.0 * comp_max / steps:.2f} comm_ms={1000.0 * comm_max / steps:.2f} "
            f"owned_min={min_owned} owned_max={max_owned}"
        )


if __name__ == "__main__":
    main()