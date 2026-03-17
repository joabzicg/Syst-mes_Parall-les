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


def build_owner_of_ix(n: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, displs = decompose_axis(n, size)
    owner_of_ix = np.empty(n, dtype=np.int64)
    for rank in range(size):
        start = int(displs[rank])
        end = int(displs[rank] + counts[rank])
        owner_of_ix[start:end] = rank
    return owner_of_ix, counts, displs


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
def accumulate_local_cells(morse_indices: np.ndarray, masses: np.ndarray, positions: np.ndarray, n_total_cells: int):
    local_cell_masses = np.zeros(n_total_cells, dtype=np.float32)
    local_weighted_positions = np.zeros((n_total_cells, 3), dtype=np.float32)
    for ibody in range(morse_indices.shape[0]):
        morse_idx = morse_indices[ibody]
        mass = masses[ibody]
        local_cell_masses[morse_idx] += mass
        local_weighted_positions[morse_idx, :] += positions[ibody, :] * mass
    return local_cell_masses, local_weighted_positions


@njit(parallel=True)
def compute_acceleration_owned_local(n_owned: int,
                                     all_ids: np.ndarray,
                                     all_positions: np.ndarray,
                                     all_masses: np.ndarray,
                                     all_cell_coords: np.ndarray,
                                     cell_start_indices: np.ndarray,
                                     body_indices: np.ndarray,
                                     cell_masses: np.ndarray,
                                     cell_com_positions: np.ndarray,
                                     n_cells: np.ndarray):
    accelerations = np.zeros((n_owned, 3), dtype=np.float32)
    for iloc in prange(n_owned):
        own_id = all_ids[iloc]
        pos = all_positions[iloc]
        cell_idx = all_cell_coords[iloc]
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
                            jlocal = body_indices[j]
                            if all_ids[jlocal] != own_id:
                                direction = all_positions[jlocal] - pos
                                distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    acc += G * direction * inv_dist3 * all_masses[jlocal]
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
        np.arange(len(masses), dtype=np.int64),
        np.array(positions, dtype=np.float32),
        np.array(velocities, dtype=np.float32),
        np.array(masses, dtype=np.float32),
        box,
    )


def empty_payload():
    return (
        np.empty((0,), dtype=np.int64),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def pack_payload(ids: np.ndarray, positions: np.ndarray, masses: np.ndarray):
    return (ids.copy(), positions.copy(), masses.copy())


def concat_payloads(base_ids: np.ndarray, base_positions: np.ndarray, base_masses: np.ndarray, payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    ids_parts = [base_ids]
    pos_parts = [base_positions]
    mass_parts = [base_masses]
    for ids, positions, masses in payloads:
        if ids.size > 0:
            ids_parts.append(ids)
            pos_parts.append(positions)
            mass_parts.append(masses)
    return (
        np.concatenate(ids_parts, axis=0),
        np.concatenate(pos_parts, axis=0),
        np.concatenate(mass_parts, axis=0),
    )


def build_subset_lists(morse_indices: np.ndarray, n_total_cells: int):
    if morse_indices.size == 0:
        cell_start_indices = np.zeros(n_total_cells + 1, dtype=np.int64)
        body_indices = np.empty((0,), dtype=np.int64)
        return cell_start_indices, body_indices
    order = np.argsort(morse_indices, kind="stable")
    body_indices = order.astype(np.int64)
    sorted_morse = morse_indices[order]
    counts = np.bincount(sorted_morse, minlength=n_total_cells)
    cell_start_indices = np.empty(n_total_cells + 1, dtype=np.int64)
    cell_start_indices[0] = 0
    cell_start_indices[1:] = np.cumsum(counts)
    return cell_start_indices, body_indices


def alltoallv_1d(comm: MPI.Comm, sendbuf: np.ndarray, sendcounts: np.ndarray, mpi_type):
    size = comm.Get_size()
    recvcounts = np.array(comm.alltoall(sendcounts.tolist()), dtype=np.int32)
    sdispls = np.zeros(size, dtype=np.int32)
    rdispls = np.zeros(size, dtype=np.int32)
    if size > 1:
        sdispls[1:] = np.cumsum(sendcounts)[:-1]
        rdispls[1:] = np.cumsum(recvcounts)[:-1]
    recvbuf = np.empty(int(recvcounts.sum()), dtype=sendbuf.dtype)
    comm.Alltoallv([sendbuf, sendcounts, sdispls, mpi_type], [recvbuf, recvcounts, rdispls, mpi_type])
    return recvbuf


def redistribute_owned(comm: MPI.Comm,
                       ids: np.ndarray,
                       positions: np.ndarray,
                       velocities: np.ndarray,
                       masses: np.ndarray,
                       grid_min: np.ndarray,
                       cell_size: np.ndarray,
                       n_cells: np.ndarray,
                       owner_of_ix: np.ndarray):
    if ids.size == 0:
        cell_coords = np.empty((0, 3), dtype=np.int64)
        send_ranks = np.empty((0,), dtype=np.int64)
    else:
        cell_coords, _ = compute_cell_data(positions, grid_min, cell_size, n_cells)
        send_ranks = owner_of_ix[cell_coords[:, 0]]

    order = np.argsort(send_ranks, kind="stable") if send_ranks.size > 0 else np.empty((0,), dtype=np.int64)
    sorted_ranks = send_ranks[order] if send_ranks.size > 0 else np.empty((0,), dtype=np.int64)
    sendcounts = np.bincount(sorted_ranks, minlength=comm.Get_size()).astype(np.int32)

    send_ids = ids[order] if ids.size > 0 else np.empty((0,), dtype=np.int64)
    send_masses = masses[order] if masses.size > 0 else np.empty((0,), dtype=np.float32)
    send_positions = positions[order].reshape(-1) if positions.size > 0 else np.empty((0,), dtype=np.float32)
    send_velocities = velocities[order].reshape(-1) if velocities.size > 0 else np.empty((0,), dtype=np.float32)

    recv_ids = alltoallv_1d(comm, send_ids, sendcounts, MPI.LONG_LONG)
    recv_masses = alltoallv_1d(comm, send_masses, sendcounts, MPI.FLOAT)
    recv_positions = alltoallv_1d(comm, send_positions, sendcounts * 3, MPI.FLOAT).reshape(-1, 3)
    recv_velocities = alltoallv_1d(comm, send_velocities, sendcounts * 3, MPI.FLOAT).reshape(-1, 3)
    return recv_ids, recv_positions, recv_velocities, recv_masses


def exchange_ghost_bodies(comm: MPI.Comm,
                          ids: np.ndarray,
                          positions: np.ndarray,
                          masses: np.ndarray,
                          ix_start: int,
                          ix_end: int,
                          grid_min: np.ndarray,
                          cell_size: np.ndarray,
                          n_cells: np.ndarray):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if ids.size == 0:
        cell_coords = np.empty((0, 3), dtype=np.int64)
    else:
        cell_coords, _ = compute_cell_data(positions, grid_min, cell_size, n_cells)

    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    if ids.size > 0:
        left_mask = cell_coords[:, 0] < (ix_start + 2)
        right_mask = cell_coords[:, 0] >= max(ix_start, ix_end - 2)
        left_payload = pack_payload(ids[left_mask], positions[left_mask], masses[left_mask])
        right_payload = pack_payload(ids[right_mask], positions[right_mask], masses[right_mask])
    else:
        left_payload = empty_payload()
        right_payload = empty_payload()

    recv_from_left = comm.sendrecv(right_payload, dest=right, sendtag=10, source=left, recvtag=10)
    recv_from_right = comm.sendrecv(left_payload, dest=left, sendtag=11, source=right, recvtag=11)

    payloads = []
    if recv_from_left is not None:
        payloads.append(recv_from_left)
    if recv_from_right is not None:
        payloads.append(recv_from_right)
    return concat_payloads(ids, positions, masses, payloads)


def compute_global_cells(comm: MPI.Comm,
                         positions: np.ndarray,
                         masses: np.ndarray,
                         grid_min: np.ndarray,
                         cell_size: np.ndarray,
                         n_cells: np.ndarray):
    n_total_cells = int(np.prod(n_cells))
    if positions.size == 0:
        local_morse = np.empty((0,), dtype=np.int64)
    else:
        _, local_morse = compute_cell_data(positions, grid_min, cell_size, n_cells)
    local_cell_masses, local_weighted_positions = accumulate_local_cells(local_morse, masses, positions, n_total_cells)
    global_cell_masses = np.empty_like(local_cell_masses)
    global_weighted_positions = np.empty_like(local_weighted_positions)
    comm.Allreduce(local_cell_masses, global_cell_masses, op=MPI.SUM)
    comm.Allreduce(local_weighted_positions, global_weighted_positions, op=MPI.SUM)
    global_cell_com_positions = np.zeros_like(global_weighted_positions)
    non_zero = global_cell_masses > 0.0
    global_cell_com_positions[non_zero] = global_weighted_positions[non_zero] / global_cell_masses[non_zero, None]
    return global_cell_masses, global_cell_com_positions


def compute_owned_accelerations(owned_ids: np.ndarray,
                                owned_positions: np.ndarray,
                                owned_masses: np.ndarray,
                                ghost_ids: np.ndarray,
                                ghost_positions: np.ndarray,
                                ghost_masses: np.ndarray,
                                grid_min: np.ndarray,
                                cell_size: np.ndarray,
                                n_cells: np.ndarray,
                                global_cell_masses: np.ndarray,
                                global_cell_com_positions: np.ndarray):
    all_ids = np.concatenate([owned_ids, ghost_ids], axis=0)
    all_positions = np.concatenate([owned_positions, ghost_positions], axis=0)
    all_masses = np.concatenate([owned_masses, ghost_masses], axis=0)
    all_cell_coords, all_morse = compute_cell_data(all_positions, grid_min, cell_size, n_cells)
    cell_start_indices, body_indices = build_subset_lists(all_morse, int(np.prod(n_cells)))
    return compute_acceleration_owned_local(
        owned_ids.shape[0],
        all_ids,
        all_positions,
        all_masses,
        all_cell_coords,
        cell_start_indices,
        body_indices,
        global_cell_masses,
        global_cell_com_positions,
        n_cells,
    )


def distributed_step_local(comm: MPI.Comm,
                           ids: np.ndarray,
                           positions: np.ndarray,
                           velocities: np.ndarray,
                           masses: np.ndarray,
                           grid_min: np.ndarray,
                           cell_size: np.ndarray,
                           n_cells: np.ndarray,
                           owner_of_ix: np.ndarray,
                           ix_start: int,
                           ix_end: int,
                           dt: float):
    comp_time = 0.0
    comm_time = 0.0

    t0 = MPI.Wtime()
    ids, positions, velocities, masses = redistribute_owned(
        comm, ids, positions, velocities, masses, grid_min, cell_size, n_cells, owner_of_ix
    )
    t1 = MPI.Wtime()
    comm_time += t1 - t0

    t0 = MPI.Wtime()
    global_cell_masses, global_cell_com_positions = compute_global_cells(
        comm, positions, masses, grid_min, cell_size, n_cells
    )
    t1 = MPI.Wtime()
    comm_time += t1 - t0

    t0 = MPI.Wtime()
    all_ids, all_positions, all_masses = exchange_ghost_bodies(
        comm, ids, positions, masses, ix_start, ix_end, grid_min, cell_size, n_cells
    )
    ghost_ids = all_ids[ids.shape[0]:]
    ghost_positions = all_positions[ids.shape[0]:]
    ghost_masses = all_masses[ids.shape[0]:]
    accelerations = compute_owned_accelerations(
        ids,
        positions,
        masses,
        ghost_ids,
        ghost_positions,
        ghost_masses,
        grid_min,
        cell_size,
        n_cells,
        global_cell_masses,
        global_cell_com_positions,
    )
    positions = positions + velocities * dt + 0.5 * accelerations * dt * dt
    t1 = MPI.Wtime()
    comp_time += t1 - t0

    t0 = MPI.Wtime()
    global_cell_masses, global_cell_com_positions = compute_global_cells(
        comm, positions, masses, grid_min, cell_size, n_cells
    )
    t1 = MPI.Wtime()
    comm_time += t1 - t0

    t0 = MPI.Wtime()
    all_ids, all_positions, all_masses = exchange_ghost_bodies(
        comm, ids, positions, masses, ix_start, ix_end, grid_min, cell_size, n_cells
    )
    ghost_ids = all_ids[ids.shape[0]:]
    ghost_positions = all_positions[ids.shape[0]:]
    ghost_masses = all_masses[ids.shape[0]:]
    new_accelerations = compute_owned_accelerations(
        ids,
        positions,
        masses,
        ghost_ids,
        ghost_positions,
        ghost_masses,
        grid_min,
        cell_size,
        n_cells,
        global_cell_masses,
        global_cell_com_positions,
    )
    velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt
    t1 = MPI.Wtime()
    comp_time += t1 - t0

    return ids, positions, velocities, masses, comp_time, comm_time


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
    global_ids, global_positions, global_velocities, global_masses, box = load_system(filename)

    n_cells = np.array(n_cells_per_dir, dtype=np.int64)
    grid_min = box[0].astype(np.float32)
    grid_max = box[1].astype(np.float32)
    cell_size = (grid_max - grid_min) / n_cells
    owner_of_ix, counts, displs = build_owner_of_ix(int(n_cells[0]), size)
    ix_start = int(displs[rank])
    ix_end = int(displs[rank] + counts[rank])

    global_cell_coords, _ = compute_cell_data(global_positions, grid_min, cell_size, n_cells)
    owner_mask = owner_of_ix[global_cell_coords[:, 0]] == rank
    ids = global_ids[owner_mask].copy()
    positions = global_positions[owner_mask].copy()
    velocities = global_velocities[owner_mask].copy()
    masses = global_masses[owner_mask].copy()

    if rank == 0:
        print(
            f"Simulation MPI distribuée locale de {filename} avec dt = {dt}, grille {n_cells_per_dir}, "
            f"processus={size}, warmup={warmup}, steps={steps}"
        )

    owned_counts = []
    for _ in range(warmup):
        ids, positions, velocities, masses, _, _ = distributed_step_local(
            comm, ids, positions, velocities, masses, grid_min, cell_size, n_cells, owner_of_ix, ix_start, ix_end, dt
        )
        owned_counts.append(ids.shape[0])

    comm.Barrier()
    total_t0 = MPI.Wtime()
    comp_time = 0.0
    comm_time = 0.0
    for _ in range(steps):
        ids, positions, velocities, masses, comp_dt, comm_dt = distributed_step_local(
            comm, ids, positions, velocities, masses, grid_min, cell_size, n_cells, owner_of_ix, ix_start, ix_end, dt
        )
        owned_counts.append(ids.shape[0])
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