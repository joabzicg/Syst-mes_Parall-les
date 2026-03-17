import sys

import numpy as np
from mpi4py import MPI

from nbodies_grid_numba_parallel import NBodySystem


TAG_INIT = 1
TAG_STEP = 2
TAG_POSITIONS = 3
TAG_STOP = 4


def build_visualization_payload(system: NBodySystem):
    colors = np.array(system.colors, dtype=np.float32)
    luminosities = np.clip(system.masses / system.max_mass, 0.5, 1.0).astype(np.float32)
    bounds = [
        [system.box[0][0], system.box[1][0]],
        [system.box[0][1], system.box[1][1]],
        [system.box[0][2], system.box[1][2]],
    ]
    return {
        "positions": np.ascontiguousarray(system.positions, dtype=np.float32),
        "colors": colors,
        "luminosities": luminosities,
        "bounds": bounds,
    }


def parse_args():
    filename = "data/galaxy_1000"
    dt = 0.001
    n_cells_per_dir = (20, 20, 1)
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        n_cells_per_dir = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    return filename, dt, n_cells_per_dir


def run_display_process(comm, dt: float):
    import visualizer3d

    payload = comm.recv(source=1, tag=TAG_INIT)
    recv_buffer = np.empty_like(payload["positions"])
    recv_buffer[:] = payload["positions"]

    visualizer = visualizer3d.Visualizer3D(
        payload["positions"],
        payload["colors"],
        payload["luminosities"],
        payload["bounds"],
    )

    def updater(step_dt: float):
        comm.send(float(step_dt), dest=1, tag=TAG_STEP)
        comm.Recv([recv_buffer, MPI.FLOAT], source=1, tag=TAG_POSITIONS)
        return recv_buffer

    try:
        visualizer.run(updater=updater, dt=dt)
    finally:
        comm.send(None, dest=1, tag=TAG_STOP)


def run_compute_process(comm, filename: str, n_cells_per_dir: tuple[int, int, int]):
    system = NBodySystem(filename, ncells_per_dir=n_cells_per_dir)
    payload = build_visualization_payload(system)
    comm.send(payload, dest=0, tag=TAG_INIT)

    while True:
        status = MPI.Status()
        message = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            break

        dt = float(message)
        system.update_positions(dt)
        positions = np.ascontiguousarray(system.positions, dtype=np.float32)
        comm.Send([positions, MPI.FLOAT], dest=0, tag=TAG_POSITIONS)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            raise SystemExit("Cette version MPI doit être lancée avec exactement 2 processus.")
        return

    filename, dt, n_cells_per_dir = parse_args()

    if rank == 0:
        print(
            f"Simulation MPI affichage/calcul de {filename} avec dt = {dt} "
            f"et grille {n_cells_per_dir}"
        )
        run_display_process(comm, dt)
    elif rank == 1:
        run_compute_process(comm, filename, n_cells_per_dir)


if __name__ == "__main__":
    main()