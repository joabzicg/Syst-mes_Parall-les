from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
import matplotlib.cm


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations

        z = 0
        for it in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return it + 1 - log(log(abs(z))) / log(2)
                return it
        return self.max_iterations


TAG_WORK = 1
TAG_STOP = 2
TAG_RESULT_META = 3
TAG_RESULT_DATA = 4


def compute_rows(mandelbrot_set, width, height, scaleX, scaleY, rows):
    local_h = len(rows)
    local_conv = np.empty((local_h, width), dtype=np.float64)
    for j, y in enumerate(rows):
        imag = -1.125 + scaleY * y
        for x in range(width):
            c = complex(-2.0 + scaleX * x, imag)
            local_conv[j, x] = mandelbrot_set.convergence(c, smooth=True)
    return local_conv


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024
    scaleX = 3.0 / width
    scaleY = 2.25 / height

    block_size = 8

    # p == 1: exécution séquentielle
    if p == 1:
        comm.Barrier()
        t0 = MPI.Wtime()
        all_rows = list(range(height))
        convergence = compute_rows(mandelbrot_set, width, height, scaleX, scaleY, all_rows)
        comm.Barrier()
        t1 = MPI.Wtime()
        t_total = t1 - t0

        print(f"[p=1] Temps total (calcul + gather): {t_total:.6f} s")
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        image.save("mandelbrot_mpi_master_slave.png")
        print("Image sauvée : mandelbrot_mpi_master_slave.png")
        return

    # p > 1 : master/worker
    comm.Barrier()
    t0 = MPI.Wtime()

    if rank == 0:
        convergence = np.empty((height, width), dtype=np.float64)
        next_row = 0
        active_workers = 0

        # Envoi initial de tâches
        for dest in range(1, p):
            if next_row >= height:
                comm.send(None, dest=dest, tag=TAG_STOP)
                continue
            n_rows = min(block_size, height - next_row)
            comm.send((next_row, n_rows), dest=dest, tag=TAG_WORK)
            next_row += n_rows
            active_workers += 1

        received_rows = 0
        status = MPI.Status()

        while active_workers > 0:
            meta = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT_META, status=status)
            src = status.Get_source()
            start_row, n_rows = meta

            buf = np.empty((n_rows, width), dtype=np.float64)
            comm.Recv([buf, MPI.DOUBLE], source=src, tag=TAG_RESULT_DATA)

            convergence[start_row:start_row + n_rows, :] = buf
            received_rows += n_rows

            if next_row < height:
                n_rows = min(block_size, height - next_row)
                comm.send((next_row, n_rows), dest=src, tag=TAG_WORK)
                next_row += n_rows
            else:
                comm.send(None, dest=src, tag=TAG_STOP)
                active_workers -= 1

    else:
        while True:
            status = MPI.Status()
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == TAG_STOP:
                break

            start_row, n_rows = task
            rows = list(range(start_row, start_row + n_rows))
            buf = compute_rows(mandelbrot_set, width, height, scaleX, scaleY, rows)

            comm.send((start_row, n_rows), dest=0, tag=TAG_RESULT_META)
            comm.Send([buf, MPI.DOUBLE], dest=0, tag=TAG_RESULT_DATA)

    comm.Barrier()
    t1 = MPI.Wtime()
    local_total = t1 - t0
    t_total = comm.reduce(local_total, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"[p={p}] Temps total (calcul + gather): {t_total:.6f} s")
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        image.save("mandelbrot_mpi_master_slave.png")
        print("Image sauvée : mandelbrot_mpi_master_slave.png")


if __name__ == "__main__":
    main()
