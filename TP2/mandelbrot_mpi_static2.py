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


def rows_static_block_cyclic(height: int, p: int, rank: int, block_size: int):
    """
    block_size = 1  -> cyclique pur (ligne par ligne)
    block_size = 8  -> block-cyclique (8 lignes consécutives)
    """
    rows = []
    step = p * block_size
    for base in range(0, height, step):
        start = base + rank * block_size
        end = min(start + block_size, height)
        if start < height:
            rows.extend(range(start, end))
    return np.array(rows, dtype=np.int32)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    # Paramètres (comme le séquentiel)
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024
    scaleX = 3.0 / width
    scaleY = 2.25 / height

    # >>> Répartition statique améliorée :
    BLOCK_SIZE = 1   # 1 = cyclique pur ; essaie 8 si tu veux block-cyclique
    my_rows = rows_static_block_cyclic(height, p, rank, BLOCK_SIZE)
    local_h = my_rows.size

    local_conv = np.empty((local_h, width), dtype=np.float64)

    # Mesure du temps de calcul (partie lourde)
    comm.Barrier()
    t0 = MPI.Wtime()

    for j, y in enumerate(my_rows):
        imag = -1.125 + scaleY * y
        for x in range(width):
            c = complex(-2.0 + scaleX * x, imag)
            local_conv[j, x] = mandelbrot_set.convergence(c, smooth=True)

    comm.Barrier()
    t1 = MPI.Wtime()

    local_compute = t1 - t0
    t_compute = comm.reduce(local_compute, op=MPI.MAX, root=0)

    # Gather des indices de lignes
    my_count_rows = np.array([local_h], dtype=np.int32)
    counts_rows = comm.gather(my_count_rows[0], root=0)

    if rank == 0:
        displs_rows = np.zeros(p, dtype=np.int32)
        displs_rows[1:] = np.cumsum(counts_rows[:-1])
        all_rows = np.empty(sum(counts_rows), dtype=np.int32)
    else:
        displs_rows = None
        all_rows = None

    comm.Gatherv(my_rows, [all_rows, counts_rows, displs_rows, MPI.INT], root=0)

    # Gather des données (convergence)
    sendbuf = local_conv.ravel()
    my_count_data = sendbuf.size
    counts_data = comm.gather(my_count_data, root=0)

    if rank == 0:
        displs_data = np.zeros(p, dtype=np.int64)
        displs_data[1:] = np.cumsum(counts_data[:-1])
        all_data = np.empty(sum(counts_data), dtype=np.float64)
    else:
        displs_data = None
        all_data = None

    comm.Barrier()
    tg0 = MPI.Wtime()
    comm.Gatherv(sendbuf, [all_data, counts_data, displs_data, MPI.DOUBLE], root=0)
    comm.Barrier()
    tg1 = MPI.Wtime()

    local_total = tg1 - t0
    t_total = comm.reduce(local_total, op=MPI.MAX, root=0)

    # Reconstruction + sauvegarde sur rank 0
    if rank == 0:
        convergence = np.empty((height, width), dtype=np.float64)

        # Remettre chaque ligne à sa place
        offset = 0
        for idx, y in enumerate(all_rows):
            convergence[y, :] = all_data[offset:offset + width]
            offset += width

        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        image.save("mandelbrot_mpi_static2.png")

        print(f"[p={p}] Temps calcul (max ranks): {t_compute:.6f} s")
        print(f"[p={p}] Temps total (calcul + gather): {t_total:.6f} s")
        print("Image sauvée : mandelbrot_mpi_static2.png")


if __name__ == "__main__":
    main()
