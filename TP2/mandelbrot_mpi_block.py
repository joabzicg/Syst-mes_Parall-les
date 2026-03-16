# mandelbrot_mpi_block.py
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
        # Zones de convergence connues (mêmes checks do seu mandelbrot.py)
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


def block_partition(total_rows: int, p: int, rank: int):
    """Divide total_rows em blocos contíguos, o mais equilibrado possível.
    Retorna (start_row, n_rows)."""
    q, r = divmod(total_rows, p)
    n_rows = q + (1 if rank < r else 0)
    start = rank * q + min(rank, r)
    return start, n_rows


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    # Mesmos parâmetros do seu mandelbrot.py (você pode ajustar)
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024

    scaleX = 3.0 / width
    scaleY = 2.25 / height

    # 1) Partição por blocos de linhas (y)
    y0, local_h = block_partition(height, p, rank)

    # Cada processo computa local_h linhas: array (local_h, width) = (y, x)
    local_conv = np.empty((local_h, width), dtype=np.float64)

    # 2) Medir tempo do CÁLCULO paralelo (somente o loop pesado)
    comm.Barrier()
    t0 = MPI.Wtime()

    for j in range(local_h):
        y = y0 + j
        imag = -1.125 + scaleY * y
        for x in range(width):
            c = complex(-2.0 + scaleX * x, imag)
            local_conv[j, x] = mandelbrot_set.convergence(c, smooth=True)

    comm.Barrier()
    t1 = MPI.Wtime()

    local_compute = t1 - t0
    # Tempo "do cálculo" do paralelo = o mais lento (MAX)
    t_compute = comm.reduce(local_compute, op=MPI.MAX, root=0)

    # 3) Reunir a imagem no rank 0 (Gatherv porque pode sobrar linhas)
    sendbuf = local_conv.ravel()

    sendcount = sendbuf.size
    counts = comm.gather(sendcount, root=0)

    if rank == 0:
        displs = np.zeros(p, dtype=np.int64)
        displs[1:] = np.cumsum(counts[:-1])
        recvbuf = np.empty(np.sum(counts), dtype=np.float64)
    else:
        displs = None
        recvbuf = None

    comm.Barrier()
    tg0 = MPI.Wtime()

    comm.Gatherv(sendbuf, [recvbuf, counts, displs, MPI.DOUBLE], root=0)

    comm.Barrier()
    tg1 = MPI.Wtime()

    local_total = tg1 - t0
    t_total = comm.reduce(local_total, op=MPI.MAX, root=0)  # cálculo + gather

    # 4) Rank 0 remonta e salva a imagem
    if rank == 0:
        convergence = recvbuf.reshape(height, width)  # (H, W)

        ti0 = MPI.Wtime()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        ti1 = MPI.Wtime()

        image.save("mandelbrot_mpi_block.png")

        print(f"[p={p}] Tempo cálculo (max ranks): {t_compute:.6f} s")
        print(f"[p={p}] Tempo total (cálculo + gather): {t_total:.6f} s")
        print(f"[p={p}] Tempo constituição imagem (rank0): {(ti1-ti0):.6f} s")
        print("Imagem salva em: mandelbrot_mpi_block.png")


if __name__ == "__main__":
    main()
