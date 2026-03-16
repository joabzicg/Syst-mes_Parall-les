# Produit matrice-vecteur v = A.u (partition par lignes)
from mpi4py import MPI
import numpy as np
import sys


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    # Dimension du problème (doit être divisible par p)
    dim = 120
    if len(sys.argv) > 1:
        dim = int(sys.argv[1])

    if dim % p != 0:
        if rank == 0:
            raise ValueError(f"dim={dim} doit être divisible par p={p}")
        return

    nloc = dim // p
    row_start = rank * nloc
    row_end = row_start + nloc

    # Construire u complet localement (petit)
    u = np.array([i + 1.0 for i in range(dim)], dtype=np.float64)

    # Construire localement les lignes de A utiles
    # A_ij = (i + j) % dim + 1
    rows = np.arange(row_start, row_end, dtype=np.int64)[:, None]
    cols = np.arange(dim, dtype=np.int64)[None, :]
    A_local = ((rows + cols) % dim + 1).astype(np.float64)

    comm.Barrier()
    t0 = MPI.Wtime()

    # Produit local : v_local de taille nloc
    v_local = A_local @ u

    # Rassembler toutes les tranches de v sur tous les rangs
    v = np.empty(dim, dtype=np.float64)
    comm.Allgather(v_local, v)

    comm.Barrier()
    t1 = MPI.Wtime()
    t_total = t1 - t0

    t_max = comm.reduce(t_total, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"[p={p}] Temps total (calcul + allgather): {t_max:.6f} s")
        print(f"v[0:5] = {v[:5]}")


if __name__ == "__main__":
    main()
