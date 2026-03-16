# Produit matrice-vecteur v = A.u (partition par colonnes)
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
    col_start = rank * nloc
    col_end = col_start + nloc

    # Construire u complet localement (petit), puis extraire u_local
    u = np.array([i + 1.0 for i in range(dim)], dtype=np.float64)
    u_local = u[col_start:col_end]

    # Construire localement les colonnes de A utiles
    # A_ij = (i + j) % dim + 1
    rows = np.arange(dim, dtype=np.int64)[:, None]
    cols = np.arange(col_start, col_end, dtype=np.int64)[None, :]
    A_local = ((rows + cols) % dim + 1).astype(np.float64)

    comm.Barrier()
    t0 = MPI.Wtime()

    # Contribution locale: somme sur les colonnes locales
    v_local = A_local @ u_local

    # Somme globale pour obtenir v complet sur tous les rangs
    v = np.empty_like(v_local)
    comm.Allreduce(v_local, v, op=MPI.SUM)

    comm.Barrier()
    t1 = MPI.Wtime()
    t_total = t1 - t0

    t_max = comm.reduce(t_total, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"[p={p}] Temps total (calcul + allreduce): {t_max:.6f} s")
        # Affichage court pour validation
        print(f"v[0:5] = {v[:5]}")


if __name__ == "__main__":
    main()
