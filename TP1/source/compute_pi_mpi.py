from mpi4py import MPI
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nb_samples = 10_000_000
if len(sys.argv) > 1:
    nb_samples = int(sys.argv[1])

base = nb_samples // size
rem = nb_samples % size
local_n = base + (1 if rank < rem else 0)

np.random.seed(int(time.time()) + rank * 12345)

comm.Barrier()
t0 = time.time()

x = 2.0 * np.random.random_sample(local_n) - 1.0
y = 2.0 * np.random.random_sample(local_n) - 1.0
hits = np.count_nonzero(x * x + y * y <= 1.0)

local_hits = np.array(hits, dtype=np.int64)
local_samples = np.array(local_n, dtype=np.int64)

global_hits = np.array(0, dtype=np.int64)
global_samples = np.array(0, dtype=np.int64)

comm.Reduce(local_hits, global_hits, op=MPI.SUM, root=0)
comm.Reduce(local_samples, global_samples, op=MPI.SUM, root=0)

comm.Barrier()
t1 = time.time()

if rank == 0:
    pi = 4.0 * global_hits / global_samples
    print(f"Temps (mpi4py): {t1 - t0:.6f} s")
    print(f"pi ≈ {pi:.8f}")
