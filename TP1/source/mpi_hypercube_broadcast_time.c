#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <dimension d> [iterations]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int d = atoi(argv[1]);
    int expected = 1 << d;
    if (size != expected) {
        if (rank == 0) fprintf(stderr, "Need exactly 2^d processes. d=%d => %d processes\n", d, expected);
        MPI_Finalize();
        return 1;
    }

    int iters = 1000;
    if (argc > 2) iters = atoi(argv[2]);

    int token = 42;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int it = 0; it < iters; ++it) {
        if (rank == 0) token = 42;
        for (int step = 0; step < d; ++step) {
            int offset = 1 << step;
            int group = 1 << (step + 1);

            if (rank < group) {
                if (rank < offset) {
                    int partner = rank + offset;
                    MPI_Send(&token, 1, MPI_INT, partner, step, MPI_COMM_WORLD);
                } else {
                    int partner = rank - offset;
                    MPI_Recv(&token, 1, MPI_INT, partner, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double avg = (t1 - t0) / iters;
    double tmax = 0.0;
    MPI_Reduce(&avg, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("d=%d, p=%d, avg_time=%.9f s\n", d, size, tmax);
    }

    MPI_Finalize();
    return 0;
}
