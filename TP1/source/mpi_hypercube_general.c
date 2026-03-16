#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <dimension d>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int d = atoi(argv[1]);
    int expected = 1 << d;
    if (size != expected) {
        if (rank == 0) {
            fprintf(stderr, "Need exactly 2^d processes. d=%d => %d processes\n", d, expected);
        }
        MPI_Finalize();
        return 1;
    }

    int token = -1;
    if (rank == 0) token = 42;

    for (int step = 0; step < d; ++step) {
        int offset = 1 << step;
        int group = 1 << (step + 1);

        if (rank < group) {
            if (rank < offset) {
                // send to partner in upper half
                int partner = rank + offset;
                MPI_Send(&token, 1, MPI_INT, partner, step, MPI_COMM_WORLD);
            } else {
                // receive from partner in lower half
                int partner = rank - offset;
                MPI_Recv(&token, 1, MPI_INT, partner, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("Rank %d has token=%d\n", rank, token);

    MPI_Finalize();
    return 0;
}
