#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 8) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly 8 processes (dim=3)\n");
        }
        MPI_Finalize();
        return 1;
    }

    int token = -1;

    // Step 0 (bit 0): 0 -> 1
    if (rank == 0) {
        token = 42;
        MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 1 (bit 1): 0->2, 1->3
    if (rank == 0) {
        MPI_Send(&token, 1, MPI_INT, 2, 1, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Send(&token, 1, MPI_INT, 3, 1, MPI_COMM_WORLD);
    } else if (rank == 2) {
        MPI_Recv(&token, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 3) {
        MPI_Recv(&token, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 2 (bit 2): 0->4,1->5,2->6,3->7
    if (rank == 0) {
        MPI_Send(&token, 1, MPI_INT, 4, 2, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Send(&token, 1, MPI_INT, 5, 2, MPI_COMM_WORLD);
    } else if (rank == 2) {
        MPI_Send(&token, 1, MPI_INT, 6, 2, MPI_COMM_WORLD);
    } else if (rank == 3) {
        MPI_Send(&token, 1, MPI_INT, 7, 2, MPI_COMM_WORLD);
    } else if (rank == 4) {
        MPI_Recv(&token, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 5) {
        MPI_Recv(&token, 1, MPI_INT, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 6) {
        MPI_Recv(&token, 1, MPI_INT, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 7) {
        MPI_Recv(&token, 1, MPI_INT, 3, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("Rank %d has token=%d\n", rank, token);

    MPI_Finalize();
    return 0;
}
