#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    long long nbSamples = 10000000LL;
    if (argc > 1) {
        nbSamples = atoll(argv[1]);
    }

    long long nbDarts = 0;
    double t0 = omp_get_wtime();

#pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)omp_get_thread_num();
        long long local_hits = 0;

#pragma omp for
        for (long long i = 0; i < nbSamples; ++i) {
            double x = 2.0 * (double)rand_r(&seed) / (double)RAND_MAX - 1.0;
            double y = 2.0 * (double)rand_r(&seed) / (double)RAND_MAX - 1.0;
            if (x * x + y * y <= 1.0) {
                local_hits++;
            }
        }

#pragma omp atomic
        nbDarts += local_hits;
    }

    double t1 = omp_get_wtime();
    double pi = 4.0 * (double)nbDarts / (double)nbSamples;

    printf("pi ≈ %.8f\n", pi);
    printf("samples = %lld\n", nbSamples);
    printf("threads = %d\n", omp_get_max_threads());
    printf("time = %.6f s\n", t1 - t0);

    return 0;
}
