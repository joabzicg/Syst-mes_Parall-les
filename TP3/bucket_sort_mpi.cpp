#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

static std::vector<int> build_displs(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size(), 0);
    std::partial_sum(counts.begin(), counts.end() - 1, displs.begin() + 1);
    return displs;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nbp = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);

    int N = 100000;
    if (argc > 1) {
        N = std::max(1, std::atoi(argv[1]));
    }

    if (N % nbp != 0) {
        if (rank == 0) {
            std::cerr << "Error: N must be divisible by nbp." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<double> global_data;
    if (rank == 0) {
        global_data.resize(N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < N; ++i) {
            global_data[i] = dist(gen);
        }
    }

    int local_n = N / nbp;
    std::vector<double> local_data(local_n);
    MPI_Scatter(global_data.data(), local_n, MPI_DOUBLE, local_data.data(), local_n,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::sort(local_data.begin(), local_data.end());

    const int sample_count = nbp + 1;
    std::vector<double> local_samples(sample_count, 0.0);
    if (!local_data.empty()) {
        for (int i = 0; i < sample_count; ++i) {
            int idx = static_cast<int>(
                (static_cast<double>(i) * (local_data.size() - 1)) / (sample_count - 1));
            local_samples[i] = local_data[idx];
        }
    }

    std::vector<double> all_samples;
    if (rank == 0) {
        all_samples.resize(sample_count * nbp);
    }

    MPI_Gather(local_samples.data(), sample_count, MPI_DOUBLE, all_samples.data(),
               sample_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> splitters;
    if (rank == 0) {
        std::sort(all_samples.begin(), all_samples.end());
        splitters.resize(nbp - 1);
        for (int i = 1; i < nbp; ++i) {
            splitters[i - 1] = all_samples[i * (nbp + 1)];
        }
    } else {
        splitters.resize(nbp - 1);
    }

    if (nbp > 1) {
        MPI_Bcast(splitters.data(), nbp - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    std::vector<std::vector<double>> buckets(nbp);
    for (double v : local_data) {
        int b = 0;
        if (!splitters.empty()) {
            b = static_cast<int>(std::upper_bound(splitters.begin(), splitters.end(), v) -
                                 splitters.begin());
        }
        buckets[b].push_back(v);
    }

    std::vector<int> sendcounts(nbp, 0);
    for (int i = 0; i < nbp; ++i) {
        sendcounts[i] = static_cast<int>(buckets[i].size());
    }

    std::vector<int> recvcounts(nbp, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> senddispls = build_displs(sendcounts);
    std::vector<int> recvdispls = build_displs(recvcounts);

    int send_total = std::accumulate(sendcounts.begin(), sendcounts.end(), 0);
    int recv_total = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

    std::vector<double> sendbuf(send_total);
    int offset = 0;
    for (int i = 0; i < nbp; ++i) {
        std::copy(buckets[i].begin(), buckets[i].end(), sendbuf.begin() + offset);
        offset += sendcounts[i];
    }

    std::vector<double> recvbuf(recv_total);
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE,
                  recvbuf.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);

    std::vector<double> merged;
    merged.reserve(recv_total);
    for (int i = 0; i < nbp; ++i) {
        int count = recvcounts[i];
        if (count == 0) {
            continue;
        }
        const double* seg_begin = recvbuf.data() + recvdispls[i];
        const double* seg_end = seg_begin + count;
        if (merged.empty()) {
            merged.assign(seg_begin, seg_end);
        } else {
            std::vector<double> temp;
            temp.resize(merged.size() + static_cast<size_t>(count));
            std::merge(merged.begin(), merged.end(), seg_begin, seg_end, temp.begin());
            merged.swap(temp);
        }
    }

    int local_sorted_size = static_cast<int>(merged.size());
    std::vector<int> final_counts;
    if (rank == 0) {
        final_counts.resize(nbp, 0);
    }

    MPI_Gather(&local_sorted_size, 1, MPI_INT, final_counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    std::vector<int> final_displs;
    if (rank == 0) {
        final_displs = build_displs(final_counts);
        global_data.resize(N);
    }

    MPI_Gatherv(merged.data(), local_sorted_size, MPI_DOUBLE, global_data.data(),
                final_counts.data(), final_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
