#include "../lib/cu_linalg.h"
#include "../lib/cu_SPMV.h"
#include "../lib/SPMV.h"
#include "../lib/adjMatrix.h"
#include "../lib/helpers.h"

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#include <cuda_profiler_api.h>

#define BLOCKSIZE 4
#define SEED 1234 // To seed RNG
#define WIDTH 81  // for formatting std::cout output

template <typename T>
void cu_linalg_test(const unsigned n, adjMatrix &A);

void cuda_start_timer(cudaEvent_t &start, cudaEvent_t &end);
float cuda_end_timer(cudaEvent_t &start, cudaEvent_t &end);

int main(void)
{
    unsigned n{10'000};

    long unsigned edges{n * 100};
    timeval start, end;
    gettimeofday(&start, NULL);
    adjMatrix A(n, edges);
    gettimeofday(&end, NULL);
    std::cout << "Time elapsed for build random adjacency matrix with n = " << n << " edges = " << edges << ": "
              << end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0 << " seconds\n\n";

    std::cout << "\nTesting CUDA vs serial execution of linalg functions for n = " << n << "\n\n";
    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
              << std::setfill(' ');
    std::cout << "SINGLE PRECISION\n";
    cu_linalg_test<float>(n, A);
    std::cout << "\n";
    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
              << std::setfill(' ');
    std::cout << "DOUBLE PRECISION\n";
    cu_linalg_test<double>(n, A);
}

template <typename T>
void cu_linalg_test(const unsigned n, adjMatrix &A)
{
    T ans;

    std::random_device rd;
    std::mt19937 gen{SEED};
    //std::mt19937 gen{rd()};
    std::uniform_real_distribution<T> U(0.0, 1.0);

    std::vector<T> x(n), y(n), ans_vec(n);

    for (auto it = x.begin(); it != x.end(); it++)
        *it = U(gen);

    for (auto it = y.begin(); it != y.end(); it++)
        *it = U(gen);

    unsigned block_size{BLOCKSIZE}, num_blocks{n / block_size + (n % block_size ? 1 : 0)}, h_blocks{n / (2 * block_size) + (n % (2 * block_size) ? 1 : 0)};

    dim3 blocks{num_blocks}, half_blocks{h_blocks}, threads{block_size}, one_block{1u};

    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
              << std::setfill(' ');
    std::cout << "\t\tCUDA\t\tSerial\t\tRelative Error\t\tSpeedup\n";
    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
              << std::setfill(' ');

    T *x_d, *y_d, *tmp_d, *ans_d, *spMV_ans_d;
    long unsigned *IA_d, *JA_d;

    cudaMalloc((void **)&x_d, sizeof(T) * n);
    cudaMalloc((void **)&y_d, sizeof(T) * n);
    cudaMalloc((void **)&tmp_d, sizeof(T) * num_blocks);
    cudaMalloc((void **)&ans_d, sizeof(T));
    cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, &y[0], sizeof(T) * n, cudaMemcpyHostToDevice);
    {
        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_dot_prod<T, BLOCKSIZE><<<half_blocks, threads, block_size * sizeof(T)>>>(x_d, y_d, n, tmp_d);
        cu_reduce<T, BLOCKSIZE><<<1, threads, block_size * sizeof(T)>>>(tmp_d, num_blocks, ans_d);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        auto serial_ans = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        std::cout << "Inner product: \t" << ans
                  << "  \t" << serial_ans << "\t\t"
                  << (serial_ans - ans) / serial_ans << "\t\t" << speedup << "\n\n";
    }
    {
        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_norm_sq<T, BLOCKSIZE><<<half_blocks, threads, block_size * sizeof(T)>>>(x_d, n, tmp_d);
        cu_reduce<T, BLOCKSIZE><<<1, threads, block_size * sizeof(T)>>>(tmp_d, num_blocks, ans_d);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        auto serial_ans = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        std::cout << "Norm squared: \t" << ans
                  << "  \t" << serial_ans << "\t\t"
                  << (serial_ans - ans) / serial_ans << "\t\t" << speedup << "\n\n";
    }
    {
        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_reduce<T, BLOCKSIZE><<<half_blocks, threads, block_size * sizeof(T)>>>(x_d, n, tmp_d);
        cu_reduce<T, BLOCKSIZE><<<1, threads, block_size * sizeof(T)>>>(tmp_d, num_blocks, ans_d);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        auto serial_ans = std::accumulate(x.begin(), x.end(), 0.0);
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        std::cout << "Reduce: \t" << ans
                  << "  \t" << serial_ans << "\t\t"
                  << (serial_ans - ans) / serial_ans << "\t\t" << speedup << "\n\n";
    }
    {
        cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, &y[0], sizeof(T) * n, cudaMemcpyHostToDevice);

        T alpha{2};
        std::vector<T> gpu_ans_vec(n);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_dpax<T><<<blocks, threads>>>(x_d, alpha, y_d, n);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&gpu_ans_vec[0], x_d, n * sizeof(T), cudaMemcpyDeviceToHost);

        auto ans{std::inner_product(gpu_ans_vec.begin(), gpu_ans_vec.end(), gpu_ans_vec.begin(), 0.0)};


        int idx{0};
        timeval start, end;
        gettimeofday(&start, NULL);
        std::for_each(x.begin(), x.end(), [&](T & a)
                      { a -= alpha * y[idx++]; });
        auto serial_ans{std::inner_product(x.begin(), x.end(), x.begin(), 0.0)};
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        std::cout << "Vector add: \t" << ans
                  << "  \t" << serial_ans << "\t\t"
                  << (serial_ans - ans) / serial_ans << "\t\t\t" << speedup << "\n\n";
    }
    {
        cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, &y[0], sizeof(T) * n, cudaMemcpyHostToDevice);

        T alpha{4};
        std::vector<T> gpu_ans_vec(n);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_dvexda<T><<<blocks, threads>>>(x_d, alpha, y_d, n);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&gpu_ans_vec[0], x_d, n * sizeof(T), cudaMemcpyDeviceToHost);

        auto ans{std::inner_product(gpu_ans_vec.begin(), gpu_ans_vec.end(), gpu_ans_vec.begin(), 0.0)};


        int idx{0};
        timeval start, end;
        gettimeofday(&start, NULL);
        std::for_each(x.begin(), x.end(), [&](T & a)
                      { a = y[idx++]/alpha; });
        auto serial_ans{std::inner_product(x.begin(), x.end(), x.begin(), 0.0)};
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        std::cout << "v=v/scalar: \t" << ans
                  << "  \t" << serial_ans << "\t\t"
                  << (serial_ans - ans) / serial_ans << "\t\t\t" << speedup << "\n\n";
    }

    cudaMalloc((void **)&IA_d, sizeof(long unsigned) * (n + 1));
    cudaMalloc((void **)&JA_d, sizeof(long unsigned) * A.edge_count * 2);
    cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
    cudaMemcpy(IA_d, A.row_offset, sizeof(long unsigned) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(JA_d, A.col_idx, sizeof(long unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
    {
        std::vector<T> gpu_ans_vec(n);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_spMV1<T><<<blocks, threads>>>(IA_d, JA_d, static_cast<unsigned long>(n), x_d, spMV_ans_d);

        cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        spMV(A, &x[0], &ans_vec[0]);
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        T relative_error{0};
        unsigned max_idx{0u};
        diff_arrays(&gpu_ans_vec[0], &ans_vec[0], n, relative_error, max_idx);

        ans = std::sqrt(std::inner_product(gpu_ans_vec.begin(), gpu_ans_vec.end(), gpu_ans_vec.begin(), 0));
        auto serial_ans = std::sqrt(std::inner_product(ans_vec.begin(), ans_vec.end(), ans_vec.begin(), 0));

        std::cout << "SPMV1: \t\t" << ans
                  << "\t\t" << serial_ans << "\t\t"
                  << relative_error / serial_ans << "\t\t\t" << speedup << "\n\n";
    }
    {
        std::vector<T> gpu_ans_vec(n);
        std::vector<long unsigned> blockrows(n);
        long unsigned blocks_needed{0u};
        get_blockrows<T>(A, block_size, &blockrows[0], blocks_needed);

        dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

        long unsigned *blockrows_d;

        cudaMalloc((void **)&blockrows_d, sizeof(long unsigned) * (blocks_needed + 1));

        cudaMemcpy(blockrows_d, &blockrows[0], sizeof(long unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_spMV2<T, long unsigned><<<blocks_IPCSR, threads>>>(IA_d, JA_d, blockrows_d, static_cast<unsigned long>(n), x_d, spMV_ans_d);

        cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        spMV(A, &x[0], &ans_vec[0]);
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        T relative_error{0};
        unsigned max_idx{0u};
        diff_arrays(&gpu_ans_vec[0], &ans_vec[0], n, relative_error, max_idx);

        ans = std::sqrt(std::inner_product(gpu_ans_vec.begin(), gpu_ans_vec.end(), gpu_ans_vec.begin(), 0));
        auto serial_ans = std::sqrt(std::inner_product(ans_vec.begin(), ans_vec.end(), ans_vec.begin(), 0));

        std::cout << "SPMV2: \t\t" << ans
                  << "\t\t" << serial_ans << "\t\t"
                  << relative_error / serial_ans << "\t\t\t" << speedup << "\n\n";
        cudaFree(blockrows_d);
    }
    {
        std::vector<T> gpu_ans_vec(n);
        std::vector<long unsigned> blockrows(n);

        T *tmp_d;

        cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_spMV3_kernel1<T, long unsigned><<<blocks, threads>>>(JA_d, A.get_edges() * 2, x_d, tmp_d);
        cu_spMV3_kernel2<T, long unsigned><<<1, threads, 49152>>>(tmp_d, IA_d, static_cast<unsigned long>(n), spMV_ans_d);

        cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        spMV(A, &x[0], &ans_vec[0]);
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        T relative_error{0};
        unsigned max_idx{0u};
        diff_arrays(&gpu_ans_vec[0], &ans_vec[0], n, relative_error, max_idx);

        ans = std::sqrt(std::inner_product(gpu_ans_vec.begin(), gpu_ans_vec.end(), gpu_ans_vec.begin(), 0));
        auto serial_ans = std::sqrt(std::inner_product(ans_vec.begin(), ans_vec.end(), ans_vec.begin(), 0));

        std::cout << "SPMV3: \t\t" << ans
                  << "\t\t" << serial_ans << "\t\t"
                  << relative_error / serial_ans << "\t\t\t" << speedup << "\n\n";

        cudaFree(tmp_d);
    }

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(tmp_d);
    cudaFree(ans_d);
    cudaFree(spMV_ans_d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    cudaProfilerStop();
}

void cuda_start_timer(cudaEvent_t &start, cudaEvent_t &end)
{
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
}

float cuda_end_timer(cudaEvent_t &start, cudaEvent_t &end)
{
    cudaEventRecord(end, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    float time_taken;
    cudaEventElapsedTime(&time_taken, start, end);
    return time_taken * 0.001;
}