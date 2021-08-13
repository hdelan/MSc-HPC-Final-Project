#include "../lib/cu_linalg.h"
#include "../lib/cu_lanczos.h"
#include "../lib/cu_SPMV.h"
#include "../lib/eigen.h"
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

#define BLOCKSIZE 32
#define SEED 1234 // To seed RNG
#define WIDTH 81  // for formatting std::cout output

void cuda_start_timer(cudaEvent_t &start, cudaEvent_t &end);
float cuda_end_timer(cudaEvent_t &start, cudaEvent_t &end);

int main(void)
{
    unsigned n{100};

    long unsigned edges{n * 5};
    timeval start, end;
    gettimeofday(&start, NULL);
    adjMatrix A(n, edges);
    gettimeofday(&end, NULL);
    std::cout << "Time elapsed to build random adjacency matrix with n = " << n << " edges = " << edges << ": "
              << end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0 << " seconds\n\n";
    
    unsigned krylov_dim {20};

    std::vector<double> x (n, 1);
    
    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);
    
    bool cuda {true};
    lanczosDecomp cu_L(A, krylov_dim, &x[0], cuda);
    eigenDecomp cu_E(cu_L);
    multOut(cu_L, cu_E, A);
    
    float gpu_time{cuda_end_timer(start_d, end_d)};
    
    timeval s, e;
    gettimeofday(&s, NULL);
    
    cuda = false;
    lanczosDecomp L(A, krylov_dim, &x[0], cuda);
    eigenDecomp E(L);
    multOut(L, E, A);

    gettimeofday(&e, NULL);
    std::cout <<  "Speedup: \t"
              << (end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0)/gpu_time << "\n\n";

    L.check_ans(cu_L);
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
