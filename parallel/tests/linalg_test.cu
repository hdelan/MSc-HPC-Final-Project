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

#include <cuda_profiler_api.h>

#define BLOCKSIZE 32
#define SEED 1234       // To seed RNG
#define WIDTH 71        // for formatting std::cout output

template <typename T> void cu_linalg_test(const unsigned n);

int main (void)
{
    unsigned n{10000};
    
    std::cout << "\nTesting CUDA vs serial execution of linalg functions for n = "<<n<<"\n\n";
    
    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n' << std::setfill(' ');
    std::cout << "SINGLE PRECISION\n";
    cu_linalg_test<float>(n);
    
    std::cout << "\n";
    
    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n' << std::setfill(' ');
    std::cout << "DOUBLE PRECISION\n";
    cu_linalg_test<double>(n);

}


template <typename T>
void cu_linalg_test(const unsigned n) {
    T ans;
    
    std::random_device rd;
    std::mt19937 gen{SEED};
    //std::mt19937 gen{rd()};
    std::uniform_real_distribution<T> U(0.0, 1.0);

    std::vector<T> x(n), y(n), ans_vec(n), gpu_ans_vec(n);


    for (auto it = x.begin(); it != x.end(); it++)
        *it = U(gen);

    for (auto it = y.begin(); it != y.end(); it++)
        *it = U(gen);

    unsigned block_size{BLOCKSIZE}, num_blocks{n / block_size + 1};
    
    dim3 blocks{num_blocks}, threads{block_size}, one_block {1u};
    
    T *x_d, *y_d, *tmp_d, *ans_d;
    cudaMalloc((void **)&x_d, sizeof(T) * n);
    cudaMalloc((void **)&y_d, sizeof(T) * n);
    cudaMalloc((void **)&tmp_d, sizeof(T) * num_blocks);
    cudaMalloc((void **)&ans_d, sizeof(T));
    cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, &y[0], sizeof(T) * n, cudaMemcpyHostToDevice);

    cu_dot_prod<T,BLOCKSIZE><<<blocks, threads, block_size*sizeof(T)>>>(x_d, y_d, n, tmp_d);
    cu_reduce<T,BLOCKSIZE><<<one_block, threads, num_blocks*sizeof(T)>>>(tmp_d, num_blocks, ans_d);

    cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);
    auto serial_ans = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);

    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n' << std::setfill(' ');
    std::cout << "\t\t\tCUDA\t\tSerial\t\tRelative Error\n";
    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n' << std::setfill(' ');

    std::cout << "Inner product: \t\t" << ans 
    << "\t\t" << serial_ans << "\t\t" 
    << (serial_ans - ans)/serial_ans << "\n\n";
    
    cu_norm_sq<T,BLOCKSIZE><<<blocks, threads, block_size*sizeof(T)>>>(x_d, n, tmp_d);
    cu_reduce<T,BLOCKSIZE><<<one_block, threads, num_blocks*sizeof(T)>>>(tmp_d, num_blocks, ans_d);

    cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);
    serial_ans = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    
    std::cout << "Norm squared: \t\t" << ans 
    << "\t\t" << serial_ans << "\t\t" 
    << (serial_ans - ans)/serial_ans << "\n\n";

    cu_reduce<T,BLOCKSIZE><<<blocks, threads, block_size*sizeof(T)>>>(x_d, n, tmp_d);
    cu_reduce<T,BLOCKSIZE><<<one_block, threads, num_blocks*sizeof(T)>>>(tmp_d, num_blocks, ans_d);

    cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

    serial_ans = std::accumulate(x.begin(), x.end(), 0.0);
    std::cout << "Reduce: \t\t" << ans 
    << "\t\t" << serial_ans << "\t\t" 
    << (serial_ans - ans)/serial_ans << "\n\n";

    long unsigned edges {n*10};
    adjMatrix A(n, edges);
    spMV(A, &x[0], &ans_vec[0]);

    long unsigned * IA_d, *JA_d;
    T * spMV_ans_d;
    cudaMalloc((void**)&IA_d, sizeof(long unsigned)*(n+1));
    cudaMalloc((void**)&JA_d, sizeof(long unsigned)*edges*2);
    cudaMalloc((void**)&spMV_ans_d, sizeof(T)*n);
    cudaMemcpy(IA_d, A.row_offset, sizeof(long unsigned)*(n+1),cudaMemcpyHostToDevice);
    cudaMemcpy(JA_d, A.col_idx, sizeof(long unsigned)*2*edges,cudaMemcpyHostToDevice);
    
    long unsigned long_n {static_cast<long unsigned>(n)};
    
    cu_spMV1<T, long unsigned><<<blocks,threads>>>(IA_d, JA_d, long_n, x_d, spMV_ans_d);
    
    std::cout << "y[0]\t" << gpu_ans_vec[0] << '\n';
    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T)*n, cudaMemcpyDeviceToHost);
    std::cout << "y[0]\t" << gpu_ans_vec[0] << '\n';

    T relative_error {0};
    unsigned max_idx {0u};
    diff_arrays(&gpu_ans_vec[0], &ans_vec[0], n, relative_error, max_idx);


    ans = std::sqrt(std::inner_product(gpu_ans_vec.begin(), gpu_ans_vec.end(), gpu_ans_vec.begin(), 0));
    serial_ans = std::sqrt(std::inner_product(ans_vec.begin(), ans_vec.end(), ans_vec.begin(), 0));
    
    std::cout << "SPMV: \t\t\t" << ans
    << "\t\t" << serial_ans << "\t\t" 
    << relative_error/serial_ans << "\n\n";

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(tmp_d);
    cudaFree(ans_d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    cudaProfilerStop();
}