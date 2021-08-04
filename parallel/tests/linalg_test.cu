#include "../lib/linalg.h"

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#define BLOCKSIZE 32

int main (void)
{

    unsigned n{1000}, seed{1234};

    std::random_device rd;
    std::mt19937 gen{seed};
    std::uniform_real_distribution<double> U(0.0, 1.0);

    std::vector<double> x(n), y(n);

    double ans;

    for (auto it = x.begin(); it != x.end(); it++)
        *it = U(gen);

    for (auto it = y.begin(); it != y.end(); it++)
        *it = U(gen);

    unsigned block_size{BLOCKSIZE}, num_blocks{n / block_size + 1};
    
    dim3 blocks{num_blocks}, threads{block_size}, one_block {1u};
    
    double *x_d, *y_d, *tmp_d, *ans_d;
    cudaMalloc((void **)&x_d, sizeof(double) * n);
    cudaMalloc((void **)&y_d, sizeof(double) * n);
    cudaMalloc((void **)&tmp_d, sizeof(double) * num_blocks);
    cudaMalloc((void **)&ans_d, sizeof(double));
    cudaMemcpy(&x[0], x_d, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(&y[0], y_d, sizeof(double) * n, cudaMemcpyHostToDevice);

    cu_dot_prod<BLOCKSIZE><<<blocks, threads, n*sizeof(double)>>>(x_d, y_d, n, tmp_d);
    cu_reduce<BLOCKSIZE><<<one_block, threads, num_blocks*sizeof(double)>>>(tmp_d, num_blocks, ans_d);

    cudaMemcpy(ans_d, &ans, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "\nTesting CUDA vs serial execution for n = 1000\n\n\t\tCUDA\tSerial\tRelative Error\n"
            << "Inner product: \t" << ans << '\t' << std::inner_product(x.begin(), x.end(), y.begin(), 0.0) << "\n\n";
}