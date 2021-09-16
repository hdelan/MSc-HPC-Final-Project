/**
 * \file:        linalg_test.cu
 * \brief:       Code to test the speedups of linalg functions
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */
#include "../lib/cu_linalg.h"
#include "../lib/cu_SPMV.h"
#include "../lib/SPMV.h"
#include "../lib/adjMatrix.h"
#include "../lib/helpers.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/time.h>

#include <cuda_profiler_api.h>

#include "cublas_v2.h"
#include "cblas.h"

#include "../lib/blocks.h"
#include "../lib/cblas_aliases.h"

#define SEED 1234 // To seed RNG
#define WIDTH 81  // for formatting std::cout output

template <typename T>
void cu_linalg_test(const unsigned n);
template <typename T>
void cu_SPMV_test(const unsigned n, adjMatrix &A);

int main(void)
{
  unsigned n{100'000};

  std::cout << "\nTesting CUDA linear algebra functions for n ="<<n<<" blocksize="<<BLOCKSIZE<<"\n\n";
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << "SINGLE PRECISION\n";
  cu_linalg_test<float>(n);
  std::cout << "\n";
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << "DOUBLE PRECISION\n";
  cu_linalg_test<double>(n);
  std::cout << '\n';

}

  template <typename T>
void cu_linalg_test(const unsigned n)
{
  unsigned width {16};
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

  unsigned num_blocks{n / BLOCKSIZE + (n % BLOCKSIZE ? 1 : 0)}, h_blocks{n / (2 * BLOCKSIZE) + (n % (2 * BLOCKSIZE) ? 1 : 0)};

  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');

  T *x_d, *y_d, *tmp_d, *ans_d, *alpha_d;

  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaMalloc((void **)&x_d, sizeof(T) * n);
  cudaMalloc((void **)&y_d, sizeof(T) * n);
  cudaMalloc((void **)&tmp_d, sizeof(T) * num_blocks);
  cudaMalloc((void **)&ans_d, sizeof(T));
  cudaMalloc((void **)&alpha_d, sizeof(T));
  cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, &y[0], sizeof(T) * n, cudaMemcpyHostToDevice);

  /*************DOT PRODUCT**********/
  {
    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);
    // These kernels outperform cublas
    cu_dot_prod<T, BLOCKSIZE><<<h_blocks, BLOCKSIZE>>>(x_d, y_d, n, tmp_d);
    //cudaDeviceSynchronize();
    cu_reduce<T, 256><<<1, 256>>>(tmp_d, h_blocks, ans_d);
    //cublasDdot(handle, static_cast<int>(n), x_d,1, y_d,1, ans_d);

    float gpu_time{cuda_end_timer(start_d, end_d)};

    cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

    timeval start, end;
    gettimeofday(&start, NULL);
    auto serial_ans = cblas_ddot(n, &x[0], 1, &y[0], 1);
    //auto serial_ans = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);

    gettimeofday(&end, NULL);
    auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

    std::cout << std::setw(width)<<std::left<< "Inner product:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      << std::setw(width) << (serial_ans - ans) / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /*************NORM*****************/
  {
    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    // These kernels outperform cublas
    cu_norm_sq<T, BLOCKSIZE><<<h_blocks, BLOCKSIZE>>>(x_d, n, tmp_d);
    cu_reduce_sqrt<T, 256><<<1, 256>>>(tmp_d, h_blocks, ans_d);

    cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

    //cublasDnrm2(handle, n, x_d, 1, &ans);

    float gpu_time{cuda_end_timer(start_d, end_d)};

    timeval start, end;
    gettimeofday(&start, NULL);
    //auto serial_ans = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));
    auto serial_ans = cblas_dnrm2(n, &x[0], 1);
    gettimeofday(&end, NULL);
    auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

    std::cout << std::setw(width)<<std::left<< "Norm:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      << std::setw(width) << (serial_ans - ans) / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /*************REDUCE*************/
  {
    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);
    
    cu_reduce<T, BLOCKSIZE><<<h_blocks, BLOCKSIZE>>>(x_d, n, tmp_d);
    cu_reduce<T, 256><<<1, 256>>>(tmp_d, h_blocks, ans_d);
    //cu_reduce<T, BLOCKSIZE><<<1, BLOCKSIZE>>>(x_d, n, ans_d);

    float gpu_time{cuda_end_timer(start_d, end_d)};

    cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

    std::vector<T> ones (n, 1);

    timeval start, end;
    gettimeofday(&start, NULL);
    //auto serial_ans = std::accumulate(x.begin(), x.end(), 0.0);
    auto serial_ans = cblas_ddot(n, &x[0], 1, &ones[0], 1);
    gettimeofday(&end, NULL);
    auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

    std::cout << std::setw(width)<<std::left<< "Reduce:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << (serial_ans - ans) / serial_ans <<std::setw(width) << speedup << "\n";

  }
  /**********VECTOR ADDITION*******/
  {
    cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, &y[0], sizeof(T) * n, cudaMemcpyHostToDevice);

    T alpha{2};
    cudaMemcpy(alpha_d, &alpha, sizeof(T), cudaMemcpyHostToDevice);

    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_dpax<T><<<num_blocks, BLOCKSIZE>>>(x_d, alpha_d, y_d, n);

    float gpu_time{cuda_end_timer(start_d, end_d)};

    cudaMemcpy(&gpu_ans_vec[0], x_d, n * sizeof(T), cudaMemcpyDeviceToHost);

    auto ans{std::inner_product(gpu_ans_vec.begin(), gpu_ans_vec.end(), gpu_ans_vec.begin(), 0.0)};

    timeval start, end;
    gettimeofday(&start, NULL);
    //std::for_each(x.begin(), x.end(), [&](T & a)
    //              { a -= alpha * y[idx++]; });
    auto alpha_neg = -alpha;
    cblas_daxpy(n,alpha_neg, &y[0], 1, &x[0], 1);
    gettimeofday(&end, NULL);
    auto serial_ans{std::inner_product(x.begin(), x.end(), x.begin(), 0.0)};
    auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

    std::cout << std::setw(width)<<std::left<< "Vector add:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << (serial_ans - ans) / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /**********VECTOR SCALING*********/
  {
    cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, &y[0], sizeof(T) * n, cudaMemcpyHostToDevice);

    T alpha{123};
    cudaMemcpy(alpha_d, &alpha, sizeof(T), cudaMemcpyHostToDevice);

    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_dvexda<T><<<num_blocks, BLOCKSIZE>>>(x_d, alpha_d, y_d, n);

    float gpu_time{cuda_end_timer(start_d, end_d)};

    cudaMemcpy(&gpu_ans_vec[0], x_d, n * sizeof(T), cudaMemcpyDeviceToHost);

    auto ans{std::inner_product(gpu_ans_vec.begin(), gpu_ans_vec.end(), gpu_ans_vec.begin(), 0.0)};


    timeval start, end;
    gettimeofday(&start, NULL);
    //std::for_each(x.begin(), x.end(), [&](T & a)
    //             { a = y[idx++]*(1/alpha); });

    cblas_daxpby(n,1/alpha, &y[0], 1, 0, &x[0], 1);
    gettimeofday(&end, NULL);
    auto serial_ans{std::inner_product(x.begin(), x.end(), x.begin(), 0.0)};
    auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

    std::cout << std::setw(width)<<std::left<< "v = v/alpha:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << (serial_ans - ans) / serial_ans <<std::setw(width) << speedup << "\n";
  }

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(tmp_d);
  cudaFree(ans_d);
  cudaFree(alpha_d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  cudaProfilerStop();
}

