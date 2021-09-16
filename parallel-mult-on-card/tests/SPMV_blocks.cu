/**
 * \file:        SPMV_blocks.cu
 * \brief:       Some code to run SPMV_test for every blocksize
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

#define SEED 1234 // To seed RNG
#define WIDTH 81  // for formatting std::cout output

template <typename T>
void cu_SPMV_test(const unsigned n, adjMatrix &A);

int main(int argc, char * argv[])
{
  unsigned n{100'000};
  unsigned edges{n * 10};

  adjMatrix A;

  bool write_to_file {true};

  char construct_matrix {'f'};

  timeval start, end;
  gettimeofday(&start, NULL);

  if (construct_matrix == 'f') {
    //std::string filename {"../data/California.mtx"};
    //std::string filename {"../data/file.txt"};
    std::string filename {"../data/bn1000000e9999944/bn1000000e9999944.mtx"};
    //std::string filename {"../data/kmer_U1a/kmer_U1a.mtx"};
    //std::string filename {"../data/kmer_P1a/kmer_P1a.mtx"};
    //std::string filename {"../data/europe_osm/europe_osm.mtx"};
    //std::string filename {"../data/delaunay_n24/delaunay_n24.mtx"};
    std::ifstream fs;
    fs.open(filename);
    assert(!fs.fail() && "Reading in file failed\n");
    fs >> n >> n >> edges;
    std::cout << "Constructing matrix for " << n << " nodes and " << edges << " edges.\n\n";
    adjMatrix B(n, edges, fs);
    A = std::move(B);
    fs.close();
  } else if (construct_matrix == 'b') {
    adjMatrix B(n, 20, 'b');
    A = std::move(B);
  } else {
    adjMatrix B(n, edges);
    A = std::move(B);
  }

  gettimeofday(&end, NULL);
  std::cout << "Time elapsed for build adjacency matrix with n = " << n << " edges = " << edges << ": "
    << end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0 << " seconds\n\n";


  std::cout << "\nTesting CUDA vs serial execution of SPMV for n = " << n << "\n\n";
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << "SINGLE PRECISION\n";
  cu_SPMV_test<float>(n, A);
  std::cout << "\n";
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << "DOUBLE PRECISION\n";
  cu_SPMV_test<double>(n, A);
  std::cout << '\n';

  if (write_to_file && construct_matrix != 'f') {
    std::cout << "Writing matrix to file...\n";
    A.write_matrix_to_file();
    std::cout << "Done\n";
  }

}

template <typename T>
void cu_SPMV_test(const unsigned n, adjMatrix &A){
  {
    std::cout << "\n4 = " << 4 << "\n\n";
    unsigned width {16};
    T ans;

    std::random_device rd;
    std::mt19937 gen{SEED};
    //std::mt19937 gen{rd()};
    std::uniform_real_distribution<T> U(0.0, 1.0);

    std::vector<T> x(n), ans_vec(n);

    for (auto it = x.begin(); it != x.end(); it++)
      *it = U(gen);

    unsigned num_blocks{n / 4 + (n % 4 ? 1 : 0)};

    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
      << std::setfill(' ');
    std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
      << std::setfill(' ');

    T *x_d, *spMV_ans_d;
    unsigned *IA_d, *JA_d;

    cudaMalloc((void **)&x_d, sizeof(T) * n);
    cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
    cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
    cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
    cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
    /**************SPMV1*********/
    {
      std::vector<T> gpu_ans_vec(n);

      cudaEvent_t start_d, end_d;
      cuda_start_timer(start_d, end_d);

      cu_spMV1<T><<<num_blocks, 4>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

      cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

      float gpu_time{cuda_end_timer(start_d, end_d)};

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

      std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
        << std::setw(width) << serial_ans 
        <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    }

    /**************SPMV2*********/ // Adaptive number of rows per block
    {
      std::vector<T> gpu_ans_vec(n);
      std::vector<unsigned> blockrows(n);
      unsigned blocks_needed{0u};
      get_blockrows<T>(A, 4, &blockrows[0], blocks_needed);

    // for (auto i=0u;i<blocks_needed+1;i++) {
    //   std::cout << blockrows[i] << " ";
    //}
    dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

    unsigned *blockrows_d;

    cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

    cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV2<T, unsigned, 4><<<blocks_IPCSR, 4>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
    << std::setw(width) << serial_ans 
    <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    cudaFree(blockrows_d);
    }
    /**************SPMV3*********/ // Dynamically parallel kernel
    {
      std::vector<T> gpu_ans_vec(n);
      std::vector<unsigned> blockrowsn;

      T *tmp_d;

      cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

      cudaEvent_t start_d, end_d;
      cuda_start_timer(start_d, end_d);

      auto avg_nnz {2lu*A.get_edges()/A.get_n()};

      auto num_blocks3 {n/4 + (n%4==0 ? 0 : 1)};

      cu_spMV3<T, unsigned, 4><<<num_blocks3, 4>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

      cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

      float gpu_time{cuda_end_timer(start_d, end_d)};

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

      std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
        << std::setw(width) << serial_ans 
        <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

      cudaFree(tmp_d);
    }
    /**************SPMV4*********/
    {
      std::vector<T> gpu_ans_vec(n);

      cudaEvent_t start_d, end_d;
      cuda_start_timer(start_d, end_d);

      cu_spMV4<T, unsigned, 4><<<n, 4>>>(IA_d, JA_d, x_d, spMV_ans_d);

      cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

      float gpu_time{cuda_end_timer(start_d, end_d)};

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

      std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
        << std::setw(width) << serial_ans 
        <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    }
    /**************SPMV1&4*********/
    {
      std::vector<T> gpu_ans_vec(n);

      cudaEvent_t start_d, end_d;
      cuda_start_timer(start_d, end_d);

      unsigned split {n/10000};

      std::vector<unsigned> tmp_hybrid(n-split+1);

      auto j {split};
      auto offset {A.row_offset[split]};
      for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
        *it = A.row_offset[j++] - offset;

      unsigned hybrid_blocks {(n-split) / 4 + ((n-split) % 4 ? 1 : 0)};

      unsigned * other_IA_d;
      cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
      cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

      cudaStream_t stream[2];

      cudaStreamCreate(&stream[0]);
      cudaStreamCreate(&stream[1]);

      cu_spMV4<T, unsigned, 4><<<split, 4,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
      cu_spMV1<T><<<hybrid_blocks, 4, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

      cudaStreamSynchronize(stream[0]);
      cudaStreamSynchronize(stream[1]);

      cudaStreamDestroy(stream[0]);
      cudaStreamDestroy(stream[1]);

      cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

      float gpu_time{cuda_end_timer(start_d, end_d)};

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

      std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
        << std::setw(width) << serial_ans 
        <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    }

    cudaFree(x_d);
    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(spMV_ans_d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    cudaProfilerStop();
  }
  {
    std::cout << "\n8 = " << 8 << "\n\n";
    unsigned width {16};
    T ans;

    std::random_device rd;
    std::mt19937 gen{SEED};
    //std::mt19937 gen{rd()};
    std::uniform_real_distribution<T> U(0.0, 1.0);

    std::vector<T> x(n), ans_vec(n);

    for (auto it = x.begin(); it != x.end(); it++)
      *it = U(gen);

    unsigned num_blocks{n / 8 + (n % 8 ? 1 : 0)};

    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
      << std::setfill(' ');
    std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
    std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
      << std::setfill(' ');

    T *x_d, *spMV_ans_d;
    unsigned *IA_d, *JA_d;

    cudaMalloc((void **)&x_d, sizeof(T) * n);
    cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
    cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
    cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
    cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
    /**************SPMV1*********/
    {
      std::vector<T> gpu_ans_vec(n);

      cudaEvent_t start_d, end_d;
      cuda_start_timer(start_d, end_d);

      cu_spMV1<T><<<num_blocks, 8>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

      cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

      float gpu_time{cuda_end_timer(start_d, end_d)};

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

      std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
        << std::setw(width) << serial_ans 
        <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    }

    /**************SPMV2*********/ // Adaptive number of rows per block
    {
      std::vector<T> gpu_ans_vec(n);
      std::vector<unsigned> blockrows(n);
      unsigned blocks_needed{0u};
      get_blockrows<T>(A, 8, &blockrows[0], blocks_needed);

    // for (auto i=0u;i<blocks_needed+1;i++) {
    //   std::cout << blockrows[i] << " ";
    //}
    dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

    unsigned *blockrows_d;

    cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

    cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV2<T, unsigned, 8><<<blocks_IPCSR, 8>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
    << std::setw(width) << serial_ans 
    <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    cudaFree(blockrows_d);
    }
    /**************SPMV3*********/ // Dynamically parallel kernel
    {
      std::vector<T> gpu_ans_vec(n);
      std::vector<unsigned> blockrowsn;

      T *tmp_d;

      cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

      cudaEvent_t start_d, end_d;
      cuda_start_timer(start_d, end_d);

      auto avg_nnz {2lu*A.get_edges()/A.get_n()};

      auto num_blocks3 {n/8 + (n%8==0 ? 0 : 1)};

      cu_spMV3<T, unsigned, 8><<<num_blocks3, 8>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

      cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

      float gpu_time{cuda_end_timer(start_d, end_d)};

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

      std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
        << std::setw(width) << serial_ans 
        <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

      cudaFree(tmp_d);
    }
    /**************SPMV4*********/
    {
      std::vector<T> gpu_ans_vec(n);

      cudaEvent_t start_d, end_d;
      cuda_start_timer(start_d, end_d);

      cu_spMV4<T, unsigned, 8><<<n, 8>>>(IA_d, JA_d, x_d, spMV_ans_d);

      cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

      float gpu_time{cuda_end_timer(start_d, end_d)};

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

      std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
        << std::setw(width) << serial_ans 
        <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    }
    /**************SPMV1&4*********/
    {
      std::vector<T> gpu_ans_vec(n);

      cudaEvent_t start_d, end_d;
      cuda_start_timer(start_d, end_d);

      unsigned split {n/10000};

      std::vector<unsigned> tmp_hybrid(n-split+1);

      auto j {split};
      auto offset {A.row_offset[split]};
      for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
        *it = A.row_offset[j++] - offset;

      unsigned hybrid_blocks {(n-split) / 8 + ((n-split) % 8 ? 1 : 0)};

      unsigned * other_IA_d;
      cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
      cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

      cudaStream_t stream[2];

      cudaStreamCreate(&stream[0]);
      cudaStreamCreate(&stream[1]);

      cu_spMV4<T, unsigned, 8><<<split, 8,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
      cu_spMV1<T><<<hybrid_blocks, 8, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

      cudaStreamSynchronize(stream[0]);
      cudaStreamSynchronize(stream[1]);

      cudaStreamDestroy(stream[0]);
      cudaStreamDestroy(stream[1]);

      cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

      float gpu_time{cuda_end_timer(start_d, end_d)};

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

      std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
        << std::setw(width) << serial_ans 
        <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    }

    cudaFree(x_d);
    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(spMV_ans_d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    cudaProfilerStop();
  }
{
  std::cout << "\n16 = " << 16 << "\n\n";
  unsigned width {16};
  T ans;

  std::random_device rd;
  std::mt19937 gen{SEED};
  //std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> U(0.0, 1.0);

  std::vector<T> x(n), ans_vec(n);

  for (auto it = x.begin(); it != x.end(); it++)
    *it = U(gen);

  unsigned num_blocks{n / 16 + (n % 16 ? 1 : 0)};

  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');

  T *x_d, *spMV_ans_d;
  unsigned *IA_d, *JA_d;

  cudaMalloc((void **)&x_d, sizeof(T) * n);
  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
  cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
  cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
  /**************SPMV1*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV1<T><<<num_blocks, 16>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  /**************SPMV2*********/ // Adaptive number of rows per block
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrows(n);
    unsigned blocks_needed{0u};
    get_blockrows<T>(A, 16, &blockrows[0], blocks_needed);

  // for (auto i=0u;i<blocks_needed+1;i++) {
  //   std::cout << blockrows[i] << " ";
  //}
  dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

  unsigned *blockrows_d;

  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

  cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start_d, end_d;
  cuda_start_timer(start_d, end_d);

  cu_spMV2<T, unsigned, 16><<<blocks_IPCSR, 16>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

  cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

  float gpu_time{cuda_end_timer(start_d, end_d)};

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

  std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
  << std::setw(width) << serial_ans 
  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  cudaFree(blockrows_d);
  }
  /**************SPMV3*********/ // Dynamically parallel kernel
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrowsn;

    T *tmp_d;

    cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    auto avg_nnz {2lu*A.get_edges()/A.get_n()};

    auto num_blocks3 {n/16 + (n%16==0 ? 0 : 1)};

    cu_spMV3<T, unsigned, 16><<<num_blocks3, 16>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

    cudaFree(tmp_d);
  }
  /**************SPMV4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV4<T, unsigned, 16><<<n, 16>>>(IA_d, JA_d, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /**************SPMV1&4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    unsigned split {n/10000};

    std::vector<unsigned> tmp_hybrid(n-split+1);

    auto j {split};
    auto offset {A.row_offset[split]};
    for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
      *it = A.row_offset[j++] - offset;

    unsigned hybrid_blocks {(n-split) / 16 + ((n-split) % 16 ? 1 : 0)};

    unsigned * other_IA_d;
    cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
    cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cu_spMV4<T, unsigned, 16><<<split, 16,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
    cu_spMV1<T><<<hybrid_blocks, 16, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  cudaFree(x_d);
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(spMV_ans_d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  cudaProfilerStop();
}
{
  std::cout << "\n32 = " << 32 << "\n\n";
  unsigned width {16};
  T ans;

  std::random_device rd;
  std::mt19937 gen{SEED};
  //std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> U(0.0, 1.0);

  std::vector<T> x(n), ans_vec(n);

  for (auto it = x.begin(); it != x.end(); it++)
    *it = U(gen);

  unsigned num_blocks{n / 32 + (n % 32 ? 1 : 0)};

  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');

  T *x_d, *spMV_ans_d;
  unsigned *IA_d, *JA_d;

  cudaMalloc((void **)&x_d, sizeof(T) * n);
  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
  cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
  cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
  /**************SPMV1*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV1<T><<<num_blocks, 32>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  /**************SPMV2*********/ // Adaptive number of rows per block
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrows(n);
    unsigned blocks_needed{0u};
    get_blockrows<T>(A, 32, &blockrows[0], blocks_needed);

  // for (auto i=0u;i<blocks_needed+1;i++) {
  //   std::cout << blockrows[i] << " ";
  //}
  dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

  unsigned *blockrows_d;

  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

  cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start_d, end_d;
  cuda_start_timer(start_d, end_d);

  cu_spMV2<T, unsigned, 32><<<blocks_IPCSR, 32>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

  cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

  float gpu_time{cuda_end_timer(start_d, end_d)};

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

  std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
  << std::setw(width) << serial_ans 
  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  cudaFree(blockrows_d);
  }
  /**************SPMV3*********/ // Dynamically parallel kernel
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrowsn;

    T *tmp_d;

    cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    auto avg_nnz {2lu*A.get_edges()/A.get_n()};

    auto num_blocks3 {n/32 + (n%32==0 ? 0 : 1)};

    cu_spMV3<T, unsigned, 32><<<num_blocks3, 32>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

    cudaFree(tmp_d);
  }
  /**************SPMV4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV4<T, unsigned, 32><<<n, 32>>>(IA_d, JA_d, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /**************SPMV1&4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    unsigned split {n/10000};

    std::vector<unsigned> tmp_hybrid(n-split+1);

    auto j {split};
    auto offset {A.row_offset[split]};
    for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
      *it = A.row_offset[j++] - offset;

    unsigned hybrid_blocks {(n-split) / 32 + ((n-split) % 32 ? 1 : 0)};

    unsigned * other_IA_d;
    cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
    cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cu_spMV4<T, unsigned, 32><<<split, 32,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
    cu_spMV1<T><<<hybrid_blocks, 32, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  cudaFree(x_d);
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(spMV_ans_d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  cudaProfilerStop();
}
{
  std::cout << "\n64 = " << 64 << "\n\n";
  unsigned width {16};
  T ans;

  std::random_device rd;
  std::mt19937 gen{SEED};
  //std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> U(0.0, 1.0);

  std::vector<T> x(n), ans_vec(n);

  for (auto it = x.begin(); it != x.end(); it++)
    *it = U(gen);

  unsigned num_blocks{n / 64 + (n % 64 ? 1 : 0)};

  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');

  T *x_d, *spMV_ans_d;
  unsigned *IA_d, *JA_d;

  cudaMalloc((void **)&x_d, sizeof(T) * n);
  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
  cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
  cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
  /**************SPMV1*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV1<T><<<num_blocks, 64>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  /**************SPMV2*********/ // Adaptive number of rows per block
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrows(n);
    unsigned blocks_needed{0u};
    get_blockrows<T>(A, 64, &blockrows[0], blocks_needed);

  // for (auto i=0u;i<blocks_needed+1;i++) {
  //   std::cout << blockrows[i] << " ";
  //}
  dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

  unsigned *blockrows_d;

  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

  cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start_d, end_d;
  cuda_start_timer(start_d, end_d);

  cu_spMV2<T, unsigned, 64><<<blocks_IPCSR, 64>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

  cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

  float gpu_time{cuda_end_timer(start_d, end_d)};

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

  std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
  << std::setw(width) << serial_ans 
  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  cudaFree(blockrows_d);
  }
  /**************SPMV3*********/ // Dynamically parallel kernel
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrowsn;

    T *tmp_d;

    cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    auto avg_nnz {2lu*A.get_edges()/A.get_n()};

    auto num_blocks3 {n/64 + (n%64==0 ? 0 : 1)};

    cu_spMV3<T, unsigned, 64><<<num_blocks3, 64>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

    cudaFree(tmp_d);
  }
  /**************SPMV4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV4<T, unsigned, 64><<<n, 64>>>(IA_d, JA_d, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /**************SPMV1&4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    unsigned split {n/10000};

    std::vector<unsigned> tmp_hybrid(n-split+1);

    auto j {split};
    auto offset {A.row_offset[split]};
    for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
      *it = A.row_offset[j++] - offset;

    unsigned hybrid_blocks {(n-split) / 64 + ((n-split) % 64 ? 1 : 0)};

    unsigned * other_IA_d;
    cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
    cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cu_spMV4<T, unsigned, 64><<<split, 64,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
    cu_spMV1<T><<<hybrid_blocks, 64, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  cudaFree(x_d);
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(spMV_ans_d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  cudaProfilerStop();
}
{
  std::cout << "\n128 = " << 128 << "\n\n";
  unsigned width {16};
  T ans;

  std::random_device rd;
  std::mt19937 gen{SEED};
  //std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> U(0.0, 1.0);

  std::vector<T> x(n), ans_vec(n);

  for (auto it = x.begin(); it != x.end(); it++)
    *it = U(gen);

  unsigned num_blocks{n / 128 + (n % 128 ? 1 : 0)};

  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');

  T *x_d, *spMV_ans_d;
  unsigned *IA_d, *JA_d;

  cudaMalloc((void **)&x_d, sizeof(T) * n);
  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
  cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
  cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
  /**************SPMV1*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV1<T><<<num_blocks, 128>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  /**************SPMV2*********/ // Adaptive number of rows per block
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrows(n);
    unsigned blocks_needed{0u};
    get_blockrows<T>(A, 128, &blockrows[0], blocks_needed);

  // for (auto i=0u;i<blocks_needed+1;i++) {
  //   std::cout << blockrows[i] << " ";
  //}
  dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

  unsigned *blockrows_d;

  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

  cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start_d, end_d;
  cuda_start_timer(start_d, end_d);

  cu_spMV2<T, unsigned, 128><<<blocks_IPCSR, 128>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

  cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

  float gpu_time{cuda_end_timer(start_d, end_d)};

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

  std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
  << std::setw(width) << serial_ans 
  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  cudaFree(blockrows_d);
  }
  /**************SPMV3*********/ // Dynamically parallel kernel
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrowsn;

    T *tmp_d;

    cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    auto avg_nnz {2lu*A.get_edges()/A.get_n()};

    auto num_blocks3 {n/128 + (n%128==0 ? 0 : 1)};

    cu_spMV3<T, unsigned, 128><<<num_blocks3, 128>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

    cudaFree(tmp_d);
  }
  /**************SPMV4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV4<T, unsigned, 128><<<n, 128>>>(IA_d, JA_d, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /**************SPMV1&4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    unsigned split {n/10000};

    std::vector<unsigned> tmp_hybrid(n-split+1);

    auto j {split};
    auto offset {A.row_offset[split]};
    for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
      *it = A.row_offset[j++] - offset;

    unsigned hybrid_blocks {(n-split) / 128 + ((n-split) % 128 ? 1 : 0)};

    unsigned * other_IA_d;
    cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
    cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cu_spMV4<T, unsigned, 128><<<split, 128,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
    cu_spMV1<T><<<hybrid_blocks, 128, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  cudaFree(x_d);
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(spMV_ans_d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  cudaProfilerStop();
}
{
  std::cout << "\n256 = " << 256 << "\n\n";
  unsigned width {16};
  T ans;

  std::random_device rd;
  std::mt19937 gen{SEED};
  //std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> U(0.0, 1.0);

  std::vector<T> x(n), ans_vec(n);

  for (auto it = x.begin(); it != x.end(); it++)
    *it = U(gen);

  unsigned num_blocks{n / 256 + (n % 256 ? 1 : 0)};

  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');

  T *x_d, *spMV_ans_d;
  unsigned *IA_d, *JA_d;

  cudaMalloc((void **)&x_d, sizeof(T) * n);
  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
  cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
  cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
  /**************SPMV1*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV1<T><<<num_blocks, 256>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  /**************SPMV2*********/ // Adaptive number of rows per block
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrows(n);
    unsigned blocks_needed{0u};
    get_blockrows<T>(A, 256, &blockrows[0], blocks_needed);

  // for (auto i=0u;i<blocks_needed+1;i++) {
  //   std::cout << blockrows[i] << " ";
  //}
  dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

  unsigned *blockrows_d;

  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

  cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start_d, end_d;
  cuda_start_timer(start_d, end_d);

  cu_spMV2<T, unsigned, 256><<<blocks_IPCSR, 256>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

  cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

  float gpu_time{cuda_end_timer(start_d, end_d)};

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

  std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
  << std::setw(width) << serial_ans 
  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  cudaFree(blockrows_d);
  }
  /**************SPMV3*********/ // Dynamically parallel kernel
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrowsn;

    T *tmp_d;

    cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    auto avg_nnz {2lu*A.get_edges()/A.get_n()};

    auto num_blocks3 {n/256 + (n%256==0 ? 0 : 1)};

    cu_spMV3<T, unsigned, 256><<<num_blocks3, 256>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

    cudaFree(tmp_d);
  }
  /**************SPMV4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV4<T, unsigned, 256><<<n, 256>>>(IA_d, JA_d, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /**************SPMV1&4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    unsigned split {n/10000};

    std::vector<unsigned> tmp_hybrid(n-split+1);

    auto j {split};
    auto offset {A.row_offset[split]};
    for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
      *it = A.row_offset[j++] - offset;

    unsigned hybrid_blocks {(n-split) / 256 + ((n-split) % 256 ? 1 : 0)};

    unsigned * other_IA_d;
    cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
    cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cu_spMV4<T, unsigned, 256><<<split, 256,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
    cu_spMV1<T><<<hybrid_blocks, 256, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  cudaFree(x_d);
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(spMV_ans_d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  cudaProfilerStop();
}
{
  std::cout << "\n512 = " << 512 << "\n\n";
  unsigned width {16};
  T ans;

  std::random_device rd;
  std::mt19937 gen{SEED};
  //std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> U(0.0, 1.0);

  std::vector<T> x(n), ans_vec(n);

  for (auto it = x.begin(); it != x.end(); it++)
    *it = U(gen);

  unsigned num_blocks{n / 512 + (n % 512 ? 1 : 0)};

  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');

  T *x_d, *spMV_ans_d;
  unsigned *IA_d, *JA_d;

  cudaMalloc((void **)&x_d, sizeof(T) * n);
  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
  cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
  cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
  /**************SPMV1*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV1<T><<<num_blocks, 512>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  /**************SPMV2*********/ // Adaptive number of rows per block
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrows(n);
    unsigned blocks_needed{0u};
    get_blockrows<T>(A, 512, &blockrows[0], blocks_needed);

  // for (auto i=0u;i<blocks_needed+1;i++) {
  //   std::cout << blockrows[i] << " ";
  //}
  dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

  unsigned *blockrows_d;

  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

  cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start_d, end_d;
  cuda_start_timer(start_d, end_d);

  cu_spMV2<T, unsigned, 512><<<blocks_IPCSR, 512>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

  cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

  float gpu_time{cuda_end_timer(start_d, end_d)};

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

  std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
  << std::setw(width) << serial_ans 
  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  cudaFree(blockrows_d);
  }
  /**************SPMV3*********/ // Dynamically parallel kernel
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrowsn;

    T *tmp_d;

    cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    auto avg_nnz {2lu*A.get_edges()/A.get_n()};

    auto num_blocks3 {n/512 + (n%512==0 ? 0 : 1)};

    cu_spMV3<T, unsigned, 512><<<num_blocks3, 512>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

    cudaFree(tmp_d);
  }
  /**************SPMV4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV4<T, unsigned, 512><<<n, 512>>>(IA_d, JA_d, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /**************SPMV1&4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    unsigned split {n/10000};

    std::vector<unsigned> tmp_hybrid(n-split+1);

    auto j {split};
    auto offset {A.row_offset[split]};
    for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
      *it = A.row_offset[j++] - offset;

    unsigned hybrid_blocks {(n-split) / 512 + ((n-split) % 512 ? 1 : 0)};

    unsigned * other_IA_d;
    cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
    cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cu_spMV4<T, unsigned, 512><<<split, 512,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
    cu_spMV1<T><<<hybrid_blocks, 512, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  cudaFree(x_d);
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(spMV_ans_d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  cudaProfilerStop();
}
{
  std::cout << "\n1024 = " << 1024 << "\n\n";
  unsigned width {16};
  T ans;

  std::random_device rd;
  std::mt19937 gen{SEED};
  //std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> U(0.0, 1.0);

  std::vector<T> x(n), ans_vec(n);

  for (auto it = x.begin(); it != x.end(); it++)
    *it = U(gen);

  unsigned num_blocks{n / 1024 + (n % 1024 ? 1 : 0)};

  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');
  std::cout << std::setw(2*width)<<"CUDA"<<std::setw(width)<<"Serial"<<std::setw(width)<<"Rel. Error"<<std::setw(width)<<"Speedup"<<std::endl;
  std::cout << std::setw(WIDTH) << std::setfill('~') << '\n'
    << std::setfill(' ');

  T *x_d, *spMV_ans_d;
  unsigned *IA_d, *JA_d;

  cudaMalloc((void **)&x_d, sizeof(T) * n);
  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * A.edge_count * 2);
  cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
  cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
  /**************SPMV1*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV1<T><<<num_blocks, 1024>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  /**************SPMV2*********/ // Adaptive number of rows per block
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrows(n);
    unsigned blocks_needed{0u};
    get_blockrows<T>(A, 1024, &blockrows[0], blocks_needed);

  // for (auto i=0u;i<blocks_needed+1;i++) {
  //   std::cout << blockrows[i] << " ";
  //}
  dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

  unsigned *blockrows_d;

  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

  cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start_d, end_d;
  cuda_start_timer(start_d, end_d);

  cu_spMV2<T, unsigned, 1024><<<blocks_IPCSR, 1024>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

  cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

  float gpu_time{cuda_end_timer(start_d, end_d)};

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

  std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
  << std::setw(width) << serial_ans 
  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  cudaFree(blockrows_d);
  }
  /**************SPMV3*********/ // Dynamically parallel kernel
  {
    std::vector<T> gpu_ans_vec(n);
    std::vector<unsigned> blockrowsn;

    T *tmp_d;

    cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    auto avg_nnz {2lu*A.get_edges()/A.get_n()};

    auto num_blocks3 {n/1024 + (n%1024==0 ? 0 : 1)};

    cu_spMV3<T, unsigned, 1024><<<num_blocks3, 1024>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV3:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";

    cudaFree(tmp_d);
  }
  /**************SPMV4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV4<T, unsigned, 1024><<<n, 1024>>>(IA_d, JA_d, x_d, spMV_ans_d);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "SPMV4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }
  /**************SPMV1&4*********/
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    unsigned split {n/10000};

    std::vector<unsigned> tmp_hybrid(n-split+1);

    auto j {split};
    auto offset {A.row_offset[split]};
    for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
      *it = A.row_offset[j++] - offset;

    unsigned hybrid_blocks {(n-split) / 1024 + ((n-split) % 1024 ? 1 : 0)};

    unsigned * other_IA_d;
    cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
    cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cu_spMV4<T, unsigned, 1024><<<split, 1024,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
    cu_spMV1<T><<<hybrid_blocks, 1024, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    cudaMemcpy(&gpu_ans_vec[0], spMV_ans_d, sizeof(T) * n, cudaMemcpyDeviceToHost);

    float gpu_time{cuda_end_timer(start_d, end_d)};

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

    std::cout << std::setw(width)<<std::left<< "Hybrid SPMV1&4:"<<std::right<<std::setw(width) << ans
      << std::setw(width) << serial_ans 
      <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
  }

  cudaFree(x_d);
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(spMV_ans_d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  cudaProfilerStop();
}
}
