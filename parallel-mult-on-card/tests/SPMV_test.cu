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

#define SEED 1234 // To seed RNG
#define WIDTH 81  // for formatting std::cout output

#define THREADS3 32

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
    //std::string filename {"../data/California/California.mtx"};
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
  
  std::cout << "\nBLOCKSIZE = " << BLOCKSIZE << "\n\n";

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
void cu_SPMV_test(const unsigned n, adjMatrix &A)
{
  unsigned width {16};
  T ans;

  std::random_device rd;
  std::mt19937 gen{SEED};
  //std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> U(0.0, 1.0);

  std::vector<T> x(n), ans_vec(n);

  for (auto it = x.begin(); it != x.end(); it++)
    *it = U(gen);

  unsigned num_blocks{n / BLOCKSIZE + (n % BLOCKSIZE ? 1 : 0)};

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


  /**************SPMV1*********/ // naive kernel
  {
    std::vector<T> gpu_ans_vec(n);

    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);

    cu_spMV1<T><<<num_blocks, BLOCKSIZE>>>(IA_d, JA_d, n, x_d, spMV_ans_d);

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
    get_blockrows<T>(A, BLOCKSIZE, &blockrows[0], blocks_needed);

  // for (auto i=0u;i<blocks_needed+1;i++) {
  //   std::cout << blockrows[i] << " ";
  //}
  dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

  unsigned *blockrows_d;

  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (blocks_needed + 1));

  cudaMemcpy(blockrows_d, &blockrows[0], sizeof(unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start_d, end_d;
  cuda_start_timer(start_d, end_d);

  cu_spMV2<T, unsigned, BLOCKSIZE><<<blocks_IPCSR, BLOCKSIZE>>>(IA_d, JA_d, blockrows_d, n, x_d, spMV_ans_d);

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

    auto avg_nnz {2u*A.get_edges()/A.get_n()};

    auto num_blocks3 {n/THREADS3 + (n%THREADS3==0 ? 0 : 1)};

    cu_spMV3<T, unsigned, THREADS3><<<num_blocks3, THREADS3>>>(IA_d,JA_d, n, avg_nnz, x_d, spMV_ans_d);

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

    cu_spMV4<T, unsigned, BLOCKSIZE><<<n, BLOCKSIZE>>>(IA_d, JA_d, x_d, spMV_ans_d);

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

    unsigned split {std::max(n/10000, 1u)};

    std::vector<unsigned> tmp_hybrid(n-split+1);

    auto j {split};
    auto offset {A.row_offset[split]};
    for (auto it=tmp_hybrid.begin();it!=tmp_hybrid.end();it++)
      *it = A.row_offset[j++] - offset;

    unsigned hybrid_blocks {(n-split) / BLOCKSIZE + ((n-split) % BLOCKSIZE ? 1 : 0)};

    unsigned * other_IA_d;
    cudaMalloc(&other_IA_d, sizeof(unsigned)*(n-split+1));
    cudaMemcpy(other_IA_d, &tmp_hybrid[0], sizeof(unsigned)*(n-split+1), cudaMemcpyHostToDevice);

    cudaStream_t stream[2];

    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    cu_spMV4<T, unsigned, BLOCKSIZE><<<split, BLOCKSIZE,0,stream[0]>>>(IA_d, JA_d, x_d, spMV_ans_d);
    cu_spMV1<T><<<hybrid_blocks, BLOCKSIZE, 0, stream[1]>>>(other_IA_d, &JA_d[A.row_offset[split]], (n-split), x_d, &spMV_ans_d[split]);
    
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

