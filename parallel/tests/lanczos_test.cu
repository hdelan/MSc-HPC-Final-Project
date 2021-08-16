#include "../lib/cu_linalg.h"
#include "../lib/cu_lanczos.h"
#include "../lib/cu_SPMV.h"
#include "../lib/cu_multiplyOut.h"
#include "../lib/eigen.h"
#include "../lib/SPMV.h"
#include "../lib/adjMatrix.h"
#include "../lib/helpers.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <string>

#include <cuda_profiler_api.h>

#define WIDTH 81  // for formatting std::cout output

double seconds_between(timeval s, timeval e);

int main(void)
{
    unsigned n{10'000};
    long unsigned edges{n * 10};
    unsigned krylov_dim {1000};
    unsigned width {17};
    adjMatrix A;
    
    timeval start, end;
    gettimeofday(&start, NULL);

    char make_or_read_matrix {'f'};
    if (make_or_read_matrix == 'f') {
      // Read matrix from file
      std::string filename {"../data/California.mtx"};
      std::ifstream fs;
      fs.open(filename);
      assert(!fs.fail() && "File opening failed\n");
      fs >> n >> n >> edges;
      adjMatrix B(n, edges, fs);
      A = std::move(B);
      fs.close();
    } else {
      // Make random matrix
      adjMatrix B(n, edges);
      A = std::move(B);
    }

    gettimeofday(&end, NULL);
    std::cout << "\nTime elapsed to build random adjacency matrix with n = " << n << " edges = " << edges << ":\n\t"
              << seconds_between(start, end) << " seconds\n\n";
    
    std::cout << "Running Lanczos algorithm for krylov_dim "<< krylov_dim << "\n\n";

    std::vector<double> x (n, 1);
    
    timeval s, e1, e2, e3, e4;
    gettimeofday(&s, NULL);
    
    // SERIAL LANCZOS
    bool cuda {false};
    lanczosDecomp L(A, krylov_dim, &x[0], cuda);
    gettimeofday(&e1, NULL);
    
    eigenDecomp E(L);
    gettimeofday(&e2, NULL);

    multOut(L, E, A);
    gettimeofday(&e3, NULL);
    
    gettimeofday(&e4, NULL);
    
    double cpu_time_lanczos {seconds_between(s, e1)};
    double cpu_time_mult {seconds_between(e2, e3)};
    double cpu_time_whole {seconds_between(s, e3)};
    
    timeval s_d, e_d;
    cudaEvent_t start1_d, start2_d, end1_d, end2_d;
    cuda_start_timer(start1_d, end1_d);
    
    // CUDA LANCZOS
    cuda = true;
    lanczosDecomp cu_L(A, krylov_dim, &x[0], cuda);
    float gpu_time_lanczos{cuda_end_timer(start1_d, end1_d)};
    
    gettimeofday(&s_d, NULL);
    eigenDecomp cu_E(cu_L);
    
    cuda_start_timer(start2_d, end2_d);
    cu_multOut(cu_L, cu_E, A);
    float gpu_time_mult{cuda_end_timer(start2_d, end2_d)};
    
    gettimeofday(&e_d, NULL);
    double gpu_time_whole {gpu_time_lanczos + (e_d.tv_sec - s_d.tv_sec + (e_d.tv_usec - s_d.tv_usec) / 1000000.0)};
    
    
  
    std::cout << std::setfill('~') << std::setw(WIDTH) << '\n' << std::setfill(' ');
    std::cout << "TIMING\n";
    std::cout << std::setfill('~') << std::setw(WIDTH) << '\n' << std::setfill(' ');
    std::cout << std::setw(2*width) << "Serial" << std::setw(width) << "CUDA" << std::setw(width) << "Speedup"<<'\n';
    std::cout << std::setfill('~') << std::setw(WIDTH) << '\n' << std::setfill(' ');
    std::cout << std::setw(width) << std::left << "Lanczos" << std::right
              << std::setw(width) << cpu_time_lanczos 
              << std::setw(width) << gpu_time_lanczos
              << std::setw(width) << cpu_time_lanczos/gpu_time_lanczos << "\n\n";
    std::cout << std::setw(width) << std::left << "Multiply Out" << std::right
              << std::setw(width) << cpu_time_mult
              << std::setw(width) << gpu_time_mult
              << std::setw(width) << cpu_time_mult/gpu_time_mult << "\n\n";
    std::cout << std::setw(width) << std::left << "Entire algorithm" << std::right
              << std::setw(width) << cpu_time_whole
              << std::setw(width) << gpu_time_whole
              << std::setw(width) << cpu_time_whole/gpu_time_whole << "\n\n";
    
    std::cout << std::setfill('~') << std::setw(WIDTH) << '\n' << std::setfill(' ');
    std::cout << "ERROR CHECKING\n";
    std::cout << std::setfill('~') << std::setw(WIDTH) << '\n' << std::setfill(' ');
    
    L.check_ans(cu_L);
    
    std::cout << std::setfill('~') << std::setw(WIDTH) << '\n' << std::setfill(' ');
}

double seconds_between(timeval s, timeval e) {
  return e.tv_sec - s.tv_sec + (e.tv_usec - s.tv_usec) / 1000000.0;
}
