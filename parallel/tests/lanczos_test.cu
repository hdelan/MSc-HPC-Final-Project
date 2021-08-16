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

#define BLOCKSIZE 32
#define SEED 1234 // To seed RNG
#define WIDTH 81  // for formatting std::cout output

int main(void)
{
    unsigned n{10'000};
    long unsigned edges{n * 10};
    unsigned krylov_dim {1000};
    
    /*
    std::string filename {"../data/California.mtx"};
    std::ifstream fs;
    fs.open(filename);
    assert(!fs.fail() && "File opening failed\n");
    fs >> n >> n >> edges;
    */
    unsigned width {17};

    timeval start, end;
    gettimeofday(&start, NULL);
    //adjMatrix A(n, edges, fs);
    //fs.close();
    adjMatrix A(n, edges);
    gettimeofday(&end, NULL);
    std::cout << "\nTime elapsed to build random adjacency matrix with n = " << n << " edges = " << edges << ":\n\t"
              << end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0 << " seconds\n\n";

    std::cout << "Running Lanczos algorithm for krylov_dim "<< krylov_dim << "\n\n";

    std::vector<double> x (n, 1);
    
    timeval s, e1, e2;
    gettimeofday(&s, NULL);
    
    // SERIAL LANCZOS
    bool cuda {false};
    lanczosDecomp L(A, krylov_dim, &x[0], cuda);
    gettimeofday(&e1, NULL);
    
    eigenDecomp E(L);
    multOut(L, E, A);
    
    gettimeofday(&e2, NULL);
    
    double cpu_time_lanczos {e1.tv_sec - s.tv_sec + (e1.tv_usec - s.tv_usec) / 1000000.0};
    double cpu_time_whole {e2.tv_sec - s.tv_sec + (e2.tv_usec - s.tv_usec) / 1000000.0};
    
    timeval s_d, e_d;
    cudaEvent_t start_d, end_d;
    cuda_start_timer(start_d, end_d);
    
    // CUDA LANCZOS
    cuda = true;
    lanczosDecomp cu_L(A, krylov_dim, &x[0], cuda);
    float gpu_time_lanczos{cuda_end_timer(start_d, end_d)};
    
    gettimeofday(&s_d, NULL);
    eigenDecomp cu_E(cu_L);
    cu_multOut(cu_L, cu_E, A);
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

