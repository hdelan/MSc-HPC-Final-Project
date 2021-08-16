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

#define BLOCKSIZE 128
#define SEED 1234 // To seed RNG
#define WIDTH 81  // for formatting std::cout output

template <typename T>
void cu_linalg_test(const unsigned n, adjMatrix &A);

int main(void)
{
    unsigned n{5'000};
    long unsigned edges{n * 100};
    
    std::string filename {"../data/California.mtx"};
    std::ifstream fs;
    fs.open(filename);
    assert(!fs.fail() && "Reading in file failed\n");
    fs >> n >> n >> edges;

    std::cout << "\nTesting CUDA linear algebra functions for n ="<<n<<" blocksize="<<BLOCKSIZE<<"\n\n";
    
    timeval start, end;
    gettimeofday(&start, NULL);
    //adjMatrix A(n, edges);
    adjMatrix A(n, edges, fs);
    fs.close();
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
    std::cout << '\n';
}

template <typename T>
void cu_linalg_test(const unsigned n, adjMatrix &A)
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

    T *x_d, *y_d, *tmp_d, *ans_d, *spMV_ans_d, *alpha_d;
    long unsigned *IA_d, *JA_d;


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

        cu_dot_prod<T, BLOCKSIZE><<<h_blocks, BLOCKSIZE, BLOCKSIZE * sizeof(T)>>>(x_d, y_d, n, tmp_d);
        cu_reduce<T, BLOCKSIZE><<<1, BLOCKSIZE, BLOCKSIZE * sizeof(T)>>>(tmp_d, num_blocks, ans_d);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        auto serial_ans = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
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

        cu_norm_sq<T, BLOCKSIZE><<<h_blocks, BLOCKSIZE, BLOCKSIZE * sizeof(T)>>>(x_d, n, tmp_d);
        cu_reduce_sqrt<T, BLOCKSIZE><<<1, BLOCKSIZE, BLOCKSIZE * sizeof(T)>>>(tmp_d, num_blocks, ans_d);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        auto serial_ans = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));
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

        cu_reduce<T, BLOCKSIZE><<<h_blocks, BLOCKSIZE, BLOCKSIZE * sizeof(T)>>>(x_d, n, tmp_d);
        cu_reduce<T, BLOCKSIZE><<<1, BLOCKSIZE, BLOCKSIZE * sizeof(T)>>>(tmp_d, num_blocks, ans_d);

        float gpu_time{cuda_end_timer(start_d, end_d)};

        cudaMemcpy(&ans, ans_d, sizeof(T), cudaMemcpyDeviceToHost);

        timeval start, end;
        gettimeofday(&start, NULL);
        auto serial_ans = std::accumulate(x.begin(), x.end(), 0.0);
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


        int idx{0};
        timeval start, end;
        gettimeofday(&start, NULL);
        std::for_each(x.begin(), x.end(), [&](T & a)
                      { a -= alpha * y[idx++]; });
        auto serial_ans{std::inner_product(x.begin(), x.end(), x.begin(), 0.0)};
        gettimeofday(&end, NULL);
        auto speedup{(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) / gpu_time};

        std::cout << std::setw(width)<<std::left<< "Vector add:"<<std::right<<std::setw(width) << ans
                  << std::setw(width) << serial_ans 
                  <<std::setw(width) << (serial_ans - ans) / serial_ans <<std::setw(width) << speedup << "\n";
    }
    /**********VECTOR SCALING*********/
    {
        cudaMemcpy(x_d, &x[0], sizeof(T) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, &y[0], sizeof(T) * n, cudaMemcpyHostToDevice);

        T alpha{4};
        cudaMemcpy(alpha_d, &alpha, sizeof(T), cudaMemcpyHostToDevice);
        
        std::vector<T> gpu_ans_vec(n);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_dvexda<T><<<num_blocks, BLOCKSIZE>>>(x_d, alpha_d, y_d, n);

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

        std::cout << std::setw(width)<<std::left<< "v = v/alpha:"<<std::right<<std::setw(width) << ans
                  << std::setw(width) << serial_ans 
                  <<std::setw(width) << (serial_ans - ans) / serial_ans <<std::setw(width) << speedup << "\n";
    }

    cudaMalloc((void **)&IA_d, sizeof(long unsigned) * (n + 1));
    cudaMalloc((void **)&JA_d, sizeof(long unsigned) * A.edge_count * 2);
    cudaMalloc((void **)&spMV_ans_d, sizeof(T) * n);
    cudaMemcpy(IA_d, A.row_offset, sizeof(long unsigned) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(JA_d, A.col_idx, sizeof(long unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);
    /**************SPMV1*********/
    {
        std::vector<T> gpu_ans_vec(n);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_spMV1<T><<<num_blocks, BLOCKSIZE>>>(IA_d, JA_d, static_cast<unsigned long>(n), x_d, spMV_ans_d);

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

        std::cout << std::setw(width)<<std::left<< "SPMV1:"<<std::right<<std::setw(width) << ans
                  << std::setw(width) << serial_ans 
                  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
    }
    /**************SPMV2*********/
    {
        std::vector<T> gpu_ans_vec(n);
        std::vector<long unsigned> blockrows(n);
        long unsigned blocks_needed{0u};
        get_blockrows<T>(A, BLOCKSIZE, &blockrows[0], blocks_needed);

        dim3 blocks_IPCSR{static_cast<unsigned>(blocks_needed)};

        long unsigned *blockrows_d;

        cudaMalloc((void **)&blockrows_d, sizeof(long unsigned) * (blocks_needed + 1));

        cudaMemcpy(blockrows_d, &blockrows[0], sizeof(long unsigned) * (blocks_needed + 1), cudaMemcpyHostToDevice);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_spMV2<T, long unsigned><<<blocks_IPCSR, BLOCKSIZE>>>(IA_d, JA_d, blockrows_d, static_cast<unsigned long>(n), x_d, spMV_ans_d);

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

        std::cout << std::setw(width)<<std::left<< "SPMV2:"<<std::right<<std::setw(width) << ans
                  << std::setw(width) << serial_ans 
                  <<std::setw(width) << relative_error / serial_ans <<std::setw(width) << speedup << "\n";
        cudaFree(blockrows_d);
    }
    /**************SPMV3*********/
    // The numerics are currently not working on this. Just included to get timings
    /*
    {
        std::vector<T> gpu_ans_vec(n);
        std::vector<long unsigned> blockrows(n);

        T *tmp_d;

        cudaMalloc((void **)&tmp_d, sizeof(T) * A.get_edges() * 2);

        cudaEvent_t start_d, end_d;
        cuda_start_timer(start_d, end_d);

        cu_spMV3_kernel1<T, long unsigned><<<num_blocks, BLOCKSIZE>>>(JA_d, A.get_edges() * 2, x_d, tmp_d);
        cu_spMV3_kernel2<T, long unsigned><<<1, BLOCKSIZE, 49152>>>(tmp_d, IA_d, static_cast<unsigned long>(n), spMV_ans_d);

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
    }*/

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(tmp_d);
    cudaFree(ans_d);
    cudaFree(alpha_d);
    cudaFree(spMV_ans_d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    cudaProfilerStop();
}

