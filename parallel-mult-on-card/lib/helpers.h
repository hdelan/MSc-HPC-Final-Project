#ifndef HELPERS_H_32423423
#define HELPERS_H_32423423

#include "cublas_v2.h"
#include "cblas.h"
#include <string>
#include <unistd.h>

int parseArguments(int, char **, std::string &, unsigned &, bool &, unsigned &, unsigned &, unsigned &);
template <typename T>
void diff_arrays(const T * const, const T * const, const unsigned n, T & , unsigned & );
template <typename T>
T norm(const T * const a, const unsigned n);

template <typename T>
void my_exp_func(T &a);

void cuda_start_timer(cudaEvent_t &start, cudaEvent_t &end);
float cuda_end_timer(cudaEvent_t &start, cudaEvent_t &end);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          float * alpha,
                          float * A, int lda,
                          float * B, int ldb,
                          float * beta,
                          float * C, int ldc);
cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t transa,
                          int m, int n,
                          float * alpha,
                          float * A, int lda,
                          float * x, int incx,
                          float * beta,
                          float * y, int incy);
void cblas_dgemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, 
                        int m, int n, int k, 
                        float alpha, 
                        float * A, int lda, 
                        float * B, int ldb, 
                        float beta, 
                        float * C, int ldc);
void cblas_dgemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE transa, 
                        int m, int n, 
                        float alpha,
                        float * A, int lda, 
                        float * x, int incx, 
                        float beta,
                        float * y, int incy);

#endif
