#include "cu_multiplyOut.h"

#include "cblas.h"

#include <iomanip>
#include <algorithm>
#include <type_traits>

#include "cublas_v2.h"

  template <typename T>
void my_exp_func(T &a)
{
  a = std::exp(a);
}

// Overloading cublas functions so I can call Sgemm through Dgemm 
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          float * alpha,
                          float * A, int lda,
                          float * B, int ldb,
                          float * beta,
                          float * C, int ldc) {
  return cublasSgemm_v2(handle, transa, transb, m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t transa,
                          int m, int n,
                          float * alpha,
                          float * A, int lda,
                          float * x, int incx,
                          float * beta,
                          float * y, int incy) {
  return cublasSgemv_v2(handle,transa,m,n,alpha,A,lda,x,incx,beta,y,incy);
}

template <typename T>
void cu_multOut(lanczosDecomp<T> &L, eigenDecomp<T> &E, adjMatrix &A)
{

  auto n{L.get_n()}, k{L.get_krylov()};

  // Applying function
  for (auto j = 0u; j < L.krylov_dim; j++)
    my_exp_func(E.eigenvalues[j]);

  // Elementwise multiplying of f(lambda) by first row of eigenvectors
  for (auto j = 0u; j < L.krylov_dim; j++)
    E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];

  //print_matrix(3, 1, &E.eigenvalues[0]);
  T *Q_d, *V_d, *QV_d, *eigvals_d, *ans_d, alpha {1.0}, beta {0.0};

  std::vector<T> QV(n*k);

  cublasStatus_t status;
  cudaError_t cudaStat;
  cublasHandle_t handle;

  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  cudaStat = cudaMalloc(&Q_d, sizeof(T) * n * k);
  if (cudaStat != cudaSuccess) {
    std::cerr << "Allocation error for Q_d\n";
    return;
  }
  cudaStat = cudaMalloc(&V_d, sizeof(T) * k * k);
  if (cudaStat != cudaSuccess) {
    std::cerr << "Allocation error for V_d.\n";
    return;
  }
  cudaStat = cudaMalloc(&QV_d, sizeof(T) * n * k);
  if (cudaStat != cudaSuccess) {
    std::cerr << "Allocation error for QV_d.\n";
    return;
  }
  cudaStat = cudaMalloc(&eigvals_d, sizeof(T) * k);
  if (cudaStat != cudaSuccess) {
    std::cerr << "Allocation error for eigvals_d.\n";
    return;
  }
  cudaStat = cudaMalloc(&ans_d, sizeof(T) * n);
  if (cudaStat != cudaSuccess) {
    std::cerr << "Allocation error for ans_d.\n";
    return;
  }

  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Cublas initialization error.\n";
    return;
  }
  status = cublasSetVectorAsync(n*k, sizeof(T),L.Q, 1, Q_d,1, stream[0]);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Device access error.\n";
    return;
  }
  status = cublasSetVectorAsync(k*k, sizeof(T),E.eigenvectors, 1, V_d,1, stream[1]);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Device access error.\n";
    return;
  }
  status = cublasSetVectorAsync(k, sizeof(T),E.eigenvalues, 1, eigvals_d,1, stream[0]);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Device access error.\n";
    return;
  }
  status = cublasSetVectorAsync(k*n, sizeof(T),&QV[0], 1, QV_d,1, stream[1]);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Device access error.\n";
    return;
  }

    // DGEMM
    status = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,k,n,k,&alpha,V_d,k,Q_d,k,&beta,QV_d,k);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Dgemm error.\n";
      return;
    }

    // DGEMV
    status = cublasDgemv_v2(handle, CUBLAS_OP_T, k, n, &alpha, QV_d, k, eigvals_d, 1,&beta ,ans_d, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Dgemv error.\n";
      return;
    }

  status = cublasGetVector(n, sizeof(T),ans_d, 1,&L.ans[0],1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error transferring from device to host.\n";
    return;
  }

  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  cublasDestroy(handle);

  cudaFree(Q_d);
  cudaFree(V_d);
  cudaFree(QV_d);
  cudaFree(eigvals_d);
  cudaFree(ans_d);
}

template void cu_multOut<float>(lanczosDecomp<float> &L, eigenDecomp<float> &E, adjMatrix &A);
template void cu_multOut<double>(lanczosDecomp<double> &L, eigenDecomp<double> &E, adjMatrix &A);
