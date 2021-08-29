#include "cu_multiplyOut.h"
#include "multiplyOut.h"
#include "helpers.h"

#include <iomanip>
#include <algorithm>
#include <type_traits>

template <typename T>
__global__ void print_some(T * a) {
  printf("%E %E %E\n\n", a[0], a[1], a[2]);
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
  T *eigvals_d, *ans_d, alpha {1.0}, beta {0.0};

  cblas_dgemv(CblasRowMajor, CblasNoTrans, L.krylov_dim, L.krylov_dim, 1, &E.eigenvectors[0], k, &E.eigenvalues[0], 1, 0, &L.ans[0],1);

  cublasStatus_t status;
  cudaError_t cudaStat;
  cublasHandle_t handle;

  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

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
  
  // Memory transfers
  status = cublasSetVectorAsync(k, sizeof(T),&L.ans[0], 1, eigvals_d,1, stream[0]);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Device access error.\n";
    return;
  }
  //printf("\nSome values from Q_d: ");
  //print_some<<<1,1>>>(L.Q_d);
  /*
    // DGEMM
    status = cublasDgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_T,
                                n,k,k,
                                &alpha,
                                L.Q_d,n,
                                V_d,k,
                                &beta,
                                QV_d,n);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Dgemm error.\n";
      return;
    }
    //printf("\nSome values from QV_d: ");
    //print_some<<<1,1>>>(QV_d);
*/
    // DGEMV
    status = cublasDgemv_v2(handle,CUBLAS_OP_N,
                                n,k,
                                &alpha,
                                L.Q_d,n,
                                eigvals_d,1,
                                &beta,ans_d,1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Dgemv error.\n";
      return;
    }
    //printf("\nSome values from ans_d: ");
    //print_some<<<1,1>>>(ans_d);

  status = cublasGetVector(n, sizeof(T),ans_d, 1,&L.ans[0],1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error transferring from device to host.\n";
    return;
  }

  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  cublasDestroy(handle);

  cudaFree(eigvals_d);
  cudaFree(ans_d);
}

template void cu_multOut(lanczosDecomp<float> &L, eigenDecomp<float> &E, adjMatrix &A);
template void cu_multOut(lanczosDecomp<double> &L, eigenDecomp<double> &E, adjMatrix &A);
