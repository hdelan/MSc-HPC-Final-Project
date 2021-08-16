#include "cu_multiplyOut.h"
#include "cblas.h"
#include <iomanip>
#include <algorithm>

#include "cublas_v2.h"

void my_exp_func(double &a)
{
        a = std::exp(a);
}

void cu_multOut(lanczosDecomp &L, eigenDecomp &E, adjMatrix &A)
{

        auto n{L.get_n()}, k{L.get_krylov()};

        // Applying function
        for (auto j = 0u; j < L.krylov_dim; j++)
                my_exp_func(E.eigenvalues[j]);

        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        for (auto j = 0u; j < L.krylov_dim; j++)
                E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];

        //print_matrix(3, 1, &E.eigenvalues[0]);
        double *Q_d, *V_d, *QV_d, *eigvals_d, *ans_d, alpha {1.0}, beta {0.0};

        std::vector<double> QV(n*k, 1.0);

        cublasStatus_t status;
        cudaError_t cudaStat;
        cublasHandle_t handle;
        
        cudaStat = cudaMalloc(&Q_d, sizeof(double) * n * k);
        if (cudaStat != cudaSuccess) {
          std::cerr << "Allocation error for Q_d\n";
          return;
        }
        cudaStat = cudaMalloc(&V_d, sizeof(double) * k * k);
        if (cudaStat != cudaSuccess) {
          std::cerr << "Allocation error for V_d.\n";
          return;
        }
        cudaStat = cudaMalloc(&QV_d, sizeof(double) * n * k);
        if (cudaStat != cudaSuccess) {
          std::cerr << "Allocation error for QV_d.\n";
          return;
        }
        cudaStat = cudaMalloc(&eigvals_d, sizeof(double) * k);
        if (cudaStat != cudaSuccess) {
          std::cerr << "Allocation error for eigvals_d.\n";
          return;
        }
        cudaStat = cudaMalloc(&ans_d, sizeof(double) * n);
        if (cudaStat != cudaSuccess) {
          std::cerr << "Allocation error for ans_d.\n";
          return;
        }

        status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cerr << "Cublas initialization error.\n";
          return;
        }
        status = cublasSetVector(n*k, sizeof(double),L.Q, 1, Q_d,1);
        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cerr << "Device access error.\n";
          return;
        }
        status = cublasSetVector(k*k, sizeof(double),E.eigenvectors, 1, V_d,1);
        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cerr << "Device access error.\n";
          return;
        }
        status = cublasSetVector(k, sizeof(double),E.eigenvalues, 1, eigvals_d,1);
        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cerr << "Device access error.\n";
          return;
        }
        status = cublasSetVector(k*n, sizeof(double),&QV[0], 1, QV_d,1);
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

        std::vector<double> tmp(n*k);
        
        status = cublasGetVector(n, sizeof(double),ans_d, 1,&L.ans[0],1);
        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cerr << "Error transferring from device to host.\n";
          return;
        }

        cublasDestroy(handle);

        cudaFree(Q_d);
        cudaFree(V_d);
        cudaFree(QV_d);
        cudaFree(eigvals_d);
        cudaFree(ans_d);
}
