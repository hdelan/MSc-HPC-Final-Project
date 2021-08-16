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
        double *Q_d, *V_d, *QV_d, *eigvals_d, *ans_d, *alpha_d, *beta_d, alpha {1}, beta {0};

        cudaMalloc(&Q_d, sizeof(double) * n * k);
        cudaMalloc(&V_d, sizeof(double) * k * k);
        cudaMalloc(&QV_d, sizeof(double) * n * k);
        cudaMalloc(&eigvals_d, sizeof(double) * k);
        cudaMalloc(&ans_d, sizeof(double) * n);
        cudaMalloc(&alpha_d, sizeof(double));
        cudaMalloc(&beta_d, sizeof(double));

        cudaMemcpy(Q_d, L.Q, sizeof(double) * n * k, cudaMemcpyHostToDevice);
        cudaMemcpy(V_d, E.eigenvectors, sizeof(double) * k * k, cudaMemcpyHostToDevice);
        cudaMemcpy(eigvals_d, E.eigenvalues, sizeof(double) * k, cudaMemcpyHostToDevice);
        cudaMemcpy(alpha_d, &alpha, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(beta_d, &beta, sizeof(double), cudaMemcpyHostToDevice);

        cublasHandle_t handle{};


        cublasDgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_T,k,n,k,alpha_d,V_d,k,Q_d,n,beta_d,QV_d,n);
        cublasDgemv_v2(handle, CUBLAS_OP_T, k, n, alpha_d, Q_d, k, eigvals_d, 1, beta_d ,ans_d, 1);

        std::vector<double> tmp(10);

        cudaMemcpy(&tmp[0], V_d, sizeof(double) * 10, cudaMemcpyDeviceToHost);
        std::cout << "CU first QV:\n";
        std::for_each(tmp.begin(), tmp.end(), [](double & a){ std::cout << a << " ";});
        std::cout << "\n";

        cudaMemcpy(&L.ans[0], ans_d, sizeof(double) * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(&tmp[0], ans_d, sizeof(double) * 10, cudaMemcpyDeviceToHost);
        std::cout << "CU first ans:\n";
        std::for_each(tmp.begin(), tmp.end(), [](double & a){ std::cout << a << " ";});
        std::cout << "\n\n";

        cudaFree(Q_d);
        cudaFree(V_d);
        cudaFree(QV_d);
        cudaFree(eigvals_d);
        cudaFree(ans_d);
        cudaFree(alpha_d);
        cudaFree(beta_d);
}
