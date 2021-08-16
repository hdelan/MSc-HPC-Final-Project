#include "cu_multiplyOut.h"
#include "cblas.h"
#include <iomanip>

#include "cublas_v2.h"

void my_exp_func(double & a) {
        a = std::exp(a);
}

void cu_multOut(lanczosDecomp & L, eigenDecomp & E, adjMatrix & A) {

        auto n {L.get_n()}, k {L.get_krylov()};
        
        // Applying function
        for (auto j=0u;j<L.krylov_dim;j++) my_exp_func(E.eigenvalues[j]);
        
        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        for (auto j=0u;j<L.krylov_dim;j++) E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];

        //print_matrix(3, 1, &E.eigenvalues[0]);
        double *Q_d, *V_d;
        double * QV_d;

        cudaMalloc(&Q_d, sizeof(double)*n*k);
        cudaMalloc(&V_d, sizeof(double)*k*k);
        cudaMalloc(&QV_d, sizeof(double)*n*k);

        cudaMemcpy(Q_d, L.Q, sizeof(double)*n*k,cudaMemcpyHostToDevice);
        cudaMemcpy(V_d, E.eigenvectors, sizeof(double)*k*k,cudaMemcpyHostToDevice);
        
        cublasHandle_t handle;

        double alpha{1}, beta{0};
        
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,n,k,k,&alpha,V_d,k,Q_d,n,&beta,QV_d,n);

        /*
        for (auto j = 0u; j < n; j++)
        {
                for (auto k = 0u; k < L.krylov_dim; k++)
                        std::cout << std::setprecision(20) << Q[k + j * L.krylov_dim] << " ";
                std::cout << '\n';
        }
        for (auto j = 0u; j < L.krylov_dim; j++)
        {
                for (auto k = 0u; k < L.krylov_dim; k++)
                        std::cout << std::setprecision(20) << E.eigenvectors[k + j * L.krylov_dim] << " ";
                std::cout << '\n';
        }
        */
/*
        // Getting QV (n x k)
        for (auto i=0u;i<n;i++) {
                for (auto j=0u;j<L.krylov_dim;j++) {
                        QV[i*L.krylov_dim+j] = 0.0;
                        for (auto k=0u;k<L.krylov_dim;k++) {
                                QV[i*L.krylov_dim+j] += L.Q[i*L.krylov_dim+k]*E.eigenvectors[k*L.krylov_dim+j];
                        }
                }
        }
        cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, k, 1, L.Q, k, E.eigenvectors, k, 0, QV, k);
        */
// PRINT OUT QV
/*
        std::cout << "\nQV\n";
        for (auto j = 0u; j < L.krylov_dim; j++)
        {
                for (auto k = 0u; k < L.krylov_dim; k++)
                        std::cout <<std::setprecision(4)<< QV[k + j * L.krylov_dim] << " ";
                std::cout << '\n';
        }
        */
/*
        */
        // Getting QV*f(lambda)
        //cblas_dgemv(CblasRowMajor, CblasNoTrans, n, L.krylov_dim, 1, QV, k, &E.eigenvalues[0], 1, 0, &L.ans[0],1);
        //naive_dgemv(QV, &E.eigenvalues[0], n, L.krylov_dim, &L.ans[0]);

        //delete[](QV);
        cudaFree(Q_d);
        cudaFree(V_d);
        cudaFree(QV_d);
        
}

