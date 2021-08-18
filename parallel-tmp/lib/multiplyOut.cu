#include "multiplyOut.h"
#include "lapacke.h"
#include <cblas.h>
#include <iomanip>

template <typename T>
void exp_func(T & a) {
        a = std::exp(a);
}

// Wrapping sgemm in dgemm and sgemv in dgemv
void cblas_dgemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, 
                        int m, int n, int k, 
                        float alpha, 
                        float * A, int lda, 
                        float * B, int ldb, 
                        float beta, 
                        float * C, int ldc) {
  cblas_sgemm (layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
void cblas_dgemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE transa, 
                        int m, int n, 
                        float alpha,
                        float * A, int lda, 
                        float * x, int incx, 
                        float beta,
                        float * y, int incy) {
  cblas_sgemv(layout, transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <typename T>
void multOut(lanczosDecomp<T> & L, eigenDecomp<T> & E, adjMatrix & A) {

        auto n {L.get_n()};
        
        // Applying function
        for (auto j=0u;j<L.krylov_dim;j++) exp_func(E.eigenvalues[j]);
        //printf("\nSome values from Q serial: %E %E %E\n", L.Q[0], L.Q[L.krylov_dim], L.Q[2*L.krylov_dim]);
        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        for (auto j=0u;j<L.krylov_dim;j++) E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];

        //print_matrix(3, 1, &E.eigenvalues[0]);
        
        T * QV {new T [n*L.krylov_dim]};
        


        auto k = L.get_krylov();
        cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, k, 1, L.Q, k, E.eigenvectors, k, 0, QV, k);
        
        //printf("\nSome values from QV serial: %E %E %E\n", QV[0], QV[L.krylov_dim], QV[2*L.krylov_dim]);
        
        // Getting QV*f(lambda)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, L.krylov_dim, 1, QV, k, &E.eigenvalues[0], 1, 0, &L.ans[0],1);

        delete[](QV);
        
}

template void multOut(lanczosDecomp<double>&, eigenDecomp<double>&, adjMatrix &);
template void multOut(lanczosDecomp<float>&, eigenDecomp<float>&, adjMatrix &);
