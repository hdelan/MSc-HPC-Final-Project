/**
 * \file:        multiplyOut.cu
 * \brief:       Multiply out in serial with multithreaded cblas.
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */
#include "multiplyOut.h"
#include "lapacke.h"
#include "cblas.h"
#include <iomanip>

void exp_func(double & a) {
        a = std::exp(a);
}

void multOut(lanczosDecomp & L, eigenDecomp & E, adjMatrix & A) {

        auto n {L.get_n()};
        
        // Applying function
        for (auto j=0u;j<L.krylov_dim;j++) exp_func(E.eigenvalues[j]);
        
        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        for (auto j=0u;j<L.krylov_dim;j++) E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];
        
        double * QV {new double [n*L.krylov_dim]};
      
        auto k = L.get_krylov();
        cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, k, 1, L.Q, k, E.eigenvectors, k, 1, QV, k);

        // Getting QV*f(lambda)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, L.krylov_dim, 1, QV, L.krylov_dim, &E.eigenvalues[0], 1, 1, &L.ans[0],1);

        delete[](QV);
        
}

void print_matrix(unsigned rows, unsigned cols, double * A) {
        std::cout << "Printing matrix for "<<rows<<" rows and " << cols<< "cols\n";
        for (auto i=0u; i<rows; ++i) {
                for (auto j=0u; j<cols; ++j){
                        std::cout << A[i*cols+j] << " ";
                }
                std::cout << '\n';
        }
}
