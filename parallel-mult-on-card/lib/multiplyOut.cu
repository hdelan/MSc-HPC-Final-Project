/**
 * \file:        multiplyOut.cu
 * \brief:       Multiply out in serial with multithreaded cblas.
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */
#include "multiplyOut.h"
#include "helpers.h"
#include <cblas.h>
#include <iomanip>

template <typename T>
void multOut(lanczosDecomp<T> & L, eigenDecomp<T> & E, adjMatrix & A) {

        auto n {L.get_n()};
        
        // Applying function
        for (auto j=0u;j<L.krylov_dim;j++) my_exp_func(E.eigenvalues[j]);
       
        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        for (auto j=0u;j<L.krylov_dim;j++) E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];

        
        auto k = L.get_krylov();
   
        std::vector<T> tmp(k);
        openblas_set_num_threads(8);
        
        cblas_dgemv(CblasRowMajor, CblasNoTrans, k, k, 1, E.eigenvectors, k, &E.eigenvalues[0], 1, 0, &tmp[0],1);
        
        // Getting QV*f(lambda)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, k, 1, L.Q, k, &tmp[0], 1, 0, &L.ans[0],1);
        
}

template void multOut(lanczosDecomp<double>&, eigenDecomp<double>&, adjMatrix &);
template void multOut(lanczosDecomp<float>&, eigenDecomp<float>&, adjMatrix &);
