/**
 * \file:        eigen.cu
 * \brief:       Computes the eigendecomposition of the tridiagonal defined by alpha, beta
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */
#include "eigen.h"
#include "cu_lanczos.h"
#include <iomanip>

// Overloaded so can be called with floats
void LAPACKE_dstevd(int layout, char job, int n, float* a, float * b, float * c, int k) {
    LAPACKE_sstevd(layout, job, n, a, b, c, k);
}

template <typename T>
void eigenDecomp<T>::decompose()
{
  LAPACKE_dstevd(LAPACK_ROW_MAJOR, 'V', L.krylov_dim, eigenvalues, L.beta, eigenvectors, L.krylov_dim);
}

template class eigenDecomp<float>;
template class eigenDecomp<double>;
