#include "eigen.h"
#include "cu_lanczos.h"
#include <iomanip>

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
