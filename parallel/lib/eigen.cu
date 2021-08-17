#include "eigen.h"
#include <iomanip>
#include <type_traits>

template <typename T>
void eigenDecomp<T>::decompose()
{
        if (std::is_same<T, double>::value) LAPACKE_dstevd(LAPACK_ROW_MAJOR, 'V', L.krylov_dim, eigenvalues, L.beta, eigenvectors, L.krylov_dim);
        if (std::is_same<T, float>::value) LAPACKE_sstevd(LAPACK_ROW_MAJOR, 'V', L.krylov_dim, eigenvalues, L.beta, eigenvectors, L.krylov_dim);

        /*
        std::cout << "Eigenvalues:\n";
        for (int i=0;i<L.get_krylov();i++) {
          std::cout << eigenvalues[i] << " ";
        }
          std::cout << '\n';
          */
}

