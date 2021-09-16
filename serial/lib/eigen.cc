/**
 * \file:        eigen.cu
 * \brief:       Computes the eigendecomposition of the tridiagonal defined by alpha, beta
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */

#include "eigen.h"
#include <iomanip>

void eigenDecomp::decompose()
{
        LAPACKE_dstevd(LAPACK_ROW_MAJOR, 'V', L.krylov_dim, eigenvalues, L.beta, eigenvectors, L.krylov_dim);
}

std::ostream &operator<<(std::ostream &os, eigenDecomp &E)
{
        auto k {E.L.get_krylov()};
        os << "Eigenvalues:\n";
        for (auto i = 0u; i <k; i++)
        {
                os << E.eigenvalues[i] << " \t";
        }
        os << "\n\n";
        os << "Eigenvectors:\n";
        for (auto i = 0u; i < k * k; i++)
        {
                os << E.eigenvectors[i] << " \t";
                if (i % k == k - 1)
                        os << '\n';
        }
        return os;
};
