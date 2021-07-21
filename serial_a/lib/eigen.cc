#include "eigen.h"

void eigenDecomp::decompose(lanczosDecomp &D)
{
        LAPACKE_dstevd(LAPACK_ROW_MAJOR, 'V', D.krylov_dim, eigenvalues, D.beta, eigenvectors, D.krylov_dim);
}

std::ostream &operator<<(std::ostream &os, eigenDecomp &E)
{
        os << "Eigenvalues:\n";
        for (auto i = 0u; i < E.k; i++)
        {
                os << E.eigenvalues[i] << " \t";
        }
        os << "\n\n";
        os << "Eigenvectors:\n";
        for (auto i = 0u; i < E.k * E.k; i++)
        {
                os << E.eigenvectors[i] << " \t";
                if (i % E.k == E.k - 1)
                        os << '\n';
        }
        return os;
};
