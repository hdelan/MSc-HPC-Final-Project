#include "eigen.h"

void eigenDecomp::decompose(lanczosDecomp &D)
{
        LAPACKE_dstevd(LAPACK_ROW_MAJOR, 'V', D.krylov_dim, &eigenvalues[0], &D.beta[0], &eigenvectors[0], D.krylov_dim);
}

std::ostream &operator<<(std::ostream &os, eigenDecomp &E)
{
        os << "Eigenvalues:\n";
        for (auto it = E.eigenvalues.begin(); it != E.eigenvalues.end(); it++)
        {
                os << *it << " \t";
        }
        os << "\n\n";
        auto n{E.eigenvalues.size()};
        os << "Eigenvectors:\n";
        for (auto i = 0u; i < n; ++i)
        {
                for (auto it = E.eigenvectors.begin() + i * n; it != E.eigenvectors.begin() + (i + 1) * n; it++)
                {
                        os << *it << " \t";
                }
                os << '\n';
        }
        return os;
};
