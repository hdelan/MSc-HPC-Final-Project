#ifndef LANCZOS_H_1238249102
#define LANCZOS_H_1238249102

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

#include "adjMatrix.h"

class lanczosDecomp
{
public:
lanczosDecomp() = delete;
lanczosDecomp(adjMatrix &A, const unsigned krylov, double * starting_vec) : krylov_dim{krylov},
                                                                                n {A.get_n()},
                                                                                alpha(new double[krylov]),
                                                                                beta(new double[krylov - 1]),
                                                                                Q(new double[krylov*A.get_n()]),
                                                                                x(new double[A.get_n()]),
                                                                                ans(new double[A.get_n()]),
                                                                                x_norm {norm(starting_vec)}
{
        for (auto i=0u;i<n;i++) x[i] = starting_vec[i];
        decompose(A);
};
lanczosDecomp(lanczosDecomp &) = delete;
lanczosDecomp &operator=(lanczosDecomp &) = delete;
~lanczosDecomp() {
        delete[] alpha; delete[] beta; delete[] Q; delete[] x; delete[] ans;
};

        void get_ans() const;

        friend class eigenDecomp;
        friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);
        friend std::ostream &operator<<(std::ostream &os, const lanczosDecomp &D);

        double norm(const double *) const;
        double inner_prod(const double * const, const double * const, const unsigned) const;

private:
        unsigned krylov_dim;                // Dimension of the krylov subspace
        unsigned n;                         // Dimension of n
        
        double * alpha;          // The diagonal of tridiag matrix
        double * beta;           // The subdiagonal of tridiag matrix
        double * Q;              // The orthonormal basis for K_n(A,v)
        double * x;              // The starting vector
        double * ans;            // The action of matrix function on x

        double x_norm;

        void decompose(const adjMatrix &);
};
#endif
