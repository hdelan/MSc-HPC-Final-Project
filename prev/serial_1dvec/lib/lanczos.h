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
        lanczosDecomp(adjMatrix &A, unsigned krylov, std::vector<double> starting_vec) : alpha(krylov),
                                                                                         beta(krylov - 1),
                                                                                         Q(krylov*A.get_n()),
                                                                                         x{starting_vec},
                                                                                         ans(starting_vec.size()),
                                                                                         krylov_dim{krylov},
                                                                                         x_norm{norm(starting_vec)}
        {
                decompose(A);
        };
        lanczosDecomp(lanczosDecomp &) = delete;
        lanczosDecomp &operator=(lanczosDecomp &) = delete;
        ~lanczosDecomp() = default;

        void get_ans() const;

        friend class eigenDecomp;
        friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);
        friend std::ostream &operator<<(std::ostream &os, const lanczosDecomp &D);

        double norm(const std::vector<double> &);

private:
        std::vector<double> alpha;          // The diagonal of tridiag matrix
        std::vector<double> beta;           // The subdiagonal of tridiag matrix
        std::vector<double> Q;              // The orthonormal basis for K_n(A,v)
        std::vector<double> x;              // The starting vector
        std::vector<double> ans;            // The action of matrix function on x

        unsigned krylov_dim;                // Dimension of the krylov subspace

        double x_norm;

        void decompose(const adjMatrix &);
};
#endif
