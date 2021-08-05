#ifndef LANCZOS_H_CU1238249102
#define LANCZOS_H_CU1238249102

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

#include "adjMatrix.h"

#include "cu_linalg.h"
#include "cu_SPMV.h"

class lanczosDecomp
{
private:
    adjMatrix &A;

    long unsigned krylov_dim;

    double *alpha; // The diagonal of tridiag matrix
    double *beta;  // The subdiagonal of tridiag matrix
    double *Q;     // The orthonormal basis for K_n(A,v)
    double *x;     // The starting vector
    double *ans;   // The action of matrix function on x

    double x_norm;

    void decompose();
    void cu_decompose();

public:
    lanczosDecomp() = delete;
    lanczosDecomp(adjMatrix &adj, const long unsigned krylov, double *starting_vec) : A{adj},
                                                                                      krylov_dim{krylov},
                                                                                      alpha(new double[krylov]),
                                                                                      beta(new double[krylov - 1]),
                                                                                      Q(new double[krylov * A.get_n()]),
                                                                                      x(new double[A.get_n()]),
                                                                                      ans(new double[A.get_n()]),
                                                                                      x_norm{norm(starting_vec)}
    {
        for (auto i = 0u; i < A.n; i++)
            x[i] = starting_vec[i];
        cu_decompose();
    };
    lanczosDecomp(lanczosDecomp &) = delete;
    lanczosDecomp &operator=(lanczosDecomp &) = delete;
    ~lanczosDecomp()
    {
        delete[] alpha;
        delete[] beta;
        delete[] Q;
        delete[] x;
        delete[] ans;
    };

    void get_ans() const;
    long unsigned get_n() const { return A.get_n(); };
    long unsigned get_krylov() const { return krylov_dim; };

    friend class eigenDecomp;
    friend void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);
    friend std::ostream &operator<<(std::ostream &os, const lanczosDecomp &D);

    void check_ans(const double *) const;

    double inner_prod(const double *const, const double *const, const long unsigned) const;

    void reorthog();

    double norm(const double *v) const
    {
        double norm{0.0};
        for (auto i = 0u; i < A.n; i++)
            norm += v[i] * v[i];
        return std::sqrt(norm);
    }
};
#endif
