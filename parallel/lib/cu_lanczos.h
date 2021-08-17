#ifndef LANCZOS_H_CU1238249102
#define LANCZOS_H_CU1238249102

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "adjMatrix.h"

#include "cu_linalg.h"
#include "cu_SPMV.h"
#include "SPMV.h"

template <typename T>
class lanczosDecomp
{
  private:
    adjMatrix &A;

    long unsigned krylov_dim;

    T *alpha; // The diagonal of tridiag matrix
    T *beta;  // The subdiagonal of tridiag matrix
    T *Q;     // The orthonormal basis for K_n(A,v)
    T *x;     // The starting vector
    T *ans;   // The action of matrix function on x

    T x_norm;

    void decompose();
    void cu_decompose();

  public:
    lanczosDecomp() = delete;
    lanczosDecomp(adjMatrix &adj, const long unsigned krylov, T *starting_vec, bool cuda) : A{adj},
      krylov_dim{krylov},
      alpha(new T[krylov]),
      beta(new T[krylov - 1]),
      Q(new T[krylov * A.get_n()]),
      x(new T[A.get_n()]),
      ans(new T[A.get_n()]),
      x_norm{norm(starting_vec)}
    {
      for (auto i = 0u; i < A.n; i++)
        x[i] = starting_vec[i];
      if (cuda) { cu_decompose(); }
      else { decompose(); }
    };
    lanczosDecomp(lanczosDecomp &) = delete;
    lanczosDecomp &operator=(lanczosDecomp &) = delete;
    ~lanczosDecomp()
    {
      if (alpha!=nullptr) { delete[] alpha; alpha=nullptr;}
      if (beta!=nullptr) { delete[] beta; beta=nullptr;}
      if (Q!=nullptr) { delete[] Q; Q=nullptr;}
      if (x!=nullptr) { delete[] x; x=nullptr;}
      if (ans!=nullptr) { delete[] ans; ans=nullptr;}
    };

    void get_ans() const;
    long unsigned get_n() const { return A.get_n(); };
    long unsigned get_krylov() const { return krylov_dim; };

    friend class eigenDecomp;
    template <typename U>
    friend void multOut(lanczosDecomp<U> &, eigenDecomp &, adjMatrix &);
    template <typename U>
    friend void cu_multOut(lanczosDecomp<U> &, eigenDecomp &, adjMatrix &);
    

    void check_ans(const T *) const;
    void check_ans(lanczosDecomp<T> &) const;
    if (std::is_same<T,double>::value) friend void check_ans(lanczosDecomp<float> &) const;
    if (std::is_same<T,float>::value) friend void check_ans(lanczosDecomp<double> &) const;
    //void check_ans(lanczosDecomp &) const;

    T inner_prod(const T *const, const T *const, const long unsigned) const;

    void reorthog();

    T norm(const T *v) const
    {
      T norm{0.0};
      for (auto i = 0u; i < A.n; i++)
        norm += v[i] * v[i];
      return std::sqrt(norm);
    }
};
#endif
