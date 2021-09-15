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

template <typename T, typename U>
T norm(const T *v, U n)
{
  T norm{0.0};
  for (auto i = 0u; i < n; i++)
    norm += v[i] * v[i];
  return std::sqrt(norm);
}

template <typename U, typename V>
void check_ans(lanczosDecomp<U> &, lanczosDecomp<V> &);

template <typename T>
class lanczosDecomp
{
  private:
    adjMatrix &A;

    unsigned krylov_dim;

    T *alpha; // The diagonal of tridiag matrix
    T *beta;  // The subdiagonal of tridiag matrix
    T *Q;     // The orthonormal basis for K_n(A,v)
    T *x;     // The starting vector
    T *ans;   // The action of matrix function on x
    
    T *Q_d=nullptr;   // Q stored on card

    T x_norm;

    void decompose();
    void cu_decompose();

  public:
    lanczosDecomp() = delete;
    lanczosDecomp(adjMatrix &adj, const unsigned krylov, T *starting_vec, bool cuda) : A{adj},
      krylov_dim{krylov},
      alpha(new T[krylov]),
      beta(new T[krylov - 1]),
      Q(new T[krylov * A.get_n()]),
      x(new T[A.get_n()]),
      ans(new T[A.get_n()]),
      x_norm{norm(starting_vec, A.get_n())}
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
      if (Q_d!=nullptr) { cudaFree(ans); ans=nullptr;}
    };

    void free_mem() {
      std::cout << 1;
      if (alpha!=nullptr) { delete[] alpha; alpha=nullptr;}
      if (beta!=nullptr) { delete[] beta; beta=nullptr;}
      if (Q!=nullptr) { delete[] Q; Q=nullptr;}
      if (x!=nullptr) { delete[] x; x=nullptr;}
      if (Q_d!=nullptr) { cudaFree(ans); ans=nullptr;}
      std::cout << 2;
    };

    void get_ans() const;
    unsigned get_n() const { return A.get_n(); };
    unsigned get_krylov() const { return krylov_dim; };

    friend class eigenDecomp<T>;
    template <typename U>
    friend void multOut(lanczosDecomp<U> &, eigenDecomp<U> &, adjMatrix &, bool);
    template <typename U, typename V>
      friend void check_ans(lanczosDecomp<U> &, lanczosDecomp<V> &);
    template <typename U, typename V>
      friend U norm(const U *const, const V);
    template <typename U, typename V>
      friend U inner_prod(const U *const, const U *const, const V);
      template <typename U>
      friend void write_ans(std::string filename, lanczosDecomp<U> &);

    void check_ans(const T *) const;

    void reorthog();

};
#endif
