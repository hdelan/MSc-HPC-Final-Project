#include "cu_lanczos.h"
#include "lanczos.h"

/* --------------------------------------------------------------------------*/
/**
 * \brief:       Decompose function
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
void lanczosDecomp<T>::decompose()
{
  unsigned n{A.get_n()}, i{0u};
  T *v{new T[n]};
  T *Q_raw(new T[2 * n]);
  T *Q_s[2]{Q_raw, &Q_raw[n]}; // Tmp contiguous columns to use before storing

  T x_norm = norm(x, n);

  for (auto k = 0u; k < n; k++)
    Q_s[i][k] = x[k] / x_norm;

  // LANCZOS ALGORITHM
  for (auto j = 0u; j < krylov_dim; j++)
  {
    spMV(A, Q_s[i], v);                   // v = A*Q(:,j)

    alpha[j] = inner_prod(v, Q_s[i], n);  // alpha = v*Q(:,j)

    for (auto k = 0u; k < n; k++)         // v = v - alpha*Q(:,j)
      v[k] -= alpha[j] * Q_s[i][k];

    if (j > 0)
    {
      for (auto k = 0u; k < n; k++) // v = v - beta*Q(:,j-1)
        v[k] -= beta[j - 1] * Q_s[1 - i][k];
    }

    if (j < krylov_dim - 1)
    {
      beta[j] = norm(v, n);
      for (auto k = 0u; k < n; k++)
        Q_s[1 - i][k] = v[k] / beta[j];
    }

    // Copying the Q_s column into the Q matrix (implemented as a 1d row maj vector)
    for (auto k = 0u; k < n; k++)
      Q[j + k * krylov_dim] = Q_s[i][k];

    i = 1 - i;
  }
 
  delete[] v;
  delete[] Q_raw;
}

/* --------------------------------------------------------------------------*/
/**
 * \brief:       Check the answer against an analytic answer
 *
 * \param:       analytic_ans
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
void lanczosDecomp<T>::check_ans(const T *analytic_ans) const
{
  std::vector<double> diff(A.n);
  for (auto i = 0u; i < A.n; i++)
  {
    diff[i] = std::abs(ans[i] - analytic_ans[i]);
  }
  auto max_it = std::max_element(diff.begin(), diff.end());
  auto max_idx = std::distance(diff.begin(), max_it);
  std::cout << "\nMax difference of " << *max_it
    << " found at index\n\tlanczos[" << max_idx << "] \t\t\t= " << ans[max_idx]
    << "\n\tanalytic_ans[" << max_idx << "] \t\t= " << analytic_ans[max_idx] << '\n';

  std::cout << "\nTotal norm of differences\t= " << std::setprecision(20) << norm(&diff[0],A.n) << '\n';
  std::cout << "Relative norm of differences\t= " << std::setprecision(20)<< norm(&diff[0], A.n)/norm(analytic_ans, A.n) << '\n';
}

  template <typename T, typename U>
T inner_prod(const T *const v, const T *const w, const U N) 
{
  T ans{0.0};
  for (auto i = 0u; i < N; i++)
  {
    ans += v[i] * w[i];
  }
  return ans;
}

template void lanczosDecomp<double>::decompose();
template void lanczosDecomp<float>::decompose();

