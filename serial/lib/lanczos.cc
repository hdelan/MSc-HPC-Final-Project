#include "lanczos.h"
#include "adjMatrix.h"
#include "SPMV.h"

#include <iomanip>

#include "lapacke.h"

void lanczosDecomp::decompose()
{
  long unsigned n{A.get_n()}, i{0u};
  double *v{new double[n]};
  double *Q_raw(new double[2 * n]);
  double *Q_s[2]{Q_raw, &Q_raw[n]}; // Tmp contiguous columns to use before storing
  // in row-major matrix
  double x_norm = norm(x);

  for (auto k = 0u; k < n; k++)
    Q_s[i][k] = x[k] / x_norm;

  for (auto j = 0u; j < krylov_dim; j++)
  {

    // v = A*Q(:,j)
    spMV(A, Q_s[i], v);

    // alpha = v*Q(:,j)
    alpha[j] = inner_prod(v, Q_s[i], n);

    // v = v - alpha*Q(:,j)
    for (auto k = 0u; k < n; k++)
      v[k] -= alpha[j] * Q_s[i][k];

    if (j > 0)
    {
      // v = v - beta*Q(:,j-1)
      for (auto k = 0u; k < n; k++)
        v[k] -= beta[j - 1] * Q_s[1 - i][k];
    }

    if (j < krylov_dim - 1)
    {
      beta[j] = norm(v);
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

void lanczosDecomp::decompose_with_arnoldi()
{
  long unsigned n{A.get_n()}, i{0u};
  double *v{new double[n]};
  double *Q_raw(new double[2 * n]);
  double *Q_s[2]{Q_raw, &Q_raw[n]}; // Tmp contiguous columns to use before storing
  
  double *Q_col_maj {new double[krylov_dim*n]};
  double *Q_ptr[krylov_dim];
  for (auto i=0u;i<krylov_dim;i++) Q_ptr[i] = &Q_col_maj[i*n];
  
  double x_norm = norm(x);

  auto reorthog_every_k {2};

  std::cout << "\n\nUsing an Arnoldi Process every " << reorthog_every_k << " iterations of Lanczos to reorthogonalize the q_i\n\n";
  
  for (auto k = 0u; k < n; k++)
    Q_s[i][k] = x[k] / x_norm;

  for (auto j = 0u; j < krylov_dim; j++)
  {

    // v = A*Q(:,j)
    spMV(A, Q_s[i], v);

    // ARNOLDI with previous vectors
    if (j % reorthog_every_k == 0 && j > 2){
      for (auto k=0u;k<j-1;k++) {
        auto dot = inner_prod(v, Q_ptr[k],n);
        for (auto i=0u;i<n;i++)
          v[i] -= dot*Q_ptr[k][i];
      }
    }

    // alpha = v*Q(:,j)
    alpha[j] = inner_prod(v, Q_s[i], n);

    // v = v - alpha*Q(:,j)
    for (auto k = 0u; k < n; k++)
      v[k] -= alpha[j] * Q_s[i][k];

    if (j > 0)
    {
      // v = v - beta*Q(:,j-1)
      for (auto k = 0u; k < n; k++)
        v[k] -= beta[j - 1] * Q_s[1 - i][k];
    }

    if (j < krylov_dim - 1)
    {
      beta[j] = norm(v);
      for (auto k = 0u; k < n; k++)
        Q_s[1 - i][k] = v[k] / beta[j];
    }

    // Copying the Q_s column into the Q matrix (implemented as a 1d row maj vector)
    for (auto k = 0u; k < n; k++) {
      Q[j + k * krylov_dim] = Q_s[i][k];
      Q_ptr[j][k] = Q_s[i][k];
    }

    i = 1 - i;
  }
  /*
  std::cout << "\nAlpha\n";
  for (auto k=0u;k<krylov_dim;k++) std::cout << alpha[k] << " ";
  std::cout << "\nBeta\n";
  for (auto k=0u;k<krylov_dim-1;k++) std::cout << beta[k] << " ";
  std::cout << "\nQ\n";
  for (auto k=(n-1)*krylov_dim;k<n*krylov_dim;k++) std::cout << Q[k] << " ";
  */
  delete[] v;
  delete[] Q_raw;
}

std::ostream &operator<<(std::ostream &os, const lanczosDecomp &D)
{
  auto n {D.A.get_n()};
  os << "\nAlpha: \n";
  for (auto i = 0u; i < n; i++)
    os << D.alpha[i] << " ";

  os << "\n\nBeta: \n";
  for (auto i = 0u; i < n - 1; i++)
    os << D.beta[i] << " ";

  os << "\n\nQ:\n";
  for (auto i = 0u; i < n - 1; i++)
  {
    os << D.Q[i] << " ";
    if (i % D.krylov_dim == D.krylov_dim - 1)
      os << '\n';
  }

  os << '\n';
  return os;
}

double lanczosDecomp::norm(const double *v) const
{
  double norm{0.0};
  for (auto i = 0u; i < A.n; i++)
    norm += v[i] * v[i];
  return std::sqrt(norm);
}

double lanczosDecomp::inner_prod(const double *const v, const double *const w, const long unsigned N) const
{
  double ans{0.0};
  for (auto i = 0u; i < N; i++)
  {
    ans += v[i] * w[i];
  }
  return ans;
}

void lanczosDecomp::get_ans() const
{
  std::cout << "Answer vector:\n";

  for (auto i = 0u; i < A.n; i++)
    std::cout << std::setprecision(20) << ans[i] << '\n';
}

void lanczosDecomp::check_ans(const double *analytic_ans) const
{
  std::vector<double> diff(A.n);
  for (auto i = 0u; i < A.n; i++)
  {
    diff[i] = std::abs(ans[i] - analytic_ans[i]);
  }
  auto max_it {diff.begin()};
  max_it = std::max_element(diff.begin(), diff.end());
  auto max_idx = std::distance(diff.begin(), max_it);
  std::cout << "\nMax difference of " << *max_it
    << " found at index\n\tlanczos[" << max_idx << "] \t\t\t= " << ans[max_idx]
    << "\n\tanalytic_ans[" << max_idx << "] \t\t= " << analytic_ans[max_idx] << '\n';

  std::cout << "\nTotal norm of differences\t= " << norm(&diff[0]) << '\n';
  std::cout << "Relative norm of differences\t= " << norm(&diff[0])/norm(analytic_ans) << '\n';
}


void lanczosDecomp::reorthog() {
  double * tau {new double [krylov_dim]};
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.n, krylov_dim, Q, krylov_dim, tau);
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, A.n, krylov_dim, krylov_dim, Q, krylov_dim, tau);
  delete[] tau;
}
