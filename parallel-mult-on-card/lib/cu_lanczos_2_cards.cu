#include "cu_lanczos.h"

#define BLOCK_SIZE 32

template <typename T, typename U>
T inner_prod(const T *const v, const T *const w, const U N);

  template <typename T>
void lanczosDecomp<T>::cu_decompose()
{
  unsigned long n{A.get_n()};
  unsigned long *IA_d, *JA_d;
  T *Q_raw_d, *v_d, *alpha_d, *beta_d, *tmp_d;

  unsigned block_size{32}, num_blocks{static_cast<unsigned>(n) / block_size + 1};
  dim3 blocks{num_blocks}, threads{block_size};

  std::cout << num_blocks << " block(s) ";
  std::cout << "Running with "<< BLOCK_SIZE << " threads per block\n";

  T *x_normed{new T[n]};
  T x_norm = norm(x, n);

  /*
     std::cout << "x\n";
     for (auto k = 0u; k < n; k++)
     std::cout << x[k] << '\n';
   */

  for (auto k = 0u; k < n; k++)
    x_normed[k] = x[k] / x_norm;

  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * 2 * A.edge_count);
  cudaMalloc((void **)&v_d, sizeof(T) * n);
  cudaMalloc((void **)&Q_raw_d, sizeof(T) * n * 2);
  cudaMalloc((void **)&alpha_d, sizeof(T) * krylov_dim);
  cudaMalloc((void **)&beta_d, sizeof(T) * (krylov_dim - 1));
  cudaMalloc((void **)&tmp_d, sizeof(T) * (num_blocks));

  T *Q_d_ptr[2] = {&Q_raw_d[0], &Q_raw_d[n]};

  cudaMemcpy(Q_d_ptr[0], x_normed, sizeof(T) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);

  std::vector<T> tmp(n);

  int i{0};

  for (auto k = 0u; k < krylov_dim; k++)
  {
    std::vector<T> tmp_vec(n);
    std::vector<unsigned> tmp2(n);

    // v = A*Q(:,j)
    cu_spMV1<T, unsigned long><<<blocks, threads>>>(IA_d, JA_d, n, Q_d_ptr[i], v_d); 

    // alpha = v*Q(:,j)
    if (num_blocks==1) { 
      cu_dot_prod<T, BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(T)>>>(v_d, Q_d_ptr[i], n, &alpha_d[k]);
    } else {
      cu_dot_prod<T, BLOCK_SIZE><<<num_blocks/2, threads, BLOCK_SIZE*sizeof(T)>>>(v_d, Q_d_ptr[i], n, tmp_d);
      cu_reduce<T, BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(T)>>>(tmp_d, num_blocks, &alpha_d[k]);
    }

    // v = v - alpha*Q(:,j)
    cu_dpax<T><<<blocks, threads>>>(v_d, &alpha_d[k], Q_d_ptr[i], n);

    if (k > 0)
    {
      // v = v - beta*Q(:,j-1)
      cu_dpax<T><<<blocks, threads>>>(v_d, &beta_d[k - 1], Q_d_ptr[1 - i], n);
    }

    if (k < krylov_dim - 1)
    {
      // beta[j] = norm(v)
      if (num_blocks==1) {
        cu_norm_sq_sqrt<T, BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(T)>>>(v_d, n, &beta_d[k]);
      } else {
        cu_norm_sq<T, BLOCK_SIZE><<<num_blocks/2, threads, BLOCK_SIZE*sizeof(T)>>>(v_d, n, tmp_d);
        cu_reduce_sqrt<T,BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(T)>>>(tmp_d, num_blocks, &beta_d[k]);
      }
      // Q(:,j) = v/beta
      cu_dvexda<T><<<blocks, threads>>>(Q_d_ptr[1 - i], &beta_d[k], v_d, n);
    }

    cudaMemcpy(&tmp[0], Q_d_ptr[i], sizeof(T) * n, cudaMemcpyDeviceToHost);

    // Loading Q into memory
    for (auto j = 0u; j < n; j++)
      Q[k + j * krylov_dim] = tmp[j];

    i = 1 - i;
  }
  cudaMemcpy(alpha, alpha_d, sizeof(T) * krylov_dim, cudaMemcpyDeviceToHost);
  cudaMemcpy(beta, beta_d, sizeof(T) * (krylov_dim - 1), cudaMemcpyDeviceToHost);

  /*
     std::cout << "cu_Q:\n";
     for (int i=0;i<krylov_dim;i++) {
     for (int j=0;j<krylov_dim;j++)
     std::cout << Q[i*krylov_dim+j] << " ";
     std::cout << '\n';
     }

     std::cout << "\ncu_Alpha:\n";
     for (int i=0;i<krylov_dim;i++) std::cout << alpha[i] << " ";

     std::cout << "\n\ncu_Beta:\n";
     for (int i=0;i<krylov_dim-1;i++) std::cout << beta[i] << " ";
     std::cout << "\n\n";

   */
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(v_d);
  cudaFree(Q_raw_d);
  cudaFree(alpha_d);
  cudaFree(beta_d);
  cudaFree(tmp_d);
}

template <typename T>
void lanczosDecomp<T>::get_ans() const
{
  std::cout << "Answer vector:\n";

  for (auto i = 0u; i < A.n; i++)
    std::cout << std::setprecision(20) << ans[i] << '\n';
}

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
  /*
     std::cout << "\nAlpha:\n";
     for (auto j = 0u; j < krylov_dim; j++)
     std::cout << alpha[j] << " ";
     std::cout << "\n\nBeta:\n";
     for (auto j = 0u; j < krylov_dim - 1; j++)
     std::cout << beta[j] << " ";
     std::cout << '\n';
     std::cout << '\n';
   */
  /* PRINT OUT Q 
     std::cout << "\nQ\n";
     for (auto j = 0u; j < krylov_dim; j++)
     {
     for (auto k = 0u; k < krylov_dim; k++)
     std::cout << Q[k + j * krylov_dim] << " ";
     std::cout << '\n';
     }*/
  delete[] v;
  delete[] Q_raw;
}

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
/*
// Doesn't work! (doesn't give better accuracy)
void lanczosDecomp::reorthog() {
T * tau {new T [krylov_dim]};
LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.n, krylov_dim, Q, krylov_dim, tau);
LAPACKE_dorgqr(LAPACK_ROW_MAJOR, A.n, krylov_dim, krylov_dim, Q, krylov_dim, tau);
delete[] tau;
}
 */
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

template class lanczosDecomp<float>;
template class lanczosDecomp<double>;

