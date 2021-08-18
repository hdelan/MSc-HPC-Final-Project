#include "cu_lanczos.h"
#include "blocks.h"

  template <typename T>
void lanczosDecomp<T>::cu_decompose()
{
  unsigned long n{A.get_n()};
  unsigned long *IA_d, *JA_d;
  T *v_d, *alpha_d, *beta_d, *tmp_d;

  unsigned block_size{BLOCKSIZE}, num_blocks{static_cast<unsigned>(n) / block_size + 1};
  dim3 blocks{num_blocks}, threads{block_size};

  std::cout << num_blocks << " block(s) ";
  std::cout << "Running with "<< BLOCKSIZE << " threads per block\n";

  T *x_normed{new T[n]};
  T x_norm = norm(x, n);

  /*
     std::cout << "x\n";
     for (auto k = 0u; k < n; k++)
     std::cout << x[k] << '\n';
   */

  for (auto k = 0u; k < n; k++)
    x_normed[k] = x[k] / x_norm;
  
  cudaMalloc((void **)&Q_d, sizeof(T) * n * krylov_dim);

  cudaMalloc((void **)&IA_d, sizeof(long unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(long unsigned) * 2 * A.edge_count);
  cudaMalloc((void **)&v_d, sizeof(T) * n);
  cudaMalloc((void **)&alpha_d, sizeof(T) * krylov_dim);
  cudaMalloc((void **)&beta_d, sizeof(T) * (krylov_dim - 1));
  cudaMalloc((void **)&tmp_d, sizeof(T) * (num_blocks));

  T *Q_d_ptr[krylov_dim];
  for (auto i=0u;i<krylov_dim;i++) Q_d_ptr[i] = &Q_d[n*i];

  std::vector<T> tmp(n);
  
  cudaStream_t stream[3];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  cudaStreamCreate(&stream[2]);

  cudaMemcpyAsync(Q_d_ptr[0], x_normed, sizeof(T) * n, cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(IA_d, A.row_offset, sizeof(long unsigned) * (n + 1), cudaMemcpyHostToDevice, stream[1]);
  cudaMemcpyAsync(JA_d, A.col_idx, sizeof(long unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice, stream[2]);

  cudaStreamSynchronize(stream[0]);
  cudaStreamSynchronize(stream[1]);
  cudaStreamSynchronize(stream[2]);

  cudaStreamDestroy(stream[2]);

  for (auto k = 0u; k < krylov_dim; k++)
  {
    // v = A*Q(:,j)
    cu_spMV1<T, unsigned long><<<blocks, threads>>>(IA_d, JA_d, n, Q_d_ptr[k], v_d); 

    // alpha = v*Q(:,j)
    if (num_blocks==1) { 
      cu_dot_prod<T, BLOCKSIZE><<<1, threads, BLOCKSIZE*sizeof(T), stream[0]>>>(v_d, Q_d_ptr[k], n, &alpha_d[k]);
    } else {
      cu_dot_prod<T, BLOCKSIZE><<<num_blocks/2, threads, BLOCKSIZE*sizeof(T), stream[0]>>>(v_d, Q_d_ptr[k], n, tmp_d);
      cu_reduce<T, BLOCKSIZE><<<1, threads, BLOCKSIZE*sizeof(T), stream[0]>>>(tmp_d, num_blocks, &alpha_d[k]);
    }

    // v = v - alpha*Q(:,j)
    cu_dpax<T><<<blocks, threads,0,stream[0]>>>(v_d, &alpha_d[k], Q_d_ptr[k], n);

    if (k > 0)
    {
      // v = v - beta*Q(:,j-1)
      cu_dpax<T><<<blocks, threads,0,stream[0]>>>(v_d, &beta_d[k - 1], Q_d_ptr[k-1], n);
    }

    if (k < krylov_dim - 1)
    {
      // beta[j] = norm(v)
      if (num_blocks==1) {
        cu_norm_sq_sqrt<T, BLOCKSIZE><<<1, threads, BLOCKSIZE*sizeof(T), stream[0]>>>(v_d, n, &beta_d[k]);
      } else {
        cu_norm_sq<T, BLOCKSIZE><<<num_blocks/2, threads, BLOCKSIZE*sizeof(T), stream[0]>>>(v_d, n, tmp_d);
        cu_reduce_sqrt<T,BLOCKSIZE><<<1, threads, BLOCKSIZE*sizeof(T), stream[0]>>>(tmp_d, num_blocks, &beta_d[k]);
      }
      // Q(:,j) = v/beta
      cu_dvexda<T><<<blocks, threads,0,stream[0]>>>(Q_d_ptr[k+1], &beta_d[k], v_d, n);
    }
  }
  cudaMemcpyAsync(alpha, alpha_d, sizeof(T) * krylov_dim, cudaMemcpyDeviceToHost, stream[0]);
  cudaMemcpyAsync(beta, beta_d, sizeof(T) * (krylov_dim - 1), cudaMemcpyDeviceToHost, stream[1]);

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
  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(v_d);
  cudaFree(alpha_d);
  cudaFree(beta_d);
  cudaFree(tmp_d);
}

template void lanczosDecomp<float>::cu_decompose();
template void lanczosDecomp<double>::cu_decompose();

