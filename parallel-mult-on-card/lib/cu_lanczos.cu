#include "cu_lanczos.h"

#define SPMV_BLOCKSIZE 1024
#define DOT_BLOCKSIZE 128
#define NORM_BLOCKSIZE 128
#define RED_BLOCKSIZE 256
#define SAX_BLOCKSIZE 256


  template <typename T>
void lanczosDecomp<T>::cu_decompose()
{
  unsigned n{A.get_n()};
  unsigned *IA_d, *JA_d, *blockrows_d;
  T *v_d, *alpha_d, *beta_d, *tmp_d;

  unsigned dot_grid {n/(DOT_BLOCKSIZE*2) + (n%(DOT_BLOCKSIZE*2)==0?0:1)};
  unsigned norm_grid {n/(NORM_BLOCKSIZE*2) + (n%(NORM_BLOCKSIZE*2)==0?0:1)};
  unsigned sax_grid {n/SAX_BLOCKSIZE + (n%SAX_BLOCKSIZE==0?0:1)};

  std::vector<unsigned> blockrows(n);
  unsigned spmv_blocks_needed {0u};

  get_blockrows<T>(A,SPMV_BLOCKSIZE,&blockrows[0],spmv_blocks_needed);

  T *x_normed{new T[n]};
  T x_norm = norm(x, n);

  for (auto k = 0u; k < n; k++)
    x_normed[k] = x[k] / x_norm;
  
  cudaMalloc((void **)&Q_d, sizeof(T) * n * krylov_dim);

  cudaMalloc((void **)&IA_d, sizeof(unsigned) * (n + 1));
  cudaMalloc((void **)&JA_d, sizeof(unsigned) * 2 * A.edge_count);
  cudaMalloc((void **)&v_d, sizeof(T) * n);
  cudaMalloc((void **)&alpha_d, sizeof(T) * krylov_dim);
  cudaMalloc((void **)&beta_d, sizeof(T) * (krylov_dim - 1));
  cudaMalloc((void **)&tmp_d, sizeof(T) * (norm_grid));
  cudaMalloc((void **)&blockrows_d, sizeof(unsigned) * (spmv_blocks_needed+1));

  auto global_memory_used { (n+1+2*A.edge_count)*sizeof(unsigned)+(n+2*krylov_dim-1+norm_grid)*sizeof(T)};

  std::cout << "\nUsing " 
        << global_memory_used << " bytes of CUDA global memory (" 
        << 100* global_memory_used/(double)11996954624 
        << "% of capacity 11996954624 bytes)\n";

  T *Q_d_ptr[krylov_dim];
  for (auto i=0u;i<krylov_dim;i++) Q_d_ptr[i] = &Q_d[n*i];

  std::vector<T> tmp(n);
  
  cudaStream_t stream[3];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  cudaStreamCreate(&stream[2]);

  cudaMemcpyAsync(Q_d_ptr[0], x_normed, sizeof(T) * n, cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(IA_d, A.row_offset, sizeof(unsigned) * (n + 1), cudaMemcpyHostToDevice, stream[1]);
  cudaMemcpyAsync(JA_d, A.col_idx, sizeof(unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice, stream[2]);
  cudaMemcpyAsync(blockrows_d, &blockrows[0], sizeof(unsigned)*(spmv_blocks_needed+1), cudaMemcpyHostToDevice, stream[0]);

  cudaStreamSynchronize(stream[0]);
  cudaStreamSynchronize(stream[1]);
  cudaStreamSynchronize(stream[2]);

  cudaStreamDestroy(stream[2]);

  for (auto k = 0u; k < krylov_dim; k++)
  {
    // v = A*Q(:,j)
    cu_spMV2<T, unsigned, SPMV_BLOCKSIZE><<<spmv_blocks_needed, SPMV_BLOCKSIZE>>>(IA_d, JA_d, blockrows_d, n, Q_d_ptr[k], v_d);
    //cu_spMV1<T><<<n/256+1,256>>>(IA_d, JA_d, n, Q_d_ptr[k], v_d);

    // alpha = v*Q(:,j)
    cu_dot_prod<T, DOT_BLOCKSIZE><<<dot_grid, DOT_BLOCKSIZE,0,stream[0]>>>(v_d, Q_d_ptr[k], n, tmp_d);
    cu_reduce<T, RED_BLOCKSIZE><<<1, RED_BLOCKSIZE, 0, stream[0]>>>(tmp_d, dot_grid, &alpha_d[k]);

    // v = v - alpha*Q(:,j)
    cu_dpax<T><<<sax_grid, SAX_BLOCKSIZE,0,stream[0]>>>(v_d, &alpha_d[k], Q_d_ptr[k], n);

    if (k > 0)
    {
      // v = v - beta*Q(:,j-1)
      cu_dpax<T><<<sax_grid, SAX_BLOCKSIZE,0,stream[0]>>>(v_d, &beta_d[k - 1], Q_d_ptr[k-1], n);
    }

    if (k < krylov_dim - 1)
    {
      // beta[j] = norm(v)
      cu_norm_sq<T, NORM_BLOCKSIZE><<<norm_grid, NORM_BLOCKSIZE, 0, stream[0]>>>(v_d, n, tmp_d);
      cu_reduce_sqrt<T,RED_BLOCKSIZE><<<1, RED_BLOCKSIZE, 0, stream[0]>>>(tmp_d, norm_grid, &beta_d[k]);
      
      // Q(:,j) = v/beta
      cu_dvexda<T><<<sax_grid,SAX_BLOCKSIZE,0,stream[0]>>>(Q_d_ptr[k+1], &beta_d[k], v_d, n);
    }
  }
  cudaMemcpyAsync(alpha, alpha_d, sizeof(T) * krylov_dim, cudaMemcpyDeviceToHost, stream[0]);
  cudaMemcpyAsync(beta, beta_d, sizeof(T) * (krylov_dim - 1), cudaMemcpyDeviceToHost, stream[1]);

  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  
  cudaFree(IA_d);
  cudaFree(JA_d);
  cudaFree(v_d);
  cudaFree(alpha_d);
  cudaFree(beta_d);
  cudaFree(tmp_d);
  cudaFree(blockrows_d);
}

template void lanczosDecomp<float>::cu_decompose();
template void lanczosDecomp<double>::cu_decompose();

