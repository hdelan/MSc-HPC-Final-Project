#include "cu_lanczos.h"
#include "blocks.h"
__global__ void change_IA_for_device1(unsigned * IA_d, const unsigned n) {
  auto tid {blockIdx.x*blockDim.x+threadIdx.x};
  if (tid < n) {
    auto offset {IA_d[0]};
    IA_d[tid] -= offset;
  }
}

/*
__global__ void print_n(float*a, const unsigned n, const int dev) {
  printf("\n");
  printf("Device %d\t", dev);
  for (auto i=0u;i<n;i++)
    printf(" %E ", a[i]);
  printf("\n");
}
__global__ void print_n(double*a, const unsigned n, const int dev) {
  printf("\n");
  printf("Device %d\t", dev);
  for (auto i=0u;i<n;i++)
    printf(" %E ", a[i]);
  printf("\n");
}
__global__ void print_n(unsigned *a, const unsigned n, const int dev) {
  printf("\n");
  printf("Device %d\t", dev);
  for (auto i=0u;i<n;i++)
    printf(" %u ", a[i]);
  printf("\n");
}
*/
  template <typename T>
void lanczosDecomp<T>::cu_decompose()
{
  unsigned n{A.get_n()};
  unsigned *IA_d0, *JA_d0;
  T *v_d0, *alpha_d0, *beta_d0, *tmp_d0, *Q_d_raw0;
  unsigned *IA_d1, *JA_d1;
  T *v_d1, *x_d1;

  T *x_normed{new T[n]};
  T x_norm = norm(x, n);

  for (auto k = 0u; k < n; k++)
    x_normed[k] = x[k] / x_norm;

  std::cout << "Running with "<< BLOCKSIZE << " threads per block\n";

  cudaStream_t stream[2];
  cudaStream_t memcpy_stream;

  int count;
  cudaGetDeviceCount(&count);
  std::cout << "Launching lanczos algorithm on " << count << " cards.\n";

  auto load_balance {0.5}; // This will determine the split of work between card one and card two
  // Card one will receive rows 0 to rows0-1, and card two will receive rows rows0 to n
  unsigned rows0 {static_cast<unsigned> (load_balance*n)};
  unsigned rows1 {n-rows0};
  unsigned edges0 {A.row_offset[rows0]};
  unsigned edges1 {2*A.edge_count - A.row_offset[rows0]};

  assert(edges0+edges1 == 2*A.edge_count);

  int i {0};

  cudaSetDevice(0);
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&memcpy_stream);

  unsigned num_blocks_total{static_cast<unsigned>(n) / BLOCKSIZE + 1};
  unsigned num_blocks0{static_cast<unsigned>(rows0) / BLOCKSIZE + 1};

  cudaMalloc((void **)&Q_d_raw0, sizeof(T)*n*2);

  cudaMalloc((void **)&IA_d0, sizeof(unsigned) * (rows0 + 1));
  cudaMalloc((void **)&JA_d0, sizeof(unsigned) * edges0);
  cudaMalloc((void **)&v_d0, sizeof(T) * n);
  cudaMalloc((void **)&alpha_d0, sizeof(T) * krylov_dim);
  cudaMalloc((void **)&beta_d0, sizeof(T) * (krylov_dim - 1));
  cudaMalloc((void **)&tmp_d0, sizeof(T) * (num_blocks_total));

  T *Q_d_ptr0[2] {&Q_d_raw0[0], &Q_d_raw0[n]};

  std::vector<T> tmp(n);

  cudaMemcpyAsync(Q_d_ptr0[0], x_normed, sizeof(T) * n, cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(IA_d0, A.row_offset, sizeof(unsigned) * (rows0 + 1), cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(JA_d0, A.col_idx, sizeof(unsigned) * edges0, cudaMemcpyHostToDevice, stream[0]);

  cudaSetDevice(1);
  cudaStreamCreate(&stream[1]);

  unsigned num_blocks1 {static_cast<unsigned>(rows1) / BLOCKSIZE + 1};

  cudaMalloc((void **)&IA_d1, sizeof(unsigned) * (rows1 + 1));
  cudaMalloc((void **)&JA_d1, sizeof(unsigned) * edges1);
  cudaMalloc((void **)&v_d1, sizeof(T) * n);
  cudaMalloc((void **)&x_d1, sizeof(T) * n);

  cudaMemcpyAsync(x_d1, x_normed, sizeof(T) * n, cudaMemcpyHostToDevice, stream[1]);
  cudaMemcpyAsync(IA_d1, &A.row_offset[rows0], sizeof(unsigned) * (rows1 + 1), cudaMemcpyHostToDevice, stream[1]);
  cudaMemcpyAsync(JA_d1, &A.col_idx[edges0], sizeof(unsigned) * edges1, cudaMemcpyHostToDevice, stream[1]);

  change_IA_for_device1<<<num_blocks1, BLOCKSIZE, 0, stream[1]>>>(IA_d1, rows1+1);
  //print_n<<<1,1,0,stream[1]>>>(&IA_d1[rows1], 1);

  for (auto k = 0u; k < krylov_dim; k++)
  {
    cudaSetDevice(1);
    cudaStreamSynchronize(stream[0]);
    // v = A*Q(:,j)
    cudaSetDevice(0);
    cu_spMV1<T, unsigned><<<num_blocks0, BLOCKSIZE, 0, stream[0]>>>(IA_d0, JA_d0, rows0, Q_d_ptr0[i], v_d0); 
    cudaSetDevice(1);
    cu_spMV1<T, unsigned><<<num_blocks1, BLOCKSIZE, 0, stream[1]>>>(IA_d1, JA_d1, rows1, x_d1, v_d1); 
    
    //if (k<10) print_n<<<1,1,0,stream[1]>>>(&v_d1[0], 2);

    cudaSetDevice(0);
    cudaMemcpyPeer(&v_d0[rows0], 0, v_d1, 1, sizeof(T)*rows1);
    
    //if (k<10) print_n<<<1,1,0,stream[0]>>>(&v_d0[0], 5);
    //if (k<10) print_n<<<1,1,0,stream[0]>>>(Q_d_ptr0[0], 5, 0);

    // alpha = v*Q(:,j)
    if (num_blocks_total==1) { 
      cu_dot_prod<T, BLOCKSIZE><<<1, BLOCKSIZE, BLOCKSIZE*sizeof(T), stream[0]>>>(v_d0, Q_d_ptr0[i], n, &alpha_d0[k]);
    } else {
      cu_dot_prod<T, BLOCKSIZE><<<num_blocks_total/2, BLOCKSIZE, BLOCKSIZE*sizeof(T), stream[0]>>>(v_d0, Q_d_ptr0[i], n, tmp_d0);
      cu_reduce<T, BLOCKSIZE><<<1, BLOCKSIZE, BLOCKSIZE*sizeof(T), stream[0]>>>(tmp_d0, num_blocks_total/2, &alpha_d0[k]);
    }

    //if (k < 10)print_n<<<1,1>>>(&alpha_d0[k], 1, 0);

    // v = v - alpha*Q(:,j)
    cu_dpax<T><<<num_blocks_total, BLOCKSIZE,0,stream[0]>>>(v_d0, &alpha_d0[k], Q_d_ptr0[i], n);

    if (k > 0)
    {
      // v = v - beta*Q(:,j-1)
      cu_dpax<T><<<num_blocks_total, BLOCKSIZE,0,stream[0]>>>(v_d0, &beta_d0[k-1], Q_d_ptr0[1-i], n);
    }

    if (k < krylov_dim - 1)
    {
      // beta[j] = norm(v)
      if (num_blocks_total==1) {
        cu_norm_sq_sqrt<T, BLOCKSIZE><<<1, BLOCKSIZE, BLOCKSIZE*sizeof(T), stream[0]>>>(v_d0, n, &beta_d0[k]);
      } else {
        cu_norm_sq<T, BLOCKSIZE><<<num_blocks_total/2, BLOCKSIZE, BLOCKSIZE*sizeof(T), stream[0]>>>(v_d0, n, tmp_d0);
        cu_reduce_sqrt<T,BLOCKSIZE><<<1, BLOCKSIZE, BLOCKSIZE*sizeof(T), stream[0]>>>(tmp_d0, num_blocks_total, &beta_d0[k]);
      }
      //if (k < 10)print_n<<<1,1>>>(&beta_d0[k], 1,0);

      // Q(:,j) = v/beta
      cu_dvexda<T><<<num_blocks_total,BLOCKSIZE,0,stream[0]>>>(Q_d_ptr0[1-i], &beta_d0[k], v_d0, n);
      //print_n<<<1,1,0,stream[0]>>>(Q_d_ptr0[1-i], 5,0);
      cudaSetDevice(1);
      cudaMemcpyPeer(x_d1,1, Q_d_ptr0[1-i], 0, sizeof(T)*n);
      //print_n<<<1,1,0,stream[1]>>>(x_d1, 5,1);
      
    }

    cudaSetDevice(0);
    cudaMemcpy(&tmp[0], Q_d_ptr0[i], sizeof(T)*n, cudaMemcpyDeviceToHost);
    i = 1-i;

    for (auto j=0u;j<n;j++)
      Q[k+j*krylov_dim] = tmp[j];
  }
  cudaMemcpyAsync(alpha, alpha_d0, sizeof(T) * krylov_dim, cudaMemcpyDeviceToHost, stream[0]);
  cudaMemcpyAsync(beta, beta_d0, sizeof(T) * (krylov_dim - 1), cudaMemcpyDeviceToHost, stream[0]);

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
  cudaStreamDestroy(memcpy_stream);

  cudaSetDevice(0);
  cudaFree(IA_d0);
  cudaFree(JA_d0);
  cudaFree(v_d0);
  cudaFree(Q_d_raw0);
  cudaFree(alpha_d0);
  cudaFree(beta_d0);
  cudaFree(tmp_d0);

  cudaSetDevice(1);
  cudaFree(IA_d1);
  cudaFree(JA_d1);
  cudaFree(v_d1);
  cudaFree(x_d1);
}

template void lanczosDecomp<float>::cu_decompose();
template void lanczosDecomp<double>::cu_decompose();

 
 
