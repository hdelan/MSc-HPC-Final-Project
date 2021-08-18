
#include "cu_SPMV.h"

#include <stdio.h>

#define SHARED_BYTES 49'152

template <typename T>
__device__ int min(T a, T b) {return a < b ? a : b;}

template <typename T> 
__device__ int max(T a, T b) { return a > b ? a : b;}

  template <typename T, typename U>
__global__ void cu_spMV1(U *const IA /*row_offset*/, U *const JA /* col_idx*/, const U n, T *const x, T *ans)
{
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  //if (idx == 0) printf("Printing from kernel! My thread id is %d\n", idx);
  if (idx > n - 1)
    return;
  T t_ans{0};
  for (auto i = IA[idx]; i < IA[idx + 1]; i++)
    t_ans += x[JA[i]];
  ans[idx] = t_ans;
}

  template <typename T, typename U>
__global__ void cu_spMV2(U *const IA /*row_offset*/, U *const JA /* col_idx*/, U *const blockrows, const U n, T *const x, T *ans)
{

  __shared__ T tmp_s[SHARED_BYTES / sizeof(T)];
  auto tid{threadIdx.x}, bid{blockIdx.x};
  auto startrow{blockrows[bid]}, endrow{blockrows[bid + 1]};
  auto firstcol{IA[startrow]};
  auto nnz{IA[endrow] - IA[startrow]};
  for (auto i = tid; i < nnz; i += blockDim.x)
    tmp_s[i] = x[JA[firstcol + i]];

  __syncthreads();

  auto num_rows{endrow - startrow};
  if (tid < num_rows)
  {
    auto sum{0.0};
    auto row_s{IA[startrow + tid] - firstcol};
    auto row_e{IA[startrow + tid + 1] - firstcol};
    for (auto i = row_s; i < row_e; i++)
      sum += tmp_s[i];
    ans[startrow + tid] = sum;
  }
}

  template <typename T, typename U>
__global__ void cu_spMV3_kernel1(U *const JA /* col_idx*/, const U total_nonzeros, T *const x, T *tmp)
{
  auto tid{threadIdx.x + blockIdx.x * blockDim.x};
  auto icr{blockDim.x + gridDim.x};
  while (tid < total_nonzeros)
  {
    tmp[tid] = x[JA[tid]];
    tid += icr;
  }
}

  template <typename T, typename U>
__global__ void cu_spMV3_kernel2(T *const tmp_v, U *const IA /* col_idx*/, const U n, T *ans)
{
  auto tid{threadIdx.x};
  auto gid{threadIdx.x + blockDim.x * blockIdx.x};

  extern __shared__ U raw_s[];
  U *IA_s{(U *)&raw_s[0]};
  T *v_s{(T *)&raw_s[blockDim.x + 1]};

  IA_s[tid] = IA[gid];
  if (tid == 0)
  {
    if (gid + blockDim.x > n)
    {
      IA_s[blockDim.x] = IA[n];
    }
    else
    {
      IA_s[blockDim.x] = IA[gid + blockDim.x];
    }
  }
  __syncthreads();
  int temp = (IA_s[blockDim.x] - IA_s[0]) / blockDim.x + 1;
  int nlen = min(static_cast<int>(temp * blockDim.x), static_cast<int>(SHARED_BYTES/sizeof(U)));
  T sum = 0;
  int maxlen = IA_s[blockDim.x];
  for (int i = IA_s[0]; i < maxlen; i += nlen)
  {
    int index = i + tid;
    __syncthreads();
    for (int j = 0; j < nlen / blockDim.x; j++)
    {
      if (index < maxlen)
      {
        v_s[tid + j * blockDim.x] = tmp_v[index];
        index += blockDim.x;
      }
    }
    __syncthreads();

    // Sum up the elements for a row
    if (!(IA_s[tid + 1] <= i || IA_s[tid] > i + nlen - 1))
    {
      int row_s = max(static_cast<int>(IA_s[tid]) - i, 0);
      int row_e = min(static_cast<int>(IA_s[tid + 1] - i), nlen);
      for (int j = row_s; j < row_e; j++)
      {
        sum += v_s[j];
      }
    }
  }
  // Write result
  ans[gid] = sum;
}

  template <typename T>
void get_blockrows(adjMatrix &A, const unsigned block_size, long unsigned *blockrows, long unsigned &blocks_needed)
{
  const unsigned shared_size{SHARED_BYTES / sizeof(T)};
  blockrows[0] = 0u;
  unsigned rows_in_this_block{0u}, sum{0u};

  long unsigned *IA{A.row_offset};

  blocks_needed = 1u;

  for (auto i = 0u; i < A.n; i++)
  {
    sum += IA[i + 1] - IA[i];
    rows_in_this_block++;
    if (sum == shared_size || (rows_in_this_block == block_size && sum <= shared_size))
    {
      blockrows[blocks_needed++] = i + 1;
      rows_in_this_block = 0u;
      sum = 0u;
    }
    else if (sum > shared_size)
    {
      if (rows_in_this_block != 1) {
        blockrows[blocks_needed++] = --i;
      } else {
        blockrows[blocks_needed++] = i;
      }
      rows_in_this_block = 0u;
      sum = 0u;
    }
  }
  if (blockrows[blocks_needed - 1] != A.n)
  {
    blockrows[blocks_needed] = A.n;
  }
  else
  {
    blocks_needed--;
  }
}

template __global__ void cu_spMV1<float, long unsigned>(long unsigned *const, long unsigned *const, const long unsigned, float *const, float *);
template __global__ void cu_spMV1<double, long unsigned>(long unsigned *const, long unsigned *const, const long unsigned, double *const, double *);
template __global__ void cu_spMV1<float, unsigned>(unsigned *const, unsigned *const, const unsigned, float *const, float *);
template __global__ void cu_spMV1<double, unsigned>(unsigned *const, unsigned *const, const unsigned, double *const, double *);

template __global__ void cu_spMV2<float, unsigned>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, long unsigned>(long unsigned *const IA /*row_offset*/, long unsigned *const JA /* col_idx*/, long unsigned *const blockrows, const long unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, long unsigned>(long unsigned *const IA /*row_offset*/, long unsigned *const JA /* col_idx*/, long unsigned *const blockrows, const long unsigned n, double *const x, double *ans);

template __global__ void cu_spMV3_kernel1<float, long unsigned>(long unsigned *const JA /* col_idx*/, const long unsigned total_nonzeros, float *const x, float *tmp);
template __global__ void cu_spMV3_kernel1<double, long unsigned>(long unsigned *const JA /* col_idx*/, const long unsigned total_nonzeros, double *const x, double *tmp);

template __global__ void cu_spMV3_kernel2<float, long unsigned>(float *const tmp, long unsigned *const IA /* col_idx*/, const long unsigned n, float *ans);
template __global__ void cu_spMV3_kernel2<double, long unsigned>(double *const tmp, long unsigned *const IA /* col_idx*/, const long unsigned n, double *ans);

template void get_blockrows<float>(adjMatrix &, const unsigned block_size, long unsigned *blockrows, long unsigned &blocks_needed);
template void get_blockrows<double>(adjMatrix &, const unsigned block_size, long unsigned *blockrows, long unsigned &blocks_needed);
