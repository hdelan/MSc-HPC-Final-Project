/**
 * \file:        cu_SPMV.cu
 * \brief:       A few CUDA SPMV function implementations
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */

#include "cu_SPMV.h"

#include <stdio.h>
#include "cu_linalg.h"

#define SHARED_BYTES 49152
#define THRESHOLD 200

  /* --------------------------------------------------------------------------*/
  /**
   * \brief:       A naive SPMV kernel
   *
   * \param:       IA           
   * \param:       JA
   * \param:       n
   * \param:       x            starting vector, normalized
   * \param:       ans
   *
   * \returns      
   */
  /* ----------------------------------------------------------------------------*/
  template <typename T, typename U>
__global__ void cu_spMV1(U *const IA /*row_offset*/, U *const JA /* col_idx*/, const U n, T *const x, T *ans)
{
  auto tid {blockDim.x * blockIdx.x + threadIdx.x};
  //if (tid == 0) printf("Printing from kernel! My thread id is %d\n", tid);
  if (tid < n) {
    T t_ans{0};
    for (auto i = IA[tid]; i < IA[tid + 1]; i++)
      t_ans += x[JA[i]];
    ans[tid] = t_ans;
  }
}


  /* --------------------------------------------------------------------------*/
  /**
   * \brief:       A SPMV kernel that assigns rows dynamically to blocks
   *
   * \param:       IA
   * \param:       JA
   * \param:       blockrows    Communicates which rows each block should be responsible for
   * \param:       n
   * \param:       x
   * \param:       ans
   *
   * \returns      
   */
  /* ----------------------------------------------------------------------------*/
  template <typename T, typename U, unsigned blockSize>
__global__ void cu_spMV2(U *const IA /*row_offset*/, U *const JA /* col_idx*/, U *const blockrows, const U n, T *const x, T *ans)
{
  __shared__ T tmp_s[SHARED_BYTES/sizeof(T)];
  auto tid{threadIdx.x}, bid{blockIdx.x};
  auto startrow{blockrows[bid]}, endrow{blockrows[bid + 1]};
  auto nnz{IA[endrow] - IA[startrow]}; // This is definitely ok since endrow is max n

  auto firstcol {IA[startrow]};

  // A single long row for a whole block
  if (endrow-startrow == 1) {
    //if (tid == 0) printf("In here! Startrow %lu\tEndrow %lu\t nnz: %lu  \tblockId: %lu\n", blockrows[bid], blockrows[bid+1], nnz, blockIdx.x);
    
    tmp_s[tid] = 0;

    for (auto i=firstcol+tid; i<IA[endrow];i+=blockSize)
      tmp_s[tid] += x[JA[i]];
  
    __syncthreads();

    // Reduction
    if (blockSize == 1024) {if (tid < 512) tmp_s[tid] += tmp_s[tid+512]; __syncthreads(); }
    if (blockSize >= 512) {if (tid < 256) tmp_s[tid] += tmp_s[tid+256];  __syncthreads(); }
    if (blockSize >= 256) {if (tid < 128) tmp_s[tid] += tmp_s[tid+128];  __syncthreads(); }
    if (blockSize >= 128) {if (tid < 64) tmp_s[tid] += tmp_s[tid+64];   __syncthreads(); }

    if (tid < 32) warpReduce<T,blockSize>(tmp_s, tid);
    if (tid == 0) ans[startrow] = tmp_s[0];
    return;
  }

  assert(nnz <= (SHARED_BYTES/sizeof(T)));
  
  for (auto i = firstcol+tid; i < IA[endrow]; i += blockDim.x)
    tmp_s[i-firstcol] = x[JA[i]];
  __syncthreads();
 
  auto num_rows{endrow - startrow};

  if (tid < num_rows && startrow+tid < n)
  {
    auto sum{0.0};
    auto row_s{IA[startrow + tid] - firstcol};
    auto row_e{IA[startrow + tid + 1] - firstcol};
    for (auto i = row_s; i < row_e; i++)
      sum += tmp_s[i];
    ans[startrow + tid] = sum;
  }
}

// This function assigns a number of rows to each block for SPMV2
  template <typename T>
void get_blockrows(adjMatrix &A, const unsigned block_size, unsigned *blockrows, unsigned &blocks_needed)
{
  const unsigned shared_size {SHARED_BYTES/sizeof(T)}; // The capacity of shared memory
  blockrows[0] = 0u;
  unsigned rows_in_this_block{0u}, sum{0u};

  unsigned *IA{A.row_offset};

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
        blockrows[blocks_needed++] = i--;
      } else {
        // If only one row is bigger than shared memory
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



template __global__ void cu_spMV1<float, unsigned>(unsigned *const, unsigned *const, const unsigned, float *const, float *);
template __global__ void cu_spMV1<double, unsigned>(unsigned *const, unsigned *const, const unsigned, double *const, double *);

template __global__ void cu_spMV2<float, unsigned, 1>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 1>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 2>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 2>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 4>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 4>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 8>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 8>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 16>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 16>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 32>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 32>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 64>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 64>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 128>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 128>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 256>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 256>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 512>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 512>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);
template __global__ void cu_spMV2<float, unsigned, 1024>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, float *const x, float *ans);
template __global__ void cu_spMV2<double, unsigned, 1024>(unsigned *const IA /*row_offset*/, unsigned *const JA /* col_idx*/, unsigned *const blockrows, const unsigned n, double *const x, double *ans);

template void get_blockrows<float>(adjMatrix &, const unsigned block_size, unsigned *blockrows, unsigned &blocks_needed);
template void get_blockrows<double>(adjMatrix &, const unsigned block_size, unsigned *blockrows, unsigned &blocks_needed);
