#ifndef SPMV_H_234234
#define SPMV_H_234234

#include "adjMatrix.h"

template <typename T, typename U>
__global__ void cu_spMV1(U *const IA /*row_offset*/, U *const JA /* col_idx*/, const U n, T *const x, T *ans);

template <typename T, typename U>
__global__ void cu_spMV2(U *const IA /*row_offset*/, U *const JA /* col_idx*/, U* const blockrows, const U n, T *const x, T *ans);

template <typename T, typename U>
__global__ void cu_spMV3_kernel1(U *const JA /* col_idx*/, const U total_nonzeros, T *const x, T *tmp);

template <typename T, typename U>
__global__ void cu_spMV3_kernel2(T*const tmp, U *const IA /* col_idx*/, const U n, T *ans);

template <typename T>
void get_blockrows(adjMatrix & A, const unsigned block_size, long unsigned * blockrows, long unsigned & blocks_needed);

#endif