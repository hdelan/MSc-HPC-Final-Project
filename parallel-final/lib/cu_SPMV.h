#ifndef SPMV_H_234234
#define SPMV_H_234234

#include "adjMatrix.h"

template <typename T, typename U>
__global__ void cu_spMV1(U *const IA /*row_offset*/, U *const JA /* col_idx*/, const U n, T *const x, T *ans);

template <typename T, typename U, unsigned blockSize>
__global__ void cu_spMV2(U *const IA /*row_offset*/, U *const JA /* col_idx*/, U* const blockrows, const U n, T *const x, T *ans);

template <typename T, typename U, unsigned blockSize>
__global__ void cu_spMV3(U *const, U * const JA, const U, const U, T *const x, T *const ans);

template <typename T, typename U, unsigned blockSize>
__global__ void cu_spMV4(U *const, U * const JA, T *const x, T *const ans);

template <typename T>
void get_blockrows(adjMatrix & A, const unsigned block_size, unsigned * blockrows, unsigned & blocks_needed);

// Some helper device functions
template <typename T, unsigned blockSize>
__device__ void warpReduce(volatile T * sdata, const unsigned tid) {
    if (blockSize >= 64) { sdata[tid] += sdata[tid+32]; __syncthreads();}
    if (blockSize >= 32 /*&& tid < 16*/) sdata[tid] += sdata[tid+16];
    if (blockSize >= 16 /*&& tid < 8*/) sdata[tid] += sdata[tid+8];
    if (blockSize >= 8  /*&& tid < 4*/) sdata[tid] += sdata[tid+4];
    if (blockSize >= 4  /*&& tid < 2*/) sdata[tid] += sdata[tid+2];
    if (blockSize >= 2  /*&& tid < 1*/) sdata[tid] += sdata[tid+1];
}

template <typename T>
__device__ int min(T a, T b) {return a < b ? a : b;}

template <typename T> 
__device__ int max(T a, T b) { return a > b ? a : b;}

#endif

