#ifndef SPMV_H_234234
#define SPMV_H_234234

template <typename T, typename U>
__global__ void cu_spMV1(U *const IA /*row_offset*/, U *const JA /* col_idx*/, U n, T *const x, T *ans);

#endif