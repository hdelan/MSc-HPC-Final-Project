#ifndef SPMV_H_234234
#define SPMV_H_234234

template <typename T> //, typename U>
__global__ void cu_spMV1(long unsigned *const IA /*row_offset*/, long unsigned *const JA /* col_idx*/, const long unsigned n, T *const x, T *ans);

#endif