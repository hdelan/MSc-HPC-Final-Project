#ifndef SPMV_H_234234
#define SPMV_H_234234

__global__ void cu_spMV1(long unsigned *const IA /*row_offset*/, long unsigned *const JA /* col_idx*/, long unsigned n, double *const x, double *ans);

#endif