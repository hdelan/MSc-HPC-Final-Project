
#include "cu_SPMV.h"

#include <stdio.h>

template <typename T, typename U>
__global__ void cu_spMV1(U *const IA /*row_offset*/, U *const JA /* col_idx*/, const U n, T *const x, T *ans)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    //if (idx == 0) printf("Printing from kernel! My thread id is %d\n", idx);
    if (idx > n-1) return;
    T t_ans{0.0};
    for (auto i = IA[idx]; i<IA[idx+1];i++)
        t_ans += x[JA[i]];
    ans[idx] = t_ans;
}

template __global__ void cu_spMV1<float, long unsigned>(long unsigned *const, long unsigned *const,long unsigned, float *const, float *);
template __global__ void cu_spMV1<double, long unsigned>(long unsigned *const, long unsigned *const, long unsigned, double *const, double *);
template __global__ void cu_spMV1<float, unsigned>(unsigned *const, unsigned *const, unsigned, float *const, float *);
template __global__ void cu_spMV1<double, unsigned>(unsigned *const, unsigned *const,  unsigned, double *const, double *);