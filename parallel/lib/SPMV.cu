
#include "SPMV.h"
#include <stdio.h>

__global__ void cu_spMV1(long unsigned *const IA /*row_offset*/, long unsigned *const JA /* col_idx*/, long unsigned n, double *const x, double *ans)
{

    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Printing from kernel! My thread id is %d\n", idx);
    if (idx > n-1) return;
    double t_ans{0.0};
    for (auto i = IA[idx]; i <IA[idx+1];i++)
        t_ans += x[JA[i]];
    ans[idx] = t_ans;
}