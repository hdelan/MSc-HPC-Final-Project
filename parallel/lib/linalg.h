#ifndef LINALG_H_234
#define LINALG_H_234

__device__ void warpReduce(volatile double * sdata, const unsigned tid);
__global__ void cu_dot_prod(double * a, double * b, const unsigned n, double * ans);
__global__ void cu_reduce(double * a, const unsigned n, double * ans);
__global__ void cu_reduce_sqrt(double * a, const unsigned n, double * ans);
__global__ void cu_norm_sq(double * a, const unsigned n, double * ans);
__global__ void cu_dpax(double * v, double alpha, double * x, const unsigned n);
__global__ void cu_dvexda(double * v, double alpha, double * x, const unsigned n);

#endif