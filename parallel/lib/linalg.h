#ifndef LINALG_H_234
#define LINALG_H_234

template <unsigned blockSize>
__device__ void warpReduce(volatile double * sdata, const unsigned tid);
template <unsigned blockSize>
__global__ void cu_dot_prod(double * a, double * b, const unsigned n, double * ans);
template <unsigned blockSize>
__global__ void cu_reduce(double * a, const unsigned n, double * ans);
template <unsigned blockSize>
__global__ void cu_reduce_sqrt(double * a, const unsigned n, double * ans);
template <unsigned blockSize>
__global__ void cu_norm_sq(double * a, const unsigned n, double * ans);
template <unsigned blockSize>
__global__ void cu_dpax(double * v, double alpha, double * x, const unsigned n);
template <unsigned blockSize>
__global__ void cu_dvexda(double * v, double alpha, double * x, const unsigned n);

#endif