#ifndef LINALG_H_234
#define LINALG_H_234

template <typename T, unsigned blockSize>
__device__ void warpReduce(volatile T * sdata, const unsigned tid);
template <typename T, unsigned blockSize>
__global__ void cu_dot_prod(T * a, T * b, const unsigned n, T * ans);
template <typename T, unsigned blockSize>
__global__ void cu_reduce(T * a, const unsigned n, T * ans);
template <typename T, unsigned blockSize>
__global__ void cu_reduce_sqrt(T * a, const unsigned n, T * ans);
template <typename T, unsigned blockSize>
__global__ void cu_norm_sq(T * a, const unsigned n, T * ans);
template <typename T, unsigned blockSize>
__global__ void cu_norm_sq_sqrt(T * a, const unsigned n, T * ans);
template <typename T>
__global__ void cu_dpax(T * v, T * alpha, T * x, const unsigned n);
template <typename T>
__global__ void cu_dvexda(T * v, T * alpha, T * x, const unsigned n);

#endif
