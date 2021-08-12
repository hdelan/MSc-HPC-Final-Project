#include "cu_linalg.h"
#include <stdio.h>

template <typename T, unsigned blockSize>
__device__ void warpReduce(volatile T * sdata, const unsigned tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid+32];
    if (blockSize >= 32 && tid < 16) sdata[tid] += sdata[tid+16];
    if (blockSize >= 16 && tid < 8) sdata[tid] += sdata[tid+8];
    if (blockSize >= 8 && tid < 4) sdata[tid] += sdata[tid+4];
    if (blockSize >= 4 && tid < 2) sdata[tid] += sdata[tid+2];
    if (blockSize >= 2 && tid < 1) sdata[tid] += sdata[tid+1];
}

template <typename T, unsigned blockSize>
__global__ void cu_dot_prod(T * a, T * b, const unsigned n, T * ans) {
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockSize*2 + tid;
    if (i >= n) return;
    unsigned gridSize = blockSize*2*gridDim.x;
    extern __shared__ unsigned char sdata_uchar[];
    T * sdata = reinterpret_cast<T *>(sdata_uchar);
    
    sdata[tid]=0.0;

    while (i+blockSize<n) {
        sdata[tid] += a[i]*b[i]+a[i+blockSize]*b[i+blockSize];
        i += gridSize;
    }
    if (i<n) sdata[tid] += a[i]*b[i];
    
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid+64]; __syncthreads(); }

    if (tid < 32) warpReduce<T,blockSize>(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
    if (tid==0) printf("Just wrote %lf\n", ans[blockIdx.x]);
}

// Reduce operation, use for 1 block to get a final answer
template <typename T, unsigned blockSize>
__global__ void cu_reduce(T * a, const unsigned n, T * ans) {
    extern __shared__ unsigned char sdata_uchar[];
    T * sdata = reinterpret_cast<T *>(sdata_uchar);
    //extern __shared__ T sdata[];
    unsigned tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + tid;
    if (i >= n) return;
    unsigned gridSize = blockSize*2*gridDim.x;

    
    sdata[tid]=0.0;

    while (i+blockSize<n) {
        sdata[tid] += a[i]+a[i+blockSize];
        i += gridSize;
    }
    if (i<n) sdata[tid] += a[i];

    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid+64]; __syncthreads(); }

    //if (tid < 32) warpReduce(sdata, tid, blockSize);
    if (tid < 32) warpReduce<T,blockSize>(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
}

template <typename T, unsigned blockSize>
__global__ void cu_reduce_sqrt(T * a, const unsigned n, T * ans) {
    extern __shared__ unsigned char sdata_uchar[];
    T * sdata = reinterpret_cast<T *>(sdata_uchar);
    //extern __shared__ T sdata[];
    unsigned tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + tid;
    if (i >= n) return;
    unsigned gridSize = blockSize*2*gridDim.x;
    
    sdata[tid]=0.0;

    while (i+blockSize<n) {
        sdata[tid] += a[i]+a[i+blockSize];
        i += gridSize;
    }
    if (i < n) sdata[tid] += a[i]+a[i+blockSize];
    
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid+64]; __syncthreads(); }

    //if (tid < 32) warpReduce(sdata, tid, blockSize);
    if (tid < 32) warpReduce<T,blockSize>(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = std::sqrt(sdata[0]);
}

template <typename T, unsigned blockSize>
__global__ void cu_norm_sq(T * a, const unsigned n, T * ans) {
    //extern __shared__ T sdata[];
    extern __shared__ unsigned char sdata_uchar[];
    T * sdata = reinterpret_cast<T *>(sdata_uchar);
    
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockSize*2 + tid;
    if (i >= n) return;
    unsigned gridSize = blockSize*2*gridDim.x;
    
    sdata[tid]=0.0;

    while (i+blockSize<n) {
        sdata[tid] += a[i]*a[i]+a[i+blockSize]*a[i+blockSize];
        i += gridSize;
    }
    if (i<n) sdata[tid] += a[i]*a[i];
    
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid+64]; __syncthreads(); }

    if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
}

template <typename T>
__global__ void cu_dpax(T * v, T alpha, T * x, const long unsigned n) {
    unsigned tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= n) return;
    v[tid] -= alpha*x[tid];
}

template <typename T>
__global__ void cu_dvexda(T * v, T alpha, T * x, const long unsigned n) {
    unsigned tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= n) return;
    v[tid] = x[tid]/alpha;
}

// EXPLICIT INSTANTIATIONS
template __global__ void cu_dpax<double>(double * v, double alpha, double * x, const long unsigned n);
template __global__ void cu_dpax<float>(float * v, float alpha, float * x, const long unsigned n);

template __global__ void cu_dvexda<double>(double * v, double alpha, double * x, const long unsigned n);
template __global__ void cu_dvexda<float>(float * v, float alpha, float * x, const long unsigned n);

template __device__ void warpReduce<float ,1>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 1>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 1>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 1>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 1>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 2>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 2>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 2>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 2>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 2>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 4>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 4>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 4>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 4>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 4>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 8>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 8>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 8>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 8>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 8>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 16>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 16>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 16>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 16>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 16>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 32>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 32>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 32>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 32>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 32>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 64>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 64>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 64>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 64>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 64>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 128>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 128>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 128>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 128>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 128>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 256>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 256>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 256>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 256>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 256>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 512>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 512>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 512>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 512>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 512>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<float, 1024>(volatile float * sdata, const unsigned tid);
template __global__ void cu_dot_prod<float, 1024>(float * a, float * b, const unsigned n, float * ans);
template __global__ void cu_reduce<float, 1024>(float * a, const unsigned n, float * ans);
template __global__ void cu_reduce_sqrt<float, 1024>(float * a, const unsigned n, float * ans);
template __global__ void cu_norm_sq<float, 1024>(float * a, const unsigned n, float * ans);

template __device__ void warpReduce<double, 1>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 1>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 1>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 1>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 1>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 2>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 2>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 2>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 2>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 2>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 4>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 4>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 4>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 4>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 4>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 8>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 8>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 8>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 8>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 8>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 16>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 16>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 16>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 16>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 16>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 32>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 32>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 32>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 32>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 32>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 64>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 64>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 64>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 64>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 64>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 128>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 128>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 128>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 128>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 128>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 256>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 256>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 256>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 256>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 256>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 512>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 512>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 512>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 512>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 512>(double * a, const unsigned n, double * ans);

template __device__ void warpReduce<double, 1024>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<double, 1024>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<double, 1024>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<double, 1024>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<double, 1024>(double * a, const unsigned n, double * ans);
