#include "linalg.h"

template <unsigned int blockSize>
__host__ __device__ void warpReduce(volatile double * sdata, const unsigned tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid+32];
    if (blockSize >= 32) sdata[tid] += sdata[tid+16];
    if (blockSize >= 16) sdata[tid] += sdata[tid+8];
    if (blockSize >= 8) sdata[tid] += sdata[tid+4];
    if (blockSize >= 4) sdata[tid] += sdata[tid+2];
    if (blockSize >= 2) sdata[tid] += sdata[tid+1];
}

template <unsigned int blockSize>
__global__ void cu_dot_prod(double * a, double * b, const unsigned n, double * ans) {
    extern __shared__ double sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockSize*2 + tid;
    unsigned gridSize = blockSize*2*gridDim.x;
    
    sdata[tid]=0.0;

    while (i<n) {
        sdata[tid] += a[i]*b[i]+a[i+blockSize]*b[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[i+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[i+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[i+64]; __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
}

// Reduce operation, use for 1 block to get a final answer
template <unsigned int blockSize>
__global__ void cu_reduce(double * a, const unsigned n, double * ans) {
    extern __shared__ double sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockSize*2 + tid;
    unsigned gridSize = blockSize*2*gridDim.x;
    
    sdata[tid]=0.0;

    while (i<n) {
        sdata[tid] += a[i]+a[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[i+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[i+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[i+64]; __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void cu_reduce_sqrt(double * a, const unsigned n, double * ans) {
    extern __shared__ double sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockSize*2 + tid;
    unsigned gridSize = blockSize*2*gridDim.x;
    
    sdata[tid]=0.0;

    while (i<n) {
        sdata[tid] += a[i]+a[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[i+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[i+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[i+64]; __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = std::sqrt(sdata[0]);
}

template <unsigned int blockSize>
__global__ void cu_norm_sq(double * a, const unsigned n, double * ans) {
    extern __shared__ double sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockSize*2 + tid;
    unsigned gridSize = blockSize*2*gridDim.x;
    
    sdata[tid]=0.0;

    while (i<n) {
        sdata[tid] += a[i]*a[i]+a[i+blockSize]*a[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[i+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[i+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[i+64]; __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void cu_dpax(double * v, double alpha, double * x, const unsigned n) {
    unsigned tid = threadIdx.x+blockIdx.x*blockSize;
    if (tid > n) return;
    v[tid] -= alpha*x[tid];
}

template <unsigned int blockSize>
__global__ void cu_dvexda(double * v, double alpha, double * x, const unsigned n) {
    unsigned tid = threadIdx.x+blockIdx.x*blockSize;
    if (tid > n) return;
    v[tid] = x[tid]/alpha;
}


template __device__ void warpReduce<32>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<32>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<32>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<32>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<32>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<32>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<32>(double * v, double alpha, double * x, const unsigned n);