#include "linalg.h"
#include <stdio.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile double * sdata, const unsigned tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid+32];
    if (blockSize >= 32) sdata[tid] += sdata[tid+16];
    if (blockSize >= 16) sdata[tid] += sdata[tid+8];
    if (blockSize >= 8) sdata[tid] += sdata[tid+4];
    if (blockSize >= 4) sdata[tid] += sdata[tid+2];
    if (blockSize >= 2) sdata[tid] += sdata[tid+1];
    printf("Hallo from in here!\n");
}

template <unsigned int blockSize>
__global__ void cu_dot_prod(double * a, double * b, const unsigned n, double * ans) {
    extern __shared__ double sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockSize*2 + tid;
    unsigned gridSize = blockSize*2*gridDim.x;
    
    printf("Hello from in here!\n");


    sdata[tid]=0.0;

    while (i<n) {
        sdata[tid] += a[i]*b[i]+a[i+blockSize]*b[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[i+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[i+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[i+64]; __syncthreads(); }

    //if (tid < 32) warpReduce(sdata, tid, blockSize);
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
}

// Reduce operation, use for 1 block to get a final answer
template <unsigned int blockSize>
__global__ void cu_reduce(double * a, const unsigned n, double * ans) {
    extern __shared__ double sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x*blockSize*2 + tid;
    unsigned gridSize = blockSize*2*gridDim.x;
    
    printf("Grello from in here!\n");
    
    sdata[tid]=0.0;

    while (i<n) {
        sdata[tid] += a[i]+a[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[i+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[i+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[i+64]; __syncthreads(); }

    //if (tid < 32) warpReduce(sdata, tid, blockSize);
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
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

    //if (tid < 32) warpReduce(sdata, tid, blockSize);
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
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

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
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

template __device__ void warpReduce<1>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<1>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<1>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<1>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<1>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<1>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<1>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<2>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<2>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<2>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<2>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<2>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<2>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<2>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<4>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<4>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<4>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<4>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<4>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<4>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<4>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<8>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<8>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<8>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<8>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<8>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<8>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<8>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<16>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<16>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<16>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<16>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<16>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<16>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<16>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<32>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<32>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<32>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<32>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<32>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<32>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<32>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<64>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<64>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<64>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<64>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<64>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<64>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<64>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<128>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<128>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<128>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<128>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<128>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<128>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<128>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<256>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<256>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<256>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<256>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<256>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<256>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<256>(double * v, double alpha, double * x, const unsigned n);

template __device__ void warpReduce<512>(volatile double * sdata, const unsigned tid);
template __global__ void cu_dot_prod<512>(double * a, double * b, const unsigned n, double * ans);
template __global__ void cu_reduce<512>(double * a, const unsigned n, double * ans);
template __global__ void cu_reduce_sqrt<512>(double * a, const unsigned n, double * ans);
template __global__ void cu_norm_sq<512>(double * a, const unsigned n, double * ans);
template __global__ void cu_dpax<512>(double * v, double alpha, double * x, const unsigned n);
template __global__ void cu_dvexda<512>(double * v, double alpha, double * x, const unsigned n);