#ifndef GOLDILOCKS_CUBIC_EXTENSION_GPU_CUH
#define GOLDILOCKS_CUBIC_EXTENSION_GPU_CUH
#ifdef __USE_CUDA__
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include <cassert>

/*
    Implementations for expressions:
*/
__device__ __forceinline__ void Goldilocks3::copy_gpu(gl64_t *c_, const gl64_t *a_)
{
    c_[threadIdx.x] = a_[threadIdx.x];
    c_[blockDim.x + threadIdx.x] = a_[blockDim.x + threadIdx.x];
    c_[blockDim.x << 1 + threadIdx.x] = a_[blockDim.x << 1 + threadIdx.x];   
}

__device__ __forceinline__ void Goldilocks3::add_gpu(gl64_t *c_, const gl64_t *a_, const gl64_t *b_)
{
    c_[threadIdx.x] = a_[threadIdx.x] + b_[threadIdx.x];
    c_[blockDim.x + threadIdx.x] = a_[blockDim.x + threadIdx.x] + b_[blockDim.x + threadIdx.x];
    c_[blockDim.x << 1 + threadIdx.x] = a_[blockDim.x << 1 + threadIdx.x] + b_[blockDim.x << 1 + threadIdx.x];
}

__device__ __forceinline__ void Goldilocks3::sub_gpu(gl64_t *c_, const gl64_t *a_, const gl64_t *b_)
{
    c_[threadIdx.x] = a_[threadIdx.x] - b_[threadIdx.x];
    c_[blockDim.x + threadIdx.x] = a_[blockDim.x + threadIdx.x] - b_[blockDim.x + threadIdx.x];
    c_[blockDim.x << 1 + threadIdx.x] = a_[blockDim.x << 1 + threadIdx.x] - b_[blockDim.x << 1 + threadIdx.x];
}

__device__ __forceinline__ void Goldilocks3::mul_gpu(uint64_t blockDim.x, gl64_t *c_, const gl64_t *a_, const gl64_t *b_)
{  
    gl64_t A = (a_[blockDim.x] + a_[blockDim.x + blockDim.x]) * (b_[blockDim.x] + b_[blockDim.x + blockDim.x]);
    gl64_t B = (a_[blockDim.x] + a_[blockDim.x << 1 + blockDim.x]) * (b_[blockDim.x] + b_[blockDim.x << 1 + blockDim.x]);
    gl64_t C = (a_[blockDim.x + blockDim.x] + a_[blockDim.x << 1 + blockDim.x]) * (b_[blockDim.x + blockDim.x] + b_[blockDim.x << 1 + blockDim.x]);
    gl64_t D = a_[blockDim.x] * b_[blockDim.x];
    gl64_t E = a_[blockDim.x + blockDim.x] * b_[blockDim.x + blockDim.x];
    gl64_t F = a_[blockDim.x << 1 + blockDim.x] * b_[blockDim.x << 1 + blockDim.x];
    gl64_t G = D - E;

    c_[blockDim.x] = (C + G) - F;
    c_[blockDim.x + blockDim.x] = ((((A + C) - E) - E) - D);
    c_[blockDim.x << 1 + blockDim.x] = B - G;
};

__device__ __forceinline__ void Goldilocks3::mul_gpu(uint64_t blockDim.x, gl64_t *c_, const gl64_t *a_, const gl64_t *challenge_, const gl64_t *challenge_ops_)
{   
        gl64_t A = (a_[blockDim.x] + a_[blockDim.x + blockDim.x]) * challenge_ops_[blockDim.x];
        gl64_t B = (a_[blockDim.x] + a_[blockDim.x << 1 + blockDim.x]) * challenge_ops_[blockDim.x + blockDim.x];
        gl64_t C = (a_[blockDim.x + blockDim.x] + a_[blockDim.x << 1 + blockDim.x]) * challenge_ops_[blockDim.x << 1 + blockDim.x];
        gl64_t D = a_[blockDim.x] * challenge_[blockDim.x];
        gl64_t E = a_[blockDim.x + blockDim.x] * challenge_[blockDim.x + blockDim.x];
        gl64_t F = a_[blockDim.x << 1 + blockDim.x] * challenge_[blockDim.x << 1 + blockDim.x];
        gl64_t G = D - E;

        c_[blockDim.x] = (C + G) - F;
        c_[blockDim.x + blockDim.x] = ((((A + C) - E) - E) - D);
        c_[blockDim.x << 1 + blockDim.x] = B - G;
};

__device__ __forceinline__ void Goldilocks3::op_gpu(uint64_t op, gl64_t *c, const Goldilocks::Element *a, const Goldilocks::Element *b)
{
    switch (op)
    {
    case 0:
        add_gpu(c, a, b);
        break;
    case 1:
        sub_gpu(c, a, b);
        break;
    case 2:
        mul_gpu(c, a, b);
        break;
    case 3:
        sub_gpu(c, b, a);
        break;
    default:
        assert(0);
        break;
    }
}

__device__ __forceinline__ void Goldilocks3::op_31_gpu(uint64_t op, Goldilocks::Element *c, const Goldilocks::Element *a, const Goldilocks::Element *b)
{
    switch (op)
    {
    case 0:
        c[blockDim.x] = a[blockDim.x] + b[blockDim.x];
        c[blockDim.x + blockDim.x] = a[blockDim.x + blockDim.x];
        c[blockDim.x << 1 + blockDim.x] = a[blockDim.x << 1 + blockDim.x];
        break;
    case 1:
        c[blockDim.x] = a[blockDim.x] - b[blockDim.x];
        c[blockDim.x + blockDim.x] = a[blockDim.x + blockDim.x];
        c[blockDim.x << 1 + blockDim.x] = a[blockDim.x << 1 + blockDim.x];
        break;
    case 2:
        c[blockDim.x] = a[blockDim.x] * b[blockDim.x];
        c[blockDim.x + blockDim.x] = a[blockDim.x + blockDim.x] * b[blockDim.x];
        c[blockDim.x << 1 + blockDim.x] = a[blockDim.x << 1 + blockDim.x] * b[blockDim.x];
        break;
    case 3:
        c[blockDim.x] = b[blockDim.x] - a[blockDim.x];
        c[blockDim.x + blockDim.x] = -a[blockDim.x + blockDim.x];
        c[blockDim.x << 1 + blockDim.x] = -a[blockDim.x << 1 + blockDim.x];
        break;
    default:
        assert(0);
        break;
    }
}
#endif
#endif