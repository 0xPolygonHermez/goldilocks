#ifndef GOLDILOCKS_GPU_CUH
#define GOLDILOCKS_GPU_CUH
#ifdef __USE_CUDA__

#include "goldilocks_base_field.hpp"
#include <cassert>
/*
    Implementations for expressions:
*/
    
    __device__ __forceinline__ void Goldilocks::op_gpu( uint64_t op, gl64_t *c, const gl64_t *a, const gl64_t *b){

        switch (op)
        {
        case 0:
            add(c[threadIdx.x], a[threadIdx.x], b[threadIdx.x]);
            break;
        case 1:
            sub(c[threadIdx.x], a[threadIdx.x], b[threadIdx.x]);
            break;
        case 2:
            mul(c[threadIdx.x], a[threadIdx.x], b[threadIdx.x]);
            break;
        case 3:
            sub(c[threadIdx.x], b[threadIdx.x], a[threadIdx.x]);
            break;
        default:
            assert(0);
            break;
        }
    }

#endif
#endif