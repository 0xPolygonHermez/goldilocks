#include <benchmark/benchmark.h>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <immintrin.h>
#include "omp.h"
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"

#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

#define NUM_HASHES 1
#define NCOLS_POS 1024 * 1024

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#endif

using namespace std;

static void ADD_BENCH(benchmark::State &state)
{

    Goldilocks::Element *x = (Goldilocks::Element *)malloc(NCOLS_POS * sizeof(Goldilocks::Element));
    Goldilocks::Element *y = (Goldilocks::Element *)malloc(NCOLS_POS * sizeof(Goldilocks::Element));
    Goldilocks::Element *z = (Goldilocks::Element *)malloc(NCOLS_POS * sizeof(Goldilocks::Element));

    for (int i = 0; i < NCOLS_POS; ++i)
    {
        x[i].fe = (u_int64_t)i;
        y[i].fe = (u_int64_t)(NCOLS_POS - i);
    }

    for (auto _ : state)
    {
        for (int i = 0; i < NUM_HASHES; ++i)
        {
            for (int k = 0; k < NCOLS_POS; ++k)
            {
                z[k] = x[k] + y[k];
            }
        }
    }
    for (int i = 0; i < NCOLS_POS; ++i)
    {
        assert(z[i].fe % GOLDILOCKS_PRIME == NCOLS_POS % GOLDILOCKS_PRIME);
    }

    free(x);
    free(y);
    free(z);
}

static void ADD_SIMPLE_AVX_BENCH(benchmark::State &state)
{
    Goldilocks::Element *x = (Goldilocks::Element *)aligned_alloc(256, NCOLS_POS * sizeof(Goldilocks::Element));
    Goldilocks::Element *y = (Goldilocks::Element *)aligned_alloc(256, NCOLS_POS * sizeof(Goldilocks::Element));
    Goldilocks::Element *z = (Goldilocks::Element *)aligned_alloc(256, NCOLS_POS * sizeof(Goldilocks::Element));

    for (int i = 0; i < NCOLS_POS; ++i)
    {
        x[i].fe = (u_int64_t)i;
        y[i].fe = (u_int64_t)(NCOLS_POS - i);
    }

    for (auto _ : state)
    {
        for (int i = 0; i < NUM_HASHES; ++i)
        {
            for (int k = 0; k < NCOLS_POS; k += 4)
            {
                const __m256i x_ = _mm256_load_si256((__m256i *)&(x[k]));
                const __m256i y_ = _mm256_load_si256((__m256i *)&(y[k]));
                const __m256i z_ = _mm256_add_epi64(x_, y_);
                _mm256_stream_si256((__m256i *)&(z[k]), z_);
            }
        }
    }
    for (int i = 0; i < NCOLS_POS; ++i)
    {
        assert(z[i].fe % GOLDILOCKS_PRIME == NCOLS_POS % GOLDILOCKS_PRIME);
    }

    free(x);
    free(y);
    free(z);
}

static void ADD_AVX_BENCH(benchmark::State &state)
{
    Goldilocks::Element *a = (Goldilocks::Element *)aligned_alloc(256, NCOLS_POS * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)aligned_alloc(256, NCOLS_POS * sizeof(Goldilocks::Element));
    Goldilocks::Element *c = (Goldilocks::Element *)aligned_alloc(256, NCOLS_POS * sizeof(Goldilocks::Element));

    for (int i = 0; i < NCOLS_POS; ++i)
    {
        a[i].fe = (u_int64_t)i;
        b[i].fe = (u_int64_t)(NCOLS_POS - i);
    }
    u_int64_t shift = 1 << 63;
    const __m256i shift_ = _mm256_set_epi64x(shift, shift, shift, shift);
    const __m256i p_ = _mm256_set_epi64x(GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME);
    const __m256i pn_ = _mm256_set_epi64x(-GOLDILOCKS_PRIME, -GOLDILOCKS_PRIME, -GOLDILOCKS_PRIME, -GOLDILOCKS_PRIME);
    const __m256i ps_ = _mm256_xor_si256(p_, shift_);

    for (auto _ : state)
    {
        for (int i = 0; i < NUM_HASHES; ++i)
        {
            for (int k = 0; k < NCOLS_POS; k += 4)
            {
                // load
                const __m256i a_ = _mm256_load_si256((__m256i *)&(a[k]));
                const __m256i b_ = _mm256_load_si256((__m256i *)&(b[k]));

                // shift a_
                __m256i as_ = _mm256_xor_si256(a_, shift_);

                // canonailze a_
                __m256i mask1_ = _mm256_cmpgt_epi64(ps_, as_);
                __m256i corr1_ = _mm256_and_si256(mask1_, pn_);
                as_ = _mm256_add_epi64(as_, corr1_);

                // addition
                const __m256i c_aux_ = _mm256_add_epi64(as_, b_); // can we use only c_

                // correction if overflow
                __m256i mask_ = _mm256_cmpgt_epi64(as_, c_aux_);
                __m256i corr_ = _mm256_and_si256(mask_, pn_); // zero ises amother  thing
                __m256i c_ = _mm256_add_epi64(c_aux_, corr_); // can we c_=c_+corr_

                // shift c_
                c_ = _mm256_xor_si256(c_, shift_);

                // stream
                _mm256_stream_si256((__m256i *)&(c[k]), c_);
            }
        }
    }
    for (int i = 0; i < NCOLS_POS; ++i)
    {
        assert(c[i].fe % GOLDILOCKS_PRIME == NCOLS_POS % GOLDILOCKS_PRIME);
    }
    free(a);
    free(b);
    free(c);
}

BENCHMARK(ADD_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(1, 1, 1)
    ->UseRealTime();
BENCHMARK(ADD_SIMPLE_AVX_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(1, 1, 1)
    ->UseRealTime();
BENCHMARK(ADD_AVX_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(1, 1, 1)
    ->UseRealTime();

BENCHMARK_MAIN();
// g++ miniapp/avx.cpp src/* -lbenchmark -mavx -mavx2 -lomp -lpthread -lgmp  -std=c++17 -Wall -pthread -fopenmp -L$(find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//') -O3 -o avx