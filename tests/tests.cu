#include <gtest/gtest.h>
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../utils/timer_gl.hpp"
#include "../utils/cuda_utils.hpp"

#define FFT_SIZE (1 << 23)
#define BLOWUP_FACTOR 1
#define NUM_COLUMNS 751

#ifdef __USE_CUDA__
TEST(GOLDILOCKS_TEST, full_gpu)
{
    Goldilocks::Element *a;
    Goldilocks::Element *b;
    Goldilocks::Element *c;
    cudaMallocManaged(&a, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    cudaMallocManaged(&b, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    cudaMallocManaged(&c, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

    NTT_Goldilocks ntt(FFT_SIZE);
    warmup_all_gpus();
    alloc_pinned_mem((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    TimerStart(LDE_MerkleTree_MultiGPU_viaCPU);
    ntt.LDE_MerkleTree_MultiGPU_viaCPU(b, a, FFT_SIZE, FFT_SIZE<<BLOWUP_FACTOR, NUM_COLUMNS, c);
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_viaCPU);

    uint64_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("free_mem: %lu, total_mem: %lu\n", free_mem, total_mem);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free_pinned_mem();
}

TEST(GOLDILOCKS_TEST, full_um)
{
    Goldilocks::Element *a;
    Goldilocks::Element *b;
    Goldilocks::Element *c;
    cudaMallocManaged(&a, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    cudaMallocManaged(&b, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    cudaMallocManaged(&c, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

    NTT_Goldilocks ntt(FFT_SIZE);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    TimerStart(LDE_MerkleTree_MultiGPU);
    ntt.LDE_MerkleTree_MultiGPU(b, a, FFT_SIZE, FFT_SIZE<<BLOWUP_FACTOR, NUM_COLUMNS, c);
    TimerStopAndLog(LDE_MerkleTree_MultiGPU);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
#endif

TEST(GOLDILOCKS_TEST, full_cpu)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

    NTT_Goldilocks ntt(FFT_SIZE);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    TimerStart(LDE_MerkleTree_CPU);
    ntt.extendPol(c, a, FFT_SIZE<<BLOWUP_FACTOR, FFT_SIZE, NUM_COLUMNS, b);
    PoseidonGoldilocks::merkletree_avx(b, c, NUM_COLUMNS, FFT_SIZE<<BLOWUP_FACTOR);
    TimerStopAndLog(LDE_MerkleTree_CPU);

    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
