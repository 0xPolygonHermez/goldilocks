#include <benchmark/benchmark.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"
#include "../src/goldilocks_base_field_avx.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/poseidon_goldilocks_avx.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/merklehash_goldilocks.hpp"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

#include <math.h> /* ceil */
#include "omp.h"

#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

#define NUM_HASHES 10000

#define FFT_SIZE (1 << 23)
#define NUM_COLUMNS 669
#define BLOWUP_FACTOR 1
#define NPHASES_NTT 3
#define NPHASES_LDE 2
#define NBLOCKS 1
#define NCOLS_HASH 100
#define NROWS_HASH FFT_SIZE

//// perf counters

#define NUM_EVENTS 3
#define PERF_COUNT_HW_DTLB_LOAD_MISSES 0x08
#define PERF_COUNT_HW_DTLB_LOADS 0x01

struct perf_event_attr events[NUM_EVENTS];
int fds[NUM_EVENTS];

int init_perf_counters()
{
    // Set up events to monitor
    events[0].type = PERF_TYPE_HARDWARE;
    events[0].config = PERF_COUNT_HW_CPU_CYCLES;

    events[1].type = PERF_TYPE_HARDWARE;
    events[1].config = PERF_COUNT_HW_CACHE_MISSES;

    events[2].type = PERF_TYPE_HARDWARE;
    events[2].config = PERF_COUNT_HW_CACHE_REFERENCES;

    /*events[3].type = PERF_TYPE_HARDWARE;
    events[3].config = PERF_COUNT_HW_DTLB_LOAD_MISSES;

    events[4].type = PERF_TYPE_HARDWARE;
    events[4].config = PERF_COUNT_HW_DTLB_LOADS;

    events[5].type = PERF_TYPE_SOFTWARE;
    events[5].config = PERF_COUNT_SW_PAGE_FAULTS;*/

    // Create the perf counters
    for (int i = 0; i < NUM_EVENTS; i++)
    {
        fds[i] = syscall(__NR_perf_event_open, &events[i], 0, -1, -1, 0);
        if (fds[i] == -1)
        {
            perror("perf_event_open failed");
            return -1;
        }
    }

    return 0;
}

void read_perf_counters(long long counters[NUM_EVENTS])
{
    for (int i = 0; i < NUM_EVENTS; i++)
    {
        long long count;
        if (read(fds[i], &count, sizeof(long long)) == -1)
        {
            std::cout << " read failed" << i << std::endl;
        }
        else
        {
            counters[i] = count;
        }
    }
}

void start_perf_counters()
{
    // Start the perf counters
    for (int i = 0; i < NUM_EVENTS; i++)
    {
        ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
        ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
    }
}

void stop_perf_counters()
{
    // Stop the perf counters
    for (int i = 0; i < NUM_EVENTS; i++)
    {
        ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
    }
}

void close_perf_counters()
{
    for (int i = 0; i < NUM_EVENTS; i++)
    {
        close(fds[i]);
    }
}

//////////

static void POSEIDON_BENCH_FULL(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)SPONGE_WIDTH;

    Goldilocks::Element fibonacci[input_size];
    Goldilocks::Element result[input_size];

    // Test vector: Fibonacci series
    // 0 1 1 2 3 5 8 13 ... NUM_HASHES * SPONGE_WIDTH ...
    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();
    for (uint64_t i = 2; i < input_size; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }
    // Benchmark
    for (auto _ : state)
    {
        // Every thread process chunks of SPONGE_WIDTH elements
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            PoseidonGoldilocks::hash_full_result_seq_old((Goldilocks::Element(&)[SPONGE_WIDTH])result[i * SPONGE_WIDTH], (Goldilocks::Element(&)[SPONGE_WIDTH])fibonacci[i * SPONGE_WIDTH]);
        }
    }
    // Check poseidon results poseidon ( 0 1 1 2 3 5 8 13 21 34 55 89 )
    assert(Goldilocks::toU64(result[0]) == 0X3095570037F4605D);
    assert(Goldilocks::toU64(result[1]) == 0X3D561B5EF1BC8B58);
    assert(Goldilocks::toU64(result[2]) == 0X8129DB5EC75C3226);
    assert(Goldilocks::toU64(result[3]) == 0X8EC2B67AFB6B87ED);

    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void POSEIDON_BENCH_FULL_(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)SPONGE_WIDTH;

    Goldilocks::Element fibonacci[input_size];
    Goldilocks::Element result[input_size];

    // Test vector: Fibonacci series
    // 0 1 1 2 3 5 8 13 ... NUM_HASHES * SPONGE_WIDTH ...
    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();
    for (uint64_t i = 2; i < input_size; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }
    // Benchmark
    for (auto _ : state)
    {
        // Every thread process chunks of SPONGE_WIDTH elements
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            PoseidonGoldilocks::hash_full_result_seq((Goldilocks::Element(&)[SPONGE_WIDTH])result[i * SPONGE_WIDTH], (Goldilocks::Element(&)[SPONGE_WIDTH])fibonacci[i * SPONGE_WIDTH]);
        }
    }
    // Check poseidon results poseidon ( 0 1 1 2 3 5 8 13 21 34 55 89 )
    assert(Goldilocks::toU64(result[0]) == 0X3095570037F4605D);
    assert(Goldilocks::toU64(result[1]) == 0X3D561B5EF1BC8B58);
    assert(Goldilocks::toU64(result[2]) == 0X8129DB5EC75C3226);
    assert(Goldilocks::toU64(result[3]) == 0X8EC2B67AFB6B87ED);

    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void POSEIDON_BENCH_FULL_AVX(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)SPONGE_WIDTH;

    Goldilocks::Element fibonacci[input_size];
    Goldilocks::Element result[input_size];

    // Test vector: Fibonacci series
    // 0 1 1 2 3 5 8 13 ... NUM_HASHES * SPONGE_WIDTH ...
    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();
    for (uint64_t i = 2; i < input_size; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }
    // Benchmark
    for (auto _ : state)
    {
        // Every thread process chunks of SPONGE_WIDTH elements
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            PoseidonGoldilocks::hash_full_result((Goldilocks::Element(&)[SPONGE_WIDTH])result[i * SPONGE_WIDTH], (Goldilocks::Element(&)[SPONGE_WIDTH])fibonacci[i * SPONGE_WIDTH]);
        }
    }
    // Check poseidon results poseidon ( 0 1 1 2 3 5 8 13 21 34 55 89 )
    /*assert(Goldilocks::toU64(result[0]) == 0X3095570037F4605D);
    assert(Goldilocks::toU64(result[1]) == 0X3D561B5EF1BC8B58);
    assert(Goldilocks::toU64(result[2]) == 0X8129DB5EC75C3226);
    assert(Goldilocks::toU64(result[3]) == 0X8EC2B67AFB6B87ED);*/

    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void POSEIDON_BENCH(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)SPONGE_WIDTH;
    uint64_t result_size = (uint64_t)NUM_HASHES * (uint64_t)CAPACITY;

    Goldilocks::Element fibonacci[input_size];
    Goldilocks::Element result[result_size];

    // Test vector: Fibonacci series
    // 0 1 1 2 3 5 8 13 ... NUM_HASHES * SPONGE_WIDTH ...
    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();
    for (uint64_t i = 2; i < input_size; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }
    // Benchmark
    for (auto _ : state)
    {
        // Every thread process chunks of SPONGE_WIDTH elements
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            PoseidonGoldilocks::hash_seq((Goldilocks::Element(&)[CAPACITY])result[i * CAPACITY], (Goldilocks::Element(&)[SPONGE_WIDTH])fibonacci[i * SPONGE_WIDTH]);
        }
    }
    // Check poseidon results poseidon ( 0 1 1 2 3 5 8 13 21 34 55 89 )
    assert(Goldilocks::toU64(result[0]) == 0X3095570037F4605D);
    assert(Goldilocks::toU64(result[1]) == 0X3D561B5EF1BC8B58);
    assert(Goldilocks::toU64(result[2]) == 0X8129DB5EC75C3226);
    assert(Goldilocks::toU64(result[3]) == 0X8EC2B67AFB6B87ED);

    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void POSEIDON_BENCH_AVX(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)SPONGE_WIDTH;
    uint64_t result_size = (uint64_t)NUM_HASHES * (uint64_t)CAPACITY;

    Goldilocks::Element fibonacci[input_size];
    Goldilocks::Element result[result_size];

    // Test vector: Fibonacci series
    // 0 1 1 2 3 5 8 13 ... NUM_HASHES * SPONGE_WIDTH ...
    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();
    for (uint64_t i = 2; i < input_size; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }
    // Benchmark
    for (auto _ : state)
    {
        // Every thread process chunks of SPONGE_WIDTH elements
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            PoseidonGoldilocks::hash((Goldilocks::Element(&)[CAPACITY])result[i * CAPACITY], (Goldilocks::Element(&)[SPONGE_WIDTH])fibonacci[i * SPONGE_WIDTH]);
        }
    }
    // Check poseidon results poseidon ( 0 1 1 2 3 5 8 13 21 34 55 89 )
    assert(Goldilocks::toU64(result[0]) == 0X3095570037F4605D);
    assert(Goldilocks::toU64(result[1]) == 0X3D561B5EF1BC8B58);
    assert(Goldilocks::toU64(result[2]) == 0X8129DB5EC75C3226);
    assert(Goldilocks::toU64(result[3]) == 0X8EC2B67AFB6B87ED);

    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void LINEAR_HASH_BENCH(benchmark::State &state)
{
    Goldilocks::Element *cols = (Goldilocks::Element *)malloc((uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH * sizeof(Goldilocks::Element));
    Goldilocks::Element *result = (Goldilocks::Element *)malloc((uint64_t)HASH_SIZE * (uint64_t)NROWS_HASH * sizeof(Goldilocks::Element));

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS

    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    // Benchmark
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NROWS_HASH; i++)
        {
            Goldilocks::Element intermediate[NCOLS_HASH];
            Goldilocks::Element temp_result[HASH_SIZE];

            std::memcpy(&intermediate[0], &cols[i * NCOLS_HASH], NCOLS_HASH * sizeof(Goldilocks::Element));
            PoseidonGoldilocks::linear_hash_seq(temp_result, intermediate, NCOLS_HASH);
            std::memcpy(&result[i * HASH_SIZE], &temp_result[0], HASH_SIZE * sizeof(Goldilocks::Element));
        }
    }
    // Rate = time to process 1 linear hash per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((float)NROWS_HASH * (float)ceil((float)NCOLS_HASH / (float)RATE) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void LINEAR_HASH_BENCH_AVX(benchmark::State &state)
{
    Goldilocks::Element *cols = (Goldilocks::Element *)malloc((uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH * sizeof(Goldilocks::Element));
    Goldilocks::Element *result = (Goldilocks::Element *)malloc((uint64_t)HASH_SIZE * (uint64_t)NROWS_HASH * sizeof(Goldilocks::Element));

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS

    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    // Benchmark
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t i = 0; i < NROWS_HASH; i++)
        {
            Goldilocks::Element intermediate[NCOLS_HASH];
            Goldilocks::Element temp_result[HASH_SIZE];

            std::memcpy(&intermediate[0], &cols[i * NCOLS_HASH], NCOLS_HASH * sizeof(Goldilocks::Element));
            PoseidonGoldilocks::linear_hash(temp_result, intermediate, NCOLS_HASH);
            std::memcpy(&result[i * HASH_SIZE], &temp_result[0], HASH_SIZE * sizeof(Goldilocks::Element));
        }
    }
    // Rate = time to process 1 linear hash per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((float)NROWS_HASH * (float)ceil((float)NCOLS_HASH / (float)RATE) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void MERKLE_TREE_BENCH(benchmark::State &state)
{
    Goldilocks::Element *cols = (Goldilocks::Element *)malloc((uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH * sizeof(Goldilocks::Element));

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS
    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(NCOLS_HASH, NROWS_HASH);
    Goldilocks::Element *tree = (Goldilocks::Element *)malloc(numElementsTree * sizeof(Goldilocks::Element));

    // Benchmark
    for (auto _ : state)
    {
        PoseidonGoldilocks::merkletree_seq(tree, cols, NCOLS_HASH, NROWS_HASH);
    }
    Goldilocks::Element root[4];
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    free(cols);
    free(tree);
    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((((float)NROWS_HASH * (float)ceil((float)NCOLS_HASH / (float)RATE)) + log2(NROWS_HASH)) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void MERKLE_TREE_BENCH_AVX(benchmark::State &state)
{
    Goldilocks::Element *cols = (Goldilocks::Element *)malloc((uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH * sizeof(Goldilocks::Element));

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS
    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(NCOLS_HASH, NROWS_HASH);
    Goldilocks::Element *tree = (Goldilocks::Element *)malloc(numElementsTree * sizeof(Goldilocks::Element));

    // Benchmark
    for (auto _ : state)
    {
        PoseidonGoldilocks::merkletree(tree, cols, NCOLS_HASH, NROWS_HASH);
    }
    Goldilocks::Element root[4];
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    free(cols);
    free(tree);
    // Rate = time to process 1 posseidon per thread
    // BytesProcessed = total bytes processed per second on every iteration
    state.counters["Rate"] = benchmark::Counter((((float)NROWS_HASH * (float)ceil((float)NCOLS_HASH / (float)RATE)) + log2(NROWS_HASH)) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

static void NTT_BENCH(benchmark::State &state)
{
    NTT_Goldilocks gntt(FFT_SIZE, state.range(0));

    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS * sizeof(Goldilocks::Element));

#pragma omp parallel for
    for (uint64_t k = 0; k < NUM_COLUMNS; k++)
    {
        uint64_t offset = k * FFT_SIZE;
        a[offset] = Goldilocks::one();
        a[offset + 1] = Goldilocks::one();
        for (uint64_t i = 2; i < FFT_SIZE; i++)
        {
            a[i] = a[i - 1] + a[i - 2];
        }
    }
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0))
        for (u_int64_t i = 0; i < NUM_COLUMNS; i++)
        {
            u_int64_t offset = i * FFT_SIZE;
            gntt.NTT(a + offset, a + offset, FFT_SIZE);
        }
    }
    free(a);
}

static void NTT_BLOCK_BENCH(benchmark::State &state)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE, state.range(0));

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
    for (auto _ : state)
    {
        gntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, NPHASES_NTT, NBLOCKS);
    }
    free(a);
}

static void LDE_BENCH(benchmark::State &state)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE, state.range(0));
    NTT_Goldilocks gntt_extension((FFT_SIZE << BLOWUP_FACTOR));

    a[0] = Goldilocks::one();
    a[1] = Goldilocks::one();
    for (uint64_t i = 2; i < (uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS; i++)
    {
        a[i] = a[i - 1] + a[i - 2];
    }

    Goldilocks::Element shift = Goldilocks::fromU64(49); // TODO: ask for this number, where to put it how to calculate it
    gntt.INTT(a, a, FFT_SIZE);

    // TODO: This can be pre-generated
    Goldilocks::Element *r = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    r[0] = Goldilocks::one();
    for (int i = 1; i < FFT_SIZE; i++)
    {
        r[i] = r[i - 1] * shift;
    }

    Goldilocks::Element *zero_array = (Goldilocks::Element *)malloc((uint64_t)((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE) * sizeof(Goldilocks::Element));
#pragma omp parallel for num_threads(state.range(0))
    for (uint i = 0; i < ((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE); i++)
    {
        zero_array[i] = Goldilocks::zero();
    }

    for (auto _ : state)
    {
        Goldilocks::Element *res = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * (uint64_t)NUM_COLUMNS * sizeof(Goldilocks::Element));

#pragma omp parallel for num_threads(state.range(0))
        for (uint64_t k = 0; k < NUM_COLUMNS; k++)
        {
            for (int i = 0; i < FFT_SIZE; i++)
            {
                a[k * FFT_SIZE + i] = a[k * FFT_SIZE + i] * r[i];
            }
            std::memcpy(res, &a[k * FFT_SIZE], FFT_SIZE);
            std::memcpy(&res[FFT_SIZE], zero_array, (FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE);

            gntt_extension.NTT(res, res, (FFT_SIZE << BLOWUP_FACTOR));
        }
        free(res);
    }
    free(zero_array);
    free(a);
    free(r);
}

static void LDE_BLOCK_BENCH(benchmark::State &state)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE, state.range(0));
    NTT_Goldilocks gntt_extension((FFT_SIZE << BLOWUP_FACTOR));

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
    for (auto _ : state)
    {
        Goldilocks::Element shift = Goldilocks::fromU64(49); // TODO: ask for this number, where to put it how to calculate it

        gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, NPHASES_NTT);

        // TODO: This can be pre-generated
        Goldilocks::Element *r = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
        r[0] = Goldilocks::one();
        for (int i = 1; i < FFT_SIZE; i++)
        {
            r[i] = r[i - 1] * shift;
        }

#pragma omp parallel for
        for (uint64_t i = 0; i < FFT_SIZE; i++)
        {
            for (uint j = 0; j < NUM_COLUMNS; j++)
            {
                a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * i + j] * r[i];
            }
        }
#pragma omp parallel for schedule(static)
        for (uint64_t i = (uint64_t)FFT_SIZE * (uint64_t)NUM_COLUMNS; i < (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * (uint64_t)NUM_COLUMNS; i++)
        {
            a[i] = Goldilocks::zero();
        }

        gntt_extension.NTT(a, a, (FFT_SIZE << BLOWUP_FACTOR), NUM_COLUMNS, NULL, NPHASES_LDE, NBLOCKS);
        free(r);
    }
    free(a);
}

static void EXTENDEDPOL_BENCH(benchmark::State &state)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

    NTT_Goldilocks ntt(FFT_SIZE, state.range(0));

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
    for (auto _ : state)
    {
        ntt.extendPol(b, a, FFT_SIZE << BLOWUP_FACTOR, FFT_SIZE, NUM_COLUMNS, c);
    }
    free(a);
    free(b);
    free(c);
}

static void EXTENDEDPOL_BENCH_2(benchmark::State &state)
{

    // Initialize perf counters

    Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

    NTT_Goldilocks ntt(FFT_SIZE, state.range(0));
    ntt.computeR(FFT_SIZE);
    int aa = 0;
    long long aux = ((long long)FFT_SIZE << BLOWUP_FACTOR) * (long long)NUM_COLUMNS;

    init_perf_counters();
    start_perf_counters();

    for (long long i = 0; i < 1000; i++)
    {
        a += i;
    }
    stop_perf_counters();

    // Read perf counters
    long long counters[NUM_EVENTS];
    read_perf_counters(counters);
    /*for (uint i = 0; i < 2; i++)
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
    }*/

    for (auto _ : state)
    {
        aa += 1;
        // ntt.extendPol(b, a, FFT_SIZE << BLOWUP_FACTOR, FFT_SIZE, 1, c);
    }

    // Print results
    // printf("\nPage faults: %lld\n", counters[NUM_EVENTS]);
    // printf("cycles: %lld\n", counters[0]);
    printf("\ncache misses: %lld %f\n", counters[1], (double)counters[1] / (double)counters[2]);
    printf("cache references: %lld %lld\n", counters[2], (aux * 8) / (64 * 1024 * 1024));

    // printf("dTLB load misses: %lld\n", counters[2]);
    // printf("dTLB loads: %lld\n", counters[4]);
    close_perf_counters();
}

BENCHMARK(POSEIDON_BENCH_FULL)
    ->Unit(benchmark::kMicrosecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    //->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->DenseRange(1, 64, 63)
    ->UseRealTime();

BENCHMARK(POSEIDON_BENCH_FULL_)
    ->Unit(benchmark::kMicrosecond)
    /*->DenseRange(1, 1, 1)
    ->RangeMultiplier(2)
    ->Range(2, omp_get_max_threads())
    ->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)*/
    //->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->DenseRange(1, 64, 63)
    ->UseRealTime();

BENCHMARK(POSEIDON_BENCH_FULL_AVX)
    ->Unit(benchmark::kMicrosecond)
    /*->DenseRange(1, 1, 1)
    ->RangeMultiplier(2)
    ->Range(2, omp_get_max_threads())
    ->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)*/
    //->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->DenseRange(1, 64, 63)
    ->UseRealTime();

BENCHMARK(POSEIDON_BENCH)
    ->Unit(benchmark::kMicrosecond)
    /*->DenseRange(1, 1, 1)
    ->RangeMultiplier(2)
    ->Range(2, omp_get_max_threads())
    ->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)*/
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();

BENCHMARK(POSEIDON_BENCH_AVX)
    ->Unit(benchmark::kMicrosecond)
    /*->DenseRange(1, 1, 1)
    ->RangeMultiplier(2)
    ->Range(2, omp_get_max_threads())
    ->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)*/
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();

BENCHMARK(LINEAR_HASH_BENCH)
    ->Unit(benchmark::kMicrosecond)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();

BENCHMARK(LINEAR_HASH_BENCH_AVX)
    ->Unit(benchmark::kMicrosecond)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();

BENCHMARK(MERKLE_TREE_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();

BENCHMARK(MERKLE_TREE_BENCH_AVX)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();
BENCHMARK(NTT_BENCH)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();

BENCHMARK(NTT_BLOCK_BENCH)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();
BENCHMARK(LDE_BENCH)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();
BENCHMARK(LDE_BLOCK_BENCH)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads() / 2, 1)
    ->UseRealTime();

BENCHMARK(EXTENDEDPOL_BENCH)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads(), omp_get_max_threads(), 1)
    ->UseRealTime();

BENCHMARK(EXTENDEDPOL_BENCH_2)
    ->Unit(benchmark::kSecond)
    //->DenseRange(1, 1, 1)
    //->RangeMultiplier(2)
    //->Range(2, omp_get_max_threads())
    //->DenseRange(omp_get_max_threads() / 2 - 8, omp_get_max_threads() / 2 + 8, 2)
    ->DenseRange(omp_get_max_threads(), omp_get_max_threads(), 1)
    ->Iterations(1)
    ->UseRealTime();

BENCHMARK_MAIN();
// Build command: g++ benchs/bench.cpp src/* -lbenchmark -lomp -lpthread -lgmp  -std=c++17 -Wall -pthread -fopenmp -mavx2 -L$(find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//') -O3 -o bench && ./bench
