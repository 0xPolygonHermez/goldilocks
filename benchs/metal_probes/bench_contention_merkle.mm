// bench_contention_merkle.mm — measure CPU NEON + GPU Metal contention on
// Merkle-tree builds.
//
// Merkle is compute-bound on M4 Pro (37% ALU, 42% occupancy per prior capture),
// so CPU NEON and GPU Metal ALUs should not contend for the same execution
// resources. But they DO share unified memory bandwidth. This probe measures
// whether running both concurrently on disjoint trees hurts, breaks even, or
// helps vs running them serially.
//
// Experiment design:
//   Allocate two independent trees A and B with the same shape.
//   Measure:
//     T_metal_alone  = time(metal builds A, then waits)
//     T_neon_alone   = time(neon  builds B)
//     T_serial       = T_metal_alone + T_neon_alone
//     T_concurrent   = time(metal commits A + neon builds B + waitMetal)
//
//   If T_concurrent ≈ max(T_metal_alone, T_neon_alone)  → perfect overlap (hybrid great)
//   If T_concurrent ≈ T_serial                           → total contention  (hybrid bad)
//   Between those two extremes: partial overlap (hybrid is a judgment call).
//
// We report the overlap efficiency:
//   eff = (T_serial - T_concurrent) / min(T_metal_alone, T_neon_alone) ∈ [0, 1]
//   eff = 1.0  → the faster task ran for free under the slower task
//   eff = 0.0  → no overlap at all
//
// Build:
//   make bench_contention_merkle
// Run:
//   ./bench_contention_merkle

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <thread>
#include <string>
#include <fstream>
#include <sstream>

#include "../../src/platform.hpp"
#include "../../src/goldilocks_base_field.hpp"
#include "../../src/poseidon_goldilocks.hpp"
#include "../../src/merklehash_goldilocks.hpp"
#include "../../src/metal/metal_context.hpp"
#include "../../src/metal/goldilocks_metal.hpp"

#ifndef MTL_KERNEL_DIR
#error "MTL_KERNEL_DIR must be defined"
#endif

using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

static std::string slurp(const std::string& p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::string strip_includes(const std::string& s) {
    std::stringstream in(s), out; std::string line;
    while (std::getline(in, line)) {
        auto p = line.find_first_not_of(" \t");
        if (p != std::string::npos && line[p] == '#'
            && line.find("include", p) != std::string::npos) continue;
        out << line << "\n";
    }
    return out.str();
}
static void load_lib() {
    MetalCtxHandle ctx = metal_context_get();
    if (metal_context_load_library(ctx, "./goldilocks.metallib") == 0) return;
    std::string dir = MTL_KERNEL_DIR;
    std::string src = strip_includes(slurp(dir + "/field.metal")) +
                      slurp(dir + "/constants.metal.inc") +
                      strip_includes(slurp(dir + "/poseidon.metal"));
    src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + src;
    if (metal_context_load_source(ctx, src.c_str()) != 0) std::abort();
}

static void fib_fill(Goldilocks::Element* cols, uint64_t ncols, uint64_t nrows) {
    for (uint64_t i = 0; i < ncols; i++) {
        cols[i]         = Goldilocks::fromU64(i + 1);
        cols[i + ncols] = Goldilocks::fromU64(i + 2);
    }
    for (uint64_t j = 2; j < nrows; j++) {
        for (uint64_t i = 0; i < ncols; i++) {
            cols[j * ncols + i] =
                cols[(j - 2) * ncols + i] + cols[(j - 1) * ncols + i];
        }
    }
}

struct Buffers {
    std::vector<Goldilocks::Element> cols_A, cols_B;
    std::vector<Goldilocks::Element> tree_A, tree_B;
};

static Buffers make_buffers(uint64_t ncols, uint64_t nrows) {
    uint64_t n_elem = ncols * nrows;
    uint64_t n_tree = MerklehashGoldilocks::getTreeNumElements(nrows);
    Buffers b;
    b.cols_A.resize(n_elem);
    b.cols_B.resize(n_elem);
    b.tree_A.resize(n_tree);
    b.tree_B.resize(n_tree);
    fib_fill(b.cols_A.data(), ncols, nrows);
    fib_fill(b.cols_B.data(), ncols, nrows);
    return b;
}

static void bench_shape(uint64_t ncols, uint64_t nrows, int iters) {
    auto b = make_buffers(ncols, nrows);
    // Warm-up once (primes pipeline + caches).
    PoseidonGoldilocks::merkletree_metal(b.tree_A.data(), b.cols_A.data(),
                                          ncols, nrows);
    PoseidonGoldilocks::merkletree_neon (b.tree_B.data(), b.cols_B.data(),
                                          ncols, nrows);

    double T_metal_alone = 0, T_neon_alone = 0, T_concurrent = 0;

    for (int it = 0; it < iters; it++) {
        // Metal alone
        auto t0 = Clock::now();
        PoseidonGoldilocks::merkletree_metal(b.tree_A.data(), b.cols_A.data(),
                                              ncols, nrows);
        auto t1 = Clock::now();
        T_metal_alone += ms(t1 - t0).count();

        // NEON alone
        auto t2 = Clock::now();
        PoseidonGoldilocks::merkletree_neon(b.tree_B.data(), b.cols_B.data(),
                                             ncols, nrows);
        auto t3 = Clock::now();
        T_neon_alone += ms(t3 - t2).count();

        // Concurrent: run NEON on the calling thread while Metal runs on a
        // helper thread. The Metal bridge's internal waitUntilCompleted is
        // inside that helper — this is the simplest way to get true
        // parallelism with the current synchronous API.
        auto t4 = Clock::now();
        std::thread metal_thread([&]{
            PoseidonGoldilocks::merkletree_metal(b.tree_A.data(), b.cols_A.data(),
                                                  ncols, nrows);
        });
        PoseidonGoldilocks::merkletree_neon(b.tree_B.data(), b.cols_B.data(),
                                             ncols, nrows);
        metal_thread.join();
        auto t5 = Clock::now();
        T_concurrent += ms(t5 - t4).count();
    }
    T_metal_alone /= iters;
    T_neon_alone  /= iters;
    T_concurrent  /= iters;

    double T_serial = T_metal_alone + T_neon_alone;
    double T_max    = std::max(T_metal_alone, T_neon_alone);
    double T_min    = std::min(T_metal_alone, T_neon_alone);
    double eff      = (T_serial - T_concurrent) / T_min;  // ∈ [0, 1] if perfect

    printf("\n=== Merkle  ncols=%llu  nrows=%llu  (iters=%d) ===\n",
           (unsigned long long)ncols, (unsigned long long)nrows, iters);
    printf("  metal alone        : %9.3f ms\n", T_metal_alone);
    printf("  neon  alone        : %9.3f ms\n", T_neon_alone);
    printf("  serial sum         : %9.3f ms\n", T_serial);
    printf("  concurrent         : %9.3f ms\n", T_concurrent);
    printf("  max(metal, neon)   : %9.3f ms  (concurrent /max = %.2fx)\n",
           T_max, T_concurrent / T_max);
    printf("  overlap efficiency : %6.2f   (1.0 = perfect, 0.0 = no overlap)\n",
           eff);
    if (eff > 0.9) {
        printf("  verdict            : GREAT — hybrid would nearly double throughput\n");
    } else if (eff > 0.5) {
        printf("  verdict            : GOOD — hybrid worth pursuing\n");
    } else if (eff > 0.1) {
        printf("  verdict            : WEAK — hybrid has partial contention\n");
    } else {
        printf("  verdict            : NO WIN — total contention, hybrid useless here\n");
    }
}

int main() {
    @autoreleasepool {
        load_lib();
        printf("bench_contention_merkle: CPU NEON vs GPU Metal under unified-memory pressure\n");
        bench_shape(128,   4096,   20);
        bench_shape(128,  65536,    5);
        bench_shape(128, 262144,    3);
    }
    return 0;
}
