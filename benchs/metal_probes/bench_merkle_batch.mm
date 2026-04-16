// bench_merkle_batch.mm — sequential vs batched Merkle (N trees).
//
// Compares:
//   A) K sequential merkletree_metal calls (each with its own commit+wait)
//   B) One merkletree_metal_batch(trees[], inputs[], K) call (one wait at end)
//
// Expected win: GPU scheduler can overlap tree N+1's leaf kernel with
// tree N's parent loop since they touch distinct buffers and live in
// separate compute encoders within the same command buffer.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "../../src/platform.hpp"
#include "../../src/goldilocks_base_field.hpp"
#include "../../src/poseidon_goldilocks.hpp"
#include "../../src/merklehash_goldilocks.hpp"
#include "../../src/metal/metal_context.hpp"
#include "../../src/metal/goldilocks_metal.hpp"

using Clock = std::chrono::high_resolution_clock;
using ms_t  = std::chrono::duration<double, std::milli>;

#ifndef MTL_KERNEL_DIR
#error "MTL_KERNEL_DIR must be defined"
#endif

static std::string slurp(const std::string& p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::string strip_inc(const std::string& src) {
    std::stringstream in(src); std::stringstream out; std::string ln;
    while (std::getline(in, ln)) {
        auto p = ln.find_first_not_of(" \t");
        if (p != std::string::npos && ln[p] == '#' &&
            ln.find("include", p) != std::string::npos) continue;
        out << ln << "\n";
    }
    return out.str();
}
static void load_lib() {
    MetalCtxHandle ctx = metal_context_get();
    std::string d = MTL_KERNEL_DIR;
    std::string src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") +
                      strip_inc(slurp(d + "/field.metal")) +
                      slurp(d + "/constants.metal.inc") +
                      strip_inc(slurp(d + "/poseidon.metal"));
    if (metal_context_load_source(ctx, src.c_str()) != 0) std::abort();
}

static void build_fibonacci(Goldilocks::Element* cols,
                            uint64_t ncols, uint64_t nrows, uint64_t seed) {
    for (uint64_t i = 0; i < ncols; i++) {
        cols[i]            = Goldilocks::fromU64(i + seed) + Goldilocks::one();
        cols[i + ncols]    = Goldilocks::fromU64(i + seed) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < nrows; j++) {
        for (uint64_t i = 0; i < ncols; i++) {
            cols[j * ncols + i] =
                cols[(j - 2) * ncols + i] + cols[(j - 1) * ncols + i];
        }
    }
}

static void bench_one(uint64_t ncols, uint64_t nrows, uint64_t count, int iters) {
    uint64_t n_elem = ncols * nrows;
    uint64_t n_tree = MerklehashGoldilocks::getTreeNumElements(nrows);

    // Allocate K input + tree buffers. Use aligned to take zero-copy path.
    std::vector<Goldilocks::Element*> inputs(count), trees(count);
    for (uint64_t k = 0; k < count; k++) {
        inputs[k] = goldilocks_metal::allocate_aligned_elements(n_elem);
        trees[k]  = goldilocks_metal::allocate_aligned_elements(n_tree);
        build_fibonacci(inputs[k], ncols, nrows, k * 7919);  // different per tree
    }

    // Warm-up
    for (uint64_t k = 0; k < count; k++) {
        PoseidonGoldilocks::merkletree_metal(trees[k], inputs[k], ncols, nrows);
    }

    // A) Sequential
    double seq_total = 0.0;
    for (int it = 0; it < iters; it++) {
        auto t0 = Clock::now();
        for (uint64_t k = 0; k < count; k++) {
            PoseidonGoldilocks::merkletree_metal(trees[k], inputs[k], ncols, nrows);
        }
        auto t1 = Clock::now();
        seq_total += ms_t(t1 - t0).count();
    }
    double seq_ms = seq_total / iters;

    // Snapshot roots from sequential run for correctness check.
    std::vector<uint64_t> roots_seq(count);
    for (uint64_t k = 0; k < count; k++) {
        roots_seq[k] = Goldilocks::toU64(trees[k][n_tree - 4]);
    }

    // B) Batched
    double batch_total = 0.0;
    for (int it = 0; it < iters; it++) {
        auto t0 = Clock::now();
        goldilocks_metal::merkletree_metal_batch(trees.data(), inputs.data(),
                                                   count, ncols, nrows);
        auto t1 = Clock::now();
        batch_total += ms_t(t1 - t0).count();
    }
    double batch_ms = batch_total / iters;

    // Correctness: roots from batched must match sequential for every tree.
    bool match = true;
    for (uint64_t k = 0; k < count; k++) {
        uint64_t r = Goldilocks::toU64(trees[k][n_tree - 4]);
        if (r != roots_seq[k]) { match = false; break; }
    }

    printf("ncols=%llu nrows=%llu count=%llu iters=%d\n",
           (unsigned long long)ncols, (unsigned long long)nrows,
           (unsigned long long)count, iters);
    printf("  sequential : %10.3f ms  (avg %7.3f ms/tree)\n",
           seq_ms, seq_ms / count);
    printf("  batched    : %10.3f ms  (avg %7.3f ms/tree)  (%5.2fx vs seq, check=%s)\n",
           batch_ms, batch_ms / count, seq_ms / batch_ms,
           match ? "match" : "DIVERGE");

    for (uint64_t k = 0; k < count; k++) {
        goldilocks_metal::free_aligned(inputs[k]);
        goldilocks_metal::free_aligned(trees[k]);
    }
}

int main() {
    @autoreleasepool {
        load_lib();
        printf("bench_merkle_batch: sequential vs batched Merkle (Apple M4 Pro)\n\n");
        bench_one(128,  4096,   8, 5);
        bench_one(128, 65536,   4, 3);
        bench_one(128, 65536,   8, 2);
        bench_one(128, 262144,  4, 2);
    }
    return 0;
}
