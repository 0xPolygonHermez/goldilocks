// probe_kernel_gpu_time.mm — time merkle_leaves ONLY (no parents).
//
// Isolates the leaf-kernel cost from the level-by-level parent dispatches.
// For 262k rows with log2(262144)=18 parent levels, we want to know if
// the 245ms total is dominated by leaves (pod12 × 16 absorb iters × 262k)
// or parents (pod12 × 262k + 131k + 65k + ... ≈ 524k pod12s total).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "../../src/platform.hpp"
#include "../../src/goldilocks_base_field.hpp"
#include "../../src/metal/metal_context.hpp"

using Clock = std::chrono::high_resolution_clock;
using ms = std::chrono::duration<double, std::milli>;

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

static double time_leaves(const char* variant, uint64_t ncols, uint64_t nrows, int iters) {
    uint64_t total_in = ncols * nrows;
    std::vector<uint64_t> inp(total_in);
    for (uint64_t i = 0; i < total_in; i++) inp[i] = i + 1;
    std::vector<uint64_t> tree(nrows * 4, 0);

    MetalCtxHandle ctx = metal_context_get();
    int is_copy_in = 0, is_copy_tree = 0;
    MetalBufHandle in_buf   = metal_buf_alias(ctx, inp.data(),
                                               total_in * sizeof(uint64_t), &is_copy_in);
    MetalBufHandle tree_buf = metal_buf_alias(ctx, tree.data(),
                                               nrows * 4 * sizeof(uint64_t), &is_copy_tree);

    uint32_t nc32 = (uint32_t)ncols;
    uint32_t d32  = 1;
    uint32_t nr32 = (uint32_t)nrows;

    bool simd = (std::string(variant) == "merkle_leaves_simd");

    // Warm-up
    if (simd) metal_dispatch_merkle_leaves_simd(ctx, in_buf, tree_buf, nc32, d32, nr32);
    else      metal_dispatch_merkle_leaves     (ctx, in_buf, tree_buf, nc32, d32, nr32);

    double total = 0.0;
    for (int it = 0; it < iters; it++) {
        auto t0 = Clock::now();
        if (simd) metal_dispatch_merkle_leaves_simd(ctx, in_buf, tree_buf, nc32, d32, nr32);
        else      metal_dispatch_merkle_leaves     (ctx, in_buf, tree_buf, nc32, d32, nr32);
        auto t1 = Clock::now();
        total += ms(t1 - t0).count();
    }

    metal_buf_release(in_buf);
    metal_buf_release(tree_buf);
    return total / iters;
}

int main() {
    @autoreleasepool {
        load_lib();
        printf("probe_kernel_gpu_time: merkle_leaves ONLY (no parents)\n");
        printf("Compare to bench_merkle's full-merkle number to see if parents matter.\n");

        struct { uint64_t nr; int it; } cases[] = {
            { 64,     50 },
            { 4096,   20 },
            { 65536,   5 },
            { 262144,  3 },
        };
        for (auto& c : cases) {
            double t_row  = time_leaves("merkle_leaves",      128, c.nr, c.it);
            double t_coop = time_leaves("merkle_leaves_simd", 128, c.nr, c.it);
            printf("\nncols=128 nrows=%llu  (iters=%d)\n",
                   (unsigned long long)c.nr, c.it);
            printf("  leaves only (row)  : %8.3f ms\n", t_row);
            printf("  leaves only (coop) : %8.3f ms\n", t_coop);
        }
    }
    return 0;
}
