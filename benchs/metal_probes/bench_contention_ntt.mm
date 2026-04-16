// bench_contention_ntt.mm — measure CPU NEON + GPU Metal contention on NTT.
//
// NTT is memory-bandwidth-bound on M4 Pro (every butterfly phase reads and
// writes the whole domain). If CPU NEON and GPU Metal share unified memory
// bandwidth, running them concurrently could serialize through the memory
// controller. This probe quantifies that.
//
// Experiment mirrors bench_contention_merkle: two independent inputs A and B
// of identical shape, time them alone and concurrent, report overlap
// efficiency.
//
// Build: make bench_contention_ntt

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
#include "../../src/ntt_goldilocks.hpp"
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
                      strip_includes(slurp(dir + "/ntt.metal"));
    src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + src;
    if (metal_context_load_source(ctx, src.c_str()) != 0) std::abort();
}

static void fill_seq(Goldilocks::Element* buf, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) buf[i] = Goldilocks::fromU64(i + 1);
}

static void bench_shape(uint64_t N, uint64_t ncols, int iters) {
    uint64_t total = N * ncols;
    std::vector<Goldilocks::Element> in_A(total), in_B(total);
    std::vector<Goldilocks::Element> out_A(total), out_B(total);
    fill_seq(in_A.data(), total);
    fill_seq(in_B.data(), total);

    NTT_Goldilocks ntt(N);
    // Warm-up (pipeline compile + twiddle upload + first-touch allocation).
    ntt.NTT_Metal(out_A.data(), in_A.data(), N, ncols, /*inverse=*/false);
    ntt.NTT      (out_B.data(), in_B.data(), N, ncols);

    double T_metal_alone = 0, T_neon_alone = 0, T_concurrent = 0;

    for (int it = 0; it < iters; it++) {
        auto t0 = Clock::now();
        ntt.NTT_Metal(out_A.data(), in_A.data(), N, ncols, false);
        auto t1 = Clock::now();
        T_metal_alone += ms(t1 - t0).count();

        auto t2 = Clock::now();
        ntt.NTT(out_B.data(), in_B.data(), N, ncols);
        auto t3 = Clock::now();
        T_neon_alone += ms(t3 - t2).count();

        auto t4 = Clock::now();
        std::thread metal_thread([&]{
            ntt.NTT_Metal(out_A.data(), in_A.data(), N, ncols, false);
        });
        ntt.NTT(out_B.data(), in_B.data(), N, ncols);
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
    double eff      = (T_serial - T_concurrent) / T_min;

    printf("\n=== NTT  N=2^%d (%llu)  ncols=%llu  (iters=%d) ===\n",
           (int)__builtin_ctzll(N),
           (unsigned long long)N, (unsigned long long)ncols, iters);
    printf("  metal alone        : %9.3f ms\n", T_metal_alone);
    printf("  neon  alone        : %9.3f ms\n", T_neon_alone);
    printf("  serial sum         : %9.3f ms\n", T_serial);
    printf("  concurrent         : %9.3f ms\n", T_concurrent);
    printf("  max(metal, neon)   : %9.3f ms  (concurrent /max = %.2fx)\n",
           T_max, T_concurrent / T_max);
    printf("  overlap efficiency : %6.2f   (1.0 = perfect, 0.0 = no overlap)\n",
           eff);
    if (eff > 0.9) {
        printf("  verdict            : GREAT — hybrid worth implementing\n");
    } else if (eff > 0.5) {
        printf("  verdict            : GOOD — partial gain available\n");
    } else if (eff > 0.1) {
        printf("  verdict            : WEAK — memory-bandwidth contention is real\n");
    } else {
        printf("  verdict            : NO WIN — fully memory-bound, hybrid useless\n");
    }
}

int main() {
    @autoreleasepool {
        load_lib();
        printf("bench_contention_ntt: CPU NEON vs GPU Metal under unified-memory pressure\n");
        bench_shape(1ULL << 18,  64, 3);
        bench_shape(1ULL << 18, 128, 2);
        bench_shape(1ULL << 20,   1, 3);
    }
    return 0;
}
