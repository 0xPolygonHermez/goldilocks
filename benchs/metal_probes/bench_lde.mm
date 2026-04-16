// bench_lde.mm — Metal vs CPU Low-Degree Extension (extendPol).
//
// Validates bit-exact parity between NTT_Goldilocks::extendPol and
// NTT_Goldilocks::extendPol_Metal over Fibonacci-seeded input, then
// measures the GPU speedup across STARK-prover shapes.
//
// Build (from this directory):
//   make bench_lde
// Run:
//   ./bench_lde               # medium shapes
//   ./bench_lde --big         # add the LDE_BENCH (2^23 × 100) shape

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "../../src/platform.hpp"
#include "../../src/goldilocks_base_field.hpp"
#include "../../src/ntt_goldilocks.hpp"
#include "../../src/metal/metal_context.hpp"

#ifndef MTL_KERNEL_DIR
#error "MTL_KERNEL_DIR must be defined (path to src/metal/kernels)"
#endif

using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static std::string strip_includes(const std::string& src) {
    std::stringstream in(src), out;
    std::string line;
    while (std::getline(in, line)) {
        auto p = line.find_first_not_of(" \t");
        if (p != std::string::npos && line[p] == '#') {
            auto q = line.find("include", p);
            if (q != std::string::npos) continue;
        }
        out << line << "\n";
    }
    return out.str();
}

static void load_metallib_or_source() {
    MetalCtxHandle ctx = metal_context_get();
    if (metal_context_load_library(ctx, "./goldilocks.metallib") == 0) return;
    std::string dir = MTL_KERNEL_DIR;
    std::string src = strip_includes(slurp(dir + "/field.metal")) +
                      strip_includes(slurp(dir + "/ntt.metal"));
    src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + src;
    if (metal_context_load_source(ctx, src.c_str()) != 0) {
        fprintf(stderr, "bench_lde: runtime source compile failed\n");
        std::abort();
    }
}

static void fill_fibonacci(Goldilocks::Element* buf, uint64_t N, uint64_t ncols) {
    for (uint64_t c = 0; c < ncols; c++) buf[c] = Goldilocks::one();
    if (N >= 2) {
        for (uint64_t c = 0; c < ncols; c++) buf[ncols + c] = Goldilocks::one();
    }
    for (uint64_t i = 2; i < N; i++) {
        for (uint64_t c = 0; c < ncols; c++) {
            buf[i * ncols + c] = buf[(i - 1) * ncols + c] + buf[(i - 2) * ncols + c];
        }
    }
}

template <class Fn>
static double time_avg(Fn&& fn, int iters) {
    fn();  // warm-up
    double total = 0.0;
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        total += ms(t1 - t0).count();
    }
    return total / iters;
}

static void bench_one(uint64_t N, uint64_t N_Ext, uint64_t ncols, int iters) {
    uint64_t in_total  = N * ncols;
    uint64_t out_total = N_Ext * ncols;

    std::vector<Goldilocks::Element> input(in_total);
    std::vector<Goldilocks::Element> cpu_out(out_total);
    std::vector<Goldilocks::Element> gpu_out(out_total);
    std::vector<Goldilocks::Element> cpu_buf(out_total);  // scratch for cpu extendPol

    fill_fibonacci(input.data(), N, ncols);

    NTT_Goldilocks ntt(N);

    double t_cpu = time_avg([&]{
        ntt.extendPol(cpu_out.data(), input.data(), N_Ext, N, ncols, cpu_buf.data());
    }, iters);

    double t_gpu = time_avg([&]{
        ntt.extendPol_Metal(gpu_out.data(), input.data(), N_Ext, N, ncols);
    }, iters);

    bool match = (std::memcmp(cpu_out.data(), gpu_out.data(),
                              out_total * sizeof(Goldilocks::Element)) == 0);

    printf("\n=== extendPol  N=2^%d → 2^%d  ncols=%llu  (iters=%d) ===\n",
           (int)__builtin_ctzll(N),
           (int)__builtin_ctzll(N_Ext),
           (unsigned long long)ncols, iters);
    printf("  cpu    : %10.3f ms\n", t_cpu);
    printf("  metal  : %10.3f ms   (%5.2fx vs cpu)\n", t_gpu, t_cpu / t_gpu);
    printf("  output bit-exact vs cpu : %s\n", match ? "match" : "DIVERGE");
}

int main(int argc, char** argv) {
    @autoreleasepool {
        load_metallib_or_source();
        printf("bench_lde: Apple M-series Goldilocks extendPol (cpu vs metal)\n");

        bool big = (argc > 1 && std::string(argv[1]) == "--big");

        // Small / medium shapes — exercise correctness and per-call overhead.
        bench_one(1ULL << 14, 1ULL << 15,   8, 10);   // N=16k  → 32k,   8 cols
        bench_one(1ULL << 16, 1ULL << 17,  16,  5);   // N=64k  → 128k, 16 cols
        bench_one(1ULL << 18, 1ULL << 19,  64,  3);   // N=256k → 512k, 64 cols
        bench_one(1ULL << 18, 1ULL << 19, 128,  3);   // N=256k → 512k, 128 cols (STARK shape)
        bench_one(1ULL << 20, 1ULL << 21,  32,  2);   // N=1M   → 2M,   32 cols

        if (big) {
            printf("\n--- LDE_BENCH scale: N=2^23 → 2^24, 100 cols (~12 GB output) ---\n");
            // Memory: input 6.25 GiB + output 12.5 GiB + cpu scratch 12.5 GiB ≈ 31 GiB.
            // Skippable if your machine doesn't have enough RAM.
            bench_one(1ULL << 23, 1ULL << 24, 100, 1);
        }
    }
    return 0;
}
