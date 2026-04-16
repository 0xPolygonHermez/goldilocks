// bench_ntt.mm — Metal vs CPU (NEON fast path) forward-NTT throughput.
//
// Single-column forward NTT across several domain sizes. The CPU NTT_iters
// has inline NEON fast paths on aarch64 (ntt_goldilocks.cpp:94+) so the CPU
// number here reflects the NEON path. There is no AVX NTT and no separate
// scalar NTT entry point — seq would require recompilation with NEON
// disabled, which this harness does not do.
//
// Correctness: for each size we also run one INTT(NTT(x)) round-trip and
// verify the CPU and Metal paths agree with the original input.
//
// Build (from this directory): make bench_ntt
// Run:                         ./bench_ntt

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
#include "../../src/metal/goldilocks_metal.hpp"

#ifndef MTL_KERNEL_DIR
#error "MTL_KERNEL_DIR must be defined (path to src/metal/kernels)"
#endif

using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

static std::string slurp(const std::string& p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static std::string strip_includes(const std::string& src) {
    std::stringstream in(src); std::stringstream out; std::string line;
    while (std::getline(in, line)) {
        auto p = line.find_first_not_of(" \t");
        if (p != std::string::npos && line[p] == '#' &&
            line.find("include", p) != std::string::npos) continue;
        out << line << "\n";
    }
    return out.str();
}

static void load_metallib_or_source() {
    MetalCtxHandle ctx = metal_context_get();
    if (metal_context_load_library(ctx, "./goldilocks.metallib") == 0) return;
    std::string dir = MTL_KERNEL_DIR;
    // All kernels referenced by NTT_Metal live in field + ntt (plus constants
    // which ntt doesn't need). Load poseidon too so merkletree_metal works if
    // someone reuses this context in tests.
    std::string src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") +
                      strip_includes(slurp(dir + "/field.metal")) +
                      slurp(dir + "/constants.metal.inc") +
                      strip_includes(slurp(dir + "/poseidon.metal")) +
                      strip_includes(slurp(dir + "/ntt.metal"));
    if (metal_context_load_source(ctx, src.c_str()) != 0) {
        fprintf(stderr, "bench_ntt: runtime source compile failed\n");
        std::abort();
    }
}

static void fill_sequence(Goldilocks::Element* a, uint64_t N, uint64_t ncols) {
    // a[row*ncols + col] = (row+1) + col*(N+1)  (deterministic, non-trivial)
    for (uint64_t r = 0; r < N; r++) {
        for (uint64_t c = 0; c < ncols; c++) {
            a[r * ncols + c] = Goldilocks::fromU64((r + 1) + c * (N + 1));
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

static void bench_one(uint64_t N, uint64_t ncols, int iters) {
    uint64_t total = N * ncols;
    std::vector<Goldilocks::Element> src(total), dst_cpu(total), dst_gpu(total);
    fill_sequence(src.data(), N, ncols);

    NTT_Goldilocks ntt(N);

    // Forward NTT timings
    double t_cpu = time_avg([&]{
        ntt.NTT(dst_cpu.data(), src.data(), N, ncols);
    }, iters);

    double t_gpu = time_avg([&]{
        ntt.NTT_Metal(dst_gpu.data(), src.data(), N, ncols, /*inverse=*/false);
    }, iters);

    bool fwd_match = (std::memcmp(dst_cpu.data(), dst_gpu.data(),
                                  total * sizeof(Goldilocks::Element)) == 0);

    // INTT timings (separate from forward). We time cpu INTT and gpu INTT on
    // the freshly-computed forward outputs. GPU INTT is in-place on a
    // working copy so repeated iterations see the same starting state.
    std::vector<Goldilocks::Element> intt_cpu_in = dst_cpu;
    std::vector<Goldilocks::Element> intt_cpu_out(total);
    double t_cpu_intt = time_avg([&]{
        ntt.INTT(intt_cpu_out.data(), intt_cpu_in.data(), N, ncols);
    }, iters);

    // Time INTT two ways: in-place (dst == src — caller-friendly, single-buffer)
    // and out-of-place (dst != src — enables the fused rev+s1+s2+s3 path that
    // saves a reverse_permutation pass). Both are content-independent, so we
    // don't restore the input between iterations.
    std::vector<Goldilocks::Element> intt_gpu_buf = dst_gpu;
    double t_gpu_intt_ip = time_avg([&]{
        ntt.NTT_Metal(intt_gpu_buf.data(), intt_gpu_buf.data(), N, ncols, /*inverse=*/true);
    }, iters);

    std::vector<Goldilocks::Element> intt_gpu_src = dst_gpu;
    std::vector<Goldilocks::Element> intt_gpu_dst(total);
    double t_gpu_intt_oop = time_avg([&]{
        ntt.NTT_Metal(intt_gpu_dst.data(), intt_gpu_src.data(), N, ncols, /*inverse=*/true);
    }, iters);

    // Round-trip correctness (separate from timing)
    std::vector<Goldilocks::Element> rt_cpu(total), rt_gpu(total);
    ntt.INTT(rt_cpu.data(), dst_cpu.data(), N, ncols);
    std::memcpy(rt_gpu.data(), dst_gpu.data(), total * sizeof(Goldilocks::Element));
    ntt.NTT_Metal(rt_gpu.data(), rt_gpu.data(), N, ncols, /*inverse=*/true);

    bool rt_cpu_ok = (std::memcmp(src.data(), rt_cpu.data(),
                                  total * sizeof(Goldilocks::Element)) == 0);
    bool rt_gpu_ok = (std::memcmp(src.data(), rt_gpu.data(),
                                  total * sizeof(Goldilocks::Element)) == 0);

    printf("\n=== NTT  N=2^%d (%llu)  ncols=%llu  (iters=%d) ===\n",
           (int)__builtin_ctzll(N),
           (unsigned long long)N,
           (unsigned long long)ncols,
           iters);
    printf("  cpu NTT             : %10.3f ms\n", t_cpu);
    printf("  metal NTT           : %10.3f ms   (%5.2fx vs cpu)\n",
           t_gpu, t_cpu / t_gpu);
    printf("  cpu INTT            : %10.3f ms\n", t_cpu_intt);
    printf("  metal INTT (in-pl)  : %10.3f ms   (%5.2fx vs cpu)\n",
           t_gpu_intt_ip,  t_cpu_intt / t_gpu_intt_ip);
    printf("  metal INTT (out-pl) : %10.3f ms   (%5.2fx vs cpu, %5.2fx vs in-pl)\n",
           t_gpu_intt_oop, t_cpu_intt / t_gpu_intt_oop,
           t_gpu_intt_ip / t_gpu_intt_oop);
    printf("  forward output bit-exact vs CPU : %s\n", fwd_match ? "match" : "DIVERGE");
    printf("  INTT(NTT(x)) == x  (cpu / gpu)  : %s / %s\n",
           rt_cpu_ok ? "ok" : "FAIL",
           rt_gpu_ok ? "ok" : "FAIL");
}

int main() {
    @autoreleasepool {
        load_metallib_or_source();
        printf("bench_ntt: Apple M-series Goldilocks NTT (cpu/NEON vs metal)\n");

        // Single-column sweep (dominated by CPU latency, shows GPU launch overhead)
        bench_one(1ULL << 14,   1, 30);   // 16k × 1
        bench_one(1ULL << 16,   1, 15);   // 64k × 1
        bench_one(1ULL << 18,   1,  8);   // 256k × 1
        bench_one(1ULL << 20,   1,  4);   // 1M × 1

        // Multi-column — where GPU parallelism actually pays off
        bench_one(1ULL << 14,  64, 10);   // 16k × 64
        bench_one(1ULL << 16,  64,  5);   // 64k × 64
        bench_one(1ULL << 18,  64,  3);   // 256k × 64
        bench_one(1ULL << 16, 256,  3);   // 64k × 256
        bench_one(1ULL << 18, 128,  2);   // 256k × 128
    }
    return 0;
}
