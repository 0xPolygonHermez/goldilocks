// Microbenchmark: GLSimd<Neon>::mul vs scalar Goldilocks::mul.
// Walltime-based, no google-benchmark dependency.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include "../src/simd/goldilocks_simd.hpp"

#ifndef GOLDILOCKS_HAS_NEON
int main() { std::puts("NEON unavailable"); return 0; }
#else

using namespace goldilocks::simd;
using N = GLSimd<Neon>;

int main() {
    const uint64_t N_ITERS = 10'000'000ULL;

    // Scalar baseline
    {
        Goldilocks::Element a{0x123456789abcdef0ULL}, b{0xfedcba9876543210ULL}, c;
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < N_ITERS; ++i) {
            Goldilocks::mul(c, a, b);
            a.fe = c.fe ^ 0x9E3779B97F4A7C15ULL;  // prevent optimization
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::printf("Scalar mul:  %10lld pairs in %.3fs = %.2f M pairs/s  (sink=%016llx)\n",
                    (long long)N_ITERS, sec, N_ITERS / sec / 1e6, (unsigned long long)c.fe);
    }

    // NEON (2 pairs per iteration)
    {
        auto va = N::set(0x123456789abcdef0ULL, 0xcafebabe01234567ULL);
        auto vb = N::set(0xfedcba9876543210ULL, 0x0fedcba987654321ULL);
        auto t0 = std::chrono::steady_clock::now();
        Goldilocks::Element sink[2];
        for (uint64_t i = 0; i < N_ITERS; ++i) {
            auto vc = N::mul(va, vb);
            va = veorq_u64(vc, vdupq_n_u64(0x9E3779B97F4A7C15ULL));
        }
        N::store(sink, va);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::printf("NEON   mul:  %10lld pairs in %.3fs = %.2f M pairs/s  (sink=%016llx,%016llx)\n",
                    (long long)(N_ITERS * 2), sec, (N_ITERS * 2) / sec / 1e6,
                    (unsigned long long)sink[0].fe, (unsigned long long)sink[1].fe);
    }

    return 0;
}
#endif
