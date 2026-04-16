// Probe 3: real SME i16 × i16 → i64 outer product accumulation + timing.
// Measures whether SME matmul is fast enough to beat our NEON Phase 1l.
//
// Build: clang++ -std=c++17 -O2 -march=armv9-a+sme+sme-i16i64 -mcpu=apple-m4 \
//                03_i16_i64_matmul.cpp -o 03_i16_i64_matmul
// Run:   ./03_i16_i64_matmul    or    sudo ./03_i16_i64_matmul
//
// On M4 Pro: streaming vector length = 512 bits → 32 lanes of i16, 8 lanes of i64.
// Each svmopa_za64_s16_m accumulates a 32-wide × 32-wide outer product of i16
// values into the 8×8 i64 ZA tile. One instruction = 256 i16×i16 muls with
// i64 accumulation.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <arm_sme.h>

__arm_new("za") __arm_locally_streaming
static uint64_t run_sme_outer_products(int iterations,
                                        const int16_t* A, const int16_t* B,
                                        int64_t* out)
{
    svbool_t pg16 = svptrue_b16();
    svbool_t pg64 = svptrue_b64();
    svint16_t a_vec = svld1_s16(pg16, A);
    svint16_t b_vec = svld1_s16(pg16, B);
    svzero_za();

    for (int i = 0; i < iterations; ++i) {
        // ZA.D[0] += A × B  (i16 × i16 → i64, one outer product)
        svmopa_za64_s16_m(0, pg64, pg64, a_vec, b_vec);
    }

    // Pull the first row of the 8-wide i64 tile out to 'out'
    svst1_hor_za64(0, 0, pg64, out);
    return (uint64_t)out[0];
}

int main() {
    printf("probe 3: SME i16×i16→i64 outer-product timing\n");
    fflush(stdout);

    alignas(64) int16_t A[32], B[32];
    alignas(64) int64_t out[8];
    for (int i = 0; i < 32; ++i) { A[i] = i * 3 + 1; B[i] = i * 5 + 7; }

    // Warm up
    run_sme_outer_products(100, A, B, out);
    printf("  warm-up ok, out[0]=%lld\n", (long long)out[0]);

    // Time varying iteration counts
    for (int iters : {1000, 10000, 100000}) {
        const int REP = 20;
        auto t0 = std::chrono::steady_clock::now();
        uint64_t sink = 0;
        for (int r = 0; r < REP; ++r) {
            sink ^= run_sme_outer_products(iters, A, B, out);
        }
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double ns_per = ns / (iters * REP);
        printf("  %d outer-products: %.2f ns each (%.2f M ops/s)  sink=%llx\n",
               iters, ns_per, 1000.0 / ns_per, (unsigned long long)sink);
    }

    // Baseline for comparison: scalar 64x64→128 mul
    const int N = 1'000'000;
    volatile uint64_t x = 0x123456789abcdef0ULL;
    volatile uint64_t y = 0xfedcba9876543210ULL;
    uint64_t acc = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        uint64_t xx = x, yy = y;
        __uint128_t p = (__uint128_t)xx * yy;
        uint64_t lo = (uint64_t)p;
        uint64_t hi = (uint64_t)(p >> 64);
        acc ^= lo ^ hi;
        x = lo; y = hi;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns_mul = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
    printf("  scalar 64x64→128 mul: %.2f ns each (%.2f M ops/s)  sink=%llx\n",
           ns_mul, 1000.0 / ns_mul, (unsigned long long)acc);

    return 0;
}
