// Fuzz harness for GLSimd<Neon>::mul / ::square.
// Compares bit-for-bit against scalar Goldilocks::mul / ::square over
// edge values + 10k random pairs.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include "../src/simd/goldilocks_simd.hpp"

#ifndef GOLDILOCKS_HAS_NEON
int main() { std::puts("NEON not available; check_neon_mul skipped."); return 0; }
#else

using namespace goldilocks::simd;
using N = GLSimd<Neon>;

static uint64_t scalar_mul(uint64_t a, uint64_t b) {
    Goldilocks::Element ea, eb, ec;
    ea.fe = a; eb.fe = b;
    Goldilocks::mul(ec, ea, eb);
    return ec.fe;
}

static uint64_t scalar_square(uint64_t a) {
    Goldilocks::Element ea, ec;
    ea.fe = a;
    Goldilocks::square(ec, ea);
    return ec.fe;
}

static bool check_mul(uint64_t a, uint64_t b, int& fail_count) {
    auto va = N::set(a, b);
    auto vb = N::set(b, a);  // lane 0: a*b, lane 1: b*a
    auto vc = N::mul(va, vb);
    Goldilocks::Element out[2];
    N::store(out, vc);

    uint64_t want0 = scalar_mul(a, b);
    uint64_t want1 = scalar_mul(b, a);
    if (out[0].fe != want0 || out[1].fe != want1) {
        if (fail_count < 10) {
            std::printf("FAIL mul: a=%016llx b=%016llx\n",
                        (unsigned long long)a, (unsigned long long)b);
            std::printf("  lane0 got %016llx want %016llx\n",
                        (unsigned long long)out[0].fe, (unsigned long long)want0);
            std::printf("  lane1 got %016llx want %016llx\n",
                        (unsigned long long)out[1].fe, (unsigned long long)want1);
        }
        ++fail_count;
        return false;
    }
    return true;
}

static bool check_square(uint64_t a, int& fail_count) {
    auto va = N::set(a, a ^ 0x5A5A5A5A5A5A5A5AULL);
    auto vc = N::square(va);
    Goldilocks::Element out[2];
    N::store(out, vc);

    uint64_t want0 = scalar_square(a);
    uint64_t want1 = scalar_square(a ^ 0x5A5A5A5A5A5A5A5AULL);
    if (out[0].fe != want0 || out[1].fe != want1) {
        if (fail_count < 10) {
            std::printf("FAIL square: a=%016llx\n", (unsigned long long)a);
            std::printf("  lane0 got %016llx want %016llx\n",
                        (unsigned long long)out[0].fe, (unsigned long long)want0);
            std::printf("  lane1 got %016llx want %016llx\n",
                        (unsigned long long)out[1].fe, (unsigned long long)want1);
        }
        ++fail_count;
        return false;
    }
    return true;
}

int main() {
    int fail = 0;
    int mul_tests = 0, sq_tests = 0;

    // 10 edge values
    const uint64_t P = GOLDILOCKS_PRIME;
    uint64_t edges[] = {
        0, 1, P - 1, P, P + 1,
        0xFFFFFFFFFFFFFFFFULL,
        0x100000000ULL,                    // 2^32
        0x00000000FFFFFFFFULL,             // 2^32 - 1
        MSB_,
        MSB_ - 1,
    };
    const int NE = sizeof(edges) / sizeof(edges[0]);

    // Edge × Edge
    for (int i = 0; i < NE; ++i)
        for (int j = 0; j < NE; ++j) {
            check_mul(edges[i], edges[j], fail);
            ++mul_tests;
        }
    for (int i = 0; i < NE; ++i) {
        check_square(edges[i], fail);
        ++sq_tests;
    }

    // 10 000 pseudorandom uniform pairs (xorshift64)
    uint64_t seed = 0xDEADBEEF12345678ULL;
    auto rnd = [&]() {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        return seed;
    };
    for (int i = 0; i < 10000; ++i) {
        uint64_t a = rnd();
        uint64_t b = rnd();
        check_mul(a, b, fail);
        ++mul_tests;
        if ((i & 3) == 0) {
            check_square(a, fail);
            ++sq_tests;
        }
    }

    // 2 000 canonical × canonical
    for (int i = 0; i < 2000; ++i) {
        uint64_t a = rnd(); if (a >= P) a -= P;
        uint64_t b = rnd(); if (b >= P) b -= P;
        check_mul(a, b, fail);
        ++mul_tests;
    }

    std::printf("mul: %d tests, square: %d tests, failures: %d\n",
                mul_tests, sq_tests, fail);
    return fail == 0 ? 0 : 1;
}
#endif
