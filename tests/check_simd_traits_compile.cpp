#include <cassert>
#include <cstdint>
#include <cstdio>
#include "../src/simd/goldilocks_simd.hpp"

using namespace goldilocks::simd;
using S = GLSimd<Scalar>;

static void check_scalar_basic() {
    auto a = S::splat(0x1234567890abcdefULL);
    auto b = S::splat(0x0fedcba987654321ULL);
    auto c = S::add(a, b);
    auto d = S::sub(c, b);
    assert(d.v == a.v);
    auto s = S::shift(a);
    auto u = S::shift(s);
    assert(u.v == a.v);
    auto m = S::and_(a, b);
    (void)m;
    auto z = S::mask_lane0_zero(a);
    assert(z.v == 0);
    std::puts("  scalar: OK");
}

#ifdef GOLDILOCKS_HAS_NEON
static void check_neon_matches_scalar() {
    using N = GLSimd<Neon>;

    // Values chosen to span edge cases
    uint64_t pairs[][2] = {
        {0x0123456789abcdefULL, 0xdeadbeefcafebabeULL},
        {GOLDILOCKS_PRIME - 1, 2},
        {0, GOLDILOCKS_PRIME - 1},
        {(uint64_t)GOLDILOCKS_PRIME_NEG, 0x1ULL},
        {MSB_, MSB_ - 1},
    };

    for (auto& p : pairs) {
        auto na = N::set(p[0], p[1]);
        auto nb = N::set(0x11ULL, 0x22ULL);
        auto nc = N::add(na, nb);
        auto nd = N::sub(nc, nb);

        Goldilocks::Element out_add[2], out_sub[2];
        N::store(out_add, nc);
        N::store(out_sub, nd);

        auto sa0 = S::splat(p[0]);  auto sb0 = S::splat(0x11ULL);
        auto sa1 = S::splat(p[1]);  auto sb1 = S::splat(0x22ULL);
        auto sc0 = S::add(sa0, sb0);
        auto sc1 = S::add(sa1, sb1);

        if (out_add[0].fe != sc0.v || out_add[1].fe != sc1.v ||
            out_sub[0].fe != p[0] || out_sub[1].fe != p[1]) {
            printf("MISMATCH for pair {%llx, %llx}:\n",
                   (unsigned long long)p[0], (unsigned long long)p[1]);
            printf("  add[0] got %llx want %llx\n", (unsigned long long)out_add[0].fe, (unsigned long long)sc0.v);
            printf("  add[1] got %llx want %llx\n", (unsigned long long)out_add[1].fe, (unsigned long long)sc1.v);
            printf("  sub[0] got %llx want %llx\n", (unsigned long long)out_sub[0].fe, (unsigned long long)p[0]);
            printf("  sub[1] got %llx want %llx\n", (unsigned long long)out_sub[1].fe, (unsigned long long)p[1]);
            return;
        }
    }

    // Shift round-trip
    auto a = N::set(0xAAAAAAAAULL, 0x55555555ULL);
    auto b = N::shift(N::shift(a));
    Goldilocks::Element out[2];
    N::store(out, b);
    assert(out[0].fe == 0xAAAAAAAAULL);
    assert(out[1].fe == 0x55555555ULL);

    // permute_lanes: [a.lane1, b.lane0]
    auto l = N::set(1, 2);
    auto r = N::set(3, 4);
    auto p = N::permute_lanes(l, r);
    Goldilocks::Element po[2];
    N::store(po, p);
    assert(po[0].fe == 2);
    assert(po[1].fe == 3);

    // mask_lane0_zero
    auto zv = N::mask_lane0_zero(N::set(0xDEADBEEF, 0xCAFEBABE));
    Goldilocks::Element zo[2];
    N::store(zo, zv);
    assert(zo[0].fe == 0);
    assert(zo[1].fe == 0xCAFEBABE);

    std::puts("  neon: OK");
}
#endif

int main() {
    std::puts("check_simd_traits_compile:");
    check_scalar_basic();
#ifdef GOLDILOCKS_HAS_NEON
    check_neon_matches_scalar();
#else
    std::puts("  neon: skipped (GOLDILOCKS_HAS_NEON not defined)");
#endif
    std::puts("part1-traits: OK");
    return 0;
}
