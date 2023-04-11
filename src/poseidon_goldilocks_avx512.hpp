#ifndef POSEIDON_GOLDILOCKS_AVX512
#define POSEIDON_GOLDILOCKS_AVX512

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

inline void PoseidonGoldilocks::pow7_avx512(__m512i &st0, __m512i &st1, __m512i &st2)
{
    __m512i pw2_0, pw2_1, pw2_2;
    Goldilocks::square_avx512(pw2_0, st0);
    Goldilocks::square_avx512(pw2_1, st1);
    Goldilocks::square_avx512(pw2_2, st2);
    __m512i pw4_0, pw4_1, pw4_2;
    Goldilocks::square_avx512(pw4_0, pw2_0);
    Goldilocks::square_avx512(pw4_1, pw2_1);
    Goldilocks::square_avx512(pw4_2, pw2_2);
    __m512i pw3_0, pw3_1, pw3_2;
    Goldilocks::mult_avx512(pw3_0, pw2_0, st0);
    Goldilocks::mult_avx512(pw3_1, pw2_1, st1);
    Goldilocks::mult_avx512(pw3_2, pw2_2, st2);

    Goldilocks::mult_avx512(st0, pw3_0, pw4_0);
    Goldilocks::mult_avx512(st1, pw3_1, pw4_1);
    Goldilocks::mult_avx512(st2, pw3_2, pw4_2);
};

inline void PoseidonGoldilocks::add_avx512(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_[SPONGE_WIDTH_2])
{
    __m512i c0, c1, c2;
    Goldilocks::load_avx512(c0, &(C_[0]));
    Goldilocks::load_avx512(c1, &(C_[8]));
    Goldilocks::load_avx512(c2, &(C_[16]));
    Goldilocks::add_avx512(st0, st0, c0);
    Goldilocks::add_avx512(st1, st1, c1);
    Goldilocks::add_avx512(st2, st2, c2);
}
// Assuming C_a is aligned
inline void PoseidonGoldilocks::add_avx512_a(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_a[SPONGE_WIDTH_2])
{
    __m512i c0, c1, c2;
    Goldilocks::load_avx512_a(c0, &(C_a[0]));
    Goldilocks::load_avx512_a(c1, &(C_a[8]));
    Goldilocks::load_avx512_a(c2, &(C_a[16]));
    Goldilocks::add_avx512(st0, st0, c0);
    Goldilocks::add_avx512(st1, st1, c1);
    Goldilocks::add_avx512(st2, st2, c2);
}
inline void PoseidonGoldilocks::add_avx512_small(__m512i &st0, __m512i &st1, __m512i &st2, const Goldilocks::Element C_small[SPONGE_WIDTH_2])
{
    __m512i c0, c1, c2;
    Goldilocks::load_avx512(c0, &(C_small[0]));
    Goldilocks::load_avx512(c1, &(C_small[8]));
    Goldilocks::load_avx512(c2, &(C_small[16]));

    Goldilocks::add_avx512_b_c(st0, st0, c0);
    Goldilocks::add_avx512_b_c(st1, st1, c1);
    Goldilocks::add_avx512_b_c(st2, st2, c2);
}
#endif