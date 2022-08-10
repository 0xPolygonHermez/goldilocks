#ifndef GOLDILOCKS_AVX
#define GOLDILOCKS_AVX
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

// notation:
//  _c means the value is in canonical form
// _s means shifted
// _n means negavita P_n = -P
#define GOLDILOCKS_PRIME_NEG 0xFFFFFFFF
#define MSB 0x8000000000000000
const __m256i MSB_ = _mm256_set_epi64x(MSB, MSB, MSB, MSB);
const __m256i P = _mm256_set_epi64x(GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME);
const __m256i P_n = _mm256_set_epi64x(GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG);
const __m256i P_s = _mm256_xor_si256(P, MSB_);

//
// set, load, stream
//
inline void Goldilocks::set(__m256i &a, const Goldilocks::Element &a0, const Goldilocks::Element &a1, const Goldilocks::Element &a2, const Goldilocks::Element &a3)
{
    a = _mm256_set_epi64x(a3.fe, a2.fe, a1.fe, a0.fe);
}
inline void Goldilocks::load(__m256i &a, const Goldilocks::Element *a4)
{
    a = _mm256_load_si256((__m256i *)(a4));
}
inline void Goldilocks::store(Goldilocks::Element *a4, const __m256i &a)
{
    _mm256_storeu_si256((__m256i *)a4, a);
}
//
// Shift, to_canonical
//
inline void Goldilocks::shift(__m256i &a_s, const __m256i &a)
{
    a_s = _mm256_xor_si256(a, MSB_);
}
// Obtain cannonical representative of a_s,
// We assume a < (a<<64)+P
// a_sc a shifted canonical
// a_s  a shifted
inline void Goldilocks::toCanonical_s(__m256i &a_sc, const __m256i &a_s)
{
    __m256i mask1_ = _mm256_cmpgt_epi64(P_s, a_s);
    __m256i corr1_ = _mm256_andnot_si256(mask1_, P_n);
    a_sc = _mm256_add_epi64(a_s, corr1_);
}
inline void Goldilocks::toCanonical(__m256i &a_c, const __m256i &a)
{
    __m256i a_s, a_sc;
    shift(a_s, a);
    toCanonical_s(a_sc, a_s);
    shift(a_c, a_sc);
}
//
// Add
//
inline void Goldilocks::add_avx_a_sc(__m256i &c, const __m256i &a_sc, const __m256i &b)
{
    // addition (if only one of the arguments is shifted the sumation is shifted)
    const __m256i caux_s = _mm256_add_epi64(a_sc, b); // can we use only c_

    // correction if overflow (if a_c > a_c+b <)
    __m256i mask_ = _mm256_cmpgt_epi64(a_sc, caux_s);
    __m256i corr_ = _mm256_and_si256(mask_, P_n);  // zero used amother thing here
    __m256i c_s = _mm256_add_epi64(caux_s, corr_); // can we c_=c_+corr_

    // shift c_
    Goldilocks::shift(c, c_s);
}
inline void Goldilocks::add_avx(__m256i &c, const __m256i &a, const __m256i &b)
{
    __m256i a_s, a_sc;
    shift(a_s, a);
    toCanonical_s(a_sc, a_s);
    add_avx_a_sc(c, a_sc, b);
}
//
// Sub: rick?? a-b = (a+1^63)-(b+1^63)=a_s-b_s
//
inline void Goldilocks::sub_avx(__m256i &c, const __m256i &a, const __m256i &b)
{
    __m256i b_s, b_sc, a_s;
    shift(b_s, b);
    shift(a_s, a);
    toCanonical_s(b_sc, b_s);
    const __m256i mask_ = _mm256_cmpgt_epi64(b_s, a_s); // when b > a so underflow
    // P > b > a =>  (a-b) < 0 and a < (P-b)+a < P
    const __m256i corr_ = _mm256_and_si256(mask_, P); // zero used amother thing here
    const __m256i c_aux = _mm256_sub_epi64(a_s, b_s);
    c = _mm256_add_epi64(c_aux, corr_);
}

//
// Mult, reduce
//
inline void Goldilocks::mult_avx_128(__m256i &c_h, __m256i &c_l, const __m256i &a, const __m256i &b)
{

    // 1. cast to 32 bits blocks
    // 2. duplicate the high parts into the lows
    // 3. cast back to 64 bits blocks
    __m256i a_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(a))); // why not use _mm256_srli_epi64?
    __m256i b_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(b)));

    // c = (a_h+a_l)*(b_h*b_l)=a_h*b_h+a_h*b_l+a_l*b_h+a_l*b_l=c_hh+c_hl+c_l_h+c_ll
    // note: _mm256_mul_epu32 uses only the lower 32bits of each chunk so a=a_l
    __m256i c_hh = _mm256_mul_epu32(a_h, b_h);
    __m256i c_hl = _mm256_mul_epu32(a_h, b);
    __m256i c_lh = _mm256_mul_epu32(a, b_h);
    __m256i c_ll = _mm256_mul_epu32(a, b);

    // Bignum addition
    // Ranges: c_hh[127:64], c_hl[95:32], c_lh[95:32], c_ll[64:0]
    // parts that intersect must be added

    // LOW PART:
    // 1: r0 = c_lh + c_ll_h //does not overflow (rick: double check why)
    __m256i c_ll_h = _mm256_srli_epi64(c_ll, 32);
    __m256i r0 = _mm256_add_epi64(c_hl, c_ll_h);

    // 2: r1 = r0_l + c_hl //does not overflow
    __m256i r0_l = _mm256_and_si256(r0, P_n);
    __m256i r1 = _mm256_add_epi64(c_lh, r0_l);

    // 3: c_l = r1_l | c_ll_l
    __m256i r1_l = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(r1))); // why not us _mm256_slli_epi64?
    c_l = _mm256_blend_epi32(c_ll, r1_l, 0xaa);

    // HIGH PART:
    // 1: r2 = r0_h + c_hh //does not overflow
    __m256i r0_h = _mm256_srli_epi64(r0, 32);
    __m256i r2 = _mm256_add_epi64(c_hh, r0_h);

    // 2: c_h = r3 + r1_h
    __m256i r1_h = _mm256_srli_epi64(r1, 32);
    c_h = _mm256_add_epi64(r2, r1_h);
}
// A % B == (((AH % B) * (2^64 % B)) + (AL % B)) % B
//      == (((AH % B) * ((2^64 - B) % B)) + (AL % B)) % B rick:try with this
inline void Goldilocks::reduce_128_64(__m256i &c, const __m256i &c_h, const __m256i &c_l)
{
    __m256i c0_s;
    shift(c0_s, c_l);
    __m256i c_hh = _mm256_srli_epi64(c_h, 32);
    __m256i c1_s;
    sub_avx(c1_s, c0_s, c_hh); // rick: this can be optimized with sum32bits
    __m256i corr_l = _mm256_mul_epu32(c_h, P_n);
    __m256i c_s;
    add_avx(c_s, c1_s, corr_l);
    shift(c, c_s);
}
inline void Goldilocks::mult_avx(__m256i &c, const __m256i &a, const __m256i &b)
{
    __m256i c_h, c_l;
    mult_avx_128(c_h, c_l, a, b);
    reduce_128_64(c, c_h, c_l);
}

//
// Square
//

//
// Dot
//

//
// Spmv
//
#endif