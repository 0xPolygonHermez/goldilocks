#ifndef GOLDILOCKS_AVX512
#define GOLDILOCKS_AVX512
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

// NOTATION:
// _c value is in canonical form
// _n negative P_n = -P
// _l low part of a variable: uint64 [31:0] or uint128 [63:0]
// _h high part of a variable: uint64 [63:32] or uint128 [127:64]
// _a alingned pointer
// _8 variable can be expressed in 8 bits (<256)

// OBSERVATIONS:
// 1.  a + b overflows iff (a + b) < a (AVX does not suport carry, this is the way to check)
// 2.  a - b underflows iff (a - b) > a (AVX does not suport carry, this is the way to check)

#define AVX512_SIZE_ 8

const __m512i P8 = _mm512_set_epi64(GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME, GOLDILOCKS_PRIME);
const __m512i P8_n = _mm512_set_epi64(GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG, GOLDILOCKS_PRIME_NEG);
const __m512i sqmask8 = _mm512_set_epi64(0x1FFFFFFFF, 0x1FFFFFFFF, 0x1FFFFFFFF, 0x1FFFFFFFF, 0x1FFFFFFFF, 0x1FFFFFFFF, 0x1FFFFFFFF, 0x1FFFFFFFF);

inline void Goldilocks::load_avx512(__m512i &a, const Goldilocks::Element *a8)
{
    a = _mm512_loadu_si512((__m512i *)(a8));
}

inline void Goldilocks::store_avx512(Goldilocks::Element *a8, const __m512i &a)
{
    _mm512_storeu_si512((__m512i *)a8, a);
}

// Obtain cannonical representative of a,
// We assume a <= a_c+P
inline void Goldilocks::toCanonical_avx512(__m512i &a_c, const __m512i &a)
{
    __mmask8 mask = -1; // all elements will be compared
    __mmask8 result_mask = _mm512_mask_cmpge_epu64_mask(mask, a, P8);
    a_c = _mm512_mask_add_epi64(a, result_mask, a, P8_n);
}

inline void Goldilocks::add_avx512(__m512i &c, const __m512i &a, const __m512i &b)
{
    // Evaluate a_c
    __m512i a_c;
    toCanonical_avx512(a_c, a);

    // Addition
    __m512i c0 = _mm512_add_epi64(a_c, b);

    // correction if overflow (iff a_c > a_c+b )
    // if correction is necessari, note that as a_c+b <= P+2^64-1, a_c+b-P <= 2^64-1
    __mmask8 mask = -1; // all elements will be compared
    __mmask8 result_mask = _mm512_mask_cmpgt_epu64_mask(mask, a_c, c0);
    c = _mm512_mask_add_epi64(c0, result_mask, c0, P8_n);
}

// Assume b is in canonical form
inline void Goldilocks::add_avx512_b_c(__m512i &c, const __m512i &a, const __m512i &b_c)
{
    const __m512i c0 = _mm512_add_epi64(a, b_c);
    __mmask8 mask = -1; // all elements will be compared
    __mmask8 result_mask = _mm512_mask_cmpgt_epu64_mask(mask, a, c0);
    c = _mm512_mask_add_epi64(c0, result_mask, c0, P8_n);
}

inline void Goldilocks::sub_avx512(__m512i &c, const __m512i &a, const __m512i &b)
{
    __m512i b_c;
    toCanonical_avx512(b_c, b);
    const __m512i c0 = _mm512_sub_epi64(a, b_c);

    // correction if underflow (iff a < b_c)
    // if correction is necessari:
    // P > b_c > a =>  {(a-b_c) < 0 and  P+(a-b_c)< P } => 0  < (P-b_c)+a < P
    __mmask8 mask = -1; // all elements will be compared
    __mmask8 result_mask = _mm512_mask_cmpgt_epu64_mask(mask, b_c, a);
    c = _mm512_mask_add_epi64(c0, result_mask, c0, P8);
}
inline void Goldilocks::sub_avx512_b_c(__m512i &c, const __m512i &a, const __m512i &b_c)
{
    const __m512i c0 = _mm512_sub_epi64(a, b_c);
    __mmask8 mask = -1; // all elements will be compared
    __mmask8 result_mask = _mm512_mask_cmpgt_epu64_mask(mask, b_c, a);
    c = _mm512_mask_add_epi64(c0, result_mask, c0, P8);
}

inline void Goldilocks::mult_avx512(__m512i &c, const __m512i &a, const __m512i &b)
{
    __m512i c_h, c_l;
    mult_avx512_128(c_h, c_l, a, b);
    reduce_avx512_128_64(c, c_h, c_l);
}

// The 128 bits of the result are stored in c_h[64:0]| c_l[64:0]
inline void Goldilocks::mult_avx512_128(__m512i &c_h, __m512i &c_l, const __m512i &a, const __m512i &b)
{

    // Obtain a_h and b_h in the lower 32 bits
    __m512i a_h = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(a)));
    __m512i b_h = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(b)));

    // c = (a_h+a_l)*(b_h+b_l)=a_h*b_h+a_h*b_l+a_l*b_h+a_l*b_l=c_hh+c_hl+cl_h+c_ll
    // note: _mm256_mul_epu32 uses only the lower 32bits of each chunk so a=a_l and b=b_l
    __m512i c_hh = _mm512_mul_epu32(a_h, b_h);
    __m512i c_hl = _mm512_mul_epu32(a_h, b);
    __m512i c_lh = _mm512_mul_epu32(a, b_h);
    __m512i c_ll = _mm512_mul_epu32(a, b);

    // Bignum addition
    // Ranges: c_hh[127:64], c_lh[95:32], c_hl[95:32], c_ll[63:0]
    // parts that intersect must be added

    // LOW PART:
    // 1: r0 = c_hl + c_ll_h
    //    does not overflow: c_hl <= (2^32-1)*(2^32-1)=2^64-2*2^32+1
    //                       c_ll_h <= 2^32-1
    //                       c_hl + c_ll_h <= 2^64-2^32
    __m512i c_ll_h = _mm512_srli_epi64(c_ll, 32);
    __m512i r0 = _mm512_add_epi64(c_hl, c_ll_h);

    // 2: r1 = r0_l + c_lh //does not overflow
    __m512i r0_l = _mm512_and_si512(r0, P8_n);
    __m512i r1 = _mm512_add_epi64(c_lh, r0_l);

    // 3: c_l = r1_l | c_ll_l
    __m512i r1_l = _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(r1)));
    __mmask16 mask = 0xAAAA;
    c_l = _mm512_mask_blend_epi32(mask, c_ll, r1_l);

    // HIGH PART: c_h = c_hh + r0_h + r1_h
    // 1: r2 = r0_h + c_hh
    //    does not overflow: c_hh <= (2^32-1)*(2^32-1)=2^64-2*2^32+1
    //                       r0_h <= 2^32-1
    //                       r0_h + c_hh <= 2^64-2^32
    __m512i r0_h = _mm512_srli_epi64(r0, 32);
    __m512i r2 = _mm512_add_epi64(c_hh, r0_h);

    // 2: c_h = r3 + r1_h
    //    does not overflow: r2 <= 2^64-2^32
    //                       r1_h <= 2^32-1
    //                       r2 + r1_h <= 2^64-1
    __m512i r1_h = _mm512_srli_epi64(r1, 32);
    c_h = _mm512_add_epi64(r2, r1_h);
}

// notes:
// P = 2^64-2^32+1
// P_n = 2^32-1
// 2^32*P_n = 2^32*(2^32-1) = 2^64-2^32 = P-1
// 2^64 = P+P_n => [2^64]=[P_n]
// process:
// c % P = [c] = [c_h*2^64+c_l] = [c_h*P_n+c_l] = [c_hh*2^32*P_n+c_hl*P_n+c_l] =
//             = [c_hh(P-1) +c_hl*P_n+c_l] = [c_l-c_hh+c_hl*P_n]
inline void Goldilocks::reduce_avx512_128_64(__m512i &c, const __m512i &c_h, const __m512i &c_l)
{
    __m512i c_hh = _mm512_srli_epi64(c_h, 32);
    __m512i c1;
    sub_avx512_b_c(c1, c_l, c_hh);
    __m512i c2 = _mm512_mul_epu32(c_h, P8_n); // c_hl*P_n (only 32bits of c_h useds)
    add_avx512_b_c(c, c1, c2);
}

inline void Goldilocks::square_avx512(__m512i &c, __m512i &a)
{
    __m512i c_h, c_l;
    square_avx512_128(c_h, c_l, a);
    reduce_avx512_128_64(c, c_h, c_l);
}

inline void Goldilocks::square_avx512_128(__m512i &c_h, __m512i &c_l, const __m512i &a)
{
    // Obtain a_h
    __m512i a_h = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(a)));

    // c = (a_h+a_l)*(b_h*a_l)=a_h*a_h+2*a_h*a_l+a_l*a_l=c_hh+2*c_hl+c_ll
    // note: _mm256_mul_epu32 uses only the lower 32bits of each chunk so a=a_l
    __m512i c_hh = _mm512_mul_epu32(a_h, a_h);
    __m512i c_lh = _mm512_mul_epu32(a, a_h); // used as 2^c_lh
    __m512i c_ll = _mm512_mul_epu32(a, a);

    // Bignum addition
    // Ranges: c_hh[127:64], c_lh[95:32], 2*c_lh[96:33],c_ll[63:0]
    //         c_ll_h[63:33]
    // parts that intersect must be added

    // LOW PART:
    // 1: r0 = c_lh + c_ll_h (31 bits)
    // Does not overflow c_lh <= (2^32-1)*(2^32-1)=2^64-2*2^32+1
    //                   c_ll_h <= 2^31-1
    //                   r0 <= 2^64-2^33+2^31
    __m512i c_ll_h = _mm512_srli_epi64(c_ll, 33); // yes 33, low part of 2*c_lh is [31:0]
    __m512i r0 = _mm512_add_epi64(c_lh, c_ll_h);

    // 2: c_l = r0_l (31 bits) | c_ll_l (33 bits)
    __m512i r0_l = _mm512_slli_epi64(r0, 33);
    __m512i c_ll_l = _mm512_and_si512(c_ll, sqmask8);
    c_l = _mm512_add_epi64(r0_l, c_ll_l);

    // HIGH PART:
    // 1: c_h = r0_h (33 bits) + c_hh (64 bits)
    // Does not overflow c_hh <= (2^32-1)*(2^32-1)=2^64-2*2^32+1
    //                   r0 <s= 2^64-2^33+2^31 => r0_h <= 2^33-2 (_h means 33 bits here!)
    //                   Dem: r0_h=2^33-1 => r0 >= r0_h*2^31=2^64-2^31!!
    //                                  contradiction with what we saw above
    //                   c_hh + c0_h <= 2^64-2^33+1+2^33-2 <= 2^64-1
    __m512i r0_h = _mm512_srli_epi64(r0, 31);
    c_h = _mm512_add_epi64(c_hh, r0_h);
}
#endif
