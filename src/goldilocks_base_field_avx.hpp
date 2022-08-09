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
    Goldilocks::shift(a_s, a);
    Goldilocks::toCanonical_s(a_sc, a_s);
    Goldilocks::shift(a_c, a_sc);
}
//
// Add... rick: afegir add_avx de goldilocs fields
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
    Goldilocks::shift(a_s, a);
    Goldilocks::toCanonical_s(a_sc, a_s);
    Goldilocks::add_avx_a_sc(c, a_sc, b);
}

//
// Mult, reduce
//

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