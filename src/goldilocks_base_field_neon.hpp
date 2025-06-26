#ifndef GOLDILOCKS_NEON
#define GOLDILOCKS_NEON

#include "goldilocks_base_field.hpp"
#include <arm_neon.h>

#ifdef __BOUNDS_CHECK__
#include <assert.h>
#define CHECK_BOUNDS_GOLDILOCKS(x) \
    { \
    Goldilocks::Element tmp[2]; \
    Goldilocks::store_neon(tmp, x); \
    assert(tmp[0].fe < GOLDILOCKS_PRIME); \
    assert(tmp[1].fe < GOLDILOCKS_PRIME); \
    }

#else
#define CHECK_BOUNDS_GOLDILOCKS(x)
#endif

static const uint64x2_t MSB = vdupq_n_u64(MSB_);
static const uint64x2_t P = vdupq_n_u64(GOLDILOCKS_PRIME);
static const uint64x2_t P_n = vdupq_n_u64(GOLDILOCKS_PRIME_NEG);
static const uint64x2_t P_s = vdupq_n_u64(0x7FFFFFFF00000001);
static const uint64x2_t LMASK = vdupq_n_u64(0xFFFFFFFF);
static const uint64x2_t SQMASK = vdupq_n_u64(0x1FFFFFFFF);

inline void Goldilocks::to_canonical_neon(uint64x2_t &b, const uint64x2_t &a)
{
    uint64x2_t mask = vcltq_u64(a, P);
    b = vbslq_u64(mask, a, vsubq_u64(a, P));
}

inline void Goldilocks::set_neon(uint64x2_t &a, const Goldilocks::Element &a0, const Goldilocks::Element &a1)
{
    a = vdupq_n_u64(a0.fe);
    a = vsetq_lane_u64(a1.fe, a, 1);
}

inline void Goldilocks::set_neon(uint64x2_t &d0, uint64x2_t &d1, const Goldilocks::Element &a0, const Goldilocks::Element &a1, const Goldilocks::Element &a2, const Goldilocks::Element &a3)
{
    d0 = vdupq_n_u64(a0.fe);
    d1 = vdupq_n_u64(a2.fe);
    d0 = vsetq_lane_u64(a1.fe, d0, 1);
    d1 = vsetq_lane_u64(a3.fe, d1, 1);
}

inline void Goldilocks::load_neon(uint64x2_t &d, const Goldilocks::Element *a2)
{
    d = vld1q_u64((const uint64_t *)a2);
}

inline void Goldilocks::load_neon(uint64x2_t &d0, uint64x2_t &d1, const Goldilocks::Element *a4)
{
    d0 = vld1q_u64((const uint64_t *)a4);
    d1 = vld1q_u64((const uint64_t *)a4 + 2);
}

inline void Goldilocks::store_neon(Goldilocks::Element *a2, const uint64x2_t &d)
{
    vst1q_u64((uint64_t *)a2, d);
}

inline void Goldilocks::store_neon(Goldilocks::Element *a4, const uint64x2_t &d0, const uint64x2_t &d1)
{
    vst1q_u64((uint64_t *)a4, d0);
    vst1q_u64((uint64_t *)a4 + 2, d1);
}

inline void Goldilocks::add_neon(uint64x2_t &c, const uint64x2_t &a, const uint64x2_t &b)
{
    uint64x2_t a_c, b_c;
    to_canonical_neon(a_c, a);
    to_canonical_neon(b_c, b);
    uint64x2_t c1 = vaddq_u64(a_c, b_c);
    uint64x2_t d = vsubq_u64(P, a_c);
    uint64x2_t c2 = vaddq_u64(c1, P_n);
    uint64x2_t mask_ = vcltq_u64(b_c, d);
    c = vbslq_u64(mask_, c1, c2);
    CHECK_BOUNDS_GOLDILOCKS(c);
}

inline void Goldilocks::sub_neon(uint64x2_t &c, const uint64x2_t &a, const uint64x2_t &b)
{
    uint64x2_t a_c, b_c;
    to_canonical_neon(a_c, a);
    to_canonical_neon(b_c, b);
    uint64x2_t c1 = vsubq_u64(a, b);
    uint64x2_t mask_ = vcgeq_u64(a, b);
    uint64x2_t c2 = vaddq_u64(c1, P);
    c = vbslq_u64(mask_, c1, c2);
    CHECK_BOUNDS_GOLDILOCKS(c);
}

inline void Goldilocks::mult_neon(uint64x2_t &c, const uint64x2_t &a, const uint64x2_t &b)
{
    uint64x2_t c_h, c_l;
    mult_neon_128(c_h, c_l, a, b);
    reduce_neon_128_64(c, c_h, c_l);
    CHECK_BOUNDS_GOLDILOCKS(c);
}

// We assume coeficients of b_8 can be expressed with 8 bits (<256)
inline void Goldilocks::mult_neon_8(uint64x2_t &c, const uint64x2_t &a, const uint64x2_t &b_8)
{
    uint64x2_t c_h, c_l;
    mult_neon_72(c_h, c_l, a, b_8);
    reduce_neon_96_64(c, c_h, c_l);
    CHECK_BOUNDS_GOLDILOCKS(c);
}

inline void Goldilocks::mult_neon_72(uint64x2_t &c_h, uint64x2_t &c_l, const uint64x2_t &a, const uint64x2_t &b)
{
    uint32x2_t a_h = vmovn_u64(vshrq_n_u64(a, 32));
    uint32x2_t a_l = vmovn_u64(a);
    uint32x2_t b_l = vmovn_u64(b);

    uint64x2_t c_hl = vmull_u32(a_h, b_l);
    uint64x2_t c_ll = vmull_u32(a_l, b_l);

    uint64x2_t c_ll_h = vshrq_n_u64(c_ll, 32);
    uint64x2_t r0 = vaddq_u64(c_hl, c_ll_h);
    uint64x2_t r0_l = vshlq_n_u64(r0, 32);

    c_l = vbslq_u64(LMASK, c_ll, r0_l);
    c_h = vshrq_n_u64(r0, 32);
}

// The 128 bits of the result are stored in c_h[64:0]| c_l[64:0]
inline void Goldilocks::mult_neon_128(uint64x2_t &c_h, uint64x2_t &c_l, const uint64x2_t &a, const uint64x2_t &b)
{
    // Split into 32 bits
    uint32x2_t a_h = vmovn_u64(vshrq_n_u64(a, 32));
    uint32x2_t b_h = vmovn_u64(vshrq_n_u64(b, 32));
    uint32x2_t a_l = vmovn_u64(a);
    uint32x2_t b_l = vmovn_u64(b);

    // c = (a_h + a_l) * (b_h + b_l)
    uint64x2_t c_hh = vmull_u32(a_h, b_h);
    uint64x2_t c_hl = vmull_u32(a_h, b_l);
    uint64x2_t c_lh = vmull_u32(a_l, b_h);
    uint64x2_t c_ll = vmull_u32(a_l, b_l);

    uint64x2_t c_ll_h = vshrq_n_u64(c_ll, 32);
    uint64x2_t r0 = vaddq_u64(c_hl, c_ll_h);
    uint64x2_t r0_l = vandq_u64(r0, P_n);
    uint64x2_t r1 = vaddq_u64(c_lh, r0_l);
    uint64x2_t r1_l = vshlq_n_u64(r1, 32);

    c_l = vbslq_u64(LMASK, c_ll, r1_l);

    uint64x2_t r0_h = vshrq_n_u64(r0, 32);
    uint64x2_t r2 = vaddq_u64(c_hh, r0_h);
    uint64x2_t r1_h = vshrq_n_u64(r1, 32);
    c_h = vaddq_u64(r2, r1_h);
}

inline void Goldilocks::reduce_neon_128_64(uint64x2_t &c, const uint64x2_t &c_h, const uint64x2_t &c_l)
{
    uint64x2_t c_hh = vshrq_n_u64(c_h, 32);
    uint64x2_t c1;
    sub_neon(c1, c_l, c_hh);
    uint64x2_t c2 = vmull_u32(vmovn_u64(c_h), vmovn_u64(P_n));
    add_neon(c, c1, c2);
}

inline void Goldilocks::reduce_neon_96_64(uint64x2_t &c, const uint64x2_t &c_h, const uint64x2_t &c_l)
{
    uint64x2_t c1 = vmull_u32(vmovn_u64(c_h), vmovn_u64(P_n));
    add_neon(c, c_l, c1);
}

inline void Goldilocks::square_neon(uint64x2_t &c, uint64x2_t &a)
{
    uint64x2_t c_h, c_l;
    square_neon_128(c_h, c_l, a);
    reduce_neon_128_64(c, c_h, c_l);
}

inline void Goldilocks::square_neon_128(uint64x2_t &c_h, uint64x2_t &c_l, const uint64x2_t &a)
{
    uint32x2_t a_h = vmovn_u64(vshrq_n_u64(a, 32));
    uint32x2_t a_l = vmovn_u64(a);

    uint64x2_t c_hh = vmull_u32(a_h, a_h);
    uint64x2_t c_lh = vmull_u32(a_l, a_h);
    uint64x2_t c_ll = vmull_u32(a_l, a_l);

    uint64x2_t c_ll_h = vshrq_n_u64(c_ll, 33); // yes 33, low part of 2*c_lh is [31:0]
    uint64x2_t r0 = vaddq_u64(c_lh, c_ll_h);
    uint64x2_t r0_l = vshlq_n_u64(r0, 33);
    uint64x2_t c_ll_l = vandq_u64(c_ll, SQMASK);
    c_l = vaddq_u64(r0_l, c_ll_l);
    uint64x2_t r0_h = vshrq_n_u64(r0, 31);
    c_h = vaddq_u64(c_hh, r0_h);
}

// Sparse matrix-vector product (4x12 sparce matrix formed of three diagonal blocks of size 4x4)
// c[i]=Sum_j(aj[i]*b[j*4+i]) 0<=i<4 0<=j<3
inline void Goldilocks::spmv_neon_4x12(uint64x2_t &c0, uint64x2_t &c1, const uint64x2_t &a0, const uint64x2_t &a1, const uint64x2_t &a2, const uint64x2_t &a3, const uint64x2_t &a4, const uint64x2_t &a5, const Goldilocks::Element b[12])
{
    uint64x2_t b0, b1, b2, b3, b4, b5;
    uint64x2_t r0, r1, r2, r3, r4, r5;
    load_neon(b0, (Goldilocks::Element *)&(b[0]));
    load_neon(b1, (Goldilocks::Element *)&(b[2]));
    load_neon(b2, (Goldilocks::Element *)&(b[4]));
    load_neon(b3, (Goldilocks::Element *)&(b[6]));
    load_neon(b4, (Goldilocks::Element *)&(b[8]));
    load_neon(b5, (Goldilocks::Element *)&(b[10]));
    mult_neon(r0, a0, b0);
    mult_neon(r1, a1, b1);
    mult_neon(r2, a2, b2);
    mult_neon(r3, a3, b3);
    mult_neon(r4, a4, b4);
    mult_neon(r5, a5, b5);
    uint64x2_t c_;
    add_neon(c_, r0, r2);
    add_neon(c0, c_, r4);
    add_neon(c_, r1, r3);
    add_neon(c1, c_, r5);
}

inline void Goldilocks::spmv_neon_4x12_8(uint64x2_t &c0, uint64x2_t &c1, const uint64x2_t &a0, const uint64x2_t &a1, const uint64x2_t &a2, const uint64x2_t &a3, const uint64x2_t &a4, const uint64x2_t &a5, const Goldilocks::Element b_8[12])
{
    uint64x2_t b0, b1, b2, b3, b4, b5;
    uint64x2_t c0_h, c1_h, c2_h, c3_h, c4_h, c5_h;
    uint64x2_t c0_l, c1_l, c2_l, c3_l, c4_l, c5_l;
    uint64x2_t r0_h, r0_l, r1_h, r1_l, aux_h, aux_l;

    load_neon(b0, (Goldilocks::Element *)&(b_8[0]));
    load_neon(b1, (Goldilocks::Element *)&(b_8[2]));
    load_neon(b2, (Goldilocks::Element *)&(b_8[4]));
    load_neon(b3, (Goldilocks::Element *)&(b_8[6]));
    load_neon(b4, (Goldilocks::Element *)&(b_8[8]));
    load_neon(b5, (Goldilocks::Element *)&(b_8[10]));

    mult_neon_72(c0_h, c0_l, a0, b0);
    mult_neon_72(c1_h, c1_l, a1, b1);
    mult_neon_72(c2_h, c2_l, a2, b2);
    mult_neon_72(c3_h, c3_l, a3, b3);
    mult_neon_72(c4_h, c4_l, a4, b4);
    mult_neon_72(c5_h, c5_l, a5, b5);

    add_neon(aux_l, c0_l, c2_l);
    add_neon(r0_l, aux_l, c4_l);
    add_neon(aux_l, c1_l, c3_l);
    add_neon(r1_l, aux_l, c5_l);

    aux_h = vaddq_u64(c0_h, c2_h);
    r0_h = vaddq_u64(aux_h, c4_h);
    aux_h = vaddq_u64(c1_h, c3_h);
    r1_h = vaddq_u64(aux_h, c5_h);

    reduce_neon_96_64(c0, r0_h, r0_l);
    reduce_neon_96_64(c1, r1_h, r1_l);
}

inline Goldilocks::Element Goldilocks::dot_neon(const uint64x2_t &a0, const uint64x2_t &a1, const uint64x2_t &a2, const uint64x2_t &a3, const uint64x2_t &a4, const uint64x2_t &a5, const Element b[12])
{
    uint64x2_t c0, c1;
    spmv_neon_4x12(c0, c1, a0, a1, a2, a3, a4, a5, b);
    Goldilocks::Element c[4];
    store_neon(c, c0, c1);
    return (c[0] + c[1]) + (c[2] + c[3]);
}

inline void Goldilocks::transpose(uint64x2_t &r00, uint64x2_t &r01, uint64x2_t &r10, uint64x2_t &r11, uint64x2_t &r20, uint64x2_t &r21, uint64x2_t &r30, uint64x2_t &r31,
                                  uint64x2_t &c00, uint64x2_t &c01, uint64x2_t &c10, uint64x2_t &c11, uint64x2_t &c20, uint64x2_t &c21, uint64x2_t &c30, uint64x2_t &c31)
{
    c00 = vzip1q_u64(r00, r10);
    c10 = vzip2q_u64(r00, r10);
    c01 = vzip1q_u64(r20, r30);
    c11 = vzip2q_u64(r20, r30);
    c20 = vzip1q_u64(r01, r11);
    c30 = vzip2q_u64(r01, r11);
    c21 = vzip1q_u64(r21, r31);
    c31 = vzip2q_u64(r21, r31);
}

inline void Goldilocks::mmult_neon_4x12(uint64x2_t &b0, uint64x2_t &b1, const uint64x2_t &a0, const uint64x2_t &a1, const uint64x2_t &a2, const uint64x2_t &a3, const uint64x2_t &a4, const uint64x2_t &a5, const Goldilocks::Element M[48])
{
    // Generate matrix 4x4
    uint64x2_t r00, r01, r10, r11, r20, r21, r30, r31;
    Goldilocks::spmv_neon_4x12(r00, r01, a0, a1, a2, a3, a4, a5, &(M[0]));
    Goldilocks::spmv_neon_4x12(r10, r11, a0, a1, a2, a3, a4, a5, &(M[12]));
    Goldilocks::spmv_neon_4x12(r20, r21, a0, a1, a2, a3, a4, a5, &(M[24]));
    Goldilocks::spmv_neon_4x12(r30, r31, a0, a1, a2, a3, a4, a5, &(M[36]));

    // Transpose: transform de 4x4 matrix stored in rows r0...r3 to the columns c0...c3
    uint64x2_t c00, c01, c10, c11, c20, c21, c30, c31;
    transpose(r00, r01, r10, r11, r20, r21, r30, r31, c00, c01, c10, c11, c20, c21, c30, c31);

    // Add columns to obtain result
    uint64x2_t sum0, sum1;
    add_neon(sum0, c00, c10);
    add_neon(sum1, c20, c30);
    add_neon(b0, sum0, sum1);
    add_neon(sum0, c01, c11);
    add_neon(sum1, c21, c31);
    add_neon(b1, sum0, sum1);
}

inline void Goldilocks::mmult_neon_4x12_8(uint64x2_t &b0, uint64x2_t &b1, const uint64x2_t &a0, const uint64x2_t &a1, const uint64x2_t &a2, const uint64x2_t &a3, const uint64x2_t &a4, const uint64x2_t &a5, const Goldilocks::Element M_8[48])
{
    // Generate matrix 4x4
    uint64x2_t r00, r01, r10, r11, r20, r21, r30, r31;
    Goldilocks::spmv_neon_4x12_8(r00, r01, a0, a1, a2, a3, a4, a5, &(M_8[0]));
    Goldilocks::spmv_neon_4x12_8(r10, r11, a0, a1, a2, a3, a4, a5, &(M_8[12]));
    Goldilocks::spmv_neon_4x12_8(r20, r21, a0, a1, a2, a3, a4, a5, &(M_8[24]));
    Goldilocks::spmv_neon_4x12_8(r30, r31, a0, a1, a2, a3, a4, a5, &(M_8[36]));

    // Transpose: transform de 4x4 matrix stored in rows r0...r3 to the columns c0...c3
    uint64x2_t c00, c01, c10, c11, c20, c21, c30, c31;
    transpose(r00, r01, r10, r11, r20, r21, r30, r31, c00, c01, c10, c11, c20, c21, c30, c31);

    // Add columns to obtain result
    uint64x2_t sum0, sum1;
    add_neon(sum0, c00, c10);
    add_neon(sum1, c20, c30);
    add_neon(b0, sum0, sum1);
    add_neon(sum0, c01, c11);
    add_neon(sum1, c21, c31);
    add_neon(b1, sum0, sum1);
}

inline void Goldilocks::mmult_neon(uint64x2_t &a0, uint64x2_t &a1, uint64x2_t &a2, uint64x2_t &a3, uint64x2_t &a4, uint64x2_t &a5, const Goldilocks::Element M[144])
{
    uint64x2_t b0, b1, b2, b3, b4, b5;
    Goldilocks::mmult_neon_4x12(b0, b1, a0, a1, a2, a3, a4, a5, &(M[0]));
    Goldilocks::mmult_neon_4x12(b2, b3, a0, a1, a2, a3, a4, a5, &(M[48]));
    Goldilocks::mmult_neon_4x12(b4, b5, a0, a1, a2, a3, a4, a5, &(M[96]));
    a0 = b0;
    a1 = b1;
    a2 = b2;
    a3 = b3;
    a4 = b4;
    a5 = b5;
}

inline void Goldilocks::mmult_neon_8(uint64x2_t &a0, uint64x2_t &a1, uint64x2_t &a2, uint64x2_t &a3, uint64x2_t &a4, uint64x2_t &a5, const Goldilocks::Element M_8[144])
{
    uint64x2_t b0, b1, b2, b3, b4, b5;
    Goldilocks::mmult_neon_4x12_8(b0, b1, a0, a1, a2, a3, a4, a5, &(M_8[0]));
    Goldilocks::mmult_neon_4x12_8(b2, b3, a0, a1, a2, a3, a4, a5, &(M_8[48]));
    Goldilocks::mmult_neon_4x12_8(b4, b5, a0, a1, a2, a3, a4, a5, &(M_8[96]));
    a0 = b0;
    a1 = b1;
    a2 = b2;
    a3 = b3;
    a4 = b4;
    a5 = b5;
}

#endif  // GOLDILOCKS_NEON