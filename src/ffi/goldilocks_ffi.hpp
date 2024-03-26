#ifndef GOLDILOCKS_FFI_HPP
#define GOLDILOCKS_FFI_HPP
    #include "goldilocks_base_field.hpp"
    #include "goldilocks_base_field_tools.hpp"
    #include "goldilocks_base_field_scalar.hpp"
    #include "goldilocks_base_field_batch.hpp"
    #include "goldilocks_base_field_avx.hpp"
    #ifdef __AVX512__
    #include "goldilocks_base_field_avx512.hpp"
    #endif

    // BASIC FIELD OPERATIONS
    inline uint64_t to_montgomery(const uint64_t &in1) { return Goldilocks::to_montgomery(in1); }
    inline uint64_t from_montgomery(const uint64_t &in1) { return Goldilocks::from_montgomery(in1); }

    inline const Element &zero() { return Goldilocks::zero(); }
    inline void zero(Element &result) { Goldilocks::zero(result); }

    inline const Element &one() { return Goldilocks::one(); }
    inline void one(Element &result) { Goldilocks::one(result); }

    inline const Element &negone() { return Goldilocks::negone(); }
    inline void negone(Element &result) { Goldilocks::negone(result); }

    inline const Element &shift() { return Goldilocks::shift(); }
    inline void shift(Element &result) { Goldilocks::shift(result); }

    inline const Element &w(uint64_t i) { return Goldilocks::w(i); }
    inline void w(Element &result, uint64_t i) { Goldilocks::w(result, i); }

    inline Element fromU64(uint64_t in1) { return Goldilocks::fromU64(in1); }
    inline void fromU64(Element &result, uint64_t in1) { Goldilocks::fromU64(result, in1); }
    inline Element fromS64(int64_t in1) { return Goldilocks::fromS64(in1); }
    inline void fromS64(Element &result, int64_t in1) { Goldilocks::fromS64(result, in1); }
    inline Element fromS32(int32_t in1) { return Goldilocks::fromS32(in1); }
    inline void fromS32(Element &result, int32_t in1) { Goldilocks::fromS32(result, in1); }
    inline Element fromString(const std::string &in1, int radix = 10) { return Goldilocks::fromString(in1, radix); }
    inline void fromString(Element &result, const std::string &in1, int radix = 10) { Goldilocks::fromString(result, in1, radix); }
    inline Element fromScalar(const mpz_class &scalar) { return Goldilocks::fromScalar(scalar); }
    inline void fromScalar(Element &result, const mpz_class &scalar) { Goldilocks::fromScalar(result, scalar); }

    inline uint64_t toU64(const Element &in1) { return Goldilocks::toU64(in1); }
    inline void toU64(uint64_t &result, const Element &in1) { Goldilocks::toU64(result, in1); }
    inline int64_t toS64(const Element &in1) { return Goldilocks::toS64(in1); }
    inline void toS64(int64_t &result, const Element &in1) { Goldilocks::toS64(result, in1); }
    inline bool toS32(int32_t &result, const Element &in1) { return Goldilocks::toS32(result, in1); }
    inline std::string toString(const Element &in1, int radix = 10) { return Goldilocks::toString(in1, radix); }
    inline void toString(std::string &result, const Element &in1, int radix = 10) { Goldilocks::toString(result, in1, radix); }
    inline std::string toString(const Element *in1, const uint64_t size, int radix = 10) { return Goldilocks::toString(in1, size, radix); }

    // SCALAR
    inline void copy(Element &dst, const Element &src) { Goldilocks::copy(dst, src); }
    inline void copy(Element *dst, const Element *src) { Goldilocks::copy(dst, src); }
    inline void parcpy(Element *dst, const Element *src, uint64_t size, int num_threads_copy = 64) { Goldilocks::parcpy(dst, src, size, num_threads_copy); }
    inline void parSetZero(Element *dst, uint64_t size, int num_threads_copy = 64) { Goldilocks::parSetZero(dst, size, num_threads_copy); }

    inline Element add(const Element &in1, const Element &in2) { return Goldilocks::add(in1, in2); }
    inline void add(Element &result, const Element &in1, const Element &in2) { Goldilocks::add(result, in1, in2); }
    inline Element inc(const Goldilocks::Element &fe) { return Goldilocks::inc(fe); }

    inline Element sub(const Element &in1, const Element &in2) { return Goldilocks::sub(in1, in2); }
    inline void sub(Element &result, const Element &in1, const Element &in2) { Goldilocks::sub(result, in1, in2); }
    inline Element dec(const Goldilocks::Element &fe) { return Goldilocks::dec(fe); }

    inline Element mul(const Element &in1, const Element &in2) { return Goldilocks::mul(in1, in2); }
    inline void mul(Element &result, const Element &in1, const Element &in2) { Goldilocks::mul(result, in1, in2); }
    inline void mul2(Element &result, const Element &in1, const Element &in2) { Goldilocks::mul2(result, in1, in2); }

    inline Element square(const Element &in1) { return Goldilocks::square(in1); }
    inline void square(Element &result, const Element &in1) { Goldilocks::square(result, in1); }

    inline Element div(const Element &in1, const Element &in2) { return Goldilocks::div(in1, in2); }
    inline void div(Element &result, const Element &in1, const Element &in2) { Goldilocks::div(result, in1, in2); }

    inline Element neg(const Element &in1) { return Goldilocks::neg(in1); }
    inline void neg(Element &result, const Element &in1) { Goldilocks::neg(result, in1); }

    inline bool isZero(const Element &in1) { return Goldilocks::isZero(in1); }
    inline bool isOne(const Element &in1) { return Goldilocks::isOne(in1); }
    inline bool isNegone(const Element &in1) { return Goldilocks::isNegone(in1); }

    inline bool equal(const Element &in1, const Element &in2) { return Goldilocks::equal(in1, in2); }

    inline Element inv(const Element &in1) { return Goldilocks::inv(in1); }
    inline void inv(Element &result, const Element &in1) { Goldilocks::inv(result, in1); }

    inline Element mulScalar(const Element &base, const uint64_t &scalar) { return Goldilocks::mulScalar(base, scalar); }
    inline void mulScalar(Element &result, const Element &base, const uint64_t &scalar) { Goldilocks::mulScalar(result, base, scalar); }

    inline Element exp(Element base, uint64_t exp) { return Goldilocks::exp(base, exp); }
    inline void exp(Element &result, Element base, uint64_t exps) { Goldilocks::exp(result, base, exps); }

    // BATCH
    inline void copy_batch(Element *dst, const Element &src) { Goldilocks::copy_batch(dst, src); }
    inline void copy_batch(Element *dst, const Element *src) { Goldilocks::copy_batch(dst, src); }
    inline void copy_batch(Element *dst, const Element *src, uint64_t stride) { Goldilocks::copy_batch(dst, src, stride); }
    inline void copy_batch(Element *dst, const Element *src, uint64_t stride[4]) { Goldilocks::copy_batch(dst, src, stride); }
    inline void copy_batch(Element *dst, uint64_t stride, const Element *src) { Goldilocks::copy_batch(dst, stride, src); }
    inline void copy_batch(Element *dst, uint64_t stride[4], const Element *src) { Goldilocks::copy_batch(dst, stride, src); }

    inline void add_batch(Element *result, const Element *in1, const Element *in2) { Goldilocks::add_batch(result, in1, in2); }
    inline void add_batch(Element *result, const Element *in1, const Element *in2, uint64_t offset2) { Goldilocks::add_batch(result, in1, in2, offset2); }
    inline void add_batch(Element *result, const Element *in1, const Element *in2, const uint64_t offsets2[4]) { Goldilocks::add_batch(result, in1, in2, offsets2); }
    inline void add_batch(Element *result, const Element *in1, const Element in2) { Goldilocks::add_batch(result, in1, in2); }
    inline void add_batch(Element *result, const Element *in1, const Element in2, uint64_t offset1) { Goldilocks::add_batch(result, in1, in2, offset1); }
    inline void add_batch(Element *result, const Element *in1, const Element *in2, uint64_t offset1, uint64_t offset2) { Goldilocks::add_batch(result, in1, in2, offset1, offset2); }
    inline void add_batch(Element *result, const Element *in1, const Element *in2, const uint64_t offsets1[4], const uint64_t offsets2[4]) { Goldilocks::add_batch(result, in1, in2, offsets1, offsets2); }
    inline void add_batch(Element *result, const Element *in1, const Element in2, const uint64_t offsets1[4]) { Goldilocks::add_batch(result, in1, in2, offsets1); }

    inline void sub_batch(Element *result, const Element *in1, const Element *in2) { Goldilocks::sub_batch(result, in1, in2); }
    inline void sub_batch(Element *result, const Element *in1, const Element *in2, uint64_t offset1, uint64_t offset2) { Goldilocks::sub_batch(result, in1, in2, offset1, offset2); }
    inline void sub_batch(Element *result, const Element *in1, const Element in2) { Goldilocks::sub_batch(result, in1, in2); }
    inline void sub_batch(Element *result, const Element in1, const Element *in2) { Goldilocks::sub_batch(result, in1, in2); }
    inline void sub_batch(Element *result, const Element *in1, const Element in2, uint64_t offset1) { Goldilocks::sub_batch(result, in1, in2, offset1); }
    inline void sub_batch(Element *result, const Element in1, const Element *in2, uint64_t offset2) { Goldilocks::sub_batch(result, in1, in2, offset2); }
    inline void sub_batch(Element *result, const Element *in1, const Element *in2, const uint64_t offsets1[4], const uint64_t offsets2[4]) { Goldilocks::sub_batch(result, in1, in2, offsets1, offsets2); }
    inline void sub_batch(Element *result, const Element in1, const Element *in2, const uint64_t offsets2[4]) { Goldilocks::sub_batch(result, in1, in2, offsets2); }
    inline void sub_batch(Element *result, const Element *in1, const Element in2, const uint64_t offsets1[4]) { Goldilocks::sub_batch(result, in1, in2, offsets1); }

    inline void mul_batch(Element *result, const Element *in1, const Element *in2) { Goldilocks::mul_batch(result, in1, in2); }
    inline void mul_batch(Element *result, const Element in1, const Element *in2) { Goldilocks::mul_batch(result, in1, in2); }
    inline void mul_batch(Element *result, const Element *in1, const Element *in2, uint64_t offset1, uint64_t offset2) { Goldilocks::mul_batch(result, in1, in2, offset1, offset2); }
    inline void mul_batch(Element *result, const Element in1, const Element *in2, uint64_t offset2) { Goldilocks::mul_batch(result, in1, in2, offset2); }
    inline void mul_batch(Element *result, const Element *in1, const Element *in2, const uint64_t offsets1[4], const uint64_t offsets2[4]) { Goldilocks::mul_batch(result, in1, in2, offsets1, offsets2); }
    
    // AVX
    inline void set_avx(__m256i &a, const Goldilocks::Element &a3, const Goldilocks::Element &a2, const Goldilocks::Element &a1, const Goldilocks::Element &a0) { Goldilocks::set_avx(a, a3, a2, a1, a0); }
    inline void load_avx(__m256i &a, const Goldilocks::Element *a4) { Goldilocks::load_avx(a, a4); }
    inline void load_avx_a(__m256i &a, const Goldilocks::Element *a4_a) { Goldilocks::load_avx_a(a, a4_a); }
    inline void store_avx(Goldilocks::Element *a4, const __m256i &a) { Goldilocks::store_avx(a4, a); }
    inline void store_avx_a(Goldilocks::Element *a4_a, const __m256i &a) { Goldilocks::store_avx_a(a4_a, a); }
    inline void shift_avx(__m256i &a_s, const __m256i &a) { Goldilocks::shift_avx(a_s, a); }
    inline void toCanonical_avx(__m256i &a_c, const __m256i &a) { Goldilocks::toCanonical_avx(a_c, a); }
    inline void toCanonical_avx_s(__m256i &a_sc, const __m256i &a_s) { Goldilocks::toCanonical_avx_s(a_sc, a_s); }

    inline void add_avx(__m256i &c, const __m256i &a, const __m256i &b) { Goldilocks::add_avx(c, a, b); }
    inline void add_avx_a_sc(__m256i &c, const __m256i &a_c, const __m256i &b) { Goldilocks::add_avx_a_sc(c, a_c, b); }
    inline void add_avx_s_b_small(__m256i &c_s, const __m256i &a_s, const __m256i &b_small) { Goldilocks::add_avx_s_b_small(c_s, a_s, b_small); }
    inline void add_avx_b_small(__m256i &c, const __m256i &a, const __m256i &b_small) { Goldilocks::add_avx_b_small(c, a, b_small); }
    inline void sub_avx(__m256i &c, const __m256i &a, const __m256i &b) { Goldilocks::sub_avx(c, a, b); }
    inline void sub_avx_s_b_small(__m256i &c_s, const __m256i &a_s, const __m256i &b_small) { Goldilocks::sub_avx_s_b_small(c_s, a_s, b_small); }

    inline void mult_avx(__m256i &c, const __m256i &a, const __m256i &b) { Goldilocks::mult_avx(c, a, b); }
    inline void mult_avx_8(__m256i &c, const __m256i &a, const __m256i &b) { Goldilocks::mult_avx_8(c, a, b); }

    inline void mult_avx_128(__m256i &c_h, __m256i &c_l, const __m256i &a, const __m256i &b) { Goldilocks::mult_avx_128(c_h, c_l, a, b); }
    inline void mult_avx_72(__m256i &c_h, __m256i &c_l, const __m256i &a, const __m256i &b) { Goldilocks::mult_avx_72(c_h, c_l, a, b); }
    inline void reduce_avx_128_64(__m256i &c, const __m256i &c_h, const __m256i &c_l) { Goldilocks::reduce_avx_128_64(c, c_h, c_l); }
    inline void reduce_avx_96_64(__m256i &c, const __m256i &c_h, const __m256i &c_l) { Goldilocks::reduce_avx_96_64(c, c_h, c_l); }

    inline void square_avx(__m256i &c, __m256i &a) { Goldilocks::square_avx(c, a); }
    inline void square_avx_128(__m256i &c_h, __m256i &c_l, const __m256i &a) { Goldilocks::square_avx_128(c_h, c_l, a); }

    inline Element dot_avx(const __m256i &a0, const __m256i &a1, const __m256i &a2, const Element b[12]) { return Goldilocks::dot_avx(a0, a1, a2, b); }
    inline Element dot_avx_a(const __m256i &a0, const __m256i &a1, const __m256i &a2, const Element b_a[12]) { return Goldilocks::dot_avx_a(a0, a1, a2, b_a); }

    inline void spmv_avx_4x12(__m256i &c, const __m256i &a0, const __m256i &a1, const __m256i &a2, const Element b[12]) { Goldilocks::spmv_avx_4x12(c, a0, a1, a2, b); }
    inline void spmv_avx_4x12_a(__m256i &c, const __m256i &a0, const __m256i &a1, const __m256i &a2, const Element b_a[12]) { Goldilocks::spmv_avx_4x12_a(c, a0, a1, a2, b_a); }
    inline void spmv_avx_4x12_8(__m256i &c, const __m256i &a0, const __m256i &a1, const __m256i &a2, const Element b_8[12]) { Goldilocks::spmv_avx_4x12_8(c, a0, a1, a2, b_8); }

    inline void mmult_avx_4x12(__m256i &b, const __m256i &a0, const __m256i &a1, const __m256i &a2, const Element M[48]) { Goldilocks::mmult_avx_4x12(b, a0, a1, a2, M); }
    inline void mmult_avx_4x12_a(__m256i &b, const __m256i &a0, const __m256i &a1, const __m256i &a2, const Element M_a[48]) { Goldilocks::mmult_avx_4x12_a(b, a0, a1, a2, M_a); }
    inline void mmult_avx_4x12_8(__m256i &b, const __m256i &a0, const __m256i &a1, const __m256i &a2, const Element M_8[48]) { Goldilocks::mmult_avx_4x12_8(b, a0, a1, a2, M_8); }

    inline void mmult_avx(__m256i &a0, __m256i &a1, __m256i &a2, const Element M[144]) { Goldilocks::mmult_avx(a0, a1, a2, M); }
    inline void mmult_avx_a(__m256i &a0, __m256i &a1, __m256i &a2, const Element M_a[144]) { Goldilocks::mmult_avx_a(a0, a1, a2, M_a); }
    inline void mmult_avx_8(__m256i &a0, __m256i &a1, __m256i &a2, const Element M_8[144]) { Goldilocks::mmult_avx_8(a0, a1, a2, M_8); }

    // implementations for expressions:

    inline void copy_avx(Element *dst, const Element &src) { Goldilocks::copy_avx(dst, src); }
    inline void copy_avx(Element *dst, const Element *src) { Goldilocks::copy_avx(dst, src); }
    inline void copy_avx(Element *dst, uint64_t stride_dst, const Element *src, uint64_t stride) { Goldilocks::copy_avx(dst, stride_dst, src, stride); }
    inline void copy_avx(Element *dst, const Element *src, uint64_t stride) { Goldilocks::copy_avx(dst, src, stride); }
    inline void copy_avx(Element *dst, const Element *src, uint64_t stride[4]) { Goldilocks::copy_avx(dst, src, stride); }
    inline void copy_avx(__m256i &dst_, const Element &src) { Goldilocks::copy_avx(dst_, src); }
    inline void copy_avx(__m256i &dst_, const __m256i &src_) { Goldilocks::copy_avx(dst_, src_); }
    inline void copy_avx(__m256i &dst_, const Element *src, uint64_t stride) { Goldilocks::copy_avx(dst_, src, stride); }
    inline void copy_avx(__m256i &dst_, const Element *src, uint64_t stride[4]) { Goldilocks::copy_avx(dst_, src, stride); }
    inline void copy_avx(Element *dst, uint64_t stride, const __m256i &src_) { Goldilocks::copy_avx(dst, stride, src_); }
    inline void copy_avx(Element *dst, uint64_t stride[4], const __m256i &src_) { Goldilocks::copy_avx(dst, stride, src_); }

    inline void add_avx(Element *c4, const Element *a4, const Element *b4) { Goldilocks::add_avx(c4, a4, b4); }
    inline void add_avx(Element *c4, const Element *a4, const Element *b4, uint64_t offset_b) { Goldilocks::add_avx(c4, a4, b4, offset_b); }
    inline void add_avx(Element *c4, const Element *a4, const Element *b4, const uint64_t offset_b[4]) { Goldilocks::add_avx(c4, a4, b4, offset_b); }
    inline void add_avx(Element *c4, const Element *a4, const Element b) { Goldilocks::add_avx(c4, a4, b); }
    inline void add_avx(Element *c4, const Element *a4, const Element b, uint64_t offset_a) { Goldilocks::add_avx(c4, a4, b, offset_a); }
    inline void add_avx(Element *c4, uint64_t offset_c, const Element *a4, uint64_t offset_a, const Element *b4,  uint64_t offset_b) { Goldilocks::add_avx(c4, offset_c, a4, offset_a, b4, offset_b); }
    inline void add_avx(Element *c4, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b) { Goldilocks::add_avx(c4, a4, b4, offset_a, offset_b); }
    inline void add_avx(Element *c4, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4]) { Goldilocks::add_avx(c4, a4, b4, offset_a, offset_b); }
    inline void add_avx(Element *c4, const Element *a4, const Element b, const uint64_t offset_a[4]) { Goldilocks::add_avx(c4, a4, b, offset_a); }

    inline void add_avx(__m256i &c_, const __m256i &a_, const Element *b4, uint64_t offset_b) { Goldilocks::add_avx(c_, a_, b4, offset_b); }
    inline void add_avx(__m256i &c_, const __m256i &a_, const Element *b4, const uint64_t offset_b[4]) { Goldilocks::add_avx(c_, a_, b4, offset_b); }
    inline void add_avx(__m256i &c_, const __m256i &a_, const Element b) { Goldilocks::add_avx(c_, a_, b); }
    inline void add_avx(__m256i &c_, const Element *a4, const Element b, uint64_t offset_a) { Goldilocks::add_avx(c_, a4, b, offset_a); }
    inline void add_avx(__m256i &c_, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b) { Goldilocks::add_avx(c_, a4, b4, offset_a, offset_b); }
    inline void add_avx(__m256i &c_, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4]) { Goldilocks::add_avx(c_, a4, b4, offset_a, offset_b); }
    inline void add_avx(__m256i &c_, const Element *a4, const Element b, const uint64_t offset_a[4]) { Goldilocks::add_avx(c_, a4, b, offset_a); }
    inline void add_avx(Element *c, uint64_t offset_c, const __m256i &a_, const __m256i &b_) { Goldilocks::add_avx(c, offset_c, a_, b_); }
    inline void add_avx(Element *c, uint64_t offset_c, const __m256i &a_, const Element *b4, uint64_t offset_b) { Goldilocks::add_avx(c, offset_c, a_, b4, offset_b); }
    inline void add_avx(Element *c, const uint64_t offset_c[4], const __m256i &a_, const __m256i &b_) { Goldilocks::add_avx(c, offset_c, a_, b_); }
    inline void add_avx(Element *c, const uint64_t offset_c[4], const __m256i &a_, const Element *b, uint64_t offset_b) { Goldilocks::add_avx(c, offset_c, a_, b, offset_b); }
    inline void add_avx(Element *c, const uint64_t offset_c[4], const __m256i &a_, const Element *b, uint64_t offset_b[4]) { Goldilocks::add_avx(c, offset_c, a_, b, offset_b); }

    inline void sub_avx(Element *c4, const Element *a4, const Element *b4) { Goldilocks::sub_avx(c4, a4, b4); }
    inline void sub_avx(Element *c4, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b) { Goldilocks::sub_avx(c4, a4, b4, offset_a, offset_b); }
    inline void sub_avx(Element *c4, const Element *a4, const Element b) { Goldilocks::sub_avx(c4, a4, b); }
    inline void sub_avx(Element *c4, const Element a, const Element *b4) { Goldilocks::sub_avx(c4, a, b4); }
    inline void sub_avx(Element *c4, const Element *a4, const Element b, uint64_t offset_a) { Goldilocks::sub_avx(c4, a4, b, offset_a); }
    inline void sub_avx(Element *c4, const Element a, const Element *b4, uint64_t offset_b) { Goldilocks::sub_avx(c4, a, b4, offset_b); }
    inline void sub_avx(Element *c4, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4]) { Goldilocks::sub_avx(c4, a4, b4, offset_a, offset_b); }
    inline void sub_avx(Element *c4, const Element a, const Element *b4, const uint64_t offset_b[4]) { Goldilocks::sub_avx(c4, a, b4, offset_b); }
    inline void sub_avx(Element *c4, const Element *a4, const Element b, const uint64_t offset_a[4]) { Goldilocks::sub_avx(c4, a4, b, offset_a); }

    inline void sub_avx(__m256i &c_, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b) { Goldilocks::sub_avx(c_, a4, b4, offset_a, offset_b); }
    inline void sub_avx(__m256i &c_, const __m256i &a_, const Element *b4, uint64_t offset_b) { Goldilocks::sub_avx(c_, a_, b4, offset_b); }
    inline void sub_avx(__m256i &c_, const Element *a4, const __m256i &b_, uint64_t offset_a) { Goldilocks::sub_avx(c_, a4, b_, offset_a); }
    inline void sub_avx(__m256i &c_, const __m256i &a_, const Element b) { Goldilocks::sub_avx(c_, a_, b); }
    inline void sub_avx(__m256i &c_, const Element a, const __m256i &b_) { Goldilocks::sub_avx(c_, a, b_); }
    inline void sub_avx(__m256i &c_, const Element *a4, const Element b, uint64_t offset_a) { Goldilocks::sub_avx(c_, a4, b, offset_a); }
    inline void sub_avx(__m256i &c_, const Element a, const Element *b4, uint64_t offset_b) { Goldilocks::sub_avx(c_, a, b4, offset_b); }
    inline void sub_avx(__m256i &c_, const Element *a4, uint64_t offset_a, const __m256i &b_) { Goldilocks::sub_avx(c_, a4, offset_a, b_); }
    inline void sub_avx(__m256i &c_, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4]) { Goldilocks::sub_avx(c_, a4, b4, offset_a, offset_b); }
    inline void sub_avx(__m256i &c_, const Element a, const Element *b4, const uint64_t offset_b[4]) { Goldilocks::sub_avx(c_, a, b4, offset_b); }
    inline void sub_avx(__m256i &c_, const Element *a4, const Element b, const uint64_t offset_a[4]) { Goldilocks::sub_avx(c_, a4, b, offset_a); }
    inline void sub_avx(__m256i &c_, const __m256i &a_, const Element *b4, uint64_t offset_b[4]) { Goldilocks::sub_avx(c_, a_, b4, offset_b); }
    inline void sub_avx(__m256i &c_, const Element *a4, const __m256i &b_, uint64_t offset_a[4]) { Goldilocks::sub_avx(c_, a4, b_, offset_a); }

    inline void sub_avx(Element *c, uint64_t offset_c, const __m256i &a_, const __m256i &b_) { Goldilocks::sub_avx(c, offset_c, a_, b_); }
    inline void sub_avx(Element *c, const uint64_t offset_c[4], const __m256i &a_, const __m256i &b_) { Goldilocks::sub_avx(c, offset_c, a_, b_); }
    inline void sub_avx(Element *c, uint64_t offset_c, const Element a, const __m256i &b_) { Goldilocks::sub_avx(c, offset_c, a, b_); }
    inline void sub_avx(Element *c, const uint64_t offset_c[4], const Element a, const __m256i &b_) { Goldilocks::sub_avx(c, offset_c, a, b_); }

    inline void mul_avx(Element *c4, const Element *a4, const Element *b4) { Goldilocks::mul_avx(c4, a4, b4); }
    inline void mul_avx(Element *c4, const Element a, const Element *b4) { Goldilocks::mul_avx(c4, a, b4); }
    inline void mul_avx(Element *c4, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b) { Goldilocks::mul_avx(c4, a4, b4, offset_a, offset_b); }
    inline void mul_avx(Element *c4, const Element a, const Element *b4, uint64_t offset_b) { Goldilocks::mul_avx(c4, a, b4, offset_b); }
    inline void mul_avx(Element *c4, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4]) { Goldilocks::mul_avx(c4, a4, b4, offset_a, offset_b); }

    inline void mul_avx(__m256i &c_, const Element a, const __m256i &b_) { Goldilocks::mul_avx(c_, a, b_); }
    inline void mul_avx(__m256i &c_, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b) { Goldilocks::mul_avx(c_, a4, b4, offset_a, offset_b); }
    inline void mul_avx(__m256i &c_, const __m256i &a_, const Element *b4, uint64_t offset_b) { Goldilocks::mul_avx(c_, a_, b4, offset_b); }
    inline void mul_avx(__m256i &c_, const Element *a4, const __m256i &b_, uint64_t offset_a) { Goldilocks::mul_avx(c_, a4, b_, offset_a); }
    inline void mul_avx(__m256i &c_, const Element a, const Element *b4, uint64_t offset_b) { Goldilocks::mul_avx(c_, a, b4, offset_b); }
    inline void mul_avx(__m256i &c_, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4]) { Goldilocks::mul_avx(c_, a4, b4, offset_a, offset_b); }
    inline void mul_avx(__m256i &c_, const __m256i &a_, const Element *b4, const uint64_t offset_b[4]) { Goldilocks::mul_avx(c_, a_, b4, offset_b); }
    inline void mul_avx(__m256i &c_, const Element *a4, const __m256i &b_, const uint64_t offset_a[4]) { Goldilocks::mul_avx(c_, a4, b_, offset_a); }
    inline void mul_avx(__m256i &c_, const Element *a4, const Element b, const uint64_t offset_a[4]) { Goldilocks::mul_avx(c_, a4, b, offset_a); }

    inline void mul_avx(Element *c, uint64_t offset_c, const Element *a, uint64_t offset_a, const Element *b, uint64_t offset_b) { Goldilocks::mul_avx(c, offset_c, a, offset_a, b, offset_b); }
    inline void mul_avx(Element *c, uint64_t offset_c, const __m256i &a_, const __m256i &b_) { Goldilocks::mul_avx(c, offset_c, a_, b_); }
    inline void mul_avx(Element *c, uint64_t offset_c, const Element *a4, const __m256i &b_, uint64_t offset_a) { Goldilocks::mul_avx(c, offset_c, a4, b_, offset_a); }
    inline void mul_avx(Element *c, uint64_t offset_c, const __m256i &a_, const Element *b, uint64_t offset_b) { Goldilocks::mul_avx(c, offset_c, a_, b, offset_b); }
    inline void mul_avx(Element *c, uint64_t offset_c, const Element *a4, const __m256i &b_, const uint64_t offset_a[4]) { Goldilocks::mul_avx(c, offset_c, a4, b_, offset_a); }
    inline void mul_avx(Element *c, uint64_t offset_c[4], const __m256i &a_, const __m256i &b_) { Goldilocks::mul_avx(c, offset_c, a_, b_); }
    inline void mul_avx(Element *c, uint64_t offset_c[4], const Element *a4, const __m256i &b_, uint64_t offset_a) { Goldilocks::mul_avx(c, offset_c, a4, b_, offset_a); }
    inline void mul_avx(Element *c, uint64_t offset_c[4], const __m256i &a_, const Element *b, uint64_t offset_b) { Goldilocks::mul_avx(c, offset_c, a_, b, offset_b); }
    inline void mul_avx(Element *c, uint64_t offset_c[4], const Element *a4, const __m256i &b_, const uint64_t offset_a[4]) { Goldilocks::mul_avx(c, offset_c, a4, b_, offset_a); }
    inline void mul_avx(Element *c, uint64_t offset_c[4], const Element *a, const Element *b, const uint64_t offset_a[4], const uint64_t offset_b[4]) { Goldilocks::mul_avx(c, offset_c, a, offset_a, b, offset_b); }

    // AVX512
#ifdef __AVX512__
    inline void load_avx512(__m512i &a, const Goldilocks::Element *a8) { Goldilocks::load_avx512(a, a8); }
    inline void load_avx512_a(__m512i &a, const Goldilocks::Element *a8_a) { Goldilocks::load_avx512_a(a, a8_a); }
    inline void store_avx512(Goldilocks::Element *a8, const __m512i &a) { Goldilocks::store_avx512(a8, a); }
    inline void store_avx512_a(Goldilocks::Element *a8_a, const __m512i &a) { Goldilocks::store_avx512_a(a8_a, a); }
    inline void toCanonical_avx512(__m512i &a_c, const __m512i &a) { Goldilocks::toCanonical_avx512(a_c, a); }

    inline void add_avx512(__m512i &c, const __m512i &a, const __m512i &b) { Goldilocks::add_avx512(c, a, b); } 
    inline void add_avx512_b_c(__m512i &c, const __m512i &a, const __m512i &b_c) { Goldilocks::add_avx512_b_c(c, a, b_c); }
    inline void sub_avx512(__m512i &c, const __m512i &a, const __m512i &b) { Goldilocks::sub_avx512(c, a, b); }
    inline void sub_avx512_b_c(__m512i &c, const __m512i &a, const __m512i &b_c) { Goldilocks::sub_avx512_b_c(c, a, b_c); }

    inline void mult_avx512(__m512i &c, const __m512i &a, const __m512i &b) { Goldilocks::mult_avx512(c, a, b); }
    inline void mult_avx512_8(__m512i &c, const __m512i &a, const __m512i &b) { Goldilocks::mult_avx512_8(c, a, b); }

    inline void mult_avx512_128(__m512i &c_h, __m512i &c_l, const __m512i &a, const __m512i &b) { Goldilocks::mult_avx512_128(c_h, c_l, a, b); }
    inline void mult_avx512_72(__m512i &c_h, __m512i &c_l, const __m512i &a, const __m512i &b) { Goldilocks::mult_avx512_72(c_h, c_l, a, b); }
    inline void reduce_avx512_128_64(__m512i &c, const __m512i &c_h, const __m512i &c_l) { Goldilocks::reduce_avx512_128_64(c, c_h, c_l); }
    inline void reduce_avx512_96_64(__m512i &c, const __m512i &c_h, const __m512i &c_l) { Goldilocks::reduce_avx512_96_64(c, c_h, c_l); }

    inline void square_avx512(__m512i &c, __m512i &a) { Goldilocks::square_avx512(c, a); }
    inline void square_avx512_128(__m512i &c_h, __m512i &c_l, const __m512i &a) { Goldilocks::square_avx512_128(c_h, c_l, a); }

    inline void dot_avx512(Element c[2], const __m512i &a0, const __m512i &a1, const __m512i &a2, const Element b[12]) { Goldilocks::dot_avx512(c, a0, a1, a2, b); }

    inline void spmv_avx512_4x12(__m512i &c, const __m512i &a0, const __m512i &a1, const __m512i &a2, const Element b[12]) { Goldilocks::spmv_avx512_4x12(c, a0, a1, a2, b); }
    inline void spmv_avx512_4x12_8(__m512i &c, const __m512i &a0, const __m512i &a1, const __m512i &a2, const Element b_8[12]) { Goldilocks::spmv_avx512_4x12_8(c, a0, a1, a2, b_8); }

    inline void mmult_avx512_4x12(__m512i &b, const __m512i &a0, const __m512i &a1, const __m512i &a2, const Element M[48]) { Goldilocks::mmult_avx512_4x12(b, a0, a1, a2, M); }
    inline void mmult_avx512_4x12_8(__m512i &b, const __m512i &a0, const __m512i &a1, const __m512i &a2, const Element M_8[48]) { Goldilocks::mmult_avx512_4x12_8(b, a0, a1, a2, M_8); }

    inline void mmult_avx512(__m512i &a0, __m512i &a1, __m512i &a2, const Element M[144]) { Goldilocks::mmult_avx512(a0, a1, a2, M); }
    inline void mmult_avx512_8(__m512i &a0, __m512i &a1, __m512i &a2, const Element M_8[144]) { Goldilocks::mmult_avx512_8(a0, a1, a2, M_8); }

    inline void copy_avx512(__m512i &dst_, const Element &src) { Goldilocks::copy_avx512(dst_, src); }
    inline void copy_avx512(__m512i &dst_, const __m512i &src_) { Goldilocks::copy_avx512(dst_, src_); }
    inline void copy_avx512(__m512i &dst_, const Element *src, uint64_t stride) { Goldilocks::copy_avx512(dst_, src, stride); }
    inline void copy_avx512(__m512i &dst_, const Element *src, uint64_t stride[AVX512_SIZE_]) { Goldilocks::copy_avx512(dst_, src, stride); }
    inline void copy_avx512(Element *dst, uint64_t stride, const __m512i &src_) { Goldilocks::copy_avx512(dst, stride, src_); }
    inline void copy_avx512(Element *dst, uint64_t stride[AVX512_SIZE_], const __m512i &src_) { Goldilocks::copy_avx512(dst, stride, src_); }

    inline void add_avx512(__m512i &c_, const __m512i &a_, const Element *b8, uint64_t offset_b) { Goldilocks::add_avx512(c_, a_, b8, offset_b); }
    inline void add_avx512(__m512i &c_, const __m512i &a_, const Element *b8, const uint64_t offset_b[AVX512_SIZE_]) { Goldilocks::add_avx512(c_, a_, b8, offset_b); }
    inline void add_avx512(__m512i &c_, const __m512i &a_, const Element b) { Goldilocks::add_avx512(c_, a_, b); }
    inline void add_avx512(__m512i &c_, const Element *a8, const Element b, uint64_t offset_a) { Goldilocks::add_avx512(c_, a8, b, offset_a); }
    inline void add_avx512(__m512i &c_, const Element *a8, const Element *b8, uint64_t offset_a, uint64_t offset_b) { Goldilocks::add_avx512(c_, a8, b8, offset_a, offset_b); }
    inline void add_avx512(__m512i &c_, const Element *a8, const Element *b8, const uint64_t offset_a[AVX512_SIZE_], const uint64_t offset_b[AVX512_SIZE_]) { Goldilocks::add_avx512(c_, a8, b8, offset_a, offset_b); }
    inline void add_avx512(__m512i &c_, const Element *a8, const Element b, const uint64_t offset_a[AVX512_SIZE_]) { Goldilocks::add_avx512(c_, a8, b, offset_a); }
    inline void add_avx512(Element *c, uint64_t offset_c, const __m512i &a_, const __m512i &b_) { Goldilocks::add_avx512(c, offset_c, a_, b_); }
    inline void add_avx512(Element *c, uint64_t offset_c, const __m512i &a_, const Element *b8, uint64_t offset_b) { Goldilocks::add_avx512(c, offset_c, a_, b8, offset_b); }
    inline void add_avx512(Element *c, const uint64_t offset_c[AVX512_SIZE_], const __m512i &a_, const __m512i &b_) { Goldilocks::add_avx512(c, offset_c, a_, b_); }
    inline void add_avx512(Element *c, const uint64_t offset_c[AVX512_SIZE_], const __m512i &a_, const Element *b, uint64_t offset_b) { Goldilocks::add_avx512(c, offset_c, a_, b, offset_b); }
    inline void add_avx512(Element *c, const uint64_t offset_c[AVX512_SIZE_], const __m512i &a_, const Element *b, uint64_t offset_b[AVX512_SIZE_]) { Goldilocks::add_avx512(c, offset_c, a_, b, offset_b); }

    inline void sub_avx512(__m512i &c_, const Element *a8, const Element *b8, uint64_t offset_a, uint64_t offset_b) { Goldilocks::sub_avx512(c_, a8, b8, offset_a, offset_b); }
    inline void sub_avx512(__m512i &c_, const __m512i &a_, const Element *b8, uint64_t offset_b) { Goldilocks::sub_avx512(c_, a_, b8, offset_b); }
    inline void sub_avx512(__m512i &c_, const Element *a8, const __m512i &b_, uint64_t offset_a) { Goldilocks::sub_avx512(c_, a8, b_, offset_a); }
    inline void sub_avx512(__m512i &c_, const __m512i &a_, const Element b) { Goldilocks::sub_avx512(c_, a_, b); }
    inline void sub_avx512(__m512i &c_, const Element a, const __m512i &b_) { Goldilocks::sub_avx512(c_, a, b_); }
    inline void sub_avx512(__m512i &c_, const Element *a8, const Element b, uint64_t offset_a) { Goldilocks::sub_avx512(c_, a8, b, offset_a); }
    inline void sub_avx512(__m512i &c_, const Element a, const Element *b8, uint64_t offset_b) { Goldilocks::sub_avx512(c_, a, b8, offset_b); }
    inline void sub_avx512(__m512i &c_, const Element *a8, const Element *b8, const uint64_t offset_a[AVX512_SIZE_], const uint64_t offset_b[4]) { Goldilocks::sub_avx512(c_, a8, b8, offset_a, offset_b); }
    inline void sub_avx512(__m512i &c_, const Element a, const Element *b8, const uint64_t offset_b[AVX512_SIZE_]) { Goldilocks::sub_avx512(c_, a, b8, offset_b); }
    inline void sub_avx512(__m512i &c_, const Element *a8, const Element b, const uint64_t offset_a[AVX512_SIZE_]) { Goldilocks::sub_avx512(c_, a8, b, offset_a); }
    inline void sub_avx512(__m512i &c_, const __m512i &a_, const Element *b8, uint64_t offset_b[AVX512_SIZE_]) { Goldilocks::sub_avx512(c_, a_, b8, offset_b); }
    inline void sub_avx512(__m512i &c_, const Element *a8, const __m512i &b_, uint64_t offset_a[AVX512_SIZE_]) { Goldilocks::sub_avx512(c_, a8, b_, offset_a); }

    inline void sub_avx512(Element *c, uint64_t offset_c, const __m512i &a_, const __m512i &b_) { Goldilocks::sub_avx512(c, offset_c, a_, b_); }
    inline void sub_avx512(Element *c, const uint64_t offset_c[AVX512_SIZE_], const __m512i &a_, const __m512i &b_) { Goldilocks::sub_avx512(c, offset_c, a_, b_); }
    inline void sub_avx512(Element *c, uint64_t offset_c, const Element a, const __m512i &b_) { Goldilocks::sub_avx512(c, offset_c, a, b_); }
    inline void sub_avx512(Element *c, const uint64_t offset_c[AVX512_SIZE_], const Element a, const __m512i &b_) { Goldilocks::sub_avx512(c, offset_c, a, b_); }

    inline void mul_avx512(__m512i &c_, const Element a, const __m512i &b_) { Goldilocks::mul_avx512(c_, a, b_); }
    inline void mul_avx512(__m512i &c_, const Element *a8, const Element *b8, uint64_t offset_a, uint64_t offset_b) { Goldilocks::mul_avx512(c_, a8, b8, offset_a, offset_b); }
    inline void mul_avx512(__m512i &c_, const __m512i &a_, const Element *b8, uint64_t offset_b) { Goldilocks::mul_avx512(c_, a_, b8, offset_b); }
    inline void mul_avx512(__m512i &c_, const Element *a8, const __m512i &b_, uint64_t offset_a) { Goldilocks::mul_avx512(c_, a8, b_, offset_a); }
    inline void mul_avx512(__m512i &c_, const Element a, const Element *b8, uint64_t offset_b) { Goldilocks::mul_avx512(c_, a, b8, offset_b); }
    inline void mul_avx512(__m512i &c_, const Element *a8, const Element *b8, const uint64_t offset_a[AVX512_SIZE_], const uint64_t offset_b[AVX512_SIZE_]) { Goldilocks::mul_avx512(c_, a8, b8, offset_a, offset_b); }
    inline void mul_avx512(__m512i &c_, const __m512i &a_, const Element *b8, const uint64_t offset_b[AVX512_SIZE_]) { Goldilocks::mul_avx512(c_, a_, b8, offset_b); }
    inline void mul_avx512(__m512i &c_, const Element *a8, const __m512i &b_, const uint64_t offset_a[AVX512_SIZE_]) { Goldilocks::mul_avx512(c_, a8, b_, offset_a); }
    inline void mul_avx512(__m512i &c_, const Element *a8, const Element b, const uint64_t offset_a[AVX512_SIZE_]) { Goldilocks::mul_avx512(c_, a8, b, offset_a); }

    inline void mul_avx512(Element *c, uint64_t offset_c, const __m512i &a_, const __m512i &b_) { Goldilocks::mul_avx512(c, offset_c, a_, b_); }
    inline void mul_avx512(Element *c, uint64_t offset_c, const Element *a8, const __m512i &b_, uint64_t offset_a) { Goldilocks::mul_avx512(c, offset_c, a8, b_, offset_a); }
    inline void mul_avx512(Element *c, uint64_t offset_c, const __m512i &a_, const Element *b, uint64_t offset_b) { Goldilocks::mul_avx512(c, offset_c, a_, b, offset_b); }
    inline void mul_avx512(Element *c, uint64_t offset_c, const Element *a8, const __m512i &b_, const uint64_t offset_a[AVX512_SIZE_]) { Goldilocks::mul_avx512(c, offset_c, a8, b_, offset_a); }
    inline void mul_avx512(Element *c, uint64_t offset_c[AVX512_SIZE_], const __m512i &a_, const __m512i &b_) { Goldilocks::mul_avx512(c, offset_c, a_, b_); }
    inline void mul_avx512(Element *c, uint64_t offset_c[AVX512_SIZE_], const Element *a8, const __m512i &b_, uint64_t offset_a) { Goldilocks::mul_avx512(c, offset_c, a8, b_, offset_a); }
    inline void mul_avx512(Element *c, uint64_t offset_c[AVX512_SIZE_], const __m512i &a_, const Element *b, uint64_t offset_b) { Goldilocks::mul_avx512(c, offset_c, a_, b, offset_b); }
    inline void mul_avx512(Element *c, uint64_t offset_c[AVX512_SIZE_], const Element *a8, const __m512i &b_, const uint64_t offset_a[AVX512_SIZE_]) { Goldilocks::mul_avx512(c, offset_c, a8, b_, offset_a); }
#endif
#endif