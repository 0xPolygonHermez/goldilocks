#ifndef GOLDILOCKS_CUBIC_EXTENSION_OPERATIONS_AVX
#define GOLDILOCKS_CUBIC_EXTENSION_OPERATIONS_AVX
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include <immintrin.h>
#include <cassert>

void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx a_, b_, c_;
    stride_a == 0 ? load_avx(a_, a) : load_avx(a_, a, stride_a);
    stride_b == 0 ? load_avx(b_, b) : load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
    stride_c == 0 ? store_avx(c, c_) : store_avx(c, c_, stride_c);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks3::Element_avx &b_)
{
    Element_avx a_, c_;
    stride_a == 0 ? load_avx(a_, a) : load_avx(a_, a, stride_a);
    op_avx(op, c_, a_, b_);
    stride_c == 0 ? store_avx(c, c_) : store_avx(c, c_, stride_c);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks3::Element_avx &a_, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx b_, c_;
    stride_b == 0 ? load_avx(b_, b) : load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
    stride_c == 0 ? store_avx(c, c_) : store_avx(c, c_, stride_c);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx a_, b_;
    stride_a == 0 ? load_avx(a_, a) : load_avx(a_, a, stride_a);
    stride_b == 0 ? load_avx(b_, b) : load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks3::Element_avx &a_, const Goldilocks3::Element_avx &b_)
{
    Element_avx c_;
    op_avx(op, c_, a_, b_);
    stride_c == 0 ? store_avx(c, c_) : store_avx(c, c_, stride_c);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks3::Element_avx &a_, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx b_;
    stride_b == 0 ? load_avx(b_, b) : load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks3::Element_avx &b_)
{
    Element_avx a_;
    stride_a == 0 ? load_avx(a_, a) : load_avx(a_, a, stride_a);
    op_avx(op, c_, a_, b_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks3::Element_avx &a_, const Goldilocks3::Element_avx &b_)
{
    switch (op)
    {
    case 0:
        Goldilocks3::add_avx(c_, a_, b_);
        break;
    case 1:
        Goldilocks3::sub_avx(c_, a_, b_);
        break;
    case 2:
        Goldilocks3::mul_avx(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
}

inline void add33c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a) {
    
}

#endif