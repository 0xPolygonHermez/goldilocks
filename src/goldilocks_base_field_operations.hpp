#ifndef GOLDILOCKS_OPERATIONS_AVX
#define GOLDILOCKS_OPERATIONS_AVX
#include "goldilocks_base_field.hpp"
#include "goldilocks_base_field_avx.hpp"
#include <immintrin.h>
#include <cassert>

//
// Operands Element* or __m256i&
//
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const __m256i &a_, const __m256i &b_)
{
    switch (op)
    {
    case 0:
        add_avx(c_, a_, b_);
        break;
    case 1:
        sub_avx(c_, a_, b_);
        break;
    case 2:
        mult_avx(c_, a_, b_);
        break;
    case 3:
        sub_avx(c_, b_, a_);
        break;
    default:
        assert(0);
        break;
    }
};

void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const Element *a4, uint64_t stride_a, const Element *b4, uint64_t stride_b)
{
    __m256i a_, b_, c_;
    stride_a == 1 ? load_avx(a_, a4) : load_avx(a_, a4, stride_a);
    stride_b == 1 ? load_avx(b_, b4) : load_avx(b_, b4, stride_b);
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
};
void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const Element *a4, uint64_t stride_a, const __m256i &b_)
{
    __m256i a_, c_;
    stride_a == 1 ? load_avx(a_, a4) : load_avx(a_, a4, stride_a);
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
};
void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const __m256i &a_, const Element *b4, uint64_t stride_b)
{
    __m256i b_, c_;
    stride_b == 1 ? load_avx(b_, b4) : load_avx(b_, b4, stride_b);
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
};
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const Element *a4, uint64_t stride_a, const Element *b4, uint64_t stride_b)
{
    __m256i a_, b_;
    stride_a == 1 ? load_avx(a_, a4) : load_avx(a_, a4, stride_a);
    stride_b == 1 ? load_avx(b_, b4) : load_avx(b_, b4, stride_b);
    op_avx(op, c_, a_, b_);
};
void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const __m256i &a_, const __m256i &b_)
{
    __m256i c_;
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
};
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const Element *a4, uint64_t stride_a, const __m256i &b_)
{
    __m256i a_;
    stride_a == 1 ? load_avx(a_, a4) : load_avx(a_, a4, stride_a);
    op_avx(op, c_, a_, b_);
};
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const __m256i &a_, const Element *b4, uint64_t stride_b)
{
    __m256i b_;
    stride_b == 1 ? load_avx(b_, b4) : load_avx(b_, b4, stride_b);
    op_avx(op, c_, a_, b_);
};

// Argument b being Element& (constant for all 4 lanes)
void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const Element *a4, uint64_t stride_a, const Element &b)
{
    __m256i a_, b_, c_;
    stride_a == 1 ? load_avx(a_, a4) : load_avx(a_, a4, stride_a);
    load_avx(b_, b);
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
};
void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const __m256i &a_, const Element &b)
{
    __m256i b_, c_;
    load_avx(b_, b);
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
};
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const Element *a4, uint64_t stride_a, const Element &b)
{
    __m256i a_, b_;
    stride_a == 1 ? load_avx(a_, a4) : load_avx(a_, a4, stride_a);
    load_avx(b_, b);
    op_avx(op, c_, a_, b_);
};
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const __m256i &a_, const Element &b)
{
    __m256i b_;
    load_avx(b_, b);
    op_avx(op, c_, a_, b_);
};

// Argument a being Element& (only to be uses with op=SUB)
void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const Element &a, const Element *b, uint64_t stride_b)
{
    __m256i a_, b_, c_;
    load_avx(a_, a);
    stride_b == 1 ? load_avx(b_, b) : load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
};
void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const Element &a, const __m256i &b_)
{
    __m256i a_, c_;
    load_avx(a_, a);
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
};
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const Element &a, const Element *b, uint64_t stride_b)
{
    __m256i a_, b_;
    load_avx(a_, a);
    stride_b == 1 ? load_avx(b_, b) : load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
};
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const Element &a, const __m256i &b_)
{
    __m256i a_;
    load_avx(a_, a);
    op_avx(op, c_, a_, b_);
};

// Argument a and b being Element&
void Goldilocks::op_avx(uint64_t op, Element *c4, uint64_t stride_c, const Element &a, const Element &b)
{
    __m256i a_, b_, c_;
    load_avx(a_, a);
    load_avx(b_, b);
    op_avx(op, c_, a_, b_);
    stride_c == 1 ? store_avx(c4, c_) : store_avx(c4, stride_c, c_);
}

void Goldilocks::op_avx(uint64_t op, __m256i &c_, const Element &a, const Element &b)
{
    __m256i a_, b_;
    load_avx(a_, a);
    load_avx(b_, b);
    op_avx(op, c_, a_, b_);
}

// generic option to be used in first and last block of rows: allways operated with Element*
void Goldilocks::op_avx(uint64_t op, Element *c4, const uint64_t offsets_c[4], const Element *a4, const uint64_t offsets_a[4], const Element *b4, const uint64_t offsets_b[4])
{
    __m256i a_, b_, c_;
    load_avx(a_, a4, offsets_a);
    load_avx(b_, b4, offsets_b);
    op_avx(op, c_, a_, b_);
    store_avx(c4, offsets_c, c_);
};
void Goldilocks::op_avx(uint64_t op, Element *c4, const uint64_t offsets_c[4], const __m256i &a_, const Element *b4, const uint64_t offsets_b[4])
{
    __m256i b_, c_;
    load_avx(b_, b4, offsets_b);
    op_avx(op, c_, a_, b_);
    store_avx(c4, offsets_c, c_);
};
void Goldilocks::op_avx(uint64_t op, Element *c4, const uint64_t offsets_c[4], const Element *a4, const uint64_t offsets_a[4], const __m256i &b_)
{
    __m256i a_, c_;
    load_avx(a_, a4, offsets_a);
    op_avx(op, c_, a_, b_);
    store_avx(c4, offsets_c, c_);
};
void Goldilocks::op_avx(uint64_t op, Element *c4, const uint64_t offsets_c[4], const __m256i &a_, const __m256i &b_)
{
    __m256i c_;
    op_avx(op, c_, a_, b_);
    store_avx(c4, offsets_c, c_);
};

void Goldilocks::op_avx(uint64_t op, __m256i &c_, const Element *a4, const uint64_t offsets_a[4], const Element *b4, const uint64_t offsets_b[4])
{
    __m256i a_, b_;
    load_avx(a_, a4, offsets_a);
    load_avx(b_, b4, offsets_b);
    op_avx(op, c_, a_, b_);
};
void Goldilocks::op_avx(uint64_t op, __m256i &c_, const __m256i &a_, const Element *b4, const uint64_t offsets_b[4])
{
    __m256i b_;
    load_avx(b_, b4, offsets_b);
    op_avx(op, c_, a_, b_);
};

void Goldilocks::op_avx(uint64_t op, __m256i &c_, const Element *a4, const uint64_t offsets_a[4], const __m256i &b_)
{
    __m256i a_;
    load_avx(a_, a4, offsets_a);
    op_avx(op, c_, a_, b_);
};

#endif