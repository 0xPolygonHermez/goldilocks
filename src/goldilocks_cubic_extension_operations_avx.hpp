#ifndef GOLDILOCKS_CUBIC_EXTENSION_OPERATIONS_AVX
#define GOLDILOCKS_CUBIC_EXTENSION_OPERATIONS_AVX
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_base_field_avx.hpp"
#include <immintrin.h>
#include <cassert>

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

void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx a_, b_, c_;
    load_avx(a_, a, stride_a);
    load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks3::Element_avx &b_)
{
    Element_avx a_, c_;
    load_avx(a_, a, stride_a);
    op_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks3::Element_avx &a_, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx b_, c_;
    load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx a_, b_;
    load_avx(a_, a, stride_a);
    load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks3::Element_avx &a_, const Goldilocks3::Element_avx &b_)
{
    Element_avx c_;
    op_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks3::Element_avx &a_, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx b_;
    load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks3::Element_avx &b_)
{
    Element_avx a_;
    load_avx(a_, a, stride_a);
    op_avx(op, c_, a_, b_);
}

///////////////////////
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    switch (op)
    {
    case 0:
        add31_4rows(c, stride_c, a, stride_a, b, stride_b);
        break;
    case 1:
        sub31_4rows(c, stride_c, a, stride_a, b, stride_b);
        break;
    case 2:
        Element_avx c_, a_;
        __m256i b_;
        load_avx(a_, a, stride_a);
        Goldilocks::load_avx(b_, b, stride_b);
        mul31_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element &b)
{
    switch (op)
    {
    case 0:
        add31_4rows(c, stride_c, a, stride_a, b);
        break;
    case 1:
        sub31_4rows(c, stride_c, a, stride_a, b);
        break;
    case 2:
        Element_avx c_, a_;
        __m256i b_;
        load_avx(a_, a, stride_a);
        Goldilocks::load_avx(b_, b);
        mul31_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    switch (op)
    {
    case 0:
        add13_4rows(c, stride_c, a, stride_a, b, stride_b);
        break;
    case 1:
        sub13_4rows(c, stride_c, a, stride_a, b, stride_b);
        break;
    case 2:
        Element_avx c_, b_;
        __m256i a_;
        Goldilocks::load_avx(a_, a, stride_a);
        load_avx(b_, b, stride_b);
        mul13_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element &a, const Goldilocks::Element *b, uint64_t stride_b)
{
    switch (op)
    {
    case 0:
        add13_4rows(c, stride_c, a, b, stride_b);
        break;
    case 1:
        sub13_4rows(c, stride_c, a, b, stride_b);
        break;
    case 2:
        Element_avx c_, b_;
        __m256i a_;
        Goldilocks::load_avx(a_, a);
        load_avx(b_, b, stride_b);
        mul13_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const __m256i &b_)
{
    Goldilocks::Element b[4];

    switch (op)
    {
    case 0:

        Goldilocks::store_avx(b, b_);
        add31_4rows(c, stride_c, a, stride_a, b, 1);
        break;
    case 1:
        Goldilocks::store_avx(b, b_);
        sub31_4rows(c, stride_c, a, stride_a, b, 1);
        break;
    case 2:
        Element_avx c_, a_;
        load_avx(a_, a, stride_a);
        mul31_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const __m256i &a_, const Goldilocks::Element *b, uint64_t stride_b)
{
    Goldilocks::Element a[4];
    switch (op)
    {
    case 0:
        Goldilocks::store_avx(a, a_);
        add13_4rows(c, stride_c, a, 1, b, stride_b);
        break;
    case 1:
        Goldilocks::store_avx(a, a_);
        sub13_4rows(c, stride_c, a, 1, b, stride_b);
        break;
    case 2:
        Element_avx c_, b_;
        load_avx(b_, b, stride_b);
        mul13_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Element_avx &a_, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx c_;
    __m256i b_;
    switch (op)
    {
    case 0:

        Goldilocks::load_avx(b_, b, stride_b);
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        store_avx(c, stride_c, c_);
        break;
    case 1:
        Goldilocks::load_avx(b_, b, stride_b);
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        store_avx(c, stride_c, c_);
        break;
    case 2:
        Goldilocks::load_avx(b_, b, stride_b);
        mul31_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Element_avx &a_, const Goldilocks::Element &b)
{
    Element_avx c_;
    __m256i b_;
    switch (op)
    {
    case 0:

        Goldilocks::load_avx(b_, b);
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        store_avx(c, stride_c, c_);
        break;
    case 1:
        Goldilocks::load_avx(b_, b);
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        store_avx(c, stride_c, c_);
        break;
    case 2:
        Goldilocks::load_avx(b_, b);
        mul31_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Element_avx &b_)
{
    Element_avx c_;
    __m256i a_;
    switch (op)
    {
    case 0:

        Goldilocks::load_avx(a_, a, stride_a);
        Goldilocks::add_avx(c_[0], b_[0], a_);
        c_[1] = b_[1];
        c_[2] = b_[2];
        store_avx(c, stride_c, c_);
        break;
    case 1:
        Goldilocks::load_avx(a_, a, stride_a);
        Goldilocks::sub_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        store_avx(c, stride_c, c_);
        break;
    case 2:
        Goldilocks::load_avx(a_, a, stride_a);
        mul13_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element &a, const Element_avx &b_)
{
    Element_avx c_;
    __m256i a_;
    switch (op)
    {
    case 0:
        Goldilocks::load_avx(a_, a);
        Goldilocks::add_avx(c_[0], b_[0], a_);
        c_[1] = b_[1];
        c_[2] = b_[2];
        store_avx(c, stride_c, c_);
        break;
    case 1:
        Goldilocks::load_avx(a_, a);
        Goldilocks::sub_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        store_avx(c, stride_c, c_);
        break;
    case 2:
        Goldilocks::load_avx(a_, a);
        mul13_4rows(c_, a_, b_);
        store_avx(c, stride_c, c_);
        break;
    default:
        assert(0);
        break;
    }
};

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx a_;
    __m256i b_;
    load_avx(a_, a, stride_a);
    Goldilocks::load_avx(b_, b, stride_b);

    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 2:
        mul31_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element &b)
{
    Element_avx a_;
    __m256i b_;
    load_avx(a_, a, stride_a);
    Goldilocks::load_avx(b_, b);

    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 2:
        mul31_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx b_;
    __m256i a_;

    switch (op)
    {
    case 0:
        Goldilocks::load_avx(a_, a, stride_a);
        load_avx(b_, b, stride_b);
        Goldilocks::add_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        break;
    case 1:
        Goldilocks::Element c[12];
        sub13_4rows(c, 3, a, stride_a, b, stride_b);
        load_avx(c_, c, 3);
        break;
    case 2:
        Goldilocks::load_avx(a_, a, stride_a);
        load_avx(b_, b, stride_b);
        mul13_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element &a, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx b_;
    __m256i a_;

    switch (op)
    {
    case 0:
        Goldilocks::load_avx(a_, a);
        load_avx(b_, b, stride_b);
        Goldilocks::add_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        break;
    case 1:
        Goldilocks::Element c[12];
        sub13_4rows(c, 3, a, b, stride_b);
        load_avx(c_, c, 3);
        break;
    case 2:
        Goldilocks::load_avx(a_, a);
        load_avx(b_, b, stride_b);
        mul13_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
};

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const Element_avx &a_, const __m256i &b_)
{
    Element_avx c_;
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 2:
        mul31_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
    store_avx(c, stride_c, c_);
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c, const __m256i &a_, const Element_avx &b_)
{
    Element_avx c_;
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_, b_[0]);
        Goldilocks::sub_avx(c_[0], P, b_[1]);
        Goldilocks::sub_avx(c_[0], P, b_[2]);
        break;
    case 2:
        mul13_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
    store_avx(c, stride_c, c_);
};

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Element_avx &a_, const Goldilocks::Element *b, uint64_t stride_b)
{
    __m256i b_;
    Goldilocks::load_avx(b_, b, stride_b);
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 2:
        mul31_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Element_avx &a_, const Goldilocks::Element &b)
{
    __m256i b_;
    Goldilocks::load_avx(b_, b);
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 2:
        mul31_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
};
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const Element_avx &b_)
{
    __m256i a_;
    Goldilocks::load_avx(a_, a, stride_a);
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_, b_[0]);
        Goldilocks::sub_avx(c_[0], P, b_[1]);
        Goldilocks::sub_avx(c_[0], P, b_[2]);
        break;
    case 2:
        mul13_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
}
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element &a, const Element_avx &b_)
{
    __m256i a_;
    Goldilocks::load_avx(a_, a);
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_, b_[0]);
        Goldilocks::sub_avx(c_[0], P, b_[1]);
        Goldilocks::sub_avx(c_[0], P, b_[2]);
        break;
    case 2:
        mul13_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
}

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a, const __m256i &b_)
{
    Element_avx a_;
    load_avx(a_, a, stride_a);
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 2:
        mul31_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
}
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const __m256i &a_, const Goldilocks::Element *b, uint64_t stride_b)
{
    Element_avx b_;
    load_avx(b_, b, stride_b);
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_, b_[0]);
        Goldilocks::sub_avx(c_[0], P, b_[1]);
        Goldilocks::sub_avx(c_[0], P, b_[2]);
        break;
    case 2:
        mul13_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
}

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Element_avx &a_, const __m256i &b_)
{
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_[0], b_);
        c_[1] = a_[1];
        c_[2] = a_[2];
        break;
    case 2:
        mul31_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
}
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const __m256i &a_, const Element_avx &b_)
{
    switch (op)
    {
    case 0:
        Goldilocks::add_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
        break;
    case 1:
        Goldilocks::sub_avx(c_[0], a_, b_[0]);
        Goldilocks::sub_avx(c_[1], P, b_[1]);
        Goldilocks::sub_avx(c_[2], P, b_[2]);
        break;
    case 2:
        mul13_4rows(c_, a_, b_);
        break;
    default:
        assert(0);
        break;
    }
}

///////// GENERIC OPTIONS //////////
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, const uint64_t stride_c[4], const Goldilocks::Element *a, const uint64_t stride_a[4], const Goldilocks::Element *b, const uint64_t stride_b[4])
{
    Element_avx a_, b_, c_;
    load_avx(a_, a, stride_a);
    load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, const uint64_t stride_c[4], const Goldilocks::Element *a, const uint64_t stride_a[4], const Element_avx &b_)
{
    Element_avx a_, c_;
    load_avx(a_, a, stride_a);
    op_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, const uint64_t stride_c[4], const Element_avx &a_, const Goldilocks::Element *b, const uint64_t stride_b[4])
{
    Element_avx b_, c_;
    load_avx(b_, b, stride_b);
    op_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a4, uint64_t const stride_a[4], const Goldilocks::Element *b4, uint64_t const stride_b[4])
{
    Element_avx a_, b_;
    load_avx(a_, a4, stride_a);
    load_avx(b_, b4, stride_b);
    op_avx(op, c_, a_, b_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks::Element *c, const uint64_t stride_c[4], const Element_avx &a_, const Element_avx &b_)
{
    Element_avx c_;
    op_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks3::Element_avx &a_, const Goldilocks::Element *b4, uint64_t const stride_b[4])
{
    Element_avx b_;
    load_avx(b_, b4, stride_b);
    op_avx(op, c_, a_, b_);
};
void Goldilocks3::op_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a4, uint64_t const stride_a[4], const Goldilocks3::Element_avx &b_)
{
    Element_avx a_;
    load_avx(a_, a4, stride_a);
    op_avx(op, c_, a_, b_);
};

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c[4], const Goldilocks::Element *a, uint64_t stride_a[4], const Goldilocks::Element *b, uint64_t stride_b[4])
{
    Element_avx a_, c_;
    __m256i b_;
    load_avx(a_, a, stride_a);
    Goldilocks::load_avx(b_, b, stride_b);
    op_31_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c[4], const Goldilocks::Element *a, uint64_t stride_a[4], const __m256i &b_)
{
    Element_avx a_, c_;
    load_avx(a_, a, stride_a);
    op_31_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
};
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c[4], const Goldilocks3::Element_avx &a_, const Goldilocks::Element *b, uint64_t stride_b[4])
{
    Element_avx c_;
    __m256i b_;
    Goldilocks::load_avx(b_, b, stride_b);
    op_31_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
};
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks::Element *c, uint64_t stride_c[4], const Goldilocks3::Element_avx &a_, const __m256i &b_)
{
    Element_avx c_;
    op_31_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
};

void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, uint64_t stride_a[4], const Goldilocks::Element *b, uint64_t stride_b[4])
{
    Element_avx a_;
    __m256i b_;
    load_avx(a_, a, stride_a);
    Goldilocks::load_avx(b_, b, stride_b);
    op_31_avx(op, c_, a_, b_);
};
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a4, uint64_t stride_a[4], const __m256i &b_)
{
    Element_avx a_;
    load_avx(a_, a4, stride_a);
    op_31_avx(op, c_, a_, b_);
};
void Goldilocks3::op_31_avx(uint64_t op, Goldilocks3::Element_avx &c_, Goldilocks3::Element_avx &a_, const Goldilocks::Element *b, uint64_t stride_b[4])
{
    __m256i b_;
    Goldilocks::load_avx(b_, b, stride_b);
    op_31_avx(op, c_, a_, b_);
};

void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, const uint64_t stride_c[4], const Goldilocks::Element *a, const uint64_t stride_a[4], const Goldilocks::Element *b, const uint64_t stride_b[4])
{
    Element_avx b_, c_;
    __m256i a_;
    Goldilocks::load_avx(a_, a, stride_a);
    load_avx(b_, b, stride_b);
    op_13_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
}
void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, const uint64_t stride_c[4], const __m256i &a_, const Goldilocks::Element *b, const uint64_t stride_b[4])
{
    Element_avx b_, c_;
    load_avx(b_, b, stride_b);
    op_13_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
};

void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, const uint64_t stride_c[4], const Goldilocks::Element *a, const uint64_t stride_a[4], const Goldilocks3::Element_avx &b_)
{
    Element_avx c_;
    __m256i a_;
    Goldilocks::load_avx(a_, a, stride_a);
    op_13_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
};

void Goldilocks3::op_13_avx(uint64_t op, Goldilocks::Element *c, const uint64_t stride_c[4], const __m256i &a_, const Goldilocks3::Element_avx &b_)
{
    Element_avx c_;
    op_13_avx(op, c_, a_, b_);
    store_avx(c, stride_c, c_);
};

void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, const uint64_t stride_a[4], const Goldilocks::Element *b, const uint64_t stride_b[4])
{
    Element_avx b_;
    __m256i a_;
    Goldilocks::load_avx(a_, a, stride_a);
    load_avx(b_, b, stride_b);
    op_13_avx(op, c_, a_, b_);
};

void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const Goldilocks::Element *a, const uint64_t stride_a[4], const Goldilocks3::Element_avx &b_)
{
    __m256i a_;
    Goldilocks::load_avx(a_, a, stride_a);
    op_13_avx(op, c_, a_, b_);
};

void Goldilocks3::op_13_avx(uint64_t op, Goldilocks3::Element_avx &c_, const __m256i &a_, const Goldilocks::Element *b, const uint64_t offsets_b[4])
{
    Element_avx b_;
    load_avx(b_, b, offsets_b);
    op_13_avx(op, c_, a_, b_);
};

///////////////////////
///////////////////////

void Goldilocks3::add31_4rows(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    for (uint64_t k = 0; k < AVX_SIZE_; ++k)
    {
        c[k * stride_c] = a[k * stride_a] + b[stride_b * k];
        c[k * stride_c + 1] = a[k * stride_a + 1];
        c[k * stride_c + 2] = a[k * stride_a + 2];
    }
}
void Goldilocks3::add13_4rows(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    add31_4rows(c, stride_c, b, stride_b, a, stride_a);
}
void Goldilocks3::add31_4rows(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element &b)
{
    for (uint64_t k = 0; k < AVX_SIZE_; ++k)
    {
        c[k * stride_c] = a[k * stride_a] + b;
        c[k * stride_c + 1] = a[k * stride_a + 1];
        c[k * stride_c + 2] = a[k * stride_a + 2];
    }
}
void Goldilocks3::add13_4rows(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element &a, const Goldilocks::Element *b, uint64_t stride_b)
{
    add31_4rows(c, stride_c, b, stride_b, a);
}

void Goldilocks3::sub31_4rows(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    for (uint64_t k = 0; k < AVX_SIZE_; ++k)
    {
        c[k * stride_c] = a[k * stride_a] - b[stride_b * k];
        c[k * stride_c + 1] = a[k * stride_a + 1];
        c[k * stride_c + 2] = a[k * stride_a + 2];
    }
}
void Goldilocks3::sub31_4rows(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element &b)
{
    for (uint64_t k = 0; k < AVX_SIZE_; ++k)
    {
        c[k * stride_c] = a[k * stride_a] - b;
        c[k * stride_c + 1] = a[k * stride_a + 1];
        c[k * stride_c + 2] = a[k * stride_a + 2];
    }
}
void Goldilocks3::sub13_4rows(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, uint64_t stride_a, const Goldilocks::Element *b, uint64_t stride_b)
{
    for (uint64_t k = 0; k < AVX_SIZE_; ++k)
    {
        c[k * stride_c] = a[stride_a * k] - b[k * stride_b];
        c[k * stride_c + 1] = -b[k * stride_b + 1];
        c[k * stride_c + 2] = -b[k * stride_b + 2];
    }
}

void Goldilocks3::sub13_4rows(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element &a, const Goldilocks::Element *b, uint64_t stride_b)
{
    for (uint64_t k = 0; k < AVX_SIZE_; ++k)
    {
        c[k * stride_c] = a - b[k * stride_b];
        c[k * stride_c + 1] = -b[k * stride_b + 1];
        c[k * stride_c + 2] = -b[k * stride_b + 2];
    }
}
void Goldilocks3::mul13_4rows(Goldilocks3::Element_avx &c_, const __m256i &a_, const Goldilocks3::Element_avx &b_)
{
    Goldilocks::mult_avx(c_[0], a_, b_[0]);
    Goldilocks::mult_avx(c_[1], a_, b_[1]);
    Goldilocks::mult_avx(c_[2], a_, b_[2]);
}
void Goldilocks3::mul31_4rows(Goldilocks3::Element_avx &c_, const Goldilocks3::Element_avx &a_, const __m256i &b_)
{
    Goldilocks::mult_avx(c_[0], b_, a_[0]);
    Goldilocks::mult_avx(c_[1], b_, a_[1]);
    Goldilocks::mult_avx(c_[2], b_, a_[2]);
}

#endif