#ifndef GOLDILOCKS_CUBIC_EXTENSION_NEON_HPP
#define GOLDILOCKS_CUBIC_EXTENSION_NEON_HPP

#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_base_field.hpp"

#ifdef GOLDILOCKS_HAS_NEON
#include <arm_neon.h>

// Auto-generated NEON siblings of the AVX2 methods in goldilocks_cubic_extension.hpp.
// Each method is defined as Goldilocks3::<name>_neon and declared in the class body
// under #ifdef GOLDILOCKS_HAS_NEON.

    inline void Goldilocks3::copy_neon(Goldilocks::Element *dst, const uint64x2_t a0_, const uint64x2_t a1_, const uint64x2_t a2_)
    {
        Goldilocks::Element buff0[2], buff1[4], buff2[4];
        Goldilocks::store_neon(buff0, a0_);
        Goldilocks::store_neon(buff1, a1_);
        Goldilocks::store_neon(buff2, a2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            Goldilocks::copy(dst[k * FIELD_EXTENSION], buff0[k]);
            Goldilocks::copy(dst[k * FIELD_EXTENSION + 1], buff1[k]);
            Goldilocks::copy(dst[k * FIELD_EXTENSION + 2], buff2[k]);
        }
    };

    inline void Goldilocks3::add_neon(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {

        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, a);
        Goldilocks::load_neon(a1_, &a[2]);
        Goldilocks::load_neon(a2_, &a[4]);
        Goldilocks::load_neon(b0_, b);
        Goldilocks::load_neon(b1_, &b[2]);
        Goldilocks::load_neon(b2_, &b[4]);

        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::add_neon(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element bb[6];
        Goldilocks::Element aa[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                bb[k * FIELD_EXTENSION + i] = b[k * stride_b + i];
                aa[k * FIELD_EXTENSION + i] = a[k * stride_a + i];
            }
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);

        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::add31_neon(Goldilocks::Element *result, Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element bb[6];
        Goldilocks::Element aa[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            bb[k * FIELD_EXTENSION] = b[stride_b * k];
            bb[k * FIELD_EXTENSION + 1] = Goldilocks::zero();
            bb[k * FIELD_EXTENSION + 2] = Goldilocks::zero();
            aa[k * FIELD_EXTENSION] = a[stride_a * k];
            aa[k * FIELD_EXTENSION + 1] = a[stride_a * k + 1];
            aa[k * FIELD_EXTENSION + 2] = a[stride_a * k + 2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);

        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::add13_neon(Goldilocks::Element *result, Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        Goldilocks::Element aa[6];
        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = Goldilocks::zero();
            aa[k * FIELD_EXTENSION + 2] = Goldilocks::zero();
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, b);
        Goldilocks::load_neon(b1_, &b[2]);
        Goldilocks::load_neon(b2_, &b[4]);

        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::add1c3c_neon(Goldilocks::Element *result, const Goldilocks::Element a, const Goldilocks::Element *b)
    {
        // does not make sense to vectorise
        Goldilocks::Element res0 = b[0] + a;
        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = res0;
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }

    inline void Goldilocks3::add13c_neon(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        // does not make sense to vectorise
        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k] + b[0];
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }

    inline void Goldilocks3::add13_neon(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a, uint64_t offset_b)
    {
        // does not make sense to vectorize
        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * offset_a] + b[k * offset_b];
            result[k * FIELD_EXTENSION + 1] = b[k * offset_b + 1];
            result[k * FIELD_EXTENSION + 2] = b[k * offset_b + 2];
        }
    }

    inline void Goldilocks3::add13c_neon(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a)
    {
        // does not make sense to vectorize
        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * offset_a] + b[0];
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }

    inline void Goldilocks3::add33c_neon(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        Goldilocks::Element bb[6];
        for (uint64_t k = 0; k < 2; ++k)
        {
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, a);
        Goldilocks::load_neon(a1_, &a[2]);
        Goldilocks::load_neon(a2_, &a[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);

        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::add33c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element bb[6];
        Goldilocks::Element aa[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
            aa[k * FIELD_EXTENSION] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            aa[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);

        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::add13_neon(Goldilocks::Element *result, const uint64x2_t &a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element a[2];
        Goldilocks::store_neon(a, a_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = Goldilocks::zero();
            aa[k * FIELD_EXTENSION + 2] = Goldilocks::zero();
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, b);
        Goldilocks::load_neon(b1_, &b[2]);
        Goldilocks::load_neon(b2_, &b[4]);

        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::add13c_neon(Goldilocks::Element *result, const uint64x2_t &a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element a[2];
        Goldilocks::store_neon(a, a_);
        // does not make sense to vectorise
        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k] + b[0];
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }

    inline void Goldilocks3::add13_neon(Goldilocks3::Element_neon c_, const uint64x2_t &a_, Goldilocks3::Element_neon b_)
    {
        Goldilocks::add_neon(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
    }

    inline void Goldilocks3::add13c_neon(Goldilocks3::Element_neon c_, const uint64x2_t &a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t b0_, b1_, b2_;
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        Goldilocks::add_neon(c_[0], a_, b0_);
        c_[1] = b1_;
        c_[2] = b2_;
    }

    inline void Goldilocks3::add1c3c_neon(Goldilocks3::Element_neon c_, const Goldilocks::Element a, const Goldilocks::Element *b)
    {
        // does not make sense to vectorise
        Goldilocks::Element res0 = b[0] + a;
        Goldilocks::Element c0[2];
        Goldilocks::Element c1[2];
        Goldilocks::Element c2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            c0[k] = res0;
            c1[k] = b[1];
            c2[k] = b[2];
        }
        Goldilocks::load_neon(c_[0], c0);
        Goldilocks::load_neon(c_[1], c1);
        Goldilocks::load_neon(c_[2], c2);
    }

    inline void Goldilocks3::add13_neon(Goldilocks3::Element_neon c_, const Goldilocks::Element *a, Goldilocks3::Element_neon b_, uint64_t offset_a)
    {
        Goldilocks::Element a0[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * offset_a];
        }
        uint64x2_t a0_;
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::add_neon(c_[0], a0_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
    }

    inline void Goldilocks3::add13c_neon(Goldilocks3::Element_neon c_, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a)
    {
        Goldilocks::Element c0[2];
        Goldilocks::Element c1[2];
        Goldilocks::Element c2[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            c0[k] = a[k * offset_a] + b[0];
            c1[k] = b[1];
            c2[k] = b[2];
        }
        Goldilocks::load_neon(c_[0], c0);
        Goldilocks::load_neon(c_[1], c1);
        Goldilocks::load_neon(c_[2], c2);
    }

    inline void Goldilocks3::add_neon(Goldilocks3::Element_neon c_, Goldilocks3::Element_neon a_, Goldilocks3::Element_neon b_)
    {
        Goldilocks::add_neon(c_[0], a_[0], b_[0]);
        Goldilocks::add_neon(c_[1], a_[1], b_[1]);
        Goldilocks::add_neon(c_[2], a_[2], b_[2]);
    }

    inline void Goldilocks3::add33c_neon(Goldilocks3::Element_neon c_, Goldilocks3::Element_neon a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        Goldilocks::add_neon(c_[0], a_[0], b0_);
        Goldilocks::add_neon(c_[1], a_[1], b1_);
        Goldilocks::add_neon(c_[2], a_[2], b2_);
    }

    inline void Goldilocks3::add_neon(Goldilocks3::Element_neon c_, const Goldilocks::Element *a, Goldilocks3::Element_neon b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[2];
        Goldilocks::Element a1[2];
        Goldilocks::Element a2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        uint64x2_t a0_, a1_, a2_;

        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);

        Goldilocks::add_neon(c_[0], a0_, b_[0]);
        Goldilocks::add_neon(c_[1], a1_, b_[1]);
        Goldilocks::add_neon(c_[2], a2_, b_[2]);
    }

    inline void Goldilocks3::add33c_neon(Goldilocks3::Element_neon c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element a0[2];
        Goldilocks::Element a1[2];
        Goldilocks::Element a2[2];
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        Goldilocks::add_neon(c_[0], a0_, b0_);
        Goldilocks::add_neon(c_[1], a1_, b1_);
        Goldilocks::add_neon(c_[2], a2_, b2_);
    }

    inline void Goldilocks3::add13_neon(Goldilocks::Element *c, uint64_t stride_c, const uint64x2_t &a_, Goldilocks3::Element_neon b_)
    {
        uint64x2_t c0_;
        Goldilocks::add_neon(c0_, a_, b_[0]);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, b_[1]);
        Goldilocks::store_neon(c2, b_[2]);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }

    inline void Goldilocks3::add_neon(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, Goldilocks3::Element_neon b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[2];
        Goldilocks::Element a1[2];
        Goldilocks::Element a2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        uint64x2_t a0_, a1_, a2_;

        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);

        uint64x2_t c0_, c1_, c2_;
        Goldilocks::add_neon(c0_, a0_, b_[0]);
        Goldilocks::add_neon(c1_, a1_, b_[1]);
        Goldilocks::add_neon(c2_, a2_, b_[2]);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }

    inline void Goldilocks3::add_neon(Goldilocks::Element *c, uint64_t stride_c, Goldilocks3::Element_neon a_, Goldilocks3::Element_neon b_)
    {
        uint64x2_t c0_, c1_, c2_;
        Goldilocks::add_neon(c0_, a_[0], b_[0]);
        Goldilocks::add_neon(c1_, a_[1], b_[1]);
        Goldilocks::add_neon(c2_, a_[2], b_[2]);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }

    inline void Goldilocks3::add33c_neon(Goldilocks::Element *c, uint64_t stride_c, Goldilocks3::Element_neon a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        uint64x2_t c0_, c1_, c2_;
        Goldilocks::add_neon(c0_, a_[0], b0_);
        Goldilocks::add_neon(c1_, a_[1], b1_);
        Goldilocks::add_neon(c2_, a_[2], b2_);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }

    inline void Goldilocks3::add13_neon(Goldilocks::Element *c, uint64_t stride_c[2], const uint64x2_t &a_, Goldilocks3::Element_neon b_)
    {
        uint64x2_t c0_;
        Goldilocks::add_neon(c0_, a_, b_[0]);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, b_[1]);
        Goldilocks::store_neon(c2, b_[2]);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }

    inline void Goldilocks3::add_neon(Goldilocks::Element *c, uint64_t stride_c[2], const Goldilocks::Element *a, Goldilocks3::Element_neon b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[2];
        Goldilocks::Element a1[2];
        Goldilocks::Element a2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        uint64x2_t a0_, a1_, a2_;

        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);

        uint64x2_t c0_, c1_, c2_;
        Goldilocks::add_neon(c0_, a0_, b_[0]);
        Goldilocks::add_neon(c1_, a1_, b_[1]);
        Goldilocks::add_neon(c2_, a2_, b_[2]);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }

    inline void Goldilocks3::add33c_neon(Goldilocks::Element *c, uint64_t stride_c[2], Goldilocks3::Element_neon a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        uint64x2_t c0_, c1_, c2_;
        Goldilocks::add_neon(c0_, a_[0], b0_);
        Goldilocks::add_neon(c1_, a_[1], b1_);
        Goldilocks::add_neon(c2_, a_[2], b2_);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }

    inline void Goldilocks3::add_neon(uint64x2_t &c0_, uint64x2_t &c1_, uint64x2_t &c2_, const uint64x2_t a0_, const uint64x2_t a1_, const uint64x2_t a2_, const uint64x2_t b0_, const uint64x2_t b1_, const uint64x2_t b2_)
    {
        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);
    }

    inline void Goldilocks3::add_neon(uint64x2_t &c0_, uint64x2_t &c1_, uint64x2_t &c2_, const uint64x2_t a0_, const uint64x2_t a1_, const uint64x2_t a2_, const Goldilocks::Element *b, uint64_t stride)
    {
        Goldilocks::Element b0[2], b1[4], b2[4];
        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[k * stride];
            b1[k] = b[k * stride + 1];
            b2[k] = b[k * stride + 2];
        }
        uint64x2_t b0_, b1_, b2_;
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);
        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);
    }

    inline void Goldilocks3::add31_neon(uint64x2_t &c0_, uint64x2_t &c1_, uint64x2_t &c2_, uint64x2_t a0_, const uint64x2_t a1_, const uint64x2_t a2_, const Goldilocks::Element *b, uint64_t stride)
    {
        Goldilocks::Element b0[2], b1[4], b2[4];
        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[k * stride];
            b1[k] = Goldilocks::zero();
            b2[k] = Goldilocks::zero();
        }
        uint64x2_t b0_, b1_, b2_;
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        Goldilocks::add_neon(c0_, a0_, b0_);
        Goldilocks::add_neon(c1_, a1_, b1_);
        Goldilocks::add_neon(c2_, a2_, b2_);
    }

    inline void Goldilocks3::sub33c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {

        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                aa[k * FIELD_EXTENSION + i] = a[k * stride_a + i];
                bb[k * FIELD_EXTENSION + i] = b[i];
            }
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);

        Goldilocks::sub_neon(c0_, a0_, b0_);
        Goldilocks::sub_neon(c1_, a1_, b1_);
        Goldilocks::sub_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::sub31_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint32_t stride_b)
    {
        // Rick: does not make sense to vectorize
        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] - b[k * stride_b];
            result[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            result[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
    }

    inline void Goldilocks3::sub31c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element b, uint64_t stride_a)
    {
        // Rick: does not make sense to vectorize
        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] - b;
            result[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            result[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
    }

    inline void Goldilocks3::sub_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, a);
        Goldilocks::load_neon(a1_, &a[2]);
        Goldilocks::load_neon(a2_, &a[4]);
        Goldilocks::load_neon(b0_, b);
        Goldilocks::load_neon(b1_, &b[2]);
        Goldilocks::load_neon(b2_, &b[4]);

        Goldilocks::sub_neon(c0_, a0_, b0_);
        Goldilocks::sub_neon(c1_, a1_, b1_);
        Goldilocks::sub_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::sub33c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                aa[k * FIELD_EXTENSION + i] = a[k * FIELD_EXTENSION + i];
                bb[k * FIELD_EXTENSION + i] = b[i];
            }
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);

        Goldilocks::sub_neon(c0_, a0_, b0_);
        Goldilocks::sub_neon(c1_, a1_, b1_);
        Goldilocks::sub_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::sub_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element bb[6];
        Goldilocks::Element aa[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                bb[k * FIELD_EXTENSION + i] = b[k * stride_b + i];
                aa[k * FIELD_EXTENSION + i] = a[k * stride_a + i];
            }
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);

        Goldilocks::sub_neon(c0_, a0_, b0_);
        Goldilocks::sub_neon(c1_, a1_, b1_);
        Goldilocks::sub_neon(c2_, a2_, b2_);

        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::sub31c_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element b, const uint64_t stride_a[2])
    {
        Goldilocks::Element c0[2], c1[4], c2[4];

        for (uint64_t k = 0; k < 2; ++k)
        {
            c0[k] = a[stride_a[k]] - b;
            c1[k] = a[stride_a[k] + 1];
            c2[k] = a[stride_a[k] + 2];
        }
        Goldilocks::load_neon(c_[0], c0);
        Goldilocks::load_neon(c_[1], c1);
        Goldilocks::load_neon(c_[2], c2);

    }

    inline void Goldilocks3::sub31c_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element b, uint64_t stride_a)
    {
        Goldilocks::Element c0[2], c1[4], c2[4];

        for (uint64_t k = 0; k < 2; ++k)
        {
            c0[k] = a[k * stride_a] - b;
            c1[k] = a[k * stride_a + 1];
            c2[k] = a[k * stride_a + 2];
        }
        Goldilocks::load_neon(c_[0], c0);
        Goldilocks::load_neon(c_[1], c1);
        Goldilocks::load_neon(c_[2], c2);
    }

    inline void Goldilocks3::sub_neon(Goldilocks3::Element_neon &c_, Goldilocks3::Element_neon a_, Goldilocks3::Element_neon b_)
    {
        Goldilocks::sub_neon(c_[0], a_[0], b_[0]);
        Goldilocks::sub_neon(c_[1], a_[1], b_[1]);
        Goldilocks::sub_neon(c_[2], a_[2], b_[2]);
    }

    inline void Goldilocks3::sub33c_neon(Goldilocks3::Element_neon &c_, Goldilocks3::Element_neon a_, Goldilocks::Element *b)
    {
        Goldilocks::Element b0[2], b1[4], b2[4];
        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        Goldilocks::sub_neon(c_[0], a_[0], b0_);
        Goldilocks::sub_neon(c_[1], a_[1], b1_);
        Goldilocks::sub_neon(c_[2], a_[2], b2_);
    }

     inline void Goldilocks3::sub33c_neon(Goldilocks::Element *c, uint64_t stride_c, Goldilocks3::Element_neon a_, Goldilocks3::Element_neon b_)
    {
        uint64x2_t c0_, c1_, c2_;
        Goldilocks::sub_neon(c0_, a_[0], b_[0]);
        Goldilocks::sub_neon(c1_, a_[1], b_[1]);
        Goldilocks::sub_neon(c2_, a_[2], b_[2]);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }

    inline void Goldilocks3::sub_neon(Goldilocks3::Element_neon &c_, Goldilocks3::Element_neon a_, Goldilocks::Element *b, uint64_t stride_b)
    {
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[k * stride_b];
            b1[k] = b[k * stride_b + 1];
            b2[k] = b[k * stride_b + 2];
        }
        uint64x2_t b0_, b1_, b2_;
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        Goldilocks::sub_neon(c_[0], a_[0], b0_);
        Goldilocks::sub_neon(c_[1], a_[1], b1_);
        Goldilocks::sub_neon(c_[2], a_[2], b2_);
    }

    inline void Goldilocks3::sub33c_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {

        Goldilocks::Element a0[2];
        Goldilocks::Element a1[2];
        Goldilocks::Element a2[2];
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        Goldilocks::sub_neon(c_[0], a0_, b0_);
        Goldilocks::sub_neon(c_[1], a1_, b1_);
        Goldilocks::sub_neon(c_[2], a2_, b2_);
    }

    inline void Goldilocks3::sub13c_neon(uint64x2_t &c0_, uint64x2_t &c1_, uint64x2_t &c2_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element nb1 = Goldilocks::neg(b[1]);
        Goldilocks::Element nb2 = Goldilocks::neg(b[2]);
        Goldilocks::Element c0[2], c1[4], c2[4];
        for (uint64_t k = 0; k < 2; ++k)
        {
            c0[k] = a[k * stride_a] - b[0];
            c1[k] = nb1;
            c2[k] = nb2;
        }
        Goldilocks::load_neon(c0_, c0);
        Goldilocks::load_neon(c1_, c1);
        Goldilocks::load_neon(c2_, c2);
    }

    inline void Goldilocks3::sub33c_neon(uint64x2_t &c0_, uint64x2_t &c1_, uint64x2_t &c2_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {

        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        Goldilocks::sub_neon(c0_, a0_, b0_);
        Goldilocks::sub_neon(c1_, a1_, b1_);
        Goldilocks::sub_neon(c2_, a2_, b2_);
    }

    inline void Goldilocks3::mul13c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Element &b, uint64_t stride_a)
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 1] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 2] = a[k * stride_a];
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);
        Goldilocks::mult_neon(c0_, a0_, b0_);
        Goldilocks::mult_neon(c1_, a1_, b1_);
        Goldilocks::mult_neon(c2_, a2_, b2_);
        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::mul1c3c_neon(Goldilocks::Element *result, Goldilocks::Element a, Element &b)
    {
        // Does not make sense to vectorize
        result[0] = b[0] * a;
        result[1] = b[1] * a;
        result[2] = b[2] * a;
        for (uint64_t k = 1; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = result[0];
            result[k * FIELD_EXTENSION + 1] = result[1];
            result[k * FIELD_EXTENSION + 2] = result[2];
        }
    }

    inline void Goldilocks3::mul13c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Element &b, const uint64_t stride_a[2])
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[stride_a[k]];
            aa[k * FIELD_EXTENSION + 1] = a[stride_a[k]];
            aa[k * FIELD_EXTENSION + 2] = a[stride_a[k]];
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);
        Goldilocks::mult_neon(c0_, a0_, b0_);
        Goldilocks::mult_neon(c1_, a1_, b1_);
        Goldilocks::mult_neon(c2_, a2_, b2_);
        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::mul_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:

        for (uint64_t k = 0; k < 2; ++k)
        {

            a0[k] = a[k * FIELD_EXTENSION];
            a1[k] = a[k * FIELD_EXTENSION + 1];
            a2[k] = a[k * FIELD_EXTENSION + 2];
            b0[k] = b[k * FIELD_EXTENSION];
            b1[k] = b[k * FIELD_EXTENSION + 1];
            b2[k] = b[k * FIELD_EXTENSION + 2];
        }
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[2], result1[4], result2[4];

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b0_, b1_);
        Goldilocks::add_neon(aux1_, b0_, b2_);
        Goldilocks::add_neon(aux2_, b1_, b2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(result0_, C_, G_);
        Goldilocks::sub_neon(result0_, result0_, F_);
        Goldilocks::add_neon(result1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(result1_, result1_, auxr_);
        Goldilocks::sub_neon(result2_, B_, G_);

        Goldilocks::store_neon(result0, result0_);
        Goldilocks::store_neon(result1, result1_);
        Goldilocks::store_neon(result2, result2_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };

    inline void Goldilocks3::mul33c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        assert(2 == 4);
        Goldilocks::Element aux0[2], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            aux0[k] = aux[0];
            aux1[k] = aux[1];
            aux2[k] = aux[2];
            a0[k] = a[k * FIELD_EXTENSION];
            a1[k] = a[k * FIELD_EXTENSION + 1];
            a2[k] = a[k * FIELD_EXTENSION + 2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        Goldilocks::load_neon(aux0_, aux0);
        Goldilocks::load_neon(aux1_, aux1);
        Goldilocks::load_neon(aux2_, aux2);
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[2], result1[4], result2[4];

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(result0_, C_, G_);
        Goldilocks::sub_neon(result0_, result0_, F_);
        Goldilocks::add_neon(result1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(result1_, result1_, auxr_);
        Goldilocks::sub_neon(result2_, B_, G_);

        Goldilocks::store_neon(result0, result0_);
        Goldilocks::store_neon(result1, result1_);
        Goldilocks::store_neon(result2, result2_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };

    inline void Goldilocks3::mul_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        assert(2 == 4);
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[k * stride_b];
            b1[k] = b[k * stride_b + 1];
            b2[k] = b[k * stride_b + 2];
        }
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[2], result1[4], result2[4];

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b0_, b1_);
        Goldilocks::add_neon(aux1_, b0_, b2_);
        Goldilocks::add_neon(aux2_, b1_, b2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(result0_, C_, G_);
        Goldilocks::sub_neon(result0_, result0_, F_);
        Goldilocks::add_neon(result1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(result1_, result1_, auxr_);
        Goldilocks::sub_neon(result2_, B_, G_);

        Goldilocks::store_neon(result0, result0_);
        Goldilocks::store_neon(result1, result1_);
        Goldilocks::store_neon(result2, result2_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };

    inline void Goldilocks3::mul_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[2], const uint64_t stride_b[2])
    {
        assert(2 == 4);
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
            b0[k] = b[stride_b[k]];
            b1[k] = b[stride_b[k] + 1];
            b2[k] = b[stride_b[k] + 2];
        }
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[2], result1[4], result2[4];

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b0_, b1_);
        Goldilocks::add_neon(aux1_, b0_, b2_);
        Goldilocks::add_neon(aux2_, b1_, b2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(result0_, C_, G_);
        Goldilocks::sub_neon(result0_, result0_, F_);
        Goldilocks::add_neon(result1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(result1_, result1_, auxr_);
        Goldilocks::sub_neon(result2_, B_, G_);

        Goldilocks::store_neon(result0, result0_);
        Goldilocks::store_neon(result1, result1_);
        Goldilocks::store_neon(result2, result2_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };

    inline void Goldilocks3::mul33c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element aux0[2], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            aux0[k] = aux[0];
            aux1[k] = aux[1];
            aux2[k] = aux[2];
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        Goldilocks::load_neon(aux0_, aux0);
        Goldilocks::load_neon(aux1_, aux1);
        Goldilocks::load_neon(aux2_, aux2);
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[2], result1[4], result2[4];

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(result0_, C_, G_);
        Goldilocks::sub_neon(result0_, result0_, F_);
        Goldilocks::add_neon(result1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(result1_, result1_, auxr_);
        Goldilocks::sub_neon(result2_, B_, G_);

        Goldilocks::store_neon(result0, result0_);
        Goldilocks::store_neon(result1, result1_);
        Goldilocks::store_neon(result2, result2_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };

    inline void Goldilocks3::mul33c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[2])
    {
        Goldilocks::Element aux0[2], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            aux0[k] = aux[0];
            aux1[k] = aux[1];
            aux2[k] = aux[2];
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        Goldilocks::load_neon(aux0_, aux0);
        Goldilocks::load_neon(aux1_, aux1);
        Goldilocks::load_neon(aux2_, aux2);
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[2], result1[4], result2[4];

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(result0_, C_, G_);
        Goldilocks::sub_neon(result0_, result0_, F_);
        Goldilocks::add_neon(result1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(result1_, result1_, auxr_);
        Goldilocks::sub_neon(result2_, B_, G_);

        Goldilocks::store_neon(result0, result0_);
        Goldilocks::store_neon(result1, result1_);
        Goldilocks::store_neon(result2, result2_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };

    inline void Goldilocks3::mul13c_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = a[k];
            aa[k * FIELD_EXTENSION + 2] = a[k];
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);
        Goldilocks::mult_neon(c0_, a0_, b0_);
        Goldilocks::mult_neon(c1_, a1_, b1_);
        Goldilocks::mult_neon(c2_, a2_, b2_);
        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::mul13_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 1] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 2] = a[k * stride_a];
            bb[k * FIELD_EXTENSION] = b[k * stride_b];
            bb[k * FIELD_EXTENSION + 1] = b[k * stride_b + 1];
            bb[k * FIELD_EXTENSION + 2] = b[k * stride_b + 2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);
        Goldilocks::mult_neon(c0_, a0_, b0_);
        Goldilocks::mult_neon(c1_, a1_, b1_);
        Goldilocks::mult_neon(c2_, a2_, b2_);
        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::mul13_neon(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[2], const uint64_t stride_b[2])
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];

        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[stride_a[k]];
            aa[k * FIELD_EXTENSION + 1] = a[stride_a[k]];
            aa[k * FIELD_EXTENSION + 2] = a[stride_a[k]];
            bb[k * FIELD_EXTENSION] = b[stride_b[k]];
            bb[k * FIELD_EXTENSION + 1] = b[stride_b[k] + 1];
            bb[k * FIELD_EXTENSION + 2] = b[stride_b[k] + 2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);
        Goldilocks::mult_neon(c0_, a0_, b0_);
        Goldilocks::mult_neon(c1_, a1_, b1_);
        Goldilocks::mult_neon(c2_, a2_, b2_);
        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::mul13c_neon(Goldilocks::Element *result, const uint64x2_t &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];
        Goldilocks::Element a[2];
        Goldilocks::store_neon(a, a_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = a[k];
            aa[k * FIELD_EXTENSION + 2] = a[k];
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);
        Goldilocks::mult_neon(c0_, a0_, b0_);
        Goldilocks::mult_neon(c1_, a1_, b1_);
        Goldilocks::mult_neon(c2_, a2_, b2_);
        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::mul13_neon(Goldilocks::Element *result, const uint64x2_t &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element aa[6];
        Goldilocks::Element bb[6];
        Goldilocks::Element a[2];
        Goldilocks::store_neon(a, a_);

        for (uint64_t k = 0; k < 2; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = a[k];
            aa[k * FIELD_EXTENSION + 2] = a[k];
            bb[k * FIELD_EXTENSION] = b[k * FIELD_EXTENSION];
            bb[k * FIELD_EXTENSION + 1] = b[k * FIELD_EXTENSION + 1];
            bb[k * FIELD_EXTENSION + 2] = b[k * FIELD_EXTENSION + 2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;
        uint64x2_t c0_, c1_, c2_;

        Goldilocks::load_neon(a0_, aa);
        Goldilocks::load_neon(a1_, &aa[2]);
        Goldilocks::load_neon(a2_, &aa[4]);
        Goldilocks::load_neon(b0_, bb);
        Goldilocks::load_neon(b1_, &bb[2]);
        Goldilocks::load_neon(b2_, &bb[4]);
        Goldilocks::mult_neon(c0_, a0_, b0_);
        Goldilocks::mult_neon(c1_, a1_, b1_);
        Goldilocks::mult_neon(c2_, a2_, b2_);
        Goldilocks::store_neon(result, c0_);
        Goldilocks::store_neon(&result[2], c1_);
        Goldilocks::store_neon(&result[4], c2_);
    }

    inline void Goldilocks3::mul13c_neon(Goldilocks3::Element_neon &c_, const uint64x2_t &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t b0_, b1_, b2_;
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);
        Goldilocks::mult_neon(c_[0], a_, b0_);
        Goldilocks::mult_neon(c_[1], a_, b1_);
        Goldilocks::mult_neon(c_[2], a_, b2_);
    }

    inline void Goldilocks3::mul13_neon(Goldilocks3::Element_neon &c_, const uint64x2_t &a_, const Goldilocks3::Element_neon &b_)
    {
        Goldilocks::mult_neon(c_[0], a_, b_[0]);
        Goldilocks::mult_neon(c_[1], a_, b_[1]);
        Goldilocks::mult_neon(c_[2], a_, b_[2]);
    }

    inline void Goldilocks3::mul13_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks3::Element_neon b_, uint64_t stride_a)
    {
        Goldilocks::Element a4[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            a4[k] = a[k * stride_a];
        }
        uint64x2_t a_;
        Goldilocks::load_neon(a_, a4);
        Goldilocks::mult_neon(c_[0], a_, b_[0]);
        Goldilocks::mult_neon(c_[1], a_, b_[1]);
        Goldilocks::mult_neon(c_[2], a_, b_[2]);
    }

    inline void Goldilocks3::mul13c_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element a0[2];
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t a_;
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(a_, a0);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);
        Goldilocks::mult_neon(c_[0], a_, b0_);
        Goldilocks::mult_neon(c_[1], a_, b1_);
        Goldilocks::mult_neon(c_[2], a_, b2_);
    }

    inline void Goldilocks3::mul13c_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[2])
    {
        Goldilocks::Element a0[2];
        Goldilocks::Element b0[2];
        Goldilocks::Element b1[2];
        Goldilocks::Element b2[2];

        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[stride_a[k]];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t a_;
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(a_, a0);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);
        Goldilocks::mult_neon(c_[0], a_, b0_);
        Goldilocks::mult_neon(c_[1], a_, b1_);
        Goldilocks::mult_neon(c_[2], a_, b2_);
    }

    inline void Goldilocks3::mul13_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks3::Element_neon b_, const uint64_t stride_a[2])
    {
        Goldilocks::Element a4[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            a4[k] = a[stride_a[k]];
        }
        uint64x2_t a_;
        Goldilocks::load_neon(a_, a4);
        Goldilocks::mult_neon(c_[0], a_, b_[0]);
        Goldilocks::mult_neon(c_[1], a_, b_[1]);
        Goldilocks::mult_neon(c_[2], a_, b_[2]);
    }

    inline void Goldilocks3::mul1c3c_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element a, Element &b)
    {
        // Does not make sense to vectorize
        Goldilocks::Element cc0 = b[0] * a;
        Goldilocks::Element cc1 = b[1] * a;
        Goldilocks::Element cc2 = b[2] * a;

        Goldilocks::Element c0[2] = {cc0, cc0};
        Goldilocks::Element c1[2] = {cc1, cc1};
        Goldilocks::Element c2[2] = {cc2, cc2};

        Goldilocks::load_neon(c_[0], c0);
        Goldilocks::load_neon(c_[1], c1);
        Goldilocks::load_neon(c_[2], c2);
    }

    inline void Goldilocks3::mul33c_neon(Goldilocks3::Element_neon &c_, Goldilocks3::Element_neon &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element aux0[2], aux1[4], aux2[4], aux[3];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            aux0[k] = aux[0];
            aux1[k] = aux[1];
            aux2[k] = aux[2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        Goldilocks::load_neon(aux0_, aux0);
        Goldilocks::load_neon(aux1_, aux1);
        Goldilocks::load_neon(aux2_, aux2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a_[0], a_[1]);
        Goldilocks::add_neon(B_, a_[0], a_[2]);
        Goldilocks::add_neon(C_, a_[1], a_[2]);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a_[0], b0_);
        Goldilocks::mult_neon(E_, a_[1], b1_);
        Goldilocks::mult_neon(F_, a_[2], b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(c_[0], C_, G_);
        Goldilocks::sub_neon(c_[0], c_[0], F_);
        Goldilocks::add_neon(c_[1], A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c_[1], c_[1], auxr_);
        Goldilocks::sub_neon(c_[2], B_, G_);
    };

    inline void Goldilocks3::mul_neon(Goldilocks3::Element_neon &c_, Goldilocks3::Element_neon &a_, Goldilocks3::Element_neon &b_)
    {

        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a_[0], a_[1]);
        Goldilocks::add_neon(B_, a_[0], a_[2]);
        Goldilocks::add_neon(C_, a_[1], a_[2]);
        Goldilocks::add_neon(aux0_, b_[0], b_[1]);
        Goldilocks::add_neon(aux1_, b_[0], b_[2]);
        Goldilocks::add_neon(aux2_, b_[1], b_[2]);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a_[0], b_[0]);
        Goldilocks::mult_neon(E_, a_[1], b_[1]);
        Goldilocks::mult_neon(F_, a_[2], b_[2]);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(c_[0], C_, G_);
        Goldilocks::sub_neon(c_[0], c_[0], F_);
        Goldilocks::add_neon(c_[1], A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c_[1], c_[1], auxr_);
        Goldilocks::sub_neon(c_[2], B_, G_);
    };

    inline void Goldilocks3::mul_neon(Goldilocks::Element *c, uint64_t stride_c, Goldilocks::Element *a, Goldilocks3::Element_neon &b_, uint64_t stride_a)
    {

        Goldilocks::Element a0[2], a1[4], a2[4];
        uint64x2_t a0_, a1_, a2_;

        // redistribute data:
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);

        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b_[0], b_[1]);
        Goldilocks::add_neon(aux1_, b_[0], b_[2]);
        Goldilocks::add_neon(aux2_, b_[1], b_[2]);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b_[0]);
        Goldilocks::mult_neon(E_, a1_, b_[1]);
        Goldilocks::mult_neon(F_, a2_, b_[2]);
        Goldilocks::sub_neon(G_, D_, E_);

        uint64x2_t c0_, c1_, c2_;

        Goldilocks::add_neon(c0_, C_, G_);
        Goldilocks::sub_neon(c0_, c0_, F_);
        Goldilocks::add_neon(c1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c1_, c1_, auxr_);
        Goldilocks::sub_neon(c2_, B_, G_);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    };

    inline void Goldilocks3::mul_neon(Goldilocks::Element *c, uint64_t stride_c, Goldilocks3::Element_neon &a_, Goldilocks3::Element_neon &b_)
    {

        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a_[0], a_[1]);
        Goldilocks::add_neon(B_, a_[0], a_[2]);
        Goldilocks::add_neon(C_, a_[1], a_[2]);
        Goldilocks::add_neon(aux0_, b_[0], b_[1]);
        Goldilocks::add_neon(aux1_, b_[0], b_[2]);
        Goldilocks::add_neon(aux2_, b_[1], b_[2]);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a_[0], b_[0]);
        Goldilocks::mult_neon(E_, a_[1], b_[1]);
        Goldilocks::mult_neon(F_, a_[2], b_[2]);
        Goldilocks::sub_neon(G_, D_, E_);

        uint64x2_t c0_, c1_, c2_;

        Goldilocks::add_neon(c0_, C_, G_);
        Goldilocks::sub_neon(c0_, c0_, F_);
        Goldilocks::add_neon(c1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c1_, c1_, auxr_);
        Goldilocks::sub_neon(c2_, B_, G_);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    };

    inline void Goldilocks3::mul_neon(Goldilocks::Element *c, uint64_t stride_c[2], Goldilocks3::Element_neon &a_, Goldilocks3::Element_neon &b_)
    {

        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a_[0], a_[1]);
        Goldilocks::add_neon(B_, a_[0], a_[2]);
        Goldilocks::add_neon(C_, a_[1], a_[2]);
        Goldilocks::add_neon(aux0_, b_[0], b_[1]);
        Goldilocks::add_neon(aux1_, b_[0], b_[2]);
        Goldilocks::add_neon(aux2_, b_[1], b_[2]);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a_[0], b_[0]);
        Goldilocks::mult_neon(E_, a_[1], b_[1]);
        Goldilocks::mult_neon(F_, a_[2], b_[2]);
        Goldilocks::sub_neon(G_, D_, E_);

        uint64x2_t c0_, c1_, c2_;

        Goldilocks::add_neon(c0_, C_, G_);
        Goldilocks::sub_neon(c0_, c0_, F_);
        Goldilocks::add_neon(c1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c1_, c1_, auxr_);
        Goldilocks::sub_neon(c2_, B_, G_);

        Goldilocks::Element c0[2], c1[4], c2[4];
        Goldilocks::store_neon(c0, c0_);
        Goldilocks::store_neon(c1, c1_);
        Goldilocks::store_neon(c2, c2_);
        for (uint64_t k = 0; k < 2; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        };
    };

    inline void Goldilocks3::mul_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[k * stride_b];
            b1[k] = b[k * stride_b + 1];
            b2[k] = b[k * stride_b + 2];
        }
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b0_, b1_);
        Goldilocks::add_neon(aux1_, b0_, b2_);
        Goldilocks::add_neon(aux2_, b1_, b2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(c_[0], C_, G_);
        Goldilocks::sub_neon(c_[0], c_[0], F_);
        Goldilocks::add_neon(c_[1], A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c_[1], c_[1], auxr_);
        Goldilocks::sub_neon(c_[2], B_, G_);
    };

    inline void Goldilocks3::mul33c_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element aux0[2], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            aux0[k] = aux[0];
            aux1[k] = aux[1];
            aux2[k] = aux[2];
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        Goldilocks::load_neon(aux0_, aux0);
        Goldilocks::load_neon(aux1_, aux1);
        Goldilocks::load_neon(aux2_, aux2);
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(c_[0], C_, G_);
        Goldilocks::sub_neon(c_[0], c_[0], F_);
        Goldilocks::add_neon(c_[1], A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c_[1], c_[1], auxr_);
        Goldilocks::sub_neon(c_[2], B_, G_);
    };

    inline void Goldilocks3::mul33c_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[2])
    {
        Goldilocks::Element aux0[2], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < 2; ++k)
        {
            aux0[k] = aux[0];
            aux1[k] = aux[1];
            aux2[k] = aux[2];
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        Goldilocks::load_neon(aux0_, aux0);
        Goldilocks::load_neon(aux1_, aux1);
        Goldilocks::load_neon(aux2_, aux2);
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(c_[0], C_, G_);
        Goldilocks::sub_neon(c_[0], c_[0], F_);
        Goldilocks::add_neon(c_[1], A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c_[1], c_[1], auxr_);
        Goldilocks::sub_neon(c_[2], B_, G_);
    };

    inline void Goldilocks3::mul_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks3::Element_neon &b_, const uint64_t stride_a[2])
    {
        Goldilocks::Element a0[2], a1[4], a2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;

        // redistribute data:
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
        }
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b_[0], b_[1]);
        Goldilocks::add_neon(aux1_, b_[0], b_[2]);
        Goldilocks::add_neon(aux2_, b_[1], b_[2]);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b_[0]);
        Goldilocks::mult_neon(E_, a1_, b_[1]);
        Goldilocks::mult_neon(F_, a2_, b_[2]);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(c_[0], C_, G_);
        Goldilocks::sub_neon(c_[0], c_[0], F_);
        Goldilocks::add_neon(c_[1], A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c_[1], c_[1], auxr_);
        Goldilocks::sub_neon(c_[2], B_, G_);
    };

    inline void Goldilocks3::mul_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks3::Element_neon &b_, const uint64_t stride_a)
    {
        Goldilocks::Element a0[2], a1[4], a2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;

        // redistribute data:
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b_[0], b_[1]);
        Goldilocks::add_neon(aux1_, b_[0], b_[2]);
        Goldilocks::add_neon(aux2_, b_[1], b_[2]);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b_[0]);
        Goldilocks::mult_neon(E_, a1_, b_[1]);
        Goldilocks::mult_neon(F_, a2_, b_[2]);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(c_[0], C_, G_);
        Goldilocks::sub_neon(c_[0], c_[0], F_);
        Goldilocks::add_neon(c_[1], A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c_[1], c_[1], auxr_);
        Goldilocks::sub_neon(c_[2], B_, G_);
    };

    inline void Goldilocks3::mul_neon(Goldilocks3::Element_neon &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[2], const uint64_t stride_b[2])
    {
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
            b0[k] = b[stride_b[k]];
            b1[k] = b[stride_b[k] + 1];
            b2[k] = b[stride_b[k] + 2];
        }
        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b0_, b1_);
        Goldilocks::add_neon(aux1_, b0_, b2_);
        Goldilocks::add_neon(aux2_, b1_, b2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(c_[0], C_, G_);
        Goldilocks::sub_neon(c_[0], c_[0], F_);
        Goldilocks::add_neon(c_[1], A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c_[1], c_[1], auxr_);
        Goldilocks::sub_neon(c_[2], B_, G_);
    };

    inline void Goldilocks3::mul13c_neon(uint64x2_t &c0_, uint64x2_t &c1_, uint64x2_t &c2_, Goldilocks::Element *a, Element &b, uint64_t stride_a)
    {
        Goldilocks::Element a0[2], a1[4], a2[4];
        Goldilocks::Element b0[2], b1[4], b2[4];

        for (uint64_t k = 0; k < 2; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a];
            a2[k] = a[k * stride_a];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        uint64x2_t a0_, a1_, a2_;
        uint64x2_t b0_, b1_, b2_;

        Goldilocks::load_neon(a0_, a0);
        Goldilocks::load_neon(a1_, a1);
        Goldilocks::load_neon(a2_, a2);
        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);
        Goldilocks::mult_neon(c0_, a0_, b0_);
        Goldilocks::mult_neon(c1_, a1_, b1_);
        Goldilocks::mult_neon(c2_, a2_, b2_);
    }

    inline void Goldilocks3::mul_neon(uint64x2_t &c0_, uint64x2_t &c1_, uint64x2_t &c2_, uint64x2_t a0_, uint64x2_t a1_, uint64x2_t a2_, uint64x2_t b0_, uint64x2_t b1_, uint64x2_t b2_, uint64x2_t aux0_, uint64x2_t aux1_, uint64x2_t aux2_)
    {
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t result0_, result1_, auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(result0_, C_, G_);
        Goldilocks::sub_neon(c0_, result0_, F_);
        Goldilocks::add_neon(result1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c1_, result1_, auxr_);
        Goldilocks::sub_neon(c2_, B_, G_);
    };

    inline void Goldilocks3::mul_neon(uint64x2_t &c0_, uint64x2_t &c1_, uint64x2_t &c2_, uint64x2_t a0_, uint64x2_t a1_, uint64x2_t a2_, Goldilocks::Element *b)
    {
        assert(2 == 4);
        Goldilocks::Element b0[2], b1[4], b2[4];
        uint64x2_t aux0_, aux1_, aux2_;
        uint64x2_t b0_, b1_, b2_;

        // redistribute data:

        for (uint64_t k = 0; k < 2; ++k)
        {
            b0[k] = b[k * FIELD_EXTENSION];
            b1[k] = b[k * FIELD_EXTENSION + 1];
            b2[k] = b[k * FIELD_EXTENSION + 2];
        }

        Goldilocks::load_neon(b0_, b0);
        Goldilocks::load_neon(b1_, b1);
        Goldilocks::load_neon(b2_, b2);

        // operations
        uint64x2_t A_, B_, C_, D_, E_, F_, G_;
        uint64x2_t result0_, result1_, auxr_;

        Goldilocks::add_neon(A_, a0_, a1_);
        Goldilocks::add_neon(B_, a0_, a2_);
        Goldilocks::add_neon(C_, a1_, a2_);
        Goldilocks::add_neon(aux0_, b0_, b1_);
        Goldilocks::add_neon(aux1_, b0_, b2_);
        Goldilocks::add_neon(aux2_, b1_, b2_);
        Goldilocks::mult_neon(A_, A_, aux0_);
        Goldilocks::mult_neon(B_, B_, aux1_);
        Goldilocks::mult_neon(C_, C_, aux2_);
        Goldilocks::mult_neon(D_, a0_, b0_);
        Goldilocks::mult_neon(E_, a1_, b1_);
        Goldilocks::mult_neon(F_, a2_, b2_);
        Goldilocks::sub_neon(G_, D_, E_);

        Goldilocks::add_neon(result0_, C_, G_);
        Goldilocks::sub_neon(c0_, result0_, F_);
        Goldilocks::add_neon(result1_, A_, C_);
        Goldilocks::add_neon(auxr_, E_, E_);
        Goldilocks::add_neon(auxr_, auxr_, D_);
        Goldilocks::sub_neon(c1_, result1_, auxr_);
        Goldilocks::sub_neon(c2_, B_, G_);
    };


#endif // GOLDILOCKS_HAS_NEON
#endif // GOLDILOCKS_CUBIC_EXTENSION_NEON_HPP
