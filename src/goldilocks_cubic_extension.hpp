#ifndef GOLDILOCKS_F3
#define GOLDILOCKS_F3

#include <stdint.h> // for uint64_t
#include "goldilocks_base_field.hpp"
#include <immintrin.h>
#include <cassert>
#include <vector>

#define FIELD_EXTENSION 3

/*
This is a field extension 3 of the goldilocks:
Prime: 0xFFFFFFFF00000001
Irreducible polynomial: x^3 - x -1
*/

class Goldilocks3
{
public:
    typedef Goldilocks::Element Element[FIELD_EXTENSION];
    typedef __m256i Element_avx[FIELD_EXTENSION];
    typedef __m512i Element_avx512[FIELD_EXTENSION];

private:
    static const Element ZERO;
    static const Element ONE;
    static const Element NEGONE;

public:
    uint64_t m = 1 * FIELD_EXTENSION;
    uint64_t p = GOLDILOCKS_PRIME;
    uint64_t n64 = 1;
    uint64_t n32 = n64 * 2;
    uint64_t n8 = n32 * 4;

    static inline const Element &zero() { return ZERO; };
    static inline void zero(Element &result)
    {
        result[0] = Goldilocks::zero();
        result[1] = Goldilocks::zero();
        result[2] = Goldilocks::zero();
    };

    static inline const Element &one() { return ONE; };
    static inline void one(Element &result)
    {
        result[0] = Goldilocks::one();
        result[1] = Goldilocks::zero();
        result[2] = Goldilocks::zero();
    };
    static inline bool isOne(Element &result)
    {
        return Goldilocks::isOne(result[0]) && Goldilocks::isOne(result[0]) && Goldilocks::isOne(result[0]);
    };

    static void copy(Element &dst, const Element &src)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            Goldilocks::copy(dst[i], src[i]);
        }
    };
    static void copy(Element *dst, const Element *src)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            Goldilocks::copy((*dst)[i], (*src)[i]);
        }
    };
    static void copy_batch(Goldilocks::Element *dst, const Goldilocks::Element *src)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION * AVX_SIZE_; i++)
        {
            Goldilocks::copy(dst[i], src[i]);
        }
    };
    static void copy_avx(Goldilocks::Element *dst, const __m256i a0_, const __m256i a1_, const __m256i a2_)
    {
        Goldilocks::Element buff0[4], buff1[4], buff2[4];
        Goldilocks::store_avx(buff0, a0_);
        Goldilocks::store_avx(buff1, a1_);
        Goldilocks::store_avx(buff2, a2_);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            Goldilocks::copy(dst[k * FIELD_EXTENSION], buff0[k]);
            Goldilocks::copy(dst[k * FIELD_EXTENSION + 1], buff1[k]);
            Goldilocks::copy(dst[k * FIELD_EXTENSION + 2], buff2[k]);
        }
    };

    static inline void fromU64(Element &result, uint64_t in1[FIELD_EXTENSION])
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::fromU64(in1[i]);
        }
    }
    static inline void fromS32(Element &result, int32_t in1[FIELD_EXTENSION])
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::fromS32(in1[i]);
        }
    }
    static inline void toU64(uint64_t (&result)[FIELD_EXTENSION], const Element &in1)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::toU64(in1[i]);
        }
    }

    static inline std::vector<Goldilocks::Element> toVector(const Element &in1)
    {
        std::vector<Goldilocks::Element> result;
        result.assign(in1, in1 + FIELD_EXTENSION);
        return result;
    }
    static inline std::vector<Goldilocks::Element> toVector(const Element *in1)
    {

        std::vector<Goldilocks::Element> result;
        result.assign(*in1, *in1 + FIELD_EXTENSION);
        return result;
    }

    static inline std::string toString(const Element &in1, int radix = 10)
    {
        std::string res;
        toString(res, in1, radix);
        return res;
    }
    static inline void toString(std::string &result, const Element &in1, int radix = 10)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result += Goldilocks::toString(in1[i]);
            (i != FIELD_EXTENSION - 1) ? result += " , " : "";
        }
    }
    static inline void toString(std::string (&result)[FIELD_EXTENSION], const Element &in1, int radix = 10)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::toString(in1[i]);
        }
    }
    static std::string toString(const Element *in1, const uint64_t size, int radix)
    {
        std::string result = "";
        for (uint64_t i = 0; i < size; i++)
        {
            result += Goldilocks3::toString(in1[i], 10);
            result += "\n";
        }
        return result;
    }
    static inline void fromString(Element &result, const std::string (&in1)[FIELD_EXTENSION], int radix = 10)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::fromString(in1[i]);
        }
    }

#ifdef __AVX512__
    static void copy_avx512(Goldilocks::Element *dst, const __m512i a0_, const __m512i a1_, const __m512i a2_)
    {
        Goldilocks::Element buff0[AVX512_SIZE_], buff1[AVX512_SIZE_], buff2[AVX512_SIZE_];
        Goldilocks::store_avx512(buff0, a0_);
        Goldilocks::store_avx512(buff1, a1_);
        Goldilocks::store_avx512(buff2, a2_);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            Goldilocks::copy(dst[k * FIELD_EXTENSION], buff0[k]);
            Goldilocks::copy(dst[k * FIELD_EXTENSION + 1], buff1[k]);
            Goldilocks::copy(dst[k * FIELD_EXTENSION + 2], buff2[k]);
        }
    };
#endif
    // ======== ADD ========
    static inline void add(Element &result, const Element &a, const uint64_t &b)
    {
        result[0] = a[0] + Goldilocks::fromU64(b);
        result[1] = a[1];
        result[2] = a[2];
    }
    static inline void add(Element &result, const Element &a, const Goldilocks::Element b)
    {
        result[0] = a[0] + b;
        result[1] = a[1];
        result[2] = a[2];
    }
    static inline void add(Element &result, const Goldilocks::Element a, const Element &b)
    {
        add(result, b, a);
    }
    static inline void add(Element &result, const Element &a, const Element &b)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = a[i] + b[i];
        }
    }

    static inline void add_batch(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION * AVX_SIZE_; i++)
        {
            result[i] = a[i] + b[i];
        }
    }
    static inline void add_batch(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                result[k * FIELD_EXTENSION + i] = a[k * stride_a + i] + b[k * stride_b + i];
            }
        }
    }
    static inline void add31_batch(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] + b[stride_b * k];
            result[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            result[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
    }
    static inline void add13_batch(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = b[k * FIELD_EXTENSION] + a[k];
            result[k * FIELD_EXTENSION + 1] = b[k * FIELD_EXTENSION + 1];
            result[k * FIELD_EXTENSION + 2] = b[k * FIELD_EXTENSION + 2];
        }
    }
    static inline void add1c3c_batch(Goldilocks::Element *result, const Goldilocks::Element a, const Goldilocks::Element *b)
    {
        Goldilocks::Element res0 = b[0] + a;
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = res0;
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }
    static inline void add13c_batch(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k] + b[0];
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }
    static inline void add13_batch(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a, uint64_t offset_b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * offset_a] + b[k * offset_b];
            result[k * FIELD_EXTENSION + 1] = b[k * offset_b + 1];
            result[k * FIELD_EXTENSION + 2] = b[k * offset_b + 2];
        }
    }
    static inline void add13c_batch(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * offset_a] + b[0];
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }
    static inline void add33c_batch(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * FIELD_EXTENSION] + b[0];
            result[k * FIELD_EXTENSION + 1] = a[k * FIELD_EXTENSION + 1] + b[1];
            result[k * FIELD_EXTENSION + 2] = a[k * FIELD_EXTENSION + 2] + b[2];
        }
    }
    static inline void add33c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                result[k * FIELD_EXTENSION + i] = b[i] + a[stride_a * k + i];
            }
        }
    }

    static inline void add_avx(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {

        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, a);
        Goldilocks::load_avx(a1_, &a[4]);
        Goldilocks::load_avx(a2_, &a[8]);
        Goldilocks::load_avx(b0_, b);
        Goldilocks::load_avx(b1_, &b[4]);
        Goldilocks::load_avx(b2_, &b[8]);

        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void add_avx(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element bb[12];
        Goldilocks::Element aa[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                bb[k * FIELD_EXTENSION + i] = b[k * stride_b + i];
                aa[k * FIELD_EXTENSION + i] = a[k * stride_a + i];
            }
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);

        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void add31_avx(Goldilocks::Element *result, Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element bb[12];
        Goldilocks::Element aa[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            bb[k * FIELD_EXTENSION] = b[stride_b * k];
            bb[k * FIELD_EXTENSION + 1] = Goldilocks::zero();
            bb[k * FIELD_EXTENSION + 2] = Goldilocks::zero();
            aa[k * FIELD_EXTENSION] = a[stride_a * k];
            aa[k * FIELD_EXTENSION + 1] = a[stride_a * k + 1];
            aa[k * FIELD_EXTENSION + 2] = a[stride_a * k + 2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);

        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void add13_avx(Goldilocks::Element *result, Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        Goldilocks::Element aa[12];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = Goldilocks::zero();
            aa[k * FIELD_EXTENSION + 2] = Goldilocks::zero();
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, b);
        Goldilocks::load_avx(b1_, &b[4]);
        Goldilocks::load_avx(b2_, &b[8]);

        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void add1c3c_avx(Goldilocks::Element *result, const Goldilocks::Element a, const Goldilocks::Element *b)
    {
        // does not make sense to vectorise
        Goldilocks::Element res0 = b[0] + a;
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = res0;
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }
    static inline void add13c_avx(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        // does not make sense to vectorise
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k] + b[0];
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }
    static inline void add13_avx(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a, uint64_t offset_b)
    {
        // does not make sense to vectorize
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * offset_a] + b[k * offset_b];
            result[k * FIELD_EXTENSION + 1] = b[k * offset_b + 1];
            result[k * FIELD_EXTENSION + 2] = b[k * offset_b + 2];
        }
    }
    static inline void add13c_avx(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a)
    {
        // does not make sense to vectorize
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * offset_a] + b[0];
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }
    static inline void add33c_avx(Goldilocks::Element *result, const Goldilocks::Element *a, const Goldilocks::Element *b)
    {
        Goldilocks::Element bb[12];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, a);
        Goldilocks::load_avx(a1_, &a[4]);
        Goldilocks::load_avx(a2_, &a[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);

        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void add33c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element bb[12];
        Goldilocks::Element aa[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
            aa[k * FIELD_EXTENSION] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            aa[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);

        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void add13_avx(Goldilocks::Element *result, const __m256i &a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element a[4];
        Goldilocks::store_avx(a, a_);

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = Goldilocks::zero();
            aa[k * FIELD_EXTENSION + 2] = Goldilocks::zero();
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, b);
        Goldilocks::load_avx(b1_, &b[4]);
        Goldilocks::load_avx(b2_, &b[8]);

        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void add13c_avx(Goldilocks::Element *result, const __m256i &a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element a[4];
        Goldilocks::store_avx(a, a_);
        // does not make sense to vectorise
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k] + b[0];
            result[k * FIELD_EXTENSION + 1] = b[1];
            result[k * FIELD_EXTENSION + 2] = b[2];
        }
    }

    static inline void add13_avx(Goldilocks3::Element_avx c_, const __m256i &a_, Goldilocks3::Element_avx b_)
    {
        Goldilocks::add_avx(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
    }
    static inline void add13c_avx(Goldilocks3::Element_avx c_, const __m256i &a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i b0_, b1_, b2_;
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        Goldilocks::add_avx(c_[0], a_, b0_);
        c_[1] = b1_;
        c_[2] = b2_;
    }
    static inline void add1c3c_avx(Goldilocks3::Element_avx c_, const Goldilocks::Element a, const Goldilocks::Element *b)
    {
        // does not make sense to vectorise
        Goldilocks::Element res0 = b[0] + a;
        Goldilocks::Element c0[4];
        Goldilocks::Element c1[4];
        Goldilocks::Element c2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c0[k] = res0;
            c1[k] = b[1];
            c2[k] = b[2];
        }
        Goldilocks::load_avx(c_[0], c0);
        Goldilocks::load_avx(c_[1], c1);
        Goldilocks::load_avx(c_[2], c2);
    }
    static inline void add13_avx(Goldilocks3::Element_avx c_, const Goldilocks::Element *a, Goldilocks3::Element_avx b_, uint64_t offset_a)
    {
        Goldilocks::Element a0[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * offset_a];
        }
        __m256i a0_;
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::add_avx(c_[0], a0_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
    }
    static inline void add13c_avx(Goldilocks3::Element_avx c_, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a)
    {
        Goldilocks::Element c0[4];
        Goldilocks::Element c1[4];
        Goldilocks::Element c2[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c0[k] = a[k * offset_a] + b[0];
            c1[k] = b[1];
            c2[k] = b[2];
        }
        Goldilocks::load_avx(c_[0], c0);
        Goldilocks::load_avx(c_[1], c1);
        Goldilocks::load_avx(c_[2], c2);
    }
    static inline void add_avx(Goldilocks3::Element_avx c_, Goldilocks3::Element_avx a_, Goldilocks3::Element_avx b_)
    {
        Goldilocks::add_avx(c_[0], a_[0], b_[0]);
        Goldilocks::add_avx(c_[1], a_[1], b_[1]);
        Goldilocks::add_avx(c_[2], a_[2], b_[2]);
    }
    static inline void add33c_avx(Goldilocks3::Element_avx c_, Goldilocks3::Element_avx a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        Goldilocks::add_avx(c_[0], a_[0], b0_);
        Goldilocks::add_avx(c_[1], a_[1], b1_);
        Goldilocks::add_avx(c_[2], a_[2], b2_);
    }
    static inline void add_avx(Goldilocks3::Element_avx c_, const Goldilocks::Element *a, Goldilocks3::Element_avx b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[4];
        Goldilocks::Element a1[4];
        Goldilocks::Element a2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m256i a0_, a1_, a2_;

        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);

        Goldilocks::add_avx(c_[0], a0_, b_[0]);
        Goldilocks::add_avx(c_[1], a1_, b_[1]);
        Goldilocks::add_avx(c_[2], a2_, b_[2]);
    }
    static inline void add33c_avx(Goldilocks3::Element_avx c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element a0[4];
        Goldilocks::Element a1[4];
        Goldilocks::Element a2[4];
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        Goldilocks::add_avx(c_[0], a0_, b0_);
        Goldilocks::add_avx(c_[1], a1_, b1_);
        Goldilocks::add_avx(c_[2], a2_, b2_);
    }

    static inline void add13_avx(Goldilocks::Element *c, uint64_t stride_c, const __m256i &a_, Goldilocks3::Element_avx b_)
    {
        __m256i c0_;
        Goldilocks::add_avx(c0_, a_, b_[0]);

        Goldilocks::Element c0[4], c1[4], c2[4];
        Goldilocks::store_avx(c0, c0_);
        Goldilocks::store_avx(c1, b_[1]);
        Goldilocks::store_avx(c2, b_[2]);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }
    static inline void add_avx(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, Goldilocks3::Element_avx b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[4];
        Goldilocks::Element a1[4];
        Goldilocks::Element a2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m256i a0_, a1_, a2_;

        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);

        __m256i c0_, c1_, c2_;
        Goldilocks::add_avx(c0_, a0_, b_[0]);
        Goldilocks::add_avx(c1_, a1_, b_[1]);
        Goldilocks::add_avx(c2_, a2_, b_[2]);

        Goldilocks::Element c0[4], c1[4], c2[4];
        Goldilocks::store_avx(c0, c0_);
        Goldilocks::store_avx(c1, c1_);
        Goldilocks::store_avx(c2, c2_);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }
    static inline void add33c_avx(Goldilocks::Element *c, uint64_t stride_c, Goldilocks3::Element_avx a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        __m256i c0_, c1_, c2_;
        Goldilocks::add_avx(c0_, a_[0], b0_);
        Goldilocks::add_avx(c1_, a_[1], b1_);
        Goldilocks::add_avx(c2_, a_[2], b2_);

        Goldilocks::Element c0[4], c1[4], c2[4];
        Goldilocks::store_avx(c0, c0_);
        Goldilocks::store_avx(c1, c1_);
        Goldilocks::store_avx(c2, c2_);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }
    static inline void add13_avx(Goldilocks::Element *c, uint64_t stride_c[4], const __m256i &a_, Goldilocks3::Element_avx b_)
    {
        __m256i c0_;
        Goldilocks::add_avx(c0_, a_, b_[0]);

        Goldilocks::Element c0[4], c1[4], c2[4];
        Goldilocks::store_avx(c0, c0_);
        Goldilocks::store_avx(c1, b_[1]);
        Goldilocks::store_avx(c2, b_[2]);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }
    static inline void add_avx(Goldilocks::Element *c, uint64_t stride_c[4], const Goldilocks::Element *a, Goldilocks3::Element_avx b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[4];
        Goldilocks::Element a1[4];
        Goldilocks::Element a2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m256i a0_, a1_, a2_;

        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);

        __m256i c0_, c1_, c2_;
        Goldilocks::add_avx(c0_, a0_, b_[0]);
        Goldilocks::add_avx(c1_, a1_, b_[1]);
        Goldilocks::add_avx(c2_, a2_, b_[2]);

        Goldilocks::Element c0[4], c1[4], c2[4];
        Goldilocks::store_avx(c0, c0_);
        Goldilocks::store_avx(c1, c1_);
        Goldilocks::store_avx(c2, c2_);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }
    static inline void add33c_avx(Goldilocks::Element *c, uint64_t stride_c[4], Goldilocks3::Element_avx a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        __m256i c0_, c1_, c2_;
        Goldilocks::add_avx(c0_, a_[0], b0_);
        Goldilocks::add_avx(c1_, a_[1], b1_);
        Goldilocks::add_avx(c2_, a_[2], b2_);

        Goldilocks::Element c0[4], c1[4], c2[4];
        Goldilocks::store_avx(c0, c0_);
        Goldilocks::store_avx(c1, c1_);
        Goldilocks::store_avx(c2, c2_);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }

    static inline void add_avx(__m256i &c0_, __m256i &c1_, __m256i &c2_, const __m256i a0_, const __m256i a1_, const __m256i a2_, const __m256i b0_, const __m256i b1_, const __m256i b2_)
    {
        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);
    }
    static inline void add_avx(__m256i &c0_, __m256i &c1_, __m256i &c2_, const __m256i a0_, const __m256i a1_, const __m256i a2_, const Goldilocks::Element *b, uint64_t stride)
    {
        Goldilocks::Element b0[4], b1[4], b2[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[k * stride];
            b1[k] = b[k * stride + 1];
            b2[k] = b[k * stride + 2];
        }
        __m256i b0_, b1_, b2_;
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);
        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);
    }
    static inline void add31_avx(__m256i &c0_, __m256i &c1_, __m256i &c2_, __m256i a0_, const __m256i a1_, const __m256i a2_, const Goldilocks::Element *b, uint64_t stride)
    {
        Goldilocks::Element b0[4], b1[4], b2[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[k * stride];
            b1[k] = Goldilocks::zero();
            b2[k] = Goldilocks::zero();
        }
        __m256i b0_, b1_, b2_;
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        Goldilocks::add_avx(c0_, a0_, b0_);
        Goldilocks::add_avx(c1_, a1_, b1_);
        Goldilocks::add_avx(c2_, a2_, b2_);
    }

#ifdef __AVX512__
    static inline void add13_avx512(Goldilocks3::Element_avx512 c_, const __m512i &a_, Goldilocks3::Element_avx512 b_)
    {
        Goldilocks::add_avx512(c_[0], a_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
    }
    static inline void add13c_avx512(Goldilocks3::Element_avx512 c_, const __m512i &a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i b0_, b1_, b2_;
        Goldilocks::load_avx512(b0_, b0); // rick: better set
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        Goldilocks::add_avx512(c_[0], a_, b0_);
        c_[1] = b1_;
        c_[2] = b2_;
    }
    static inline void add1c3c_avx512(Goldilocks3::Element_avx512 c_, const Goldilocks::Element a, const Goldilocks::Element *b)
    {
        // does not make sense to vectorise
        Goldilocks::Element res0 = b[0] + a;
        Goldilocks::Element c0[AVX512_SIZE_];
        Goldilocks::Element c1[AVX512_SIZE_];
        Goldilocks::Element c2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c0[k] = res0;
            c1[k] = b[1];
            c2[k] = b[2];
        }
        Goldilocks::load_avx512(c_[0], c0); // rick: better set
        Goldilocks::load_avx512(c_[1], c1);
        Goldilocks::load_avx512(c_[2], c2);
    }
    static inline void add13_avx512(Goldilocks3::Element_avx512 c_, const Goldilocks::Element *a, Goldilocks3::Element_avx512 b_, uint64_t offset_a)
    {
        Goldilocks::Element a0[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * offset_a];
        }
        __m512i a0_;
        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::add_avx512(c_[0], a0_, b_[0]);
        c_[1] = b_[1];
        c_[2] = b_[2];
    }
    static inline void add13c_avx512(Goldilocks3::Element_avx512 c_, const Goldilocks::Element *a, const Goldilocks::Element *b, uint64_t offset_a)
    {
        Goldilocks::Element c0[AVX512_SIZE_];
        Goldilocks::Element c1[AVX512_SIZE_];
        Goldilocks::Element c2[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c0[k] = a[k * offset_a] + b[0];
            c1[k] = b[1];
            c2[k] = b[2];
        }
        Goldilocks::load_avx512(c_[0], c0);
        Goldilocks::load_avx512(c_[1], c1);
        Goldilocks::load_avx512(c_[2], c2);
    }
    static inline void add_avx512(Goldilocks3::Element_avx512 c_, Goldilocks3::Element_avx512 a_, Goldilocks3::Element_avx512 b_)
    {
        Goldilocks::add_avx512(c_[0], a_[0], b_[0]);
        Goldilocks::add_avx512(c_[1], a_[1], b_[1]);
        Goldilocks::add_avx512(c_[2], a_[2], b_[2]);
    }
    static inline void add33c_avx512(Goldilocks3::Element_avx512 c_, Goldilocks3::Element_avx512 a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(b0_, b0); // rick: better set
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        Goldilocks::add_avx512(c_[0], a_[0], b0_);
        Goldilocks::add_avx512(c_[1], a_[1], b1_);
        Goldilocks::add_avx512(c_[2], a_[2], b2_);
    }
    static inline void add_avx512(Goldilocks3::Element_avx512 c_, const Goldilocks::Element *a, Goldilocks3::Element_avx512 b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[AVX512_SIZE_];
        Goldilocks::Element a1[AVX512_SIZE_];
        Goldilocks::Element a2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m512i a0_, a1_, a2_;

        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);

        Goldilocks::add_avx512(c_[0], a0_, b_[0]);
        Goldilocks::add_avx512(c_[1], a1_, b_[1]);
        Goldilocks::add_avx512(c_[2], a2_, b_[2]);
    }
    static inline void add33c_avx512(Goldilocks3::Element_avx512 c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element a0[AVX512_SIZE_];
        Goldilocks::Element a1[AVX512_SIZE_];
        Goldilocks::Element a2[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m512i a0_, a1_, a2_;
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        Goldilocks::add_avx512(c_[0], a0_, b0_);
        Goldilocks::add_avx512(c_[1], a1_, b1_);
        Goldilocks::add_avx512(c_[2], a2_, b2_);
    }
    static inline void add13_avx512(Goldilocks::Element *c, uint64_t stride_c, const __m512i &a_, Goldilocks3::Element_avx512 b_)
    {
        __m512i c0_;
        Goldilocks::add_avx512(c0_, a_, b_[0]);

        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        Goldilocks::store_avx512(c0, c0_);
        Goldilocks::store_avx512(c1, b_[1]);
        Goldilocks::store_avx512(c2, b_[2]);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }
    static inline void add_avx512(Goldilocks::Element *c, uint64_t stride_c, const Goldilocks::Element *a, Goldilocks3::Element_avx512 b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[AVX512_SIZE_];
        Goldilocks::Element a1[AVX512_SIZE_];
        Goldilocks::Element a2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m512i a0_, a1_, a2_;

        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);

        __m512i c0_, c1_, c2_;
        Goldilocks::add_avx512(c0_, a0_, b_[0]);
        Goldilocks::add_avx512(c1_, a1_, b_[1]);
        Goldilocks::add_avx512(c2_, a2_, b_[2]);

        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        Goldilocks::store_avx512(c0, c0_);
        Goldilocks::store_avx512(c1, c1_);
        Goldilocks::store_avx512(c2, c2_);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }
    static inline void add33c_avx512(Goldilocks::Element *c, uint64_t stride_c, Goldilocks3::Element_avx512 a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        __m512i c0_, c1_, c2_;
        Goldilocks::add_avx512(c0_, a_[0], b0_);
        Goldilocks::add_avx512(c1_, a_[1], b1_);
        Goldilocks::add_avx512(c2_, a_[2], b2_);

        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        Goldilocks::store_avx512(c0, c0_);
        Goldilocks::store_avx512(c1, c1_);
        Goldilocks::store_avx512(c2, c2_);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    }
    static inline void add13_avx512(Goldilocks::Element *c, uint64_t stride_c[AVX512_SIZE_], const __m512i &a_, Goldilocks3::Element_avx512 b_)
    {
        __m512i c0_;
        Goldilocks::add_avx512(c0_, a_, b_[0]);

        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        Goldilocks::store_avx512(c0, c0_);
        Goldilocks::store_avx512(c1, b_[1]);
        Goldilocks::store_avx512(c2, b_[2]);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }
    static inline void add_avx512(Goldilocks::Element *c, uint64_t stride_c[AVX512_SIZE_], const Goldilocks::Element *a, Goldilocks3::Element_avx512 b_, uint64_t stride_a)
    {
        Goldilocks::Element a0[AVX512_SIZE_];
        Goldilocks::Element a1[AVX512_SIZE_];
        Goldilocks::Element a2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m512i a0_, a1_, a2_;

        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);

        __m512i c0_, c1_, c2_;
        Goldilocks::add_avx512(c0_, a0_, b_[0]);
        Goldilocks::add_avx512(c1_, a1_, b_[1]);
        Goldilocks::add_avx512(c2_, a2_, b_[2]);

        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        Goldilocks::store_avx512(c0, c0_);
        Goldilocks::store_avx512(c1, c1_);
        Goldilocks::store_avx512(c2, c2_);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }
    static inline void add33c_avx512(Goldilocks::Element *c, uint64_t stride_c[AVX512_SIZE_], Goldilocks3::Element_avx512 a_, const Goldilocks::Element *b)
    {
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        __m512i c0_, c1_, c2_;
        Goldilocks::add_avx512(c0_, a_[0], b0_);
        Goldilocks::add_avx512(c1_, a_[1], b1_);
        Goldilocks::add_avx512(c2_, a_[2], b2_);

        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        Goldilocks::store_avx512(c0, c0_);
        Goldilocks::store_avx512(c1, c1_);
        Goldilocks::store_avx512(c2, c2_);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        }
    }

    static inline void add_avx512(__m512i &c0_, __m512i &c1_, __m512i &c2_, const __m512i a0_, const __m512i a1_, const __m512i a2_, const __m512i b0_, const __m512i b1_, const __m512i b2_)
    {
        Goldilocks::add_avx512(c0_, a0_, b0_);
        Goldilocks::add_avx512(c1_, a1_, b1_);
        Goldilocks::add_avx512(c2_, a2_, b2_);
    }
    static inline void add_avx512(__m512i &c0_, __m512i &c1_, __m512i &c2_, const __m512i a0_, const __m512i a1_, const __m512i a2_, const Goldilocks::Element *b, uint64_t stride)
    {
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[k * stride];
            b1[k] = b[k * stride + 1];
            b2[k] = b[k * stride + 2];
        }
        __m512i b0_, b1_, b2_;
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);
        Goldilocks::add_avx512(c0_, a0_, b0_);
        Goldilocks::add_avx512(c1_, a1_, b1_);
        Goldilocks::add_avx512(c2_, a2_, b2_);
    }
    static inline void add31_avx512(__m512i &c0_, __m512i &c1_, __m512i &c2_, __m512i a0_, const __m512i a1_, const __m512i a2_, const Goldilocks::Element *b, uint64_t stride)
    {
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[k * stride];
            b1[k] = Goldilocks::zero();
            b2[k] = Goldilocks::zero();
        }
        __m512i b0_, b1_, b2_;
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        Goldilocks::add_avx512(c0_, a0_, b0_);
        Goldilocks::add_avx512(c1_, a1_, b1_);
        Goldilocks::add_avx512(c2_, a2_, b2_);
    }

#endif

    // ======== SUB ========
    static inline void sub(Element &result, Element &a, uint64_t &b)
    {
        result[0] = a[0] - Goldilocks::fromU64(b);
        result[1] = a[1];
        result[2] = a[2];
    }
    static inline void sub(Element &result, Goldilocks::Element a, Element &b)
    {
        result[0] = a - b[0];
        result[1] = Goldilocks::neg(b[1]);
        result[2] = Goldilocks::neg(b[2]);
    }
    static inline void sub(Element &result, Element &a, Goldilocks::Element b)
    {
        result[0] = a[0] - b;
        result[1] = a[1];
        result[2] = a[2];
    }
    static inline void sub(Element &result, Element &a, Element &b)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = a[i] - b[i];
        }
    }

    static inline void sub13c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element nb1 = Goldilocks::neg(b[1]);
        Goldilocks::Element nb2 = Goldilocks::neg(b[2]);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] - b[0];
            result[k * FIELD_EXTENSION + 1] = nb1;
            result[k * FIELD_EXTENSION + 2] = nb2;
        }
    }
    static inline void sub33c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                result[k * FIELD_EXTENSION + i] = a[k * stride_a + i] - b[i];
            }
        }
    }
    static inline void sub31_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint32_t stride_b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] - b[k * stride_b];
            result[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            result[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
    }
    static inline void sub31c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element b, uint64_t stride_a)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] - b;
            result[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            result[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
    }
    static inline void sub_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_ * FIELD_EXTENSION; ++k)
        {
            result[k] = a[k] - b[k];
        }
    }
    static inline void sub33c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                result[k * FIELD_EXTENSION + i] = a[k * FIELD_EXTENSION + i] - b[i];
            }
        }
    }
    static inline void sub_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                result[k * FIELD_EXTENSION + i] = a[k * stride_a + i] - b[k * stride_b + i];
            }
        }
    }

    static inline void sub33c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {

        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                aa[k * FIELD_EXTENSION + i] = a[k * stride_a + i];
                bb[k * FIELD_EXTENSION + i] = b[i];
            }
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);

        Goldilocks::sub_avx(c0_, a0_, b0_);
        Goldilocks::sub_avx(c1_, a1_, b1_);
        Goldilocks::sub_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void sub31_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint32_t stride_b)
    {
        // Rick: does not make sense to vectorize
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] - b[k * stride_b];
            result[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            result[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
    }
    static inline void sub31c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element b, uint64_t stride_a)
    {
        // Rick: does not make sense to vectorize
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] - b;
            result[k * FIELD_EXTENSION + 1] = a[k * stride_a + 1];
            result[k * FIELD_EXTENSION + 2] = a[k * stride_a + 2];
        }
    }
    static inline void sub_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, a);
        Goldilocks::load_avx(a1_, &a[4]);
        Goldilocks::load_avx(a2_, &a[8]);
        Goldilocks::load_avx(b0_, b);
        Goldilocks::load_avx(b1_, &b[4]);
        Goldilocks::load_avx(b2_, &b[8]);

        Goldilocks::sub_avx(c0_, a0_, b0_);
        Goldilocks::sub_avx(c1_, a1_, b1_);
        Goldilocks::sub_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void sub33c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                aa[k * FIELD_EXTENSION + i] = a[k * FIELD_EXTENSION + i];
                bb[k * FIELD_EXTENSION + i] = b[i];
            }
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);

        Goldilocks::sub_avx(c0_, a0_, b0_);
        Goldilocks::sub_avx(c1_, a1_, b1_);
        Goldilocks::sub_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void sub_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element bb[12];
        Goldilocks::Element aa[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
            {
                bb[k * FIELD_EXTENSION + i] = b[k * stride_b + i];
                aa[k * FIELD_EXTENSION + i] = a[k * stride_a + i];
            }
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);

        Goldilocks::sub_avx(c0_, a0_, b0_);
        Goldilocks::sub_avx(c1_, a1_, b1_);
        Goldilocks::sub_avx(c2_, a2_, b2_);

        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }

    static inline void sub31c_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks::Element b, uint64_t stride_a)
    {
        Goldilocks::Element c0[4], c1[4], c2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c0[k] = a[k * stride_a] - b;
            c1[k] = a[k * stride_a + 1];
            c2[k] = a[k * stride_a + 2];
        }
        Goldilocks::load_avx(c_[0], c0);
        Goldilocks::load_avx(c_[1], c1);
        Goldilocks::load_avx(c_[2], c2);
    }
    static inline void sub_avx(Goldilocks3::Element_avx &c_, Goldilocks3::Element_avx a_, Goldilocks3::Element_avx b_)
    {
        Goldilocks::sub_avx(c_[0], a_[0], b_[0]);
        Goldilocks::sub_avx(c_[1], a_[1], b_[1]);
        Goldilocks::sub_avx(c_[2], a_[2], b_[2]);
    }
    static inline void sub33c_avx(Goldilocks3::Element_avx &c_, Goldilocks3::Element_avx a_, Goldilocks::Element *b)
    {
        Goldilocks::Element b0[4], b1[4], b2[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        Goldilocks::sub_avx(c_[0], a_[0], b0_);
        Goldilocks::sub_avx(c_[1], a_[1], b1_);
        Goldilocks::sub_avx(c_[2], a_[2], b2_);
    }
    static inline void sub_avx(Goldilocks3::Element_avx &c_, Goldilocks3::Element_avx a_, Goldilocks::Element *b, uint64_t stride_b)
    {
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[k * stride_b];
            b1[k] = b[k * stride_b + 1];
            b2[k] = b[k * stride_b + 2];
        }
        __m256i b0_, b1_, b2_;
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        Goldilocks::sub_avx(c_[0], a_[0], b0_);
        Goldilocks::sub_avx(c_[1], a_[1], b1_);
        Goldilocks::sub_avx(c_[2], a_[2], b2_);
    }
    static inline void sub33c_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {

        Goldilocks::Element a0[4];
        Goldilocks::Element a1[4];
        Goldilocks::Element a2[4];
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        Goldilocks::sub_avx(c_[0], a0_, b0_);
        Goldilocks::sub_avx(c_[1], a1_, b1_);
        Goldilocks::sub_avx(c_[2], a2_, b2_);
    }

    static inline void sub13c_avx(__m256i &c0_, __m256i &c1_, __m256i &c2_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element nb1 = Goldilocks::neg(b[1]);
        Goldilocks::Element nb2 = Goldilocks::neg(b[2]);
        Goldilocks::Element c0[4], c1[4], c2[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c0[k] = a[k * stride_a] - b[0];
            c1[k] = nb1;
            c2[k] = nb2;
        }
        Goldilocks::load_avx(c0_, c0);
        Goldilocks::load_avx(c1_, c1);
        Goldilocks::load_avx(c2_, c2);
    }
    static inline void sub33c_avx(__m256i &c0_, __m256i &c1_, __m256i &c2_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {

        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        Goldilocks::sub_avx(c0_, a0_, b0_);
        Goldilocks::sub_avx(c1_, a1_, b1_);
        Goldilocks::sub_avx(c2_, a2_, b2_);
    }

#ifdef __AVX512__
    static inline void sub31c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks::Element b, uint64_t stride_a)
    {
        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c0[k] = a[k * stride_a] - b;
            c1[k] = a[k * stride_a + 1];
            c2[k] = a[k * stride_a + 2];
        }
        Goldilocks::load_avx512(c_[0], c0);
        Goldilocks::load_avx512(c_[1], c1);
        Goldilocks::load_avx512(c_[2], c2);
    }
    static inline void sub_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks3::Element_avx512 a_, Goldilocks3::Element_avx512 b_)
    {
        Goldilocks::sub_avx512(c_[0], a_[0], b_[0]);
        Goldilocks::sub_avx512(c_[1], a_[1], b_[1]);
        Goldilocks::sub_avx512(c_[2], a_[2], b_[2]);
    }
    static inline void sub33c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks3::Element_avx512 a_, Goldilocks::Element *b)
    {
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        Goldilocks::sub_avx512(c_[0], a_[0], b0_);
        Goldilocks::sub_avx512(c_[1], a_[1], b1_);
        Goldilocks::sub_avx512(c_[2], a_[2], b2_);
    }
    static inline void sub_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks3::Element_avx512 a_, Goldilocks::Element *b, uint64_t stride_b)
    {
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[k * stride_b];
            b1[k] = b[k * stride_b + 1];
            b2[k] = b[k * stride_b + 2];
        }
        __m512i b0_, b1_, b2_;
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        Goldilocks::sub_avx512(c_[0], a_[0], b0_);
        Goldilocks::sub_avx512(c_[1], a_[1], b1_);
        Goldilocks::sub_avx512(c_[2], a_[2], b2_);
    }
    static inline void sub33c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {

        Goldilocks::Element a0[AVX512_SIZE_];
        Goldilocks::Element a1[AVX512_SIZE_];
        Goldilocks::Element a2[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        __m512i a0_, a1_, a2_;
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        Goldilocks::sub_avx512(c_[0], a0_, b0_);
        Goldilocks::sub_avx512(c_[1], a1_, b1_);
        Goldilocks::sub_avx512(c_[2], a2_, b2_);
    }

    static inline void sub13c_avx512(__m512i &c0_, __m512i &c1_, __m512i &c2_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element nb1 = Goldilocks::neg(b[1]);
        Goldilocks::Element nb2 = Goldilocks::neg(b[2]);
        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c0[k] = a[k * stride_a] - b[0];
            c1[k] = nb1;
            c2[k] = nb2;
        }
        Goldilocks::load_avx512(c0_, c0);
        Goldilocks::load_avx512(c1_, c1);
        Goldilocks::load_avx512(c2_, c2);
    }
    static inline void sub33c_avx512(__m512i &c0_, __m512i &c1_, __m512i &c2_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {

        Goldilocks::Element a0[AVX512_SIZE_], a1[AVX512_SIZE_], a2[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i a0_, a1_, a2_;
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        Goldilocks::sub_avx512(c0_, a0_, b0_);
        Goldilocks::sub_avx512(c1_, a1_, b1_);
        Goldilocks::sub_avx512(c2_, a2_, b2_);
    }

#endif
    // ======== NEG ========
    static inline void neg(Element &result, Element &a)
    {
        sub(result, (Element &)zero(), a);
    }

    // ======== MUL ========
    static inline void mul(Element *result, Element *a, Element *b)
    {
        mul(*result, *a, *b);
    }
    static inline void mul(Element &result, Element &a, Element &b)
    {
        Goldilocks::Element A = (a[0] + a[1]) * (b[0] + b[1]);
        Goldilocks::Element B = (a[0] + a[2]) * (b[0] + b[2]);
        Goldilocks::Element C = (a[1] + a[2]) * (b[1] + b[2]);
        Goldilocks::Element D = a[0] * b[0];
        Goldilocks::Element E = a[1] * b[1];
        Goldilocks::Element F = a[2] * b[2];
        Goldilocks::Element G = D - E;

        result[0] = (C + G) - F;
        result[1] = ((((A + C) - E) - E) - D);
        result[2] = B - G;
    };
    static inline void mul(Element &result, Element &a, Goldilocks::Element &b)
    {
        result[0] = a[0] * b;
        result[1] = a[1] * b;
        result[2] = a[2] * b;
    }
    static inline void mul(Element &result, Goldilocks::Element a, Element &b)
    {
        mul(result, b, a);
    }
    static inline void mul(Element &result, Element &a, uint64_t b)
    {
        result[0] = a[0] * Goldilocks::fromU64(b);
        result[1] = a[1] * Goldilocks::fromU64(b);
        result[2] = a[2] * Goldilocks::fromU64(b);
    }

    static inline void mul13c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Element &b, uint64_t stride_a)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = b[0] * a[k * stride_a];
            result[k * FIELD_EXTENSION + 1] = b[1] * a[k * stride_a];
            result[k * FIELD_EXTENSION + 2] = b[2] * a[k * stride_a];
        }
    }
    static inline void mul1c3c_batch(Goldilocks::Element *result, Goldilocks::Element a, Element &b)
    {
        result[0] = b[0] * a;
        result[1] = b[1] * a;
        result[2] = b[2] * a;
        for (uint64_t k = 1; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = result[0];
            result[k * FIELD_EXTENSION + 1] = result[1];
            result[k * FIELD_EXTENSION + 2] = result[2];
        }
    }
    static inline void mul13c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Element &b, const uint64_t stride_a[4])
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = b[0] * a[stride_a[k]];
            result[k * FIELD_EXTENSION + 1] = b[1] * a[stride_a[k]];
            result[k * FIELD_EXTENSION + 2] = b[2] * a[stride_a[k]];
        }
    }
    static inline void mul_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, Goldilocks::Element b_[3])
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            Goldilocks::Element A = (a[k * FIELD_EXTENSION] + a[k * FIELD_EXTENSION + 1]) * b_[0];
            Goldilocks::Element B = (a[k * FIELD_EXTENSION] + a[k * FIELD_EXTENSION + 2]) * b_[1];
            Goldilocks::Element C = (a[k * FIELD_EXTENSION + 1] + a[k * FIELD_EXTENSION + 2]) * b_[2];
            Goldilocks::Element D = a[k * FIELD_EXTENSION] * b[0];
            Goldilocks::Element E = a[k * FIELD_EXTENSION + 1] * b[1];
            Goldilocks::Element F = a[k * FIELD_EXTENSION + 2] * b[2];
            Goldilocks::Element G = D - E;

            result[k * FIELD_EXTENSION] = (C + G) - F;
            result[k * FIELD_EXTENSION + 1] = ((((A + C) - E) - E) - D);
            result[k * FIELD_EXTENSION + 2] = B - G;
        }
    };
    static inline void mul_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            Goldilocks::Element A = (a[k * FIELD_EXTENSION] + a[k * FIELD_EXTENSION + 1]) * (b[k * FIELD_EXTENSION] + b[k * FIELD_EXTENSION + 1]);
            Goldilocks::Element B = (a[k * FIELD_EXTENSION] + a[k * FIELD_EXTENSION + 2]) * (b[k * FIELD_EXTENSION] + b[k * FIELD_EXTENSION + 2]);
            Goldilocks::Element C = (a[k * FIELD_EXTENSION + 1] + a[k * FIELD_EXTENSION + 2]) * (b[k * FIELD_EXTENSION + 1] + b[k * FIELD_EXTENSION + 2]);
            Goldilocks::Element D = a[k * FIELD_EXTENSION] * b[k * FIELD_EXTENSION];
            Goldilocks::Element E = a[k * FIELD_EXTENSION + 1] * b[k * FIELD_EXTENSION + 1];
            Goldilocks::Element F = a[k * FIELD_EXTENSION + 2] * b[k * FIELD_EXTENSION + 2];
            Goldilocks::Element G = D - E;

            result[k * FIELD_EXTENSION] = (C + G) - F;
            result[k * FIELD_EXTENSION + 1] = ((((A + C) - E) - E) - D);
            result[k * FIELD_EXTENSION + 2] = B - G;
        }
    };
    static inline void mul33c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        Goldilocks::Element aux[3];
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            Goldilocks::Element A = (a[k * FIELD_EXTENSION] + a[k * FIELD_EXTENSION + 1]) * aux[0];
            Goldilocks::Element B = (a[k * FIELD_EXTENSION] + a[k * FIELD_EXTENSION + 2]) * aux[1];
            Goldilocks::Element C = (a[k * FIELD_EXTENSION + 1] + a[k * FIELD_EXTENSION + 2]) * aux[2];
            Goldilocks::Element D = a[k * FIELD_EXTENSION] * b[0];
            Goldilocks::Element E = a[k * FIELD_EXTENSION + 1] * b[1];
            Goldilocks::Element F = a[k * FIELD_EXTENSION + 2] * b[2];
            Goldilocks::Element G = D - E;

            result[k * FIELD_EXTENSION] = (C + G) - F;
            result[k * FIELD_EXTENSION + 1] = ((((A + C) - E) - E) - D);
            result[k * FIELD_EXTENSION + 2] = B - G;
        }
    };
    static inline void mul_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride0, uint64_t stride1)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            Goldilocks::Element A = (a[k * stride0] + a[k * stride0 + 1]) * (b[k * stride1] + b[k * stride1 + 1]);
            Goldilocks::Element B = (a[k * stride0] + a[k * stride0 + 2]) * (b[k * stride1] + b[k * stride1 + 2]);
            Goldilocks::Element C = (a[k * stride0 + 1] + a[k * stride0 + 2]) * (b[k * stride1 + 1] + b[k * stride1 + 2]);
            Goldilocks::Element D = a[k * stride0] * b[k * stride1];
            Goldilocks::Element E = a[k * stride0 + 1] * b[k * stride1 + 1];
            Goldilocks::Element F = a[k * stride0 + 2] * b[k * stride1 + 2];
            Goldilocks::Element G = D - E;

            result[k * FIELD_EXTENSION] = (C + G) - F;
            result[k * FIELD_EXTENSION + 1] = ((((A + C) - E) - E) - D);
            result[k * FIELD_EXTENSION + 2] = B - G;
        }
    };
    static inline void mul_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride0[4], const uint64_t stride1[4])
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            Goldilocks::Element A = (a[stride0[k]] + a[stride0[k] + 1]) * (b[stride1[k]] + b[stride1[k] + 1]);
            Goldilocks::Element B = (a[stride0[k]] + a[stride0[k] + 2]) * (b[stride1[k]] + b[stride1[k] + 2]);
            Goldilocks::Element C = (a[stride0[k] + 1] + a[stride0[k] + 2]) * (b[stride1[k] + 1] + b[stride1[k] + 2]);
            Goldilocks::Element D = a[stride0[k]] * b[stride1[k]];
            Goldilocks::Element E = a[stride0[k] + 1] * b[stride1[k] + 1];
            Goldilocks::Element F = a[stride0[k] + 2] * b[stride1[k] + 2];
            Goldilocks::Element G = D - E;

            result[k * FIELD_EXTENSION] = (C + G) - F;
            result[k * FIELD_EXTENSION + 1] = ((((A + C) - E) - E) - D);
            result[k * FIELD_EXTENSION + 2] = B - G;
        }
    };
    static inline void mul33c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element aux[3];
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            Goldilocks::Element A = (a[k * stride_a] + a[k * stride_a + 1]) * aux[0];
            Goldilocks::Element B = (a[k * stride_a] + a[k * stride_a + 2]) * aux[1];
            Goldilocks::Element C = (a[k * stride_a + 1] + a[k * stride_a + 2]) * aux[2];
            Goldilocks::Element D = a[k * stride_a] * b[0];
            Goldilocks::Element E = a[k * stride_a + 1] * b[1];
            Goldilocks::Element F = a[k * stride_a + 2] * b[2];
            Goldilocks::Element G = D - E;

            result[k * FIELD_EXTENSION] = (C + G) - F;
            result[k * FIELD_EXTENSION + 1] = ((((A + C) - E) - E) - D);
            result[k * FIELD_EXTENSION + 2] = B - G;
        }
    };
    static inline void mul33c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[4])
    {
        Goldilocks::Element aux[3];
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            Goldilocks::Element A = (a[stride_a[k]] + a[stride_a[k] + 1]) * aux[0];
            Goldilocks::Element B = (a[stride_a[k]] + a[stride_a[k] + 2]) * aux[1];
            Goldilocks::Element C = (a[stride_a[k] + 1] + a[stride_a[k] + 2]) * aux[2];
            Goldilocks::Element D = a[stride_a[k]] * b[0];
            Goldilocks::Element E = a[stride_a[k] + 1] * b[1];
            Goldilocks::Element F = a[stride_a[k] + 2] * b[2];
            Goldilocks::Element G = D - E;

            result[k * FIELD_EXTENSION] = (C + G) - F;
            result[k * FIELD_EXTENSION + 1] = ((((A + C) - E) - E) - D);
            result[k * FIELD_EXTENSION + 2] = B - G;
        }
    };
    static inline void mul13c_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = b[0] * a[k];
            result[k * FIELD_EXTENSION + 1] = b[1] * a[k];
            result[k * FIELD_EXTENSION + 2] = b[2] * a[k];
        }
    }
    static inline void mul13_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[k * stride_a] * b[k * stride_b];
            result[k * FIELD_EXTENSION + 1] = a[k * stride_a] * b[k * stride_b + 1];
            result[k * FIELD_EXTENSION + 2] = a[k * stride_a] * b[k * stride_b + 2];
        }
    }
    static inline void mul13_batch(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[4], const uint64_t stride_b[4])
    {
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = a[stride_a[k]] * b[stride_b[k]];
            result[k * FIELD_EXTENSION + 1] = a[stride_a[k]] * b[stride_b[k] + 1];
            result[k * FIELD_EXTENSION + 2] = a[stride_a[k]] * b[stride_b[k] + 2];
        }
    }

    static inline void mul13c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Element &b, uint64_t stride_a)
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 1] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 2] = a[k * stride_a];
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);
        Goldilocks::mult_avx(c0_, a0_, b0_);
        Goldilocks::mult_avx(c1_, a1_, b1_);
        Goldilocks::mult_avx(c2_, a2_, b2_);
        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void mul1c3c_avx(Goldilocks::Element *result, Goldilocks::Element a, Element &b)
    {
        // Does not make sense to vectorize
        result[0] = b[0] * a;
        result[1] = b[1] * a;
        result[2] = b[2] * a;
        for (uint64_t k = 1; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = result[0];
            result[k * FIELD_EXTENSION + 1] = result[1];
            result[k * FIELD_EXTENSION + 2] = result[2];
        }
    }
    static inline void mul13c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Element &b, const uint64_t stride_a[4])
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[stride_a[k]];
            aa[k * FIELD_EXTENSION + 1] = a[stride_a[k]];
            aa[k * FIELD_EXTENSION + 2] = a[stride_a[k]];
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);
        Goldilocks::mult_avx(c0_, a0_, b0_);
        Goldilocks::mult_avx(c1_, a1_, b1_);
        Goldilocks::mult_avx(c2_, a2_, b2_);
        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void mul_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {

            a0[k] = a[k * FIELD_EXTENSION];
            a1[k] = a[k * FIELD_EXTENSION + 1];
            a2[k] = a[k * FIELD_EXTENSION + 2];
            b0[k] = b[k * FIELD_EXTENSION];
            b1[k] = b[k * FIELD_EXTENSION + 1];
            b2[k] = b[k * FIELD_EXTENSION + 2];
        }
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[4], result1[4], result2[4];

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::add_avx(aux0_, b0_, b1_);
        Goldilocks::add_avx(aux1_, b0_, b2_);
        Goldilocks::add_avx(aux2_, b1_, b2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(result0_, C_, G_);
        Goldilocks::sub_avx(result0_, result0_, F_);
        Goldilocks::add_avx(result1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(result1_, result1_, auxr_);
        Goldilocks::sub_avx(result2_, B_, G_);

        Goldilocks::store_avx(result0, result0_);
        Goldilocks::store_avx(result1, result1_);
        Goldilocks::store_avx(result2, result2_);

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };

    static inline void mul33c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        assert(AVX_SIZE_ == 4);
        Goldilocks::Element aux0[4], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
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
        Goldilocks::load_avx(aux0_, aux0);
        Goldilocks::load_avx(aux1_, aux1);
        Goldilocks::load_avx(aux2_, aux2);
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[4], result1[4], result2[4];

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(result0_, C_, G_);
        Goldilocks::sub_avx(result0_, result0_, F_);
        Goldilocks::add_avx(result1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(result1_, result1_, auxr_);
        Goldilocks::sub_avx(result2_, B_, G_);

        Goldilocks::store_avx(result0, result0_);
        Goldilocks::store_avx(result1, result1_);
        Goldilocks::store_avx(result2, result2_);

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };
    static inline void mul_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        assert(AVX_SIZE_ == 4);
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[k * stride_b];
            b1[k] = b[k * stride_b + 1];
            b2[k] = b[k * stride_b + 2];
        }
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[4], result1[4], result2[4];

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::add_avx(aux0_, b0_, b1_);
        Goldilocks::add_avx(aux1_, b0_, b2_);
        Goldilocks::add_avx(aux2_, b1_, b2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(result0_, C_, G_);
        Goldilocks::sub_avx(result0_, result0_, F_);
        Goldilocks::add_avx(result1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(result1_, result1_, auxr_);
        Goldilocks::sub_avx(result2_, B_, G_);

        Goldilocks::store_avx(result0, result0_);
        Goldilocks::store_avx(result1, result1_);
        Goldilocks::store_avx(result2, result2_);

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };
    static inline void mul_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[4], const uint64_t stride_b[4])
    {
        assert(AVX_SIZE_ == 4);
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
            b0[k] = b[stride_b[k]];
            b1[k] = b[stride_b[k] + 1];
            b2[k] = b[stride_b[k] + 2];
        }
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[4], result1[4], result2[4];

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::add_avx(aux0_, b0_, b1_);
        Goldilocks::add_avx(aux1_, b0_, b2_);
        Goldilocks::add_avx(aux2_, b1_, b2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(result0_, C_, G_);
        Goldilocks::sub_avx(result0_, result0_, F_);
        Goldilocks::add_avx(result1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(result1_, result1_, auxr_);
        Goldilocks::sub_avx(result2_, B_, G_);

        Goldilocks::store_avx(result0, result0_);
        Goldilocks::store_avx(result1, result1_);
        Goldilocks::store_avx(result2, result2_);

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };
    static inline void mul33c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element aux0[4], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
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
        Goldilocks::load_avx(aux0_, aux0);
        Goldilocks::load_avx(aux1_, aux1);
        Goldilocks::load_avx(aux2_, aux2);
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[4], result1[4], result2[4];

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(result0_, C_, G_);
        Goldilocks::sub_avx(result0_, result0_, F_);
        Goldilocks::add_avx(result1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(result1_, result1_, auxr_);
        Goldilocks::sub_avx(result2_, B_, G_);

        Goldilocks::store_avx(result0, result0_);
        Goldilocks::store_avx(result1, result1_);
        Goldilocks::store_avx(result2, result2_);

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };
    static inline void mul33c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[4])
    {
        Goldilocks::Element aux0[4], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
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
        Goldilocks::load_avx(aux0_, aux0);
        Goldilocks::load_avx(aux1_, aux1);
        Goldilocks::load_avx(aux2_, aux2);
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i result0_, result1_, result2_, auxr_;
        Goldilocks::Element result0[4], result1[4], result2[4];

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(result0_, C_, G_);
        Goldilocks::sub_avx(result0_, result0_, F_);
        Goldilocks::add_avx(result1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(result1_, result1_, auxr_);
        Goldilocks::sub_avx(result2_, B_, G_);

        Goldilocks::store_avx(result0, result0_);
        Goldilocks::store_avx(result1, result1_);
        Goldilocks::store_avx(result2, result2_);

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            result[k * FIELD_EXTENSION] = result0[k];
            result[k * FIELD_EXTENSION + 1] = result1[k];
            result[k * FIELD_EXTENSION + 2] = result2[k];
        }
    };
    static inline void mul13c_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b)
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = a[k];
            aa[k * FIELD_EXTENSION + 2] = a[k];
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);
        Goldilocks::mult_avx(c0_, a0_, b0_);
        Goldilocks::mult_avx(c1_, a1_, b1_);
        Goldilocks::mult_avx(c2_, a2_, b2_);
        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void mul13_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 1] = a[k * stride_a];
            aa[k * FIELD_EXTENSION + 2] = a[k * stride_a];
            bb[k * FIELD_EXTENSION] = b[k * stride_b];
            bb[k * FIELD_EXTENSION + 1] = b[k * stride_b + 1];
            bb[k * FIELD_EXTENSION + 2] = b[k * stride_b + 2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);
        Goldilocks::mult_avx(c0_, a0_, b0_);
        Goldilocks::mult_avx(c1_, a1_, b1_);
        Goldilocks::mult_avx(c2_, a2_, b2_);
        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void mul13_avx(Goldilocks::Element *result, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[4], const uint64_t stride_b[4])
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[stride_a[k]];
            aa[k * FIELD_EXTENSION + 1] = a[stride_a[k]];
            aa[k * FIELD_EXTENSION + 2] = a[stride_a[k]];
            bb[k * FIELD_EXTENSION] = b[stride_b[k]];
            bb[k * FIELD_EXTENSION + 1] = b[stride_b[k] + 1];
            bb[k * FIELD_EXTENSION + 2] = b[stride_b[k] + 2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);
        Goldilocks::mult_avx(c0_, a0_, b0_);
        Goldilocks::mult_avx(c1_, a1_, b1_);
        Goldilocks::mult_avx(c2_, a2_, b2_);
        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }

    static inline void mul13c_avx(Goldilocks::Element *result, const __m256i &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];
        Goldilocks::Element a[4];
        Goldilocks::store_avx(a, a_);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = a[k];
            aa[k * FIELD_EXTENSION + 2] = a[k];
            bb[k * FIELD_EXTENSION] = b[0];
            bb[k * FIELD_EXTENSION + 1] = b[1];
            bb[k * FIELD_EXTENSION + 2] = b[2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);
        Goldilocks::mult_avx(c0_, a0_, b0_);
        Goldilocks::mult_avx(c1_, a1_, b1_);
        Goldilocks::mult_avx(c2_, a2_, b2_);
        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void mul13_avx(Goldilocks::Element *result, const __m256i &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element aa[12];
        Goldilocks::Element bb[12];
        Goldilocks::Element a[4];
        Goldilocks::store_avx(a, a_);

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aa[k * FIELD_EXTENSION] = a[k];
            aa[k * FIELD_EXTENSION + 1] = a[k];
            aa[k * FIELD_EXTENSION + 2] = a[k];
            bb[k * FIELD_EXTENSION] = b[k * FIELD_EXTENSION];
            bb[k * FIELD_EXTENSION + 1] = b[k * FIELD_EXTENSION + 1];
            bb[k * FIELD_EXTENSION + 2] = b[k * FIELD_EXTENSION + 2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;
        __m256i c0_, c1_, c2_;

        Goldilocks::load_avx(a0_, aa);
        Goldilocks::load_avx(a1_, &aa[4]);
        Goldilocks::load_avx(a2_, &aa[8]);
        Goldilocks::load_avx(b0_, bb);
        Goldilocks::load_avx(b1_, &bb[4]);
        Goldilocks::load_avx(b2_, &bb[8]);
        Goldilocks::mult_avx(c0_, a0_, b0_);
        Goldilocks::mult_avx(c1_, a1_, b1_);
        Goldilocks::mult_avx(c2_, a2_, b2_);
        Goldilocks::store_avx(result, c0_);
        Goldilocks::store_avx(&result[4], c1_);
        Goldilocks::store_avx(&result[8], c2_);
    }
    static inline void mul13c_avx(Goldilocks3::Element_avx &c_, const __m256i &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i b0_, b1_, b2_;
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);
        Goldilocks::mult_avx(c_[0], a_, b0_);
        Goldilocks::mult_avx(c_[1], a_, b1_);
        Goldilocks::mult_avx(c_[2], a_, b2_);
    }
    static inline void mul13_avx(Goldilocks3::Element_avx &c_, const __m256i &a_, const Goldilocks3::Element_avx &b_)
    {
        Goldilocks::mult_avx(c_[0], a_, b_[0]);
        Goldilocks::mult_avx(c_[1], a_, b_[1]);
        Goldilocks::mult_avx(c_[2], a_, b_[2]);
    }
    static inline void mul13_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks3::Element_avx b_, uint64_t stride_a)
    {
        Goldilocks::Element a4[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a4[k] = a[k * stride_a];
        }
        __m256i a_;
        Goldilocks::load_avx(a_, a4);
        Goldilocks::mult_avx(c_[0], a_, b_[0]);
        Goldilocks::mult_avx(c_[1], a_, b_[1]);
        Goldilocks::mult_avx(c_[2], a_, b_[2]);
    }
    static inline void mul13c_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element a0[4];
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i a_;
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(a_, a0);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);
        Goldilocks::mult_avx(c_[0], a_, b0_);
        Goldilocks::mult_avx(c_[1], a_, b1_);
        Goldilocks::mult_avx(c_[2], a_, b2_);
    }
    static inline void mul13c_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[4])
    {
        Goldilocks::Element a0[4];
        Goldilocks::Element b0[4];
        Goldilocks::Element b1[4];
        Goldilocks::Element b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[stride_a[k]];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i a_;
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(a_, a0);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);
        Goldilocks::mult_avx(c_[0], a_, b0_);
        Goldilocks::mult_avx(c_[1], a_, b1_);
        Goldilocks::mult_avx(c_[2], a_, b2_);
    }
    static inline void mul13_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks3::Element_avx b_, const uint64_t stride_a[4])
    {
        Goldilocks::Element a4[4];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a4[k] = a[stride_a[k]];
        }
        __m256i a_;
        Goldilocks::load_avx(a_, a4);
        Goldilocks::mult_avx(c_[0], a_, b_[0]);
        Goldilocks::mult_avx(c_[1], a_, b_[1]);
        Goldilocks::mult_avx(c_[2], a_, b_[2]);
    }
    static inline void mul1c3c_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element a, Element &b)
    {
        // Does not make sense to vectorize
        Goldilocks::Element cc0 = b[0] * a;
        Goldilocks::Element cc1 = b[1] * a;
        Goldilocks::Element cc2 = b[2] * a;

        Goldilocks::Element c0[4] = {cc0, cc0, cc0, cc0};
        Goldilocks::Element c1[4] = {cc1, cc1, cc1, cc1};
        Goldilocks::Element c2[4] = {cc2, cc2, cc2, cc2};

        Goldilocks::load_avx(c_[0], c0);
        Goldilocks::load_avx(c_[1], c1);
        Goldilocks::load_avx(c_[2], c2);
    }
    static inline void mul33c_avx(Goldilocks3::Element_avx &c_, Goldilocks3::Element_avx &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element aux0[4], aux1[4], aux2[4], aux[3];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            aux0[k] = aux[0];
            aux1[k] = aux[1];
            aux2[k] = aux[2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        Goldilocks::load_avx(aux0_, aux0);
        Goldilocks::load_avx(aux1_, aux1);
        Goldilocks::load_avx(aux2_, aux2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a_[0], a_[1]);
        Goldilocks::add_avx(B_, a_[0], a_[2]);
        Goldilocks::add_avx(C_, a_[1], a_[2]);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a_[0], b0_);
        Goldilocks::mult_avx(E_, a_[1], b1_);
        Goldilocks::mult_avx(F_, a_[2], b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };
    static inline void mul_avx(Goldilocks3::Element_avx &c_, Goldilocks3::Element_avx &a_, Goldilocks3::Element_avx &b_)
    {

        __m256i aux0_, aux1_, aux2_;
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a_[0], a_[1]);
        Goldilocks::add_avx(B_, a_[0], a_[2]);
        Goldilocks::add_avx(C_, a_[1], a_[2]);
        Goldilocks::add_avx(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a_[0], b_[0]);
        Goldilocks::mult_avx(E_, a_[1], b_[1]);
        Goldilocks::mult_avx(F_, a_[2], b_[2]);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };

    static inline void mul_avx(Goldilocks::Element *c, uint64_t stride_c, Goldilocks3::Element_avx &a_, Goldilocks3::Element_avx &b_)
    {

        __m256i aux0_, aux1_, aux2_;
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a_[0], a_[1]);
        Goldilocks::add_avx(B_, a_[0], a_[2]);
        Goldilocks::add_avx(C_, a_[1], a_[2]);
        Goldilocks::add_avx(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a_[0], b_[0]);
        Goldilocks::mult_avx(E_, a_[1], b_[1]);
        Goldilocks::mult_avx(F_, a_[2], b_[2]);
        Goldilocks::sub_avx(G_, D_, E_);

        __m256i c0_, c1_, c2_;

        Goldilocks::add_avx(c0_, C_, G_);
        Goldilocks::sub_avx(c0_, c0_, F_);
        Goldilocks::add_avx(c1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c1_, c1_, auxr_);
        Goldilocks::sub_avx(c2_, B_, G_);

        Goldilocks::Element c0[4], c1[4], c2[4];
        Goldilocks::store_avx(c0, c0_);
        Goldilocks::store_avx(c1, c1_);
        Goldilocks::store_avx(c2, c2_);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    };
    static inline void mul_avx(Goldilocks::Element *c, uint64_t stride_c[4], Goldilocks3::Element_avx &a_, Goldilocks3::Element_avx &b_)
    {

        __m256i aux0_, aux1_, aux2_;
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a_[0], a_[1]);
        Goldilocks::add_avx(B_, a_[0], a_[2]);
        Goldilocks::add_avx(C_, a_[1], a_[2]);
        Goldilocks::add_avx(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a_[0], b_[0]);
        Goldilocks::mult_avx(E_, a_[1], b_[1]);
        Goldilocks::mult_avx(F_, a_[2], b_[2]);
        Goldilocks::sub_avx(G_, D_, E_);

        __m256i c0_, c1_, c2_;

        Goldilocks::add_avx(c0_, C_, G_);
        Goldilocks::sub_avx(c0_, c0_, F_);
        Goldilocks::add_avx(c1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c1_, c1_, auxr_);
        Goldilocks::sub_avx(c2_, B_, G_);

        Goldilocks::Element c0[4], c1[4], c2[4];
        Goldilocks::store_avx(c0, c0_);
        Goldilocks::store_avx(c1, c1_);
        Goldilocks::store_avx(c2, c2_);
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        };
    };

    static inline void mul_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[k * stride_b];
            b1[k] = b[k * stride_b + 1];
            b2[k] = b[k * stride_b + 2];
        }
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::add_avx(aux0_, b0_, b1_);
        Goldilocks::add_avx(aux1_, b0_, b2_);
        Goldilocks::add_avx(aux2_, b1_, b2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };
    static inline void mul33c_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element aux0[4], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
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
        Goldilocks::load_avx(aux0_, aux0);
        Goldilocks::load_avx(aux1_, aux1);
        Goldilocks::load_avx(aux2_, aux2);
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };
    static inline void mul33c_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[4])
    {
        Goldilocks::Element aux0[4], aux1[4], aux2[4], aux[3];
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
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
        Goldilocks::load_avx(aux0_, aux0);
        Goldilocks::load_avx(aux1_, aux1);
        Goldilocks::load_avx(aux2_, aux2);
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };
    static inline void mul_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks3::Element_avx &b_, const uint64_t stride_a[4])
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
        }
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::add_avx(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b_[0]);
        Goldilocks::mult_avx(E_, a1_, b_[1]);
        Goldilocks::mult_avx(F_, a2_, b_[2]);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };
    static inline void mul_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks3::Element_avx &b_, const uint64_t stride_a)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::add_avx(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b_[0]);
        Goldilocks::mult_avx(E_, a1_, b_[1]);
        Goldilocks::mult_avx(F_, a2_, b_[2]);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };
    static inline void mul_avx(Goldilocks3::Element_avx &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[4], const uint64_t stride_b[4])
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
            b0[k] = b[stride_b[k]];
            b1[k] = b[stride_b[k] + 1];
            b2[k] = b[stride_b[k] + 2];
        }
        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::add_avx(aux0_, b0_, b1_);
        Goldilocks::add_avx(aux1_, b0_, b2_);
        Goldilocks::add_avx(aux2_, b1_, b2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };

    static inline void mul13c_avx(__m256i &c0_, __m256i &c1_, __m256i &c2_, Goldilocks::Element *a, Element &b, uint64_t stride_a)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        Goldilocks::Element b0[4], b1[4], b2[4];

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a];
            a2[k] = a[k * stride_a];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m256i a0_, a1_, a2_;
        __m256i b0_, b1_, b2_;

        Goldilocks::load_avx(a0_, a0);
        Goldilocks::load_avx(a1_, a1);
        Goldilocks::load_avx(a2_, a2);
        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);
        Goldilocks::mult_avx(c0_, a0_, b0_);
        Goldilocks::mult_avx(c1_, a1_, b1_);
        Goldilocks::mult_avx(c2_, a2_, b2_);
    }
    static inline void mul_avx(__m256i &c0_, __m256i &c1_, __m256i &c2_, __m256i a0_, __m256i a1_, __m256i a2_, __m256i b0_, __m256i b1_, __m256i b2_, __m256i aux0_, __m256i aux1_, __m256i aux2_)
    {
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i result0_, result1_, auxr_;

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(result0_, C_, G_);
        Goldilocks::sub_avx(c0_, result0_, F_);
        Goldilocks::add_avx(result1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c1_, result1_, auxr_);
        Goldilocks::sub_avx(c2_, B_, G_);
    };
    static inline void mul_avx(__m256i &c0_, __m256i &c1_, __m256i &c2_, __m256i a0_, __m256i a1_, __m256i a2_, Goldilocks::Element *b)
    {
        assert(AVX_SIZE_ == 4);
        Goldilocks::Element b0[4], b1[4], b2[4];
        __m256i aux0_, aux1_, aux2_;
        __m256i b0_, b1_, b2_;

        // redistribute data:

        for (uint64_t k = 0; k < AVX_SIZE_; ++k)
        {
            b0[k] = b[k * FIELD_EXTENSION];
            b1[k] = b[k * FIELD_EXTENSION + 1];
            b2[k] = b[k * FIELD_EXTENSION + 2];
        }

        Goldilocks::load_avx(b0_, b0);
        Goldilocks::load_avx(b1_, b1);
        Goldilocks::load_avx(b2_, b2);

        // operations
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i result0_, result1_, auxr_;

        Goldilocks::add_avx(A_, a0_, a1_);
        Goldilocks::add_avx(B_, a0_, a2_);
        Goldilocks::add_avx(C_, a1_, a2_);
        Goldilocks::add_avx(aux0_, b0_, b1_);
        Goldilocks::add_avx(aux1_, b0_, b2_);
        Goldilocks::add_avx(aux2_, b1_, b2_);
        Goldilocks::mult_avx(A_, A_, aux0_);
        Goldilocks::mult_avx(B_, B_, aux1_);
        Goldilocks::mult_avx(C_, C_, aux2_);
        Goldilocks::mult_avx(D_, a0_, b0_);
        Goldilocks::mult_avx(E_, a1_, b1_);
        Goldilocks::mult_avx(F_, a2_, b2_);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(result0_, C_, G_);
        Goldilocks::sub_avx(c0_, result0_, F_);
        Goldilocks::add_avx(result1_, A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c1_, result1_, auxr_);
        Goldilocks::sub_avx(c2_, B_, G_);
    };

#ifdef __AVX512__
    static inline void mul13c_avx512(Goldilocks3::Element_avx512 &c_, const __m512i &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i b0_, b1_, b2_;
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);
        Goldilocks::mult_avx512(c_[0], a_, b0_);
        Goldilocks::mult_avx512(c_[1], a_, b1_);
        Goldilocks::mult_avx512(c_[2], a_, b2_);
    }
    static inline void mul13_avx512(Goldilocks3::Element_avx512 &c_, const __m512i &a_, const Goldilocks3::Element_avx512 &b_)
    {
        Goldilocks::mult_avx512(c_[0], a_, b_[0]);
        Goldilocks::mult_avx512(c_[1], a_, b_[1]);
        Goldilocks::mult_avx512(c_[2], a_, b_[2]);
    }
    static inline void mul13_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks3::Element_avx512 b_, uint64_t stride_a)
    {
        Goldilocks::Element a8[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a8[k] = a[k * stride_a];
        }
        __m512i a_;
        Goldilocks::load_avx512(a_, a8);
        Goldilocks::mult_avx512(c_[0], a_, b_[0]);
        Goldilocks::mult_avx512(c_[1], a_, b_[1]);
        Goldilocks::mult_avx512(c_[2], a_, b_[2]);
    }
    static inline void mul13c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element a0[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i a_;
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(a_, a0); // rick: set
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);
        Goldilocks::mult_avx512(c_[0], a_, b0_);
        Goldilocks::mult_avx512(c_[1], a_, b1_);
        Goldilocks::mult_avx512(c_[2], a_, b2_);
    }
    static inline void mul13c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[AVX512_SIZE_])
    {
        Goldilocks::Element a0[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_];
        Goldilocks::Element b1[AVX512_SIZE_];
        Goldilocks::Element b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[stride_a[k]];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i a_;
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(a_, a0); // rick: set
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);
        Goldilocks::mult_avx512(c_[0], a_, b0_);
        Goldilocks::mult_avx512(c_[1], a_, b1_);
        Goldilocks::mult_avx512(c_[2], a_, b2_);
    }
    static inline void mul13_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks3::Element_avx512 b_, const uint64_t stride_a[AVX512_SIZE_])
    {
        Goldilocks::Element a8[AVX512_SIZE_];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a8[k] = a[stride_a[k]];
        }
        __m512i a_;
        Goldilocks::load_avx512(a_, a8);
        Goldilocks::mult_avx512(c_[0], a_, b_[0]);
        Goldilocks::mult_avx512(c_[1], a_, b_[1]);
        Goldilocks::mult_avx512(c_[2], a_, b_[2]);
    }
    static inline void mul1c3c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element a, Element &b)
    {
        // Does not make sense to vectorize
        Goldilocks::Element cc0 = b[0] * a;
        Goldilocks::Element cc1 = b[1] * a;
        Goldilocks::Element cc2 = b[2] * a;

        Goldilocks::Element c0[AVX512_SIZE_] = {cc0, cc0, cc0, cc0, cc0, cc0, cc0, cc0};
        Goldilocks::Element c1[AVX512_SIZE_] = {cc1, cc1, cc1, cc1, cc1, cc1, cc1, cc1};
        Goldilocks::Element c2[AVX512_SIZE_] = {cc2, cc2, cc2, cc2, cc2, cc2, cc2, cc2};

        Goldilocks::load_avx512(c_[0], c0);
        Goldilocks::load_avx512(c_[1], c1);
        Goldilocks::load_avx512(c_[2], c2);
    }
    static inline void mul33c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks3::Element_avx512 &a_, Goldilocks::Element *b)
    {
        Goldilocks::Element aux0[AVX512_SIZE_], aux1[AVX512_SIZE_], aux2[AVX512_SIZE_], aux[3];
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        __m512i aux0_, aux1_, aux2_;
        __m512i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            aux0[k] = aux[0];
            aux1[k] = aux[1];
            aux2[k] = aux[2];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        Goldilocks::load_avx512(aux0_, aux0);
        Goldilocks::load_avx512(aux1_, aux1);
        Goldilocks::load_avx512(aux2_, aux2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        // operations
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a_[0], a_[1]);
        Goldilocks::add_avx512(B_, a_[0], a_[2]);
        Goldilocks::add_avx512(C_, a_[1], a_[2]);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a_[0], b0_);
        Goldilocks::mult_avx512(E_, a_[1], b1_);
        Goldilocks::mult_avx512(F_, a_[2], b2_);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(c_[0], C_, G_);
        Goldilocks::sub_avx512(c_[0], c_[0], F_);
        Goldilocks::add_avx512(c_[1], A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx512(c_[2], B_, G_);
    };
    static inline void mul_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks3::Element_avx512 &a_, Goldilocks3::Element_avx512 &b_)
    {

        __m512i aux0_, aux1_, aux2_;
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a_[0], a_[1]);
        Goldilocks::add_avx512(B_, a_[0], a_[2]);
        Goldilocks::add_avx512(C_, a_[1], a_[2]);
        Goldilocks::add_avx512(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx512(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx512(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a_[0], b_[0]);
        Goldilocks::mult_avx512(E_, a_[1], b_[1]);
        Goldilocks::mult_avx512(F_, a_[2], b_[2]);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(c_[0], C_, G_);
        Goldilocks::sub_avx512(c_[0], c_[0], F_);
        Goldilocks::add_avx512(c_[1], A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx512(c_[2], B_, G_);
    };

    static inline void mul_avx512(Goldilocks::Element *c, uint64_t stride_c, Goldilocks3::Element_avx512 &a_, Goldilocks3::Element_avx512 &b_)
    {

        __m512i aux0_, aux1_, aux2_;
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a_[0], a_[1]);
        Goldilocks::add_avx512(B_, a_[0], a_[2]);
        Goldilocks::add_avx512(C_, a_[1], a_[2]);
        Goldilocks::add_avx512(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx512(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx512(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a_[0], b_[0]);
        Goldilocks::mult_avx512(E_, a_[1], b_[1]);
        Goldilocks::mult_avx512(F_, a_[2], b_[2]);
        Goldilocks::sub_avx512(G_, D_, E_);

        __m512i c0_, c1_, c2_;

        Goldilocks::add_avx512(c0_, C_, G_);
        Goldilocks::sub_avx512(c0_, c0_, F_);
        Goldilocks::add_avx512(c1_, A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c1_, c1_, auxr_);
        Goldilocks::sub_avx512(c2_, B_, G_);

        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        Goldilocks::store_avx512(c0, c0_);
        Goldilocks::store_avx512(c1, c1_);
        Goldilocks::store_avx512(c2, c2_);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c[k * stride_c] = c0[k];
            c[k * stride_c + 1] = c1[k];
            c[k * stride_c + 2] = c2[k];
        }
    };
    static inline void mul_avx512(Goldilocks::Element *c, uint64_t stride_c[AVX512_SIZE_], Goldilocks3::Element_avx512 &a_, Goldilocks3::Element_avx512 &b_)
    {

        __m512i aux0_, aux1_, aux2_;
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a_[0], a_[1]);
        Goldilocks::add_avx512(B_, a_[0], a_[2]);
        Goldilocks::add_avx512(C_, a_[1], a_[2]);
        Goldilocks::add_avx512(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx512(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx512(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a_[0], b_[0]);
        Goldilocks::mult_avx512(E_, a_[1], b_[1]);
        Goldilocks::mult_avx512(F_, a_[2], b_[2]);
        Goldilocks::sub_avx512(G_, D_, E_);

        __m512i c0_, c1_, c2_;

        Goldilocks::add_avx512(c0_, C_, G_);
        Goldilocks::sub_avx512(c0_, c0_, F_);
        Goldilocks::add_avx512(c1_, A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c1_, c1_, auxr_);
        Goldilocks::sub_avx512(c2_, B_, G_);

        Goldilocks::Element c0[AVX512_SIZE_], c1[AVX512_SIZE_], c2[AVX512_SIZE_];
        Goldilocks::store_avx512(c0, c0_);
        Goldilocks::store_avx512(c1, c1_);
        Goldilocks::store_avx512(c2, c2_);
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            c[stride_c[k]] = c0[k];
            c[stride_c[k] + 1] = c1[k];
            c[stride_c[k] + 2] = c2[k];
        };
    };

    static inline void mul_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a, uint64_t stride_b)
    {
        Goldilocks::Element a0[AVX512_SIZE_], a1[AVX512_SIZE_], a2[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        __m512i aux0_, aux1_, aux2_;
        __m512i a0_, a1_, a2_;
        __m512i b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
            b0[k] = b[k * stride_b];
            b1[k] = b[k * stride_b + 1];
            b2[k] = b[k * stride_b + 2];
        }
        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        // operations
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a0_, a1_);
        Goldilocks::add_avx512(B_, a0_, a2_);
        Goldilocks::add_avx512(C_, a1_, a2_);
        Goldilocks::add_avx512(aux0_, b0_, b1_);
        Goldilocks::add_avx512(aux1_, b0_, b2_);
        Goldilocks::add_avx512(aux2_, b1_, b2_);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a0_, b0_);
        Goldilocks::mult_avx512(E_, a1_, b1_);
        Goldilocks::mult_avx512(F_, a2_, b2_);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(c_[0], C_, G_);
        Goldilocks::sub_avx512(c_[0], c_[0], F_);
        Goldilocks::add_avx512(c_[1], A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx512(c_[2], B_, G_);
    };
    static inline void mul33c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks::Element *b, uint64_t stride_a)
    {
        Goldilocks::Element aux0[AVX512_SIZE_], aux1[AVX512_SIZE_], aux2[AVX512_SIZE_], aux[3];
        Goldilocks::Element a0[AVX512_SIZE_], a1[AVX512_SIZE_], a2[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        __m512i aux0_, aux1_, aux2_;
        __m512i a0_, a1_, a2_;
        __m512i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
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
        Goldilocks::load_avx512(aux0_, aux0);
        Goldilocks::load_avx512(aux1_, aux1);
        Goldilocks::load_avx512(aux2_, aux2);
        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        // operations
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a0_, a1_);
        Goldilocks::add_avx512(B_, a0_, a2_);
        Goldilocks::add_avx512(C_, a1_, a2_);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a0_, b0_);
        Goldilocks::mult_avx512(E_, a1_, b1_);
        Goldilocks::mult_avx512(F_, a2_, b2_);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(c_[0], C_, G_);
        Goldilocks::sub_avx512(c_[0], c_[0], F_);
        Goldilocks::add_avx512(c_[1], A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx512(c_[2], B_, G_);
    };
    static inline void mul33c_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[AVX512_SIZE_])
    {
        Goldilocks::Element aux0[AVX512_SIZE_], aux1[AVX512_SIZE_], aux2[AVX512_SIZE_], aux[3];
        Goldilocks::Element a0[AVX512_SIZE_], a1[AVX512_SIZE_], a2[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        __m512i aux0_, aux1_, aux2_;
        __m512i a0_, a1_, a2_;
        __m512i b0_, b1_, b2_;

        // redistribute data:
        aux[0] = b[0] + b[1];
        aux[1] = b[0] + b[2];
        aux[2] = b[1] + b[2];
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
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
        Goldilocks::load_avx512(aux0_, aux0);
        Goldilocks::load_avx512(aux1_, aux1);
        Goldilocks::load_avx512(aux2_, aux2);
        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        // operations
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a0_, a1_);
        Goldilocks::add_avx512(B_, a0_, a2_);
        Goldilocks::add_avx512(C_, a1_, a2_);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a0_, b0_);
        Goldilocks::mult_avx512(E_, a1_, b1_);
        Goldilocks::mult_avx512(F_, a2_, b2_);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(c_[0], C_, G_);
        Goldilocks::sub_avx512(c_[0], c_[0], F_);
        Goldilocks::add_avx512(c_[1], A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx512(c_[2], B_, G_);
    };
    static inline void mul_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks3::Element_avx512 &b_, const uint64_t stride_a[AVX512_SIZE_])
    {
        Goldilocks::Element a0[AVX512_SIZE_], a1[AVX512_SIZE_], a2[AVX512_SIZE_];
        __m512i aux0_, aux1_, aux2_;
        __m512i a0_, a1_, a2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
        }
        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);

        // operations
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a0_, a1_);
        Goldilocks::add_avx512(B_, a0_, a2_);
        Goldilocks::add_avx512(C_, a1_, a2_);
        Goldilocks::add_avx512(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx512(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx512(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a0_, b_[0]);
        Goldilocks::mult_avx512(E_, a1_, b_[1]);
        Goldilocks::mult_avx512(F_, a2_, b_[2]);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(c_[0], C_, G_);
        Goldilocks::sub_avx512(c_[0], c_[0], F_);
        Goldilocks::add_avx512(c_[1], A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx512(c_[2], B_, G_);
    };
    static inline void mul_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks3::Element_avx512 &b_, const uint64_t stride_a)
    {
        Goldilocks::Element a0[AVX512_SIZE_], a1[AVX512_SIZE_], a2[AVX512_SIZE_];
        __m512i aux0_, aux1_, aux2_;
        __m512i a0_, a1_, a2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a + 1];
            a2[k] = a[k * stride_a + 2];
        }
        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);

        // operations
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a0_, a1_);
        Goldilocks::add_avx512(B_, a0_, a2_);
        Goldilocks::add_avx512(C_, a1_, a2_);
        Goldilocks::add_avx512(aux0_, b_[0], b_[1]);
        Goldilocks::add_avx512(aux1_, b_[0], b_[2]);
        Goldilocks::add_avx512(aux2_, b_[1], b_[2]);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a0_, b_[0]);
        Goldilocks::mult_avx512(E_, a1_, b_[1]);
        Goldilocks::mult_avx512(F_, a2_, b_[2]);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(c_[0], C_, G_);
        Goldilocks::sub_avx512(c_[0], c_[0], F_);
        Goldilocks::add_avx512(c_[1], A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx512(c_[2], B_, G_);
    };
    static inline void mul_avx512(Goldilocks3::Element_avx512 &c_, Goldilocks::Element *a, Goldilocks::Element *b, const uint64_t stride_a[AVX512_SIZE_], const uint64_t stride_b[AVX512_SIZE_])
    {
        Goldilocks::Element a0[AVX512_SIZE_], a1[AVX512_SIZE_], a2[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        __m512i aux0_, aux1_, aux2_;
        __m512i a0_, a1_, a2_;
        __m512i b0_, b1_, b2_;

        // redistribute data:
        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[stride_a[k]];
            a1[k] = a[stride_a[k] + 1];
            a2[k] = a[stride_a[k] + 2];
            b0[k] = b[stride_b[k]];
            b1[k] = b[stride_b[k] + 1];
            b2[k] = b[stride_b[k] + 2];
        }
        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        // operations
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i auxr_;

        Goldilocks::add_avx512(A_, a0_, a1_);
        Goldilocks::add_avx512(B_, a0_, a2_);
        Goldilocks::add_avx512(C_, a1_, a2_);
        Goldilocks::add_avx512(aux0_, b0_, b1_);
        Goldilocks::add_avx512(aux1_, b0_, b2_);
        Goldilocks::add_avx512(aux2_, b1_, b2_);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a0_, b0_);
        Goldilocks::mult_avx512(E_, a1_, b1_);
        Goldilocks::mult_avx512(F_, a2_, b2_);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(c_[0], C_, G_);
        Goldilocks::sub_avx512(c_[0], c_[0], F_);
        Goldilocks::add_avx512(c_[1], A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx512(c_[2], B_, G_);
    };

    static inline void mul13c_avx512(__m512i &c0_, __m512i &c1_, __m512i &c2_, Goldilocks::Element *a, Element &b, uint64_t stride_a)
    {
        Goldilocks::Element a0[AVX512_SIZE_], a1[AVX512_SIZE_], a2[AVX512_SIZE_];
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            a0[k] = a[k * stride_a];
            a1[k] = a[k * stride_a];
            a2[k] = a[k * stride_a];
            b0[k] = b[0];
            b1[k] = b[1];
            b2[k] = b[2];
        }
        __m512i a0_, a1_, a2_;
        __m512i b0_, b1_, b2_;

        Goldilocks::load_avx512(a0_, a0);
        Goldilocks::load_avx512(a1_, a1);
        Goldilocks::load_avx512(a2_, a2);
        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);
        Goldilocks::mult_avx512(c0_, a0_, b0_);
        Goldilocks::mult_avx512(c1_, a1_, b1_);
        Goldilocks::mult_avx512(c2_, a2_, b2_);
    }
    static inline void mul_avx512(__m512i &c0_, __m512i &c1_, __m512i &c2_, __m512i a0_, __m512i a1_, __m512i a2_, __m512i b0_, __m512i b1_, __m512i b2_, __m512i aux0_, __m512i aux1_, __m512i aux2_)
    {
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i result0_, result1_, auxr_;

        Goldilocks::add_avx512(A_, a0_, a1_);
        Goldilocks::add_avx512(B_, a0_, a2_);
        Goldilocks::add_avx512(C_, a1_, a2_);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a0_, b0_);
        Goldilocks::mult_avx512(E_, a1_, b1_);
        Goldilocks::mult_avx512(F_, a2_, b2_);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(result0_, C_, G_);
        Goldilocks::sub_avx512(c0_, result0_, F_);
        Goldilocks::add_avx512(result1_, A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c1_, result1_, auxr_);
        Goldilocks::sub_avx512(c2_, B_, G_);
    };
    static inline void mul_avx512(__m512i &c0_, __m512i &c1_, __m512i &c2_, __m512i a0_, __m512i a1_, __m512i a2_, Goldilocks::Element *b)
    {
        Goldilocks::Element b0[AVX512_SIZE_], b1[AVX512_SIZE_], b2[AVX512_SIZE_];
        __m512i aux0_, aux1_, aux2_;
        __m512i b0_, b1_, b2_;

        // redistribute data:

        for (uint64_t k = 0; k < AVX512_SIZE_; ++k)
        {
            b0[k] = b[k * FIELD_EXTENSION];
            b1[k] = b[k * FIELD_EXTENSION + 1];
            b2[k] = b[k * FIELD_EXTENSION + 2];
        }

        Goldilocks::load_avx512(b0_, b0);
        Goldilocks::load_avx512(b1_, b1);
        Goldilocks::load_avx512(b2_, b2);

        // operations
        __m512i A_, B_, C_, D_, E_, F_, G_;
        __m512i result0_, result1_, auxr_;

        Goldilocks::add_avx512(A_, a0_, a1_);
        Goldilocks::add_avx512(B_, a0_, a2_);
        Goldilocks::add_avx512(C_, a1_, a2_);
        Goldilocks::add_avx512(aux0_, b0_, b1_);
        Goldilocks::add_avx512(aux1_, b0_, b2_);
        Goldilocks::add_avx512(aux2_, b1_, b2_);
        Goldilocks::mult_avx512(A_, A_, aux0_);
        Goldilocks::mult_avx512(B_, B_, aux1_);
        Goldilocks::mult_avx512(C_, C_, aux2_);
        Goldilocks::mult_avx512(D_, a0_, b0_);
        Goldilocks::mult_avx512(E_, a1_, b1_);
        Goldilocks::mult_avx512(F_, a2_, b2_);
        Goldilocks::sub_avx512(G_, D_, E_);

        Goldilocks::add_avx512(result0_, C_, G_);
        Goldilocks::sub_avx512(c0_, result0_, F_);
        Goldilocks::add_avx512(result1_, A_, C_);
        Goldilocks::add_avx512(auxr_, E_, E_);
        Goldilocks::add_avx512(auxr_, auxr_, D_);
        Goldilocks::sub_avx512(c1_, result1_, auxr_);
        Goldilocks::sub_avx512(c2_, B_, G_);
    };

#endif
    // ======== DIV ========
    static inline void div(Element &result, Element &a, Goldilocks::Element b)
    {
        Goldilocks::Element b_inv = Goldilocks::inv(b);
        mul(result, a, b_inv);
    }

    // ======== MULSCALAR ========
    static inline void mulScalar(Element &result, Element &a, std::string &b)
    {
        result[0] = a[0] * Goldilocks::fromString(b);
        result[1] = a[1] * Goldilocks::fromString(b);
        result[2] = a[2] * Goldilocks::fromString(b);
    }

    // ======== SQUARE ========
    static inline void square(Element &result, Element &a)
    {
        mul(result, a, a);
    }

    // ======== INV ========
    static inline void inv(Element *result, Element *a)
    {
        inv(*result, *a);
    }
    static inline void inv(Element &result, Element &a)
    {
        Goldilocks::Element aa = a[0] * a[0];
        Goldilocks::Element ac = a[0] * a[2];
        Goldilocks::Element ba = a[1] * a[0];
        Goldilocks::Element bb = a[1] * a[1];
        Goldilocks::Element bc = a[1] * a[2];
        Goldilocks::Element cc = a[2] * a[2];

        Goldilocks::Element aaa = aa * a[0];
        Goldilocks::Element aac = aa * a[2];
        Goldilocks::Element abc = ba * a[2];
        Goldilocks::Element abb = ba * a[1];
        Goldilocks::Element acc = ac * a[2];
        Goldilocks::Element bbb = bb * a[1];
        Goldilocks::Element bcc = bc * a[2];
        Goldilocks::Element ccc = cc * a[2];

        Goldilocks::Element t = abc + abc + abc + abb - aaa - aac - aac - acc - bbb + bcc - ccc;

        Goldilocks::Element tinv = Goldilocks::inv(t);
        Goldilocks::Element i1 = (bc + bb - aa - ac - ac - cc) * tinv;

        Goldilocks::Element i2 = (ba - cc) * tinv;
        Goldilocks::Element i3 = (ac + cc - bb) * tinv;

        result[0] = i1;
        result[1] = i2;
        result[2] = i3;
    }

    static void batchInverse(Goldilocks3::Element *res, Goldilocks3::Element *src, uint64_t size)
    {
        Goldilocks3::Element aux[size];
        Goldilocks3::Element tmp[size];
        Goldilocks3::copy(tmp[0], src[0]);

        for (uint64_t i = 1; i < size; i++)
        {
            Goldilocks3::mul(tmp[i], tmp[i - 1], src[i]);
        }

        Goldilocks3::Element z;
        inv(z, tmp[size - 1]);

        for (uint64_t i = size - 1; i > 0; i--)
        {
            Goldilocks3::mul(aux[i], z, tmp[i - 1]);
            Goldilocks3::mul(z, z, src[i]);
        }
        copy(aux[0], z);
        std::memcpy(res, &aux[0], size * sizeof(Goldilocks3::Element));
    }
};

#endif // GOLDILOCKS_F3
