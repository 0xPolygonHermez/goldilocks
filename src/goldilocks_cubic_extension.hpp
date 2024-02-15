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

    static inline void load_avx(Element_avx &a_, const Goldilocks::Element *a, uint64_t stride_a)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        a0[0] = a[0];
        a1[0] = a[1];
        a2[0] = a[2];
        a0[1] = a[stride_a];
        a1[1] = a[stride_a + 1];
        a2[1] = a[stride_a + 2];
        int ind = stride_a << 1;
        a0[2] = a[ind];
        a1[2] = a[ind + 1];
        a2[2] = a[ind + 2];
        ind = ind + stride_a;
        a0[3] = a[ind];
        a1[3] = a[ind + 1];
        a2[3] = a[ind + 2];
        a_[0] = _mm256_loadu_si256((__m256i *)(a0));
        a_[1] = _mm256_loadu_si256((__m256i *)(a1));
        a_[2] = _mm256_loadu_si256((__m256i *)(a2));
    };
    static inline void load_avx(Element_avx &a_, const Goldilocks::Element *a, const uint64_t offsets_a[4])
    {
        Goldilocks::Element a0[4], a1[4], a2[4];

        a0[0] = a[offsets_a[0]];
        a1[0] = a[offsets_a[0] + 1];
        a2[0] = a[offsets_a[0] + 2];
        a0[1] = a[offsets_a[1]];
        a1[1] = a[offsets_a[1] + 1];
        a2[1] = a[offsets_a[1] + 2];
        a0[2] = a[offsets_a[2]];
        a1[2] = a[offsets_a[2] + 1];
        a2[2] = a[offsets_a[2] + 2];
        a0[3] = a[offsets_a[3]];
        a1[3] = a[offsets_a[3] + 1];
        a2[3] = a[offsets_a[3] + 2];
        a_[0] = _mm256_loadu_si256((__m256i *)(a0));
        a_[1] = _mm256_loadu_si256((__m256i *)(a1));
        a_[2] = _mm256_loadu_si256((__m256i *)(a2));
    };

    static inline void store_avx(Goldilocks::Element *a, uint64_t stride_a, const Element_avx a_)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        _mm256_storeu_si256((__m256i *)a0, a_[0]);
        _mm256_storeu_si256((__m256i *)a1, a_[1]);
        _mm256_storeu_si256((__m256i *)a2, a_[2]);
        a[0] = a0[0];
        a[1] = a1[0];
        a[2] = a2[0];
        a[stride_a] = a0[1];
        a[stride_a + 1] = a1[1];
        a[stride_a + 2] = a2[1];
        int ind = stride_a << 1;
        a[ind] = a0[2];
        a[ind + 1] = a1[2];
        a[ind + 2] = a2[2];
        ind = ind + stride_a;
        a[ind] = a0[3];
        a[ind + 1] = a1[3];
        a[ind + 2] = a2[3];
    };
    static inline void store_avx(Goldilocks::Element *a, const uint64_t offsets_a[4], const Element_avx &a_)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        _mm256_storeu_si256((__m256i *)a0, a_[0]);
        _mm256_storeu_si256((__m256i *)a1, a_[1]);
        _mm256_storeu_si256((__m256i *)a2, a_[2]);
        a[offsets_a[0]] = a0[0];
        a[offsets_a[0] + 1] = a1[0];
        a[offsets_a[0] + 2] = a2[0];
        a[offsets_a[1]] = a0[1];
        a[offsets_a[1] + 1] = a1[1];
        a[offsets_a[1] + 2] = a2[1];
        a[offsets_a[2]] = a0[2];
        a[offsets_a[2] + 1] = a1[2];
        a[offsets_a[2] + 2] = a2[2];
        a[offsets_a[3]] = a0[3];
        a[offsets_a[3] + 1] = a1[3];
        a[offsets_a[3] + 2] = a2[3];
    };

    static inline void store_avx(Goldilocks::Element *a, const uint64_t offset_a, const __m256i *a_, uint64_t stride_a)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        _mm256_storeu_si256((__m256i *)a0, a_[0]);
        _mm256_storeu_si256((__m256i *)a1, a_[stride_a]);
        _mm256_storeu_si256((__m256i *)a2, a_[2*stride_a]);
        a[0] = a0[0];
        a[1] = a1[0];
        a[2] = a2[0];
        a[offset_a] = a0[1];
        a[offset_a + 1] = a1[1];
        a[offset_a + 2] = a2[1];
        int ind = offset_a << 1;
        a[ind] = a0[2];
        a[ind + 1] = a1[2];
        a[ind + 2] = a2[2];
        ind = ind + offset_a;
        a[ind] = a0[3];
        a[ind + 1] = a1[3];
        a[ind + 2] = a2[3];
    };

    static inline void store_avx(Goldilocks::Element *a, const uint64_t offsets_a[4], const __m256i *a_, uint64_t stride_a)
    {
        Goldilocks::Element a0[4], a1[4], a2[4];
        _mm256_storeu_si256((__m256i *)a0, a_[0]);
        _mm256_storeu_si256((__m256i *)a1, a_[stride_a]);
        _mm256_storeu_si256((__m256i *)a2, a_[2 * stride_a]);
        a[offsets_a[0]] = a0[0];
        a[offsets_a[0] + 1] = a1[0];
        a[offsets_a[0] + 2] = a2[0];
        a[offsets_a[1]] = a0[1];
        a[offsets_a[1] + 1] = a1[1];
        a[offsets_a[1] + 2] = a2[1];
        a[offsets_a[2]] = a0[2];
        a[offsets_a[2] + 1] = a1[2];
        a[offsets_a[2] + 2] = a2[2];
        a[offsets_a[3]] = a0[3];
        a[offsets_a[3] + 1] = a1[3];
        a[offsets_a[3] + 2] = a2[3];
    };

    static inline void mul_avx(Element_avx &c_, const Element_avx &a_, const Element_avx &challenge_, const Element_avx &challenge_ops_)
    {
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a_[0], a_[1]);
        Goldilocks::add_avx(B_, a_[0], a_[2]);
        Goldilocks::add_avx(C_, a_[1], a_[2]);
        Goldilocks::mult_avx(A_, A_, challenge_ops_[0]);
        Goldilocks::mult_avx(B_, B_, challenge_ops_[1]);
        Goldilocks::mult_avx(C_, C_, challenge_ops_[2]);
        Goldilocks::mult_avx(D_, a_[0], challenge_[0]);
        Goldilocks::mult_avx(E_, a_[1], challenge_[1]);
        Goldilocks::mult_avx(F_, a_[2], challenge_[2]);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[1], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[1], c_[1], auxr_);
        Goldilocks::sub_avx(c_[2], B_, G_);
    };

    static inline void mul_avx(__m256i *c_, uint64_t stride_c, const __m256i *a_, uint64_t stride_a, const Element_avx &challenge_, const Element_avx &challenge_ops_)
    {
        __m256i A_, B_, C_, D_, E_, F_, G_;
        __m256i auxr_;

        Goldilocks::add_avx(A_, a_[0], a_[stride_a]);
        Goldilocks::add_avx(B_, a_[0], a_[2 * stride_a]);
        Goldilocks::add_avx(C_, a_[stride_a], a_[2 * stride_a]);
        Goldilocks::mult_avx(A_, A_, challenge_ops_[0]);
        Goldilocks::mult_avx(B_, B_, challenge_ops_[1]);
        Goldilocks::mult_avx(C_, C_, challenge_ops_[2]);
        Goldilocks::mult_avx(D_, a_[0], challenge_[0]);
        Goldilocks::mult_avx(E_, a_[stride_a], challenge_[1]);
        Goldilocks::mult_avx(F_, a_[2 * stride_a], challenge_[2]);
        Goldilocks::sub_avx(G_, D_, E_);

        Goldilocks::add_avx(c_[0], C_, G_);
        Goldilocks::sub_avx(c_[0], c_[0], F_);
        Goldilocks::add_avx(c_[stride_c], A_, C_);
        Goldilocks::add_avx(auxr_, E_, E_);
        Goldilocks::add_avx(auxr_, auxr_, D_);
        Goldilocks::sub_avx(c_[stride_c], c_[stride_c], auxr_);
        Goldilocks::sub_avx(c_[2 * stride_c], B_, G_);
    };

    static inline void copy(Element &dst, const Element &src)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            Goldilocks::copy(dst[i], src[i]);
        }
    };
    static inline void copy(Element *dst, const Element *src)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            Goldilocks::copy((*dst)[i], (*src)[i]);
        }
    };

    static inline void copy_avx(Element_avx c_, Element_avx a_)
    {
        c_[0] = a_[0];
        c_[1] = a_[1];
        c_[2] = a_[2];
    }

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
            result += toString(in1[i], 10);
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

    static inline void add_avx(Element_avx c_, const Element_avx a_, const Element_avx b_)
    {
        Goldilocks::add_avx(c_[0], a_[0], b_[0]);
        Goldilocks::add_avx(c_[1], a_[1], b_[1]);
        Goldilocks::add_avx(c_[2], a_[2], b_[2]);
    }

#ifdef __AVX512__

    static inline void add_avx512(Element_avx512 c_, Element_avx512 a_, Element_avx512 b_)
    {
        Goldilocks::add_avx512(c_[0], a_[0], b_[0]);
        Goldilocks::add_avx512(c_[1], a_[1], b_[1]);
        Goldilocks::add_avx512(c_[2], a_[2], b_[2]);
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

    static inline void sub_avx(Element_avx &c_, const Element_avx a_, const Element_avx b_)
    {
        Goldilocks::sub_avx(c_[0], a_[0], b_[0]);
        Goldilocks::sub_avx(c_[1], a_[1], b_[1]);
        Goldilocks::sub_avx(c_[2], a_[2], b_[2]);
    }

#ifdef __AVX512__

    static inline void sub_avx512(Element_avx512 &c_, Element_avx512 a_, Element_avx512 b_)
    {
        Goldilocks::sub_avx512(c_[0], a_[0], b_[0]);
        Goldilocks::sub_avx512(c_[1], a_[1], b_[1]);
        Goldilocks::sub_avx512(c_[2], a_[2], b_[2]);
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

    static inline void mul_avx(Element_avx &c_, const Element_avx &a_, const Element_avx &b_)
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

#ifdef __AVX512__

    static inline void mul_avx512(Element_avx512 &c_, Element_avx512 &a_, Element_avx512 &b_)
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

    static void batchInverse(Element *res, Element *src, uint64_t size)
    {
        Element aux[size];
        Element tmp[size];
        copy(tmp[0], src[0]);

        for (uint64_t i = 1; i < size; i++)
        {
            mul(tmp[i], tmp[i - 1], src[i]);
        }

        Element z;
        inv(z, tmp[size - 1]);

        for (uint64_t i = size - 1; i > 0; i--)
        {
            mul(aux[i], z, tmp[i - 1]);
            mul(z, z, src[i]);
        }
        copy(aux[0], z);
        std::memcpy(res, &aux[0], size * sizeof(Element));
    }

    // ======== OPERATIONS ========

    static inline void op_avx(uint64_t op, __m256i *c_, uint64_t stride_c, const __m256i *a_, uint64_t stride_a, const __m256i *b_, uint64_t stride_b)
    {
        switch (op)
        {
        case 0:
            Goldilocks::add_avx(c_[0], a_[0], b_[0]);
            Goldilocks::add_avx(c_[stride_c], a_[stride_a], b_[stride_b]);
            Goldilocks::add_avx(c_[2 * stride_c], a_[2 * stride_a], b_[2 * stride_b]);
            break;
        case 1:
            Goldilocks::sub_avx(c_[0], a_[0], b_[0]);
            Goldilocks::sub_avx(c_[stride_c], a_[stride_a], b_[stride_b]);
            Goldilocks::sub_avx(c_[2 * stride_c], a_[2 * stride_a], b_[2 * stride_b]);
            break;
        case 2:
            __m256i aux0_, aux1_, aux2_;
            __m256i A_, B_, C_, D_, E_, F_, G_;
            __m256i auxr_;

            Goldilocks::add_avx(A_, a_[0], a_[stride_a]);
            Goldilocks::add_avx(B_, a_[0], a_[2 * stride_a]);
            Goldilocks::add_avx(C_, a_[stride_a], a_[2 * stride_a]);
            Goldilocks::add_avx(aux0_, b_[0], b_[stride_b]);
            Goldilocks::add_avx(aux1_, b_[0], b_[2 * stride_b]);
            Goldilocks::add_avx(aux2_, b_[stride_b], b_[2 * stride_b]);
            Goldilocks::mult_avx(A_, A_, aux0_);
            Goldilocks::mult_avx(B_, B_, aux1_);
            Goldilocks::mult_avx(C_, C_, aux2_);
            Goldilocks::mult_avx(D_, a_[0], b_[0]);
            Goldilocks::mult_avx(E_, a_[stride_a], b_[stride_b]);
            Goldilocks::mult_avx(F_, a_[2 * stride_a], b_[2 * stride_b]);
            Goldilocks::sub_avx(G_, D_, E_);

            Goldilocks::add_avx(c_[0], C_, G_);
            Goldilocks::sub_avx(c_[0], c_[0], F_);
            Goldilocks::add_avx(c_[stride_c], A_, C_);
            Goldilocks::add_avx(auxr_, E_, E_);
            Goldilocks::add_avx(auxr_, auxr_, D_);
            Goldilocks::sub_avx(c_[stride_c], c_[stride_c], auxr_);
            Goldilocks::sub_avx(c_[2 * stride_c], B_, G_);
            break;
        case 3:
            Goldilocks::sub_avx(c_[0], b_[0], a_[0]);
            Goldilocks::sub_avx(c_[stride_c], b_[stride_b], a_[stride_a]);
            Goldilocks::sub_avx(c_[2 * stride_c], b_[2 * stride_b], a_[2 * stride_a]);
            break;
        default:
            assert(0);
            break;
        }
    }

    static inline void op_avx(uint64_t op, Element_avx &c_, const Element_avx &a_, const Element_avx &b_)
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
            mul_avx(c_, a_, b_);
            break;
        case 3:
            sub_avx(c_, b_, a_);
            break;
        default:
            assert(0);
            break;
        }
    }

    static inline void op_31_avx(uint64_t op, Element_avx &c_, const Element_avx &a_, const __m256i &b_)
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
            Goldilocks::mult_avx(c_[0], b_, a_[0]);
            Goldilocks::mult_avx(c_[1], b_, a_[1]);
            Goldilocks::mult_avx(c_[2], b_, a_[2]);
            break;
        case 3:
            Goldilocks::sub_avx(c_[0], b_, a_[0]);
            Goldilocks::sub_avx(c_[1], P, a_[1]);
            Goldilocks::sub_avx(c_[2], P, a_[2]);
            break;
        default:
            assert(0);
            break;
        }
    }

    static inline void op_31_avx(uint64_t op, __m256i *c_, uint64_t stride_c, const __m256i *a_, uint64_t stride_a, const __m256i &b_)
    {
        switch (op)
        {
        case 0:
            Goldilocks::add_avx(c_[0], a_[0], b_);
            c_[stride_c] = a_[stride_a];
            c_[2 * stride_c] = a_[2 * stride_a];
            break;
        case 1:
            Goldilocks::sub_avx(c_[0], a_[0], b_);
            c_[stride_c] = a_[stride_a];
            c_[2 * stride_c] = a_[2 * stride_a];
            break;
        case 2:
            Goldilocks::mult_avx(c_[0], b_, a_[0]);
            Goldilocks::mult_avx(c_[stride_c], b_, a_[stride_a]);
            Goldilocks::mult_avx(c_[2 * stride_c], b_, a_[2 * stride_a]);
            break;
        case 3:
            Goldilocks::sub_avx(c_[0], b_, a_[0]);
            Goldilocks::sub_avx(c_[stride_c], P, a_[stride_a]);
            Goldilocks::sub_avx(c_[2 * stride_c], P, a_[2 * stride_a]);
            break;
        default:
            assert(0);
            break;
        }
    }
};

#endif // GOLDILOCKS_F3
