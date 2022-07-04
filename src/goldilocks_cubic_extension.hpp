#ifndef GOLDILOCKS_F3
#define GOLDILOCKS_F3

#include <stdint.h> // for uint64_t
#include "goldilocks_base_field.hpp"
#include <cassert>

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

private:
    static const Element ZERO;
    static const Element ONE;
    static const Element NEGONE;

public:
    uint64_t m = 1 * FIELD_EXTENSION;
    uint64_t p = GOLDILOCKS_PRIME;
    uint64_t n64 = 1;
    uint64_t n32 = n64 * 2;
    uint64_t n8 = n32 * 8;

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
        result[1] = Goldilocks::one();
        result[2] = Goldilocks::one();
    };

    // static inline const Element &negone() { return NEGONE; };
    // static inline void negone(Element &result) { result.fe = NEGONE.fe; };

    static inline void fromU64(Element &result, uint64_t in1[FIELD_EXTENSION])
    {
        for (uint i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::fromU64(in1[i]);
        }
    }

    static inline void fromS32(Element &result, int32_t in1[FIELD_EXTENSION])
    {
        for (uint i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::fromS32(in1[i]);
        }
    }

    static inline void toU64(uint64_t (&result)[FIELD_EXTENSION], const Element &in1)
    {
        for (uint i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::toU64(in1[i]);
        }
    }

    static inline void toS32(int32_t (&result)[FIELD_EXTENSION], const Element &in1)
    {
        for (uint i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::toS32(in1[i]);
        }
    }

    static inline std::string toString(const Element &in1, int radix = 10)
    {
        std::string res;
        toString(res, in1, radix);
        return res;
    }
    static inline void toString(std::string &result, const Element &in1, int radix = 10)
    {
        result = "{ ";
        for (uint i = 0; i < FIELD_EXTENSION; i++)
        {
            result += Goldilocks::toString(in1[i]);
            (i != FIELD_EXTENSION - 1) ? result += " , " : "";
        }
        result = " }";
    }

    static inline void fromString(Element &result, const std::string (&in1)[FIELD_EXTENSION], int radix = 10)
    {
        for (uint i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = Goldilocks::fromString(in1[i]);
        }
    }

    // ======== ADD ========
    static inline void add(Element &result, Element &a, uint64_t &b)
    {
        result[0] = a[0] + Goldilocks::fromU64(b);
        result[1] = a[1];
        result[2] = a[2];
    }
    static inline void add(Element &result, Element &a, Goldilocks::Element &b)
    {
        result[0] = a[0] + b;
        result[1] = a[1];
        result[2] = a[2];
    }

    static inline void add(Element &result, Element &a, Element &b)
    {
        for (uint i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = a[i] + b[i];
        }
    }

    // ======== SUB ========
    static inline void sub(uint64_t &result, uint64_t &a, uint64_t &b)
    {
        result = a - b;
    }

    static inline void sub(Element &result, Element &a, uint64_t &b)
    {
        result[0] = a[0] - Goldilocks::fromU64(b);
        result[1] = a[1];
        result[2] = a[2];
    }

    static inline void sub(Element &result, Element &a, Element &b)
    {
        for (uint i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = a[i] - b[i];
        }
    }

    // ======== NEG ========
    static inline void neg(Element &result, Element &a)
    {
        sub(result, (Element &)zero(), a);
    }

    // ======== MUL ========
    static inline void mul(Element &result, Element &a, Element &b)
    {
        Goldilocks::Element A = (a[0] + a[1]) * (b[0] + b[1]);
        Goldilocks::Element B = (a[0] + a[2]) * (b[0] + b[2]);
        Goldilocks::Element C = (a[1] + a[2]) * (b[1] + b[2]);
        Goldilocks::Element D = a[0] + b[0];
        Goldilocks::Element E = a[1] + b[1];
        Goldilocks::Element F = a[2] + b[2];
        Goldilocks::Element G = D - E;

        result[0] = (C + G) - F;
        result[1] = ((((A + C) - E) - E) - D);
        result[2] = B - G;
    };

    static inline void mul(Element &result, Element &a, uint64_t &b)
    {
        result[0] = a[0] * Goldilocks::fromU64(b);
        result[1] = a[1] * Goldilocks::fromU64(b);
        result[2] = a[2] * Goldilocks::fromU64(b);
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
    inline void inv(Element &result, Element &a)
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
};

#endif // GOLDILOCKS_F3