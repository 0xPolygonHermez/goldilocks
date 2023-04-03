#ifndef GOLDILOCKS_SCALAR
#define GOLDILOCKS_SCALAR
#include "goldilocks_base_field.hpp"

inline void Goldilocks::copy(Element &dst, const Element &src) { dst.fe = src.fe; };

inline void Goldilocks::copy(Element *dst, const Element *src) { dst->fe = src->fe; };

inline Goldilocks::Element Goldilocks::add(const Element &in1, const Element &in2)
{
    Goldilocks::Element result;
    Goldilocks::add(result, in1, in2);
    return result;
}

inline void Goldilocks::add(Element &result, const Element &in1, const Element &in2)
{
    uint64_t in_1 = in1.fe;
    uint64_t in_2 = in2.fe;
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %0\n\t"
            "add   %2, %0\n\t"
            "cmovc %3, %%r10\n\t"
            "add   %%r10, %0\n\t"
            "jnc  1f\n\t"
            "add   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(result.fe)
            : "r"(in_1), "r"(in_2), "m"(CQ), "m"(ZR)
            : "%r10");

#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
}

inline Goldilocks::Element Goldilocks::inc(const Goldilocks::Element &fe)
{
    Goldilocks::Element result;
    if (fe.fe < GOLDILOCKS_PRIME - 2)
    {
        result.fe = fe.fe + 1;
    }
    else if (fe.fe == GOLDILOCKS_PRIME - 1)
    {
        result.fe = 0;
    }
    else
    {
        result = Goldilocks::add(fe, Goldilocks::one());
    }
    return result;
}

inline Goldilocks::Element Goldilocks::sub(const Element &in1, const Element &in2)
{
    Goldilocks::Element result;
    Goldilocks::sub(result, in1, in2);
    return result;
}

inline void Goldilocks::sub(Element &result, const Element &in1, const Element &in2)
{
    uint64_t in_1 = in1.fe;
    uint64_t in_2 = in2.fe;
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %0\n\t"
            "sub   %2, %0\n\t"
            "cmovc %3, %%r10\n\t"
            "sub   %%r10, %0\n\t"
            "jnc  1f\n\t"
            "sub   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(result.fe)
            : "r"(in_1), "r"(in_2), "m"(CQ), "m"(ZR)
            : "%r10");
#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
}

inline Goldilocks::Element Goldilocks::dec(const Goldilocks::Element &fe)
{
    Goldilocks::Element result;
    if (fe.fe > 0)
    {
        result.fe = fe.fe - 1;
    }
    else
    {
        result.fe = GOLDILOCKS_PRIME - 1;
    }
    return result;
}

inline Goldilocks::Element Goldilocks::mul(const Element &in1, const Element &in2)
{
    Goldilocks::Element result;
    Goldilocks::mul(result, in1, in2);
    return result;
}

inline void Goldilocks::mul(Element &result, const Element &in1, const Element &in2)
{
#if USE_MONTGOMERY == 1
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %%rax\n\t"
            "mul   %2\n\t"
            "mov   %%rdx, %%r8\n\t"
            "mov   %%rax, %%r9\n\t"
            "mulq   %3\n\t"
            "mulq   %4\n\t"
            "add    %%r9, %%rax\n\t"
            "adc    %%r8, %%rdx\n\t"
            "cmovc %5, %%r10\n\t"
            "add   %%r10, %%rdx\n\t"
            //"cmovnc %6, %%r10\n\t"
            //"add   %%r10, %0\n\t"
            "jnc  1f\n\t"
            "add   %5, %0\n\t"
            "1: \n\t"
            : "=&d"(result.fe)
            : "r"(in1.fe), "r"(in2.fe), "m"(MM), "m"(Q), "m"(CQ), "m"(ZR)
            : "%rax", "%r8", "%r9", "%r10");

#else
    __asm__("mov   %1, %0\n\t"
            "mul   %2\n\t"
            // "xor   %%rbx, %%rbx\n\t"
            "mov   %%edx, %%ebx\n\t"
            "sub   %4, %%rbx\n\t"
            "rol   $32, %%rdx\n\t"
            //"xor   %%rcx, %%rcx;\n\t"
            "mov   %%edx, %%ecx\n\t"
            "sub   %%rcx, %%rdx\n\t"
            "add   %4, %%rcx\n\t"
            "sub   %%rbx, %%rdx\n\t"
            //"mov   %3,%%r10 \n\t"
            "xor   %%rbx, %%rbx\n\t"
            "add   %%rdx, %0\n\t"
            "cmovc %3, %%rbx\n\t"
            "add   %%rbx, %0\n\t"
            // TODO: migrate to labels
            //"xor   %%rbx, %%rbx\n\t"
            //"sub   %%rcx, %0\n\t"
            //"cmovc %%r10, %%rbx\n\t"
            //"sub   %%rbx, %0\n\t"
            "sub   %%rcx, %0\n\t"
            "jnc  1f\n\t"
            "sub   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(result.fe)
            : "r"(in1.fe), "r"(in2.fe), "m"(CQ), "m"(TWO32)
            : "%rbx", "%rcx", "%rdx");

#endif
#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
}

inline void Goldilocks::mul2(Element &result, const Element &in1, const Element &in2)
{
#if USE_MONTGOMERY == 1
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %%rax\n\t"
            "mul   %2\n\t"
            "mov   %%rdx, %%r8\n\t"
            "mov   %%rax, %%r9\n\t"
            "mulq   %3\n\t"
            "mulq   %4\n\t"
            "add    %%r9, %%rax\n\t"
            "adc    %%r8, %%rdx\n\t"
            "cmovc %5, %%r10\n\t"
            "add   %%r10, %%rdx\n\t"
            : "=&d"(result.fe)
            : "r"(in1.fe), "r"(in2.fe), "m"(MM), "m"(Q), "m"(CQ)
            : "%rax", "%r8", "%r9", "%r10");
#else
    __asm__(
        "mov   %1, %%rax\n\t"
        "mul   %2\n\t"
        "divq   %3\n\t"
        : "=&d"(result.fe)
        : "r"(in1.fe), "r"(in2.fe), "m"(Q)
        : "%rax");
#endif
#if GOLDILOCKS_DEBUG == 1 && USE_MONTGOMERY == 0
    result.fe = result.fe % GOLDILOCKS_PRIME;
#endif
}

inline Goldilocks::Element Goldilocks::square(const Element &in1) { return mul(in1, in1); };

inline void Goldilocks::square(Element &result, const Element &in1) { return mul(result, in1, in1); };

inline Goldilocks::Element Goldilocks::div(const Element &in1, const Element &in2) { return mul(in1, inv(in2)); };

inline void Goldilocks::div(Element &result, const Element &in1, const Element &in2) { mul(result, in1, inv(in2)); };

inline Goldilocks::Element Goldilocks::neg(const Element &in1) { return sub(Goldilocks::zero(), in1); };

inline void Goldilocks::neg(Element &result, const Element &in1) { return sub(result, Goldilocks::zero(), in1); };

inline bool Goldilocks::isZero(const Element &in1) { return equal(in1, Goldilocks::zero()); };

inline bool Goldilocks::isOne(const Element &in1) { return equal(in1, Goldilocks::one()); };

inline bool Goldilocks::isNegone(const Element &in1) { return equal(in1, Goldilocks::negone()); };

inline bool Goldilocks::equal(const Element &in1, const Element &in2) { return Goldilocks::toU64(in1) == Goldilocks::toU64(in2); }

inline Goldilocks::Element Goldilocks::inv(const Element &in1)
{
    Goldilocks::Element result;
    Goldilocks::inv(result, in1);
    return result;
};

inline Goldilocks::Element Goldilocks::mulScalar(const Element &base, const uint64_t &scalar)
{
    Goldilocks::Element result;
    Goldilocks::mulScalar(result, base, scalar);
    return result;
};
inline void Goldilocks::mulScalar(Element &result, const Element &base, const uint64_t &scalar)
{
    Element eScalar = fromU64(scalar);
    mul(result, base, eScalar);
};

inline Goldilocks::Element Goldilocks::exp(Element base, uint64_t exp)
{
    Goldilocks::Element result;
    Goldilocks::exp(result, base, exp);
    return result;
};

inline void Goldilocks::exp(Element &result, Element base, uint64_t exp)
{
    result = Goldilocks::one();

    for (;;)
    {
        if (exp & 1)
            mul(result, result, base);
        exp >>= 1;
        if (!exp)
            break;
        mul(base, base, base);
    }
};
#endif