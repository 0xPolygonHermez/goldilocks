#ifndef GOLDILOCKS
#define GOLDILOCKS

#include <stdint.h> // uint64_t

#define USE_MONTGOMERY 0

class Goldilocks
{
public:
    typedef struct
    {
        uint64_t fe;
    } Element;

private:
    static const Element P;
    static const Element Q;
    static const Element MM;
    static const Element CQ;
    static const Element R2;

    static const Element ZERO;
    static const Element ONE;
    static const Element NEGONE;

public:
    static inline const Element &zero() { return ZERO; };
    static inline void zero(Element &result) { result.fe = ZERO.fe; };

    static inline const Element &one() { return ONE; };
    static inline void one(Element &result) { result.fe = ONE.fe; };

    static inline const Element &negone() { return NEGONE; };
    static inline void negone(Element &result) { result.fe = NEGONE.fe; };

    static Element fromU64(uint64_t in1);
    static uint64_t toU64(const Element &in1); // Normalizado

    static Element fromS32(int32_t in1);
    static int32_t toS32(const Element &in1); // Normalizado

    static std::string toString(const Element &in1, int radix);
    static Element fromString(const std::string &in1, int radix);
    static Element fromString(Element &result, const std::string &in1, int radix);

    static Element add(const Element &in1, const Element &in2);
    static void add(Element &result, const Element &in1, const Element &in2);

    static Element sub(const Element &in1, const Element &in2);
    static void sub(Element &result, const Element &in1, const Element &in2);

    static Element mul(const Element &in1, const Element &in2);
    static void mul(Element &result, const Element &in1, const Element &in2);

    static Element div(const Element &in1, const Element &in2) { return mul(in1, inv(in2)); };
    static void div(Element &result, const Element &in1, const Element &in2) { mul(result, in1, inv(in2)); };

    static Element neg(const Element &in1);
    static void neg(Element &result, const Element &in1);

    static Element inv(const Element &in1);
    static void inv(Element &result, const Element &in1);

    static Element square(const Element &in1) { return mul(in1, in1); };
    static void square(Element &result, const Element &in1) { return mul(result, in1, in1); };

    static Element mulScalar(const Element &base, const uint64_t &scalar);
    static void mulScalar(Element &result, const Element &base, const uint64_t &scalar);

    static Element exp(const Element &base, const uint64_t &exp);
    static void exp(Element &result, const Element &base, const uint64_t &exp);

    static bool isZero(const Element &in1);
    static bool equal(const Element &in1, const Element &in2);
};

/*
    static Element add_gl(Element in1, Element in2);

    static Element gl_add_2(Element in1, Element in2);
    static Element gl_add(Element in1, Element in2);
    static void gl_add(Element &res, Element &in1, Element &in2);

    static Element gl_sub(Element in1, Element in2);

    static Element gl_mul(Element a, Element b);

    static Element gl_mmul2(Element in1, Element in2);
    static Element gl_mmul(Element in1, Element in2);
inline Goldilocks::Element Goldilocks::add_gl(Element in1, Element in2)
{
    Element res = 0;
    if (__builtin_add_overflow(in1, in2, &res))
    {
        res += CQ;
    }
    return res;
}

inline Goldilocks::Element Goldilocks::gl_mul(Goldilocks::Element a, Goldilocks::Element b)
{
    Element r;
    Element q;
    __asm__(
        "mulq   %3\n\t"
        "divq   %4\n\t"
        : "=a"(r), "=&d"(q)
        : "a"(a), "rm"(b), "rm"(P));
    return q;
}

inline Goldilocks::Element Goldilocks::gl_tom(Goldilocks::Element in1)
{
    Element res;
    __asm__(
        "xor   %%r10, %%r10\n\t"
        "mov   %1, %%rax\n\t"
        "mulq   %5\n\t"
        "mov   %%rdx, %%r8\n\t"
        "mov   %%rax, %%r9\n\t"
        "mulq   %2\n\t"
        "mulq   %3\n\t"
        "add    %%r9, %%rax\n\t"
        "adc    %%r8, %%rdx\n\t"
        "cmovc %4, %%r10\n\t"
        "add   %%r10, %%rdx\n\t"
        : "=&d"(res)
        : "r"(in1), "m"(MM), "m"(Q), "m"(CQ), "m"(R2)
        : "%rax", "%r8", "%r9", "%r10");
    return res;
}

inline Goldilocks::Element Goldilocks::gl_fromm(Goldilocks::Element in1)
{
    Element res;

    __asm__(
        "xor   %%r10, %%r10\n\t"
        "mov   %1, %%rax\n\t"
        "mov   %%rax, %%r9\n\t"
        "mulq   %2\n\t"
        "mulq   %3\n\t"
        "add    %%r9, %%rax\n\t"
        "adc    %%r10, %%rdx\n\t"
        "cmovc %4, %%r10\n\t"
        "add   %%r10, %%rdx\n\t"
        : "=&d"(res)
        : "r"(in1), "m"(MM), "m"(Q), "m"(CQ)
        : "%rax", "%r8", "%r9", "%r10");
    return res;
}

inline Goldilocks::Element Goldilocks::gl_add_2(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
    __asm__("mov   %1, %0\n\t"
            "add   %2, %0\n\t"
            "jnc  1f\n\t"
            "add   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(res)
            : "r"(in1), "r"(in2), "m"(CQ)
            :);
    return res;
};

inline Goldilocks::Element Goldilocks::gl_add(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
    __asm__("xor   %%r10, %%r10\n\t"
            "mov   %1, %0\n\t"
            "add   %2, %0\n\t"
            "cmovc %3, %%r10\n\t"
            "add   %%r10, %0\n\t"
            : "=&a"(res)
            : "r"(in1), "r"(in2), "m"(CQ)
            : "%r10");
    return res;
};

inline void Goldilocks::gl_add(Element &res, Element &in1, Element &in2)
{
    res = gl_add(in1, in2);
}

inline Goldilocks::Element Goldilocks::gl_sub(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
    __asm__("mov   %1, %0\n\t"
            "sub   %2, %0\n\t"
            "jnc  1f\n\t"
            "add   %3, %0\n\t"
            "1: \n\t"
            : "=&a"(res)
            : "r"(in1), "r"(in2), "m"(Q)
            :);
    return res;
}

inline Goldilocks::Element Goldilocks::gl_mmul2(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
    __asm__("mov   %1, %%rax\n\t"
            "mul   %2\n\t"
            "mov   %%rdx, %%r8\n\t"
            "mov   %%rax, %%r9\n\t"
            "mulq   %3\n\t"
            "mulq   %4\n\t"
            "add    %%r9, %%rax\n\t"
            "adc    %%r8, %%rdx\n\t"
            "jnc  1f\n\t"
            "add   %5, %%rdx\n\t"
            "1:"
            : "=&d"(res)
            : "r"(in1), "r"(in2), "m"(MM), "m"(Q), "m"(CQ)
            : "%rax", "%r8", "%r9");
    return res;
}

inline Goldilocks::Element Goldilocks::gl_mmul(Goldilocks::Element in1, Goldilocks::Element in2)
{
    Element res;
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
            : "=&d"(res)
            : "r"(in1), "r"(in2), "m"(MM), "m"(Q), "m"(CQ)
            : "%rax", "%r8", "%r9", "%r10");
    return res;
}

#endif // GOLDILOCKS
