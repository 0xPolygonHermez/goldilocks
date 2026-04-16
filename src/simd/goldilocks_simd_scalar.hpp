#ifndef GOLDILOCKS_SIMD_SCALAR_HPP
#define GOLDILOCKS_SIMD_SCALAR_HPP

#include <cstdint>
#include <cstring>
#include "../goldilocks_base_field.hpp"
#include "goldilocks_simd_backends.hpp"

namespace goldilocks {
namespace simd {

template <> struct GLSimd<Scalar> {
    using Element = Goldilocks::Element;

    struct Vec { uint64_t v; };
    static constexpr size_t Lanes = 1;

    // Constants
    static inline const Vec P      { GOLDILOCKS_PRIME };
    static inline const Vec P_n    { GOLDILOCKS_PRIME_NEG };
    static inline const Vec MSB    { MSB_ };
    static inline const Vec P_s    { (uint64_t)GOLDILOCKS_PRIME ^ (uint64_t)MSB_ };
    static inline const Vec sqmask { 0x1FFFFFFFFULL };

    // Memory
    static inline __attribute__((always_inline)) Vec load(const Element* p)         { Vec r; std::memcpy(&r.v, p, sizeof(uint64_t)); return r; }
    static inline __attribute__((always_inline)) Vec load_aligned(const Element* p) { return load(p); }
    static inline __attribute__((always_inline)) void store(Element* p, Vec a)         { std::memcpy(p, &a.v, sizeof(uint64_t)); }
    static inline __attribute__((always_inline)) void store_aligned(Element* p, Vec a) { store(p, a); }

    // Construction
    static inline __attribute__((always_inline)) Vec set(uint64_t a0) { return Vec{a0}; }
    static inline __attribute__((always_inline)) Vec splat(uint64_t x) { return Vec{x}; }
    static inline __attribute__((always_inline)) Vec copy(Vec a)       { return a; }

    // Arithmetic (defers to scalar Goldilocks)
    static inline __attribute__((always_inline)) Vec add(Vec a, Vec b) {
        Element ea, eb, ec;
        ea.fe = a.v; eb.fe = b.v;
        Goldilocks::add(ec, ea, eb);
        return Vec{ec.fe};
    }
    static inline __attribute__((always_inline)) Vec sub(Vec a, Vec b) {
        Element ea, eb, ec;
        ea.fe = a.v; eb.fe = b.v;
        Goldilocks::sub(ec, ea, eb);
        return Vec{ec.fe};
    }
    static inline __attribute__((always_inline)) Vec add_small(Vec a, Vec b) {
        return add(a, b);  // scalar: no optimization
    }

    static inline __attribute__((always_inline)) Vec mul(Vec a, Vec b) {
        Element ea, eb, ec;
        ea.fe = a.v; eb.fe = b.v;
        Goldilocks::mul(ec, ea, eb);
        return Vec{ec.fe};
    }
    static inline __attribute__((always_inline)) Vec square(Vec a) {
        Element ea, ec;
        ea.fe = a.v;
        Goldilocks::square(ec, ea);
        return Vec{ec.fe};
    }

    // Bit / comparison
    static inline __attribute__((always_inline)) Vec shift(Vec a)           { return Vec{a.v ^ MSB_}; }
    static inline __attribute__((always_inline)) Vec toCanonical_s(Vec a_s) {
        const bool lt = (int64_t)a_s.v < (int64_t)P_s.v;
        return Vec{ a_s.v + (lt ? 0ULL : (uint64_t)GOLDILOCKS_PRIME_NEG) };
    }
    static inline __attribute__((always_inline)) Vec blend(Vec a, Vec b, Vec mask) {
        return Vec{ (a.v & ~mask.v) | (b.v & mask.v) };
    }
    static inline __attribute__((always_inline)) Vec cmpgt_biased(Vec a, Vec b) {
        return Vec{ ((int64_t)a.v > (int64_t)b.v) ? ~0ULL : 0ULL };
    }
    static inline __attribute__((always_inline)) Vec and_(Vec a, Vec b) {
        return Vec{ a.v & b.v };
    }

    // Composite / lane mixing
    static inline __attribute__((always_inline)) Vec permute_lanes(Vec /*a*/, Vec b) { return b; }
    static inline __attribute__((always_inline)) Vec mask_lane0_zero(Vec /*v*/)      { return Vec{0}; }
};

} // namespace simd
} // namespace goldilocks

#endif // GOLDILOCKS_SIMD_SCALAR_HPP
