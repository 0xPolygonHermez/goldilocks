#ifndef GOLDILOCKS_SIMD_HPP
#define GOLDILOCKS_SIMD_HPP

// ============================================================================
// Goldilocks SIMD traits layer (Phase 1 — Architect B).
//
// One primary template, one specialization per backend. Consumers write
// generic code over GLSimd<B>; backend-specific intrinsics stay in the
// per-backend headers.
//
// Vocabulary (every specialization MUST provide):
//
//   Types / constants:
//     using Vec; using Element = Goldilocks::Element;
//     static constexpr size_t Lanes;
//     static const Vec P, P_n, P_s, MSB, sqmask;
//
//   Memory:
//     static Vec  load        (const Element* p);
//     static Vec  load_aligned(const Element* p);
//     static void store       (Element* p, Vec v);
//     static void store_aligned(Element* p, Vec v);
//
//   Construction:
//     static Vec  set   (uint64_t a0 [, a1, a2, a3]);  // lane 0 = a0 (low)
//     static Vec  splat (uint64_t x);
//     static Vec  copy  (Vec v);
//
//   Arithmetic (Part 1 — except mul/square, which are Part 2):
//     static Vec  add       (Vec a, Vec b);
//     static Vec  sub       (Vec a, Vec b);
//     static Vec  add_small (Vec a, Vec b);   // assumes b in [0, 2^32)
//     static Vec  mul       (Vec a, Vec b);   // *** Part 2 ***
//     static Vec  square    (Vec a);          // *** Part 2 ***
//
//   Bit / comparison:
//     static Vec  shift        (Vec a);              // XOR with MSB broadcast
//     static Vec  toCanonical_s(Vec a_s);            // shifted-canonical form
//     static Vec  blend        (Vec a, Vec b, mask);
//     static Vec  cmpgt_biased (Vec a, Vec b);       // signed cmp on a_s,b_s
//     static Vec  and_         (Vec a, Vec b);
//
//   Lane mixing / matrix composites (composite: no single backend opcode):
//     static Vec  permute_lanes  (Vec a, Vec b);
//     static Vec  mask_lane0_zero(Vec v);            // lane 0 -> 0
//     (dot / mmult_4x12 are Poseidon-shaped helpers; live in
//      poseidon_goldilocks_simd.hpp as composites over the above.)
//
// LANE-ORDER CONVENTION:
//   set(a0, a1, ..., aN-1) puts a0 in lane 0 (low). NEON: vld1q_u64 order.
//   This differs from x86 _mm256_set_epi64x (high-to-low). AVX2 specialization
//   (Phase 6) must account for this in its set() wrapper.
//
// CORRECTNESS CONTRACT (binding on Part 2 and all backends):
//   For canonical inputs a, b in [0, P):
//     mul(a, b)  == scalar Goldilocks::mul(a, b)   bit-for-bit, lane-wise.
//     square(a)  == scalar Goldilocks::square(a)   bit-for-bit, lane-wise.
//   For non-canonical inputs in [0, 2^64), behavior must match existing AVX2.
// ============================================================================

#include "../goldilocks_base_field.hpp"
#include "goldilocks_simd_backends.hpp"

namespace goldilocks {
namespace simd {

// Primary template — intentionally undefined. Specializations follow.
template <class Backend> struct GLSimd;

} // namespace simd
} // namespace goldilocks

#include "goldilocks_simd_scalar.hpp"
#ifdef GOLDILOCKS_HAS_NEON
#  include "goldilocks_simd_neon.hpp"
#endif

namespace goldilocks {
namespace simd {

// DefaultBackend mirrors the existing dispatch order. Avx2 / Avx512 stay
// unspecialized in Phase 1 — using them as default would be a compile error,
// which is intentional (x86 code continues using the legacy _avx.hpp path).
#if defined(GOLDILOCKS_HAS_AVX512)
using DefaultBackend = Avx512;
#elif defined(GOLDILOCKS_HAS_AVX2)
using DefaultBackend = Avx2;
#elif defined(GOLDILOCKS_HAS_NEON)
using DefaultBackend = Neon;
#else
using DefaultBackend = Scalar;
#endif

} // namespace simd
} // namespace goldilocks

// Public Goldilocks::*_neon wrappers (ARM64 only). Live here so they see the
// full GLSimd<Neon> specialization; placing them in goldilocks_base_field.hpp
// causes an ordering issue because the umbrella includes that header first.
#ifdef GOLDILOCKS_HAS_NEON
inline void Goldilocks::load_neon(uint64x2_t &a, const Goldilocks::Element *p) {
    a = goldilocks::simd::GLSimd<goldilocks::simd::Neon>::load(p);
}
inline void Goldilocks::store_neon(Goldilocks::Element *p, const uint64x2_t &a) {
    goldilocks::simd::GLSimd<goldilocks::simd::Neon>::store(p, a);
}
inline void Goldilocks::add_neon(uint64x2_t &c, const uint64x2_t &a, const uint64x2_t &b) {
    c = goldilocks::simd::GLSimd<goldilocks::simd::Neon>::add(a, b);
}
inline void Goldilocks::sub_neon(uint64x2_t &c, const uint64x2_t &a, const uint64x2_t &b) {
    c = goldilocks::simd::GLSimd<goldilocks::simd::Neon>::sub(a, b);
}
inline void Goldilocks::mult_neon(uint64x2_t &c, const uint64x2_t &a, const uint64x2_t &b) {
    c = goldilocks::simd::GLSimd<goldilocks::simd::Neon>::mul(a, b);
}
inline void Goldilocks::square_neon(uint64x2_t &c, const uint64x2_t &a) {
    c = goldilocks::simd::GLSimd<goldilocks::simd::Neon>::square(a);
}
#endif

#endif // GOLDILOCKS_SIMD_HPP
