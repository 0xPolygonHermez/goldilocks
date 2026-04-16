#ifndef GOLDILOCKS_SIMD_NEON_HPP
#define GOLDILOCKS_SIMD_NEON_HPP

#include "../platform.hpp"
#ifndef GOLDILOCKS_HAS_NEON
#  error "goldilocks_simd_neon.hpp included without GOLDILOCKS_HAS_NEON"
#endif

#include <arm_neon.h>
#include <cstdint>
#include "../goldilocks_base_field.hpp"
#include "goldilocks_simd_backends.hpp"

namespace goldilocks {
namespace simd {

template <> struct GLSimd<Neon> {
    using Element = Goldilocks::Element;
    using Vec     = uint64x2_t;
    static constexpr size_t Lanes = 2;

    // Constants (broadcast across both lanes)
    static inline const Vec P      = vdupq_n_u64(GOLDILOCKS_PRIME);
    static inline const Vec P_n    = vdupq_n_u64(GOLDILOCKS_PRIME_NEG);
    static inline const Vec MSB    = vdupq_n_u64(MSB_);
    static inline const Vec P_s    = veorq_u64(vdupq_n_u64(GOLDILOCKS_PRIME), vdupq_n_u64(MSB_));
    static inline const Vec sqmask = vdupq_n_u64(0x1FFFFFFFFULL);

    // Memory
    static inline __attribute__((always_inline)) Vec load(const Element* p) {
        return vld1q_u64(reinterpret_cast<const uint64_t*>(p));
    }
    static inline __attribute__((always_inline)) Vec load_aligned(const Element* p) {
        return vld1q_u64(reinterpret_cast<const uint64_t*>(__builtin_assume_aligned(p, 16)));
    }
    static inline __attribute__((always_inline)) void store(Element* p, Vec a) {
        vst1q_u64(reinterpret_cast<uint64_t*>(p), a);
    }
    static inline __attribute__((always_inline)) void store_aligned(Element* p, Vec a) {
        vst1q_u64(reinterpret_cast<uint64_t*>(__builtin_assume_aligned(p, 16)), a);
    }

    // Construction
    // Lane-order convention: set(a0, a1) puts a0 in lane 0 (low).
    static inline __attribute__((always_inline)) Vec set(uint64_t a0, uint64_t a1) {
        uint64_t tmp[2] = {a0, a1};
        return vld1q_u64(tmp);
    }
    static inline __attribute__((always_inline)) Vec splat(uint64_t x) { return vdupq_n_u64(x); }
    static inline __attribute__((always_inline)) Vec copy(Vec a)       { return a; }

    // Bit / comparison primitives (declared before arithmetic for inline usage)
    static inline __attribute__((always_inline)) Vec shift(Vec a) { return veorq_u64(a, MSB); }
    static inline __attribute__((always_inline)) Vec cmpgt_biased(Vec a, Vec b) {
        // Signed compare on already-biased (shifted) values gives unsigned cmp.
        return vreinterpretq_u64_s64(
                    vcgtq_s64(vreinterpretq_s64_u64(a),
                              vreinterpretq_s64_u64(b)));
    }
    static inline __attribute__((always_inline)) Vec toCanonical_s(Vec a_s) {
        // If a_s < P_s (signed), then a < P (unsigned), leave it; else add P_n (2^32-1).
        Vec mask = cmpgt_biased(P_s, a_s);        // a_s < P_s ? all-ones : 0
        Vec corr = vbicq_u64(P_n, mask);           // P_n AND NOT mask -> 0 if mask=1, P_n if mask=0
        return vaddq_u64(a_s, corr);
    }
    static inline __attribute__((always_inline)) Vec blend(Vec a, Vec b, Vec mask) {
        return vbslq_u64(mask, b, a);
    }
    static inline __attribute__((always_inline)) Vec and_(Vec a, Vec b) {
        return vandq_u64(a, b);
    }

    // Phase 1d: simplified add/sub using direct unsigned comparison (vcgtq_u64,
    // available in ARMv8). Saves ~3 NEON ops per add vs the shifted-canonical
    // trick that AVX2 needs for its signed-only 64-bit comparison.
    // Matches scalar Goldilocks::add bit-for-bit (both produce non-canonical
    // output in [0, 2^64)).
    static inline __attribute__((always_inline)) Vec add(Vec a, Vec b) {
        Vec r = vaddq_u64(a, b);
        Vec wrap1 = vcgtq_u64(a, r);               // r wrapped iff r < a
        Vec corr1 = vandq_u64(wrap1, P_n);
        r = vaddq_u64(r, corr1);
        Vec wrap2 = vcgtq_u64(corr1, r);           // rare second wrap
        Vec corr2 = vandq_u64(wrap2, P_n);
        return vaddq_u64(r, corr2);
    }
    static inline __attribute__((always_inline)) Vec sub(Vec a, Vec b) {
        Vec r = vsubq_u64(a, b);
        Vec borrow1 = vcgtq_u64(r, a);             // borrow iff r > a unsigned
        Vec corr1 = vandq_u64(borrow1, P_n);
        Vec prev = r;
        r = vsubq_u64(r, corr1);
        Vec borrow2 = vcgtq_u64(r, prev);          // rare second underflow
        Vec corr2 = vandq_u64(borrow2, P_n);
        return vsubq_u64(r, corr2);
    }
    static inline __attribute__((always_inline)) Vec add_small(Vec a, Vec b) {
        // b assumed in [0, 2^32): c0_s cannot wrap (a_sc < 2^64 - 2^32).
        Vec a_s  = shift(a);
        Vec a_sc = toCanonical_s(a_s);
        Vec c0_s = vaddq_u64(a_sc, b);
        return shift(c0_s);
    }

    // ------------------------------------------------------------------------
    // NEON-native 64x64 → 128 multiply + Goldilocks reduction.
    //
    // For each lane k ∈ {0, 1}:
    //   Given a, b ∈ [0, 2^64),
    //   (1) Compute the 128-bit product c_hi:c_lo = a * b using 4× vmull_u32:
    //          a = a_hi * 2^32 + a_lo,  b = b_hi * 2^32 + b_lo
    //          ll = a_lo * b_lo   (fits 64 bits)
    //          lh = a_lo * b_hi   (fits 64 bits)
    //          hl = a_hi * b_lo   (fits 64 bits)
    //          hh = a_hi * b_hi   (fits 64 bits)
    //          mid     = lh + hl                              (may carry)
    //          c_lo    = ll + (mid << 32)                     (may carry)
    //          c_hi    = hh + (mid >> 32) + carry(mid) * 2^32
    //                       + carry(c_lo)
    //
    //   (2) Reduce using 2^64 ≡ 2^32 - 1 (mod p):
    //          Let c_hi_lo = c_hi & (2^32 - 1), c_hi_hi = c_hi >> 32.
    //          s  = c_lo - c_hi                               (borrow b1)
    //          s2 = s + (c_hi_lo << 32)                       (carry c2)
    //          s3 = s2 + c_hi_hi * CQ          where CQ=2^32-1 (carry c3)
    //          adj = c2 + c3 - b1   ∈ {-1, 0, 1, 2}
    //          r  = s3 + adj * CQ
    //          if (r >= p) r -= p
    //
    //   This matches the existing scalar Goldilocks::mul algorithm bit-for-bit.
    //   No __uint128_t is used. No movehdup-style 32-bit lane extraction tricks.
    // ------------------------------------------------------------------------
    // Phase 1e: scalar mul+umulh per lane via __uint128_t. Matches Plonky3's
    // aarch64_neon/packing.rs:295-395 approach. Apple Silicon has 2 integer
    // multiply pipes; out-of-order execution interleaves the two lane muls
    // to hit ~2x integer-mul throughput. NEON vmull_u32 has worse latency and
    // fewer issue slots on M-series than scalar mul+umulh.
    //
    // Reduction follows scalar Goldilocks::mul: 2^64 ≡ 2^32 - 1 (mod p).
    // Non-canonical output in [0, 2^64); public mul() canonicalizes.
    // Scalar Goldilocks mul reduction (non-canonical output). Used by both the
    // C++ and inline-ASM mul paths.
    static inline __attribute__((always_inline)) uint64_t mul_scalar(uint64_t a, uint64_t b) {
        __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
        uint64_t c_lo = (uint64_t)prod;
        uint64_t c_hi = (uint64_t)(prod >> 64);
        uint64_t c_hi_lo = (uint32_t)c_hi;
        uint64_t c_hi_hi = c_hi >> 32;

        uint64_t s = c_lo - c_hi;
        bool borrow = s > c_lo;
        uint64_t s2 = s + (c_hi_lo << 32);
        bool carry2 = s2 < s;
        uint64_t s3 = s2 + ((c_hi_hi << 32) - c_hi_hi);
        bool carry3 = s3 < s2;

        int adj = (int)carry2 + (int)carry3 - (int)borrow;
        uint64_t r = s3 + (uint64_t)adj * (uint64_t)GOLDILOCKS_PRIME_NEG;
        if (adj < 0) r = s3 - (uint64_t)GOLDILOCKS_PRIME_NEG;
        return r;
    }

    // Phase 1k: full dual-lane reduction in one asm block (Plonky3 pattern).
    // Every carry/borrow uses csetm for branchless correction. The two lanes
    // are interleaved by the compiler at register-allocation time; Apple
    // Silicon's two integer-mul pipes issue one mul+umulh pair per cycle.
    static inline __attribute__((always_inline)) Vec mul_reduced(Vec a, Vec b) {
        uint64_t a0 = vgetq_lane_u64(a, 0);
        uint64_t a1 = vgetq_lane_u64(a, 1);
        uint64_t b0 = vgetq_lane_u64(b, 0);
        uint64_t b1 = vgetq_lane_u64(b, 1);
        const uint64_t EPS = (uint64_t)GOLDILOCKS_PRIME_NEG;
        uint64_t r0, r1;
        uint64_t lo0, hi0, lo1, hi1;
        uint64_t hh0, hh1, hl0, hl1, s0, s1, he0, he1;
        uint64_t adj0, adj1, t0, t1;

        asm(
            "mul   %[lo0], %[a0], %[b0]\n\t"
            "mul   %[lo1], %[a1], %[b1]\n\t"
            "umulh %[hi0], %[a0], %[b0]\n\t"
            "umulh %[hi1], %[a1], %[b1]\n\t"
            "lsr   %[hh0], %[hi0], #32\n\t"
            "lsr   %[hh1], %[hi1], #32\n\t"
            "subs  %[t0],  %[lo0], %[hh0]\n\t"
            "csetm %w[adj0], cc\n\t"
            "subs  %[t1],  %[lo1], %[hh1]\n\t"
            "csetm %w[adj1], cc\n\t"
            "sub   %[t0],  %[t0], %[adj0]\n\t"
            "sub   %[t1],  %[t1], %[adj1]\n\t"
            "and   %[hl0], %[hi0], %[eps]\n\t"
            "and   %[hl1], %[hi1], %[eps]\n\t"
            "lsl   %[s0],  %[hl0], #32\n\t"
            "lsl   %[s1],  %[hl1], #32\n\t"
            "sub   %[he0], %[s0], %[hl0]\n\t"
            "sub   %[he1], %[s1], %[hl1]\n\t"
            "adds  %[r0],  %[t0], %[he0]\n\t"
            "csetm %w[adj0], cs\n\t"
            "adds  %[r1],  %[t1], %[he1]\n\t"
            "csetm %w[adj1], cs\n\t"
            "add   %[r0],  %[r0], %[adj0]\n\t"
            "add   %[r1],  %[r1], %[adj1]\n\t"
            : [r0]"=&r"(r0),   [r1]"=&r"(r1),
              [lo0]"=&r"(lo0), [lo1]"=&r"(lo1),
              [hi0]"=&r"(hi0), [hi1]"=&r"(hi1),
              [hh0]"=&r"(hh0), [hh1]"=&r"(hh1),
              [hl0]"=&r"(hl0), [hl1]"=&r"(hl1),
              [s0]"=&r"(s0),   [s1]"=&r"(s1),
              [he0]"=&r"(he0), [he1]"=&r"(he1),
              [adj0]"=&r"(adj0), [adj1]"=&r"(adj1),
              [t0]"=&r"(t0),   [t1]"=&r"(t1)
            : [a0]"r"(a0), [b0]"r"(b0),
              [a1]"r"(a1), [b1]"r"(b1),
              [eps]"r"(EPS)
            : "cc"
        );
        (void)lo0; (void)hi0; (void)lo1; (void)hi1;
        (void)hh0; (void)hh1; (void)hl0; (void)hl1;
        (void)s0; (void)s1; (void)he0; (void)he1;
        (void)adj0; (void)adj1; (void)t0; (void)t1;

        uint64_t tmp[2] = {r0, r1};
        return vld1q_u64(tmp);
    }

    // Branchless Goldilocks mul reduction per lane.
    // adj ∈ {-1, 0, 1, 2}; encode into (pos_count, borrow) where pos * EPSILON
    // is added and borrow * EPSILON is subtracted. Both paths merge to one add-sub.
    static inline __attribute__((always_inline)) uint64_t mul_scalar_branchless(uint64_t a, uint64_t b) {
        __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
        uint64_t c_lo = (uint64_t)prod;
        uint64_t c_hi = (uint64_t)(prod >> 64);
        uint64_t c_hi_lo = (uint32_t)c_hi;
        uint64_t c_hi_hi = c_hi >> 32;
        const uint64_t EPS = (uint64_t)GOLDILOCKS_PRIME_NEG;

        uint64_t s = c_lo - c_hi;
        uint64_t borrow = (uint64_t)(s > c_lo);
        uint64_t s2 = s + (c_hi_lo << 32);
        uint64_t carry2 = (uint64_t)(s2 < s);
        uint64_t s3 = s2 + ((c_hi_hi << 32) - c_hi_hi);
        uint64_t carry3 = (uint64_t)(s3 < s2);

        // r = s3 + (carry2 + carry3) * EPS - borrow * EPS  (non-canonical).
        uint64_t pos_cq = (carry2 + carry3) * EPS;
        uint64_t neg_cq = borrow * EPS;
        return s3 + pos_cq - neg_cq;
    }

    // Phase 1m: bulk-accumulate helper for mvp.
    // Computes: (a * m)_lo + (a * m)_hi into running sums, without per-step
    // reduction. Caller tracks carries separately and applies the combined
    // reduction once per output position at the end of the 12-element inner
    // loop. Saves ~2 ops per mul vs mul_small_reduced + add.
    //
    // Math: each product a_j * m_j yields (lo_j, hi_j) with hi_j ≤ 41.
    //   SL_true = Σ lo_j  (may exceed 2^64; tracked as sum_lo + carries*2^64)
    //   SH      = Σ hi_j  (< 12*41 = 492, never overflows 64 bits)
    //   Result = SL + SH * 2^64 ≡ sum_lo + (carries + SH) * EPSILON (mod p)
    static inline __attribute__((always_inline)) void accumulate_small_pair(
        uint64_t a0, uint64_t a1, uint64_t m,
        uint64_t& sum_lo0, uint64_t& sum_hi0, uint64_t& carries0,
        uint64_t& sum_lo1, uint64_t& sum_hi1, uint64_t& carries1)
    {
        uint64_t lo0, hi0, lo1, hi1;
        asm(
            "mul   %[lo0], %[a0], %[m]\n\t"
            "mul   %[lo1], %[a1], %[m]\n\t"
            "umulh %[hi0], %[a0], %[m]\n\t"
            "umulh %[hi1], %[a1], %[m]\n\t"
            "adds  %[slo0], %[slo0], %[lo0]\n\t"
            "adc   %[c0],   %[c0],   xzr\n\t"
            "add   %[shi0], %[shi0], %[hi0]\n\t"
            "adds  %[slo1], %[slo1], %[lo1]\n\t"
            "adc   %[c1],   %[c1],   xzr\n\t"
            "add   %[shi1], %[shi1], %[hi1]\n\t"
            : [lo0]"=&r"(lo0), [lo1]"=&r"(lo1),
              [hi0]"=&r"(hi0), [hi1]"=&r"(hi1),
              [slo0]"+&r"(sum_lo0), [shi0]"+&r"(sum_hi0), [c0]"+&r"(carries0),
              [slo1]"+&r"(sum_lo1), [shi1]"+&r"(sum_hi1), [c1]"+&r"(carries1)
            : [a0]"r"(a0), [a1]"r"(a1), [m]"r"(m)
            : "cc"
        );
        (void)lo0; (void)lo1; (void)hi0; (void)hi1;
    }

    // Phase 1o: bulk-accumulate for FULL 64-bit × 64-bit products (used in
    // partial-round dot, where the S constants are full-range 64-bit values
    // that don't fit the mul_small path).
    //
    // Math: for each mul a*s, (lo, hi) are the 128-bit parts. hi can be any
    // 64-bit value, so sum_hi itself overflows across the 12 accumulations.
    // Track two carry counters:
    //   cl_carries = count of sum_lo wraps
    //   ch_carries = count of sum_hi wraps
    //
    // Final reduction (mod p):
    //   Total = sum_lo + cl_carries*2^64 + (sum_hi + ch_carries*2^64) * 2^64
    //         ≡ sum_lo + (cl_carries + sum_hi) * EPSILON + ch_carries * EPSILON²  (mod p)
    //
    // EPSILON² mod p = p - 2^32 = -2^32 (mod p), so ch_carries * EPSILON²
    // ≡ -ch_carries * 2^32. See finalize_full_sum.
    static inline __attribute__((always_inline)) void accumulate_full_pair(
        uint64_t a0, uint64_t a1, uint64_t s,
        uint64_t& sum_lo0, uint64_t& sum_hi0, uint64_t& cl0, uint64_t& ch0,
        uint64_t& sum_lo1, uint64_t& sum_hi1, uint64_t& cl1, uint64_t& ch1)
    {
        uint64_t lo0, hi0, lo1, hi1;
        asm(
            "mul   %[lo0], %[a0], %[s]\n\t"
            "mul   %[lo1], %[a1], %[s]\n\t"
            "umulh %[hi0], %[a0], %[s]\n\t"
            "umulh %[hi1], %[a1], %[s]\n\t"
            "adds  %[slo0], %[slo0], %[lo0]\n\t"
            "adc   %[cl0],  %[cl0],  xzr\n\t"
            "adds  %[shi0], %[shi0], %[hi0]\n\t"
            "adc   %[ch0],  %[ch0],  xzr\n\t"
            "adds  %[slo1], %[slo1], %[lo1]\n\t"
            "adc   %[cl1],  %[cl1],  xzr\n\t"
            "adds  %[shi1], %[shi1], %[hi1]\n\t"
            "adc   %[ch1],  %[ch1],  xzr\n\t"
            : [lo0]"=&r"(lo0), [lo1]"=&r"(lo1),
              [hi0]"=&r"(hi0), [hi1]"=&r"(hi1),
              [slo0]"+&r"(sum_lo0), [shi0]"+&r"(sum_hi0),
              [cl0]"+&r"(cl0),     [ch0]"+&r"(ch0),
              [slo1]"+&r"(sum_lo1), [shi1]"+&r"(sum_hi1),
              [cl1]"+&r"(cl1),     [ch1]"+&r"(ch1)
            : [a0]"r"(a0), [a1]"r"(a1), [s]"r"(s)
            : "cc"
        );
        (void)lo0; (void)lo1; (void)hi0; (void)hi1;
    }

    // Phase 1p: single-lane variant for single-hash dot.
    // Processes 2 muls per call (two different a,s pairs) to keep both integer
    // mul pipes busy; accumulates separately into a single lane's running sums.
    static inline __attribute__((always_inline)) void accumulate_full_single_pair(
        uint64_t a0, uint64_t s0, uint64_t a1, uint64_t s1,
        uint64_t& sum_lo, uint64_t& sum_hi, uint64_t& cl, uint64_t& ch)
    {
        uint64_t lo0, hi0, lo1, hi1;
        asm(
            "mul   %[lo0], %[a0], %[s0]\n\t"
            "mul   %[lo1], %[a1], %[s1]\n\t"
            "umulh %[hi0], %[a0], %[s0]\n\t"
            "umulh %[hi1], %[a1], %[s1]\n\t"
            "adds  %[slo], %[slo], %[lo0]\n\t"
            "adc   %[cl],  %[cl],  xzr\n\t"
            "adds  %[shi], %[shi], %[hi0]\n\t"
            "adc   %[ch],  %[ch],  xzr\n\t"
            "adds  %[slo], %[slo], %[lo1]\n\t"
            "adc   %[cl],  %[cl],  xzr\n\t"
            "adds  %[shi], %[shi], %[hi1]\n\t"
            "adc   %[ch],  %[ch],  xzr\n\t"
            : [lo0]"=&r"(lo0), [lo1]"=&r"(lo1),
              [hi0]"=&r"(hi0), [hi1]"=&r"(hi1),
              [slo]"+&r"(sum_lo), [shi]"+&r"(sum_hi),
              [cl]"+&r"(cl), [ch]"+&r"(ch)
            : [a0]"r"(a0), [s0]"r"(s0), [a1]"r"(a1), [s1]"r"(s1)
            : "cc"
        );
        (void)lo0; (void)lo1; (void)hi0; (void)hi1;
    }

    // Final reduction for the full-mul bulk accumulator.
    // r ≡ sum_lo + (cl_carries + sum_hi) * EPSILON - ch_carries * 2^32  (mod p)
    // Since ch_carries ≤ 12, ch_carries * 2^32 is small. Output non-canonical in [0, 2^64).
    static inline __attribute__((always_inline)) uint64_t finalize_full_sum(
        uint64_t sum_lo, uint64_t sum_hi, uint64_t cl_carries, uint64_t ch_carries)
    {
        // + (sum_hi + cl_carries) * EPSILON
        uint64_t total_eps_coeff = sum_hi + cl_carries;  // sum_hi adds only up to 2^64, cl_carries ≤ 12
        // total_eps_coeff can be any 64-bit value (sum_hi is 64-bit)
        // So this isn't small. Compute full-width multiply by EPSILON.
        // (total_eps_coeff * EPSILON) as 128-bit, reduce again via 2^64 ≡ EPSILON.
        __uint128_t eps_prod = (__uint128_t)total_eps_coeff * (uint64_t)GOLDILOCKS_PRIME_NEG;
        uint64_t ep_lo = (uint64_t)eps_prod;
        uint64_t ep_hi = (uint64_t)(eps_prod >> 64);

        // Subtract ch_carries * 2^32 (small; ≤ 12 * 2^32)
        uint64_t sub_term = ch_carries << 32;

        // r = sum_lo + ep_lo + ep_hi * EPSILON - sub_term  (mod p, via 2^64 ≡ EPSILON)
        // ep_hi is small (≤ about 2 * EPSILON) because total_eps_coeff < 2^64 and EPSILON < 2^32,
        // so ep_prod < 2^96, ep_hi < 2^32. So ep_hi * EPSILON fits in ~2^64.
        uint64_t ep_hi_eps = (ep_hi << 32) - ep_hi;  // ep_hi * EPSILON via shift-sub

        // Accumulate: r = sum_lo + ep_lo + ep_hi_eps - sub_term  (with wrap handling)
        uint64_t r, adj1, adj2, adj3;
        asm(
            "adds  %[r], %[slo], %[elo]\n\t"
            "csetm %w[a1], cs\n\t"
            "add   %[r], %[r], %[a1]\n\t"
            "adds  %[r], %[r], %[ehe]\n\t"
            "csetm %w[a2], cs\n\t"
            "add   %[r], %[r], %[a2]\n\t"
            "subs  %[r], %[r], %[sub]\n\t"
            "csetm %w[a3], cc\n\t"
            "sub   %[r], %[r], %[a3]\n\t"
            : [r]"=&r"(r), [a1]"=&r"(adj1), [a2]"=&r"(adj2), [a3]"=&r"(adj3)
            : [slo]"r"(sum_lo), [elo]"r"(ep_lo), [ehe]"r"(ep_hi_eps), [sub]"r"(sub_term)
            : "cc"
        );
        return r;
    }

    // Final reduction: given (sum_lo, sum_hi, carries), compute
    //   r = sum_lo + (sum_hi + carries) * EPSILON  (mod 2^64, with wrap-fix)
    // Result is non-canonical [0, 2^64).
    static inline __attribute__((always_inline)) uint64_t finalize_small_sum(
        uint64_t sum_lo, uint64_t sum_hi, uint64_t carries)
    {
        uint64_t total_hi = sum_hi + carries;              // < 492 + 12 = 504
        uint64_t eps_term = (total_hi << 32) - total_hi;   // total_hi * EPSILON
        uint64_t r, adj;
        asm(
            "adds  %[r], %[slo], %[et]\n\t"
            "csetm %w[adj], cs\n\t"
            "add   %[r], %[r], %[adj]\n\t"
            : [r]"=&r"(r), [adj]"=&r"(adj)
            : [slo]"r"(sum_lo), [et]"r"(eps_term)
            : "cc"
        );
        return r;
    }

    // Phase 1l: mul_small — Goldilocks multiplication when one operand is known
    // to fit in [0, 2^32). Poseidon's MDS matrix entries are all ≤ 41, so this
    // is heavily used inside mvp_neon/mvp_neon_2.
    //
    // Simplified reduction: c = a * small ≤ 2^64 * 2^32 = 2^96, so c_hi < 2^32.
    // This means c_hi_hi (= c_hi >> 32) is always 0. The reduction collapses:
    //   result = c_lo + c_hi * EPSILON (mod p) = c_lo + (c_hi << 32) - c_hi
    // Only 1 carry-correction step needed (the second branch in the full mul is
    // always no-op because c_hi < 2^32).
    static inline __attribute__((always_inline)) Vec mul_small_reduced(Vec a, uint64_t small0, uint64_t small1) {
        uint64_t a0 = vgetq_lane_u64(a, 0);
        uint64_t a1 = vgetq_lane_u64(a, 1);
        uint64_t r0, r1;
        uint64_t lo0, hi0, lo1, hi1, shl0, shl1, adj0, adj1, tmp0, tmp1;

        asm(
            "mul   %[lo0], %[a0], %[s0]\n\t"
            "mul   %[lo1], %[a1], %[s1]\n\t"
            "umulh %[hi0], %[a0], %[s0]\n\t"
            "umulh %[hi1], %[a1], %[s1]\n\t"
            // hi_times_eps = (hi << 32) - hi  (since EPSILON = 2^32 - 1)
            "lsl   %[shl0], %[hi0], #32\n\t"
            "lsl   %[shl1], %[hi1], #32\n\t"
            "sub   %[tmp0], %[shl0], %[hi0]\n\t"
            "sub   %[tmp1], %[shl1], %[hi1]\n\t"
            // result = lo + hi_times_eps  (may wrap)
            "adds  %[r0], %[lo0], %[tmp0]\n\t"
            "csetm %w[adj0], cs\n\t"
            "adds  %[r1], %[lo1], %[tmp1]\n\t"
            "csetm %w[adj1], cs\n\t"
            "add   %[r0], %[r0], %[adj0]\n\t"
            "add   %[r1], %[r1], %[adj1]\n\t"
            : [r0]"=&r"(r0), [r1]"=&r"(r1),
              [lo0]"=&r"(lo0), [lo1]"=&r"(lo1),
              [hi0]"=&r"(hi0), [hi1]"=&r"(hi1),
              [shl0]"=&r"(shl0), [shl1]"=&r"(shl1),
              [tmp0]"=&r"(tmp0), [tmp1]"=&r"(tmp1),
              [adj0]"=&r"(adj0), [adj1]"=&r"(adj1)
            : [a0]"r"(a0), [a1]"r"(a1),
              [s0]"r"(small0), [s1]"r"(small1)
            : "cc"
        );
        (void)lo0; (void)hi0; (void)lo1; (void)hi1;
        (void)shl0; (void)shl1; (void)tmp0; (void)tmp1;
        (void)adj0; (void)adj1;
        uint64_t tmp[2] = {r0, r1};
        return vld1q_u64(tmp);
    }

    // Public canonical mul — preserves the [0, P) output contract.
    static inline __attribute__((always_inline)) Vec mul(Vec a, Vec b) {
        uint64x2_t r = mul_reduced(a, b);
        uint64x2_t ge_p_mask = vcgeq_u64(r, P);
        return vsubq_u64(r, vandq_u64(ge_p_mask, P));
    }

    // Non-canonical square. Alias for mul_reduced(a, a).
    static inline __attribute__((always_inline)) Vec square_reduced(Vec a) {
        return mul_reduced(a, a);
    }
    static inline __attribute__((always_inline)) Vec square(Vec a) {
        return mul(a, a);
    }

    // Final canonicalizer for use right before store. Non-canonical → canonical.
    // Must handle inputs up to [0, 2^64). If r > P but r < 2P, one subtract suffices.
    // Wrap-around corrections in add can push intermediate values close to 2^64; the
    // double-subtract ensures r ∈ [0, P) on exit.
    static inline __attribute__((always_inline)) Vec canonicalize(Vec r) {
        uint64x2_t ge_p = vcgeq_u64(r, P);
        r = vsubq_u64(r, vandq_u64(ge_p, P));
        ge_p = vcgeq_u64(r, P);
        return vsubq_u64(r, vandq_u64(ge_p, P));
    }

    // Composite / lane mixing
    static inline __attribute__((always_inline)) Vec permute_lanes(Vec a, Vec b) {
        return vextq_u64(a, b, 1);
    }
    static inline __attribute__((always_inline)) Vec mask_lane0_zero(Vec v) {
        // Zero lane 0, keep lane 1. AVX2 needs a 4-lane mask; NEON uses blend with {0, all_ones}.
        uint64_t mask[2] = {0ULL, ~0ULL};
        Vec m = vld1q_u64(mask);
        return vandq_u64(v, m);
    }
};

} // namespace simd
} // namespace goldilocks

#endif // GOLDILOCKS_SIMD_NEON_HPP
