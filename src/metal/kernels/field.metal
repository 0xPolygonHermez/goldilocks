// field.metal — Goldilocks field arithmetic for Apple Metal GPU
//
// Prime:  p = 0xFFFFFFFF00000001  (2^64 - 2^32 + 1)
// CQ:     0xFFFFFFFF             (= 2^32 - 1 = -p mod 2^64 + 1 = epsilon)
//
// Lazy-reduce contract (mirrors NEON goldilocks_simd_neon.hpp:143-161):
//   - gl_mul returns in [0, 2p), NOT [0, p).
//   - gl_add and gl_sub return fully reduced in [0, p).
//   - gl_canonicalize brings [0, 2p) -> [0, p); call only at kernel exit.
//
// Metal 3 / Apple Silicon: ulong is 64-bit; metal::mulhi(ulong, ulong) returns
// the high 64 bits of the 128-bit product.

#pragma once
#include <metal_stdlib>
using namespace metal;

constant ulong GL_PRIME = 0xFFFFFFFF00000001UL;
constant ulong GL_CQ    = 0xFFFFFFFFUL;  // 2^32 - 1

// gl_add: fully reduced result in [0, p)
// Algorithm: s = a + b; on 64-bit overflow add CQ (= 2^32-1);
// if that also wraps add CQ again. Two corrective steps suffice
// because inputs are in [0, p) so sum is in [0, 2p) ⊂ [0, 2^65).
inline ulong gl_add(ulong a, ulong b) {
    ulong s = a + b;
    // carry from a+b
    ulong carry = (s < a) ? 1UL : 0UL;
    // adjust: if carry, add CQ; if that also wraps, add CQ again.
    s += carry * GL_CQ;
    ulong carry2 = (carry != 0 && s < GL_CQ) ? 1UL : 0UL;
    s += carry2 * GL_CQ;
    return s;
}

// gl_sub: fully reduced result in [0, p)
// Algorithm: s = a - b; on borrow (s > a), subtract CQ (≡ add p mod 2^64).
// Two subtractive steps suffice for same reason.
inline ulong gl_sub(ulong a, ulong b) {
    ulong s = a - b;
    // borrow from a-b
    ulong borrow = (s > a) ? 1UL : 0UL;
    ulong prev = s;
    s -= borrow * GL_CQ;
    ulong borrow2 = (borrow != 0 && s > prev) ? 1UL : 0UL;
    s -= borrow2 * GL_CQ;
    return s;
}

// gl_mul: LAZY reduce — result in [0, 2p), NOT canonicalized.
//
// Port of mul_scalar (goldilocks_simd_neon.hpp:143-161) to MSL.
// Uses metal::mulhi(a,b) for high 64 bits of 64x64 product.
//
// Algorithm (identical to NEON path — deliberately omits final if r>=p):
//   prod = a * b  (128-bit conceptual)
//   c_lo = low64(prod), c_hi = high64(prod)
//   c_hi_lo = c_hi[31:0], c_hi_hi = c_hi[63:32]
//   s  = c_lo - c_hi              -- may borrow
//   s2 = s + c_hi_lo * 2^32       -- may carry
//   s3 = s2 + c_hi_hi * CQ        -- may carry  (CQ = 2^32-1)
//   adj in {-1,0,1,2}: r = s3 + adj*CQ
//   NO final if (r >= p) r -= p   -- lazy reduce
inline ulong gl_mul(ulong a, ulong b) {
    ulong c_lo  = a * b;
    ulong c_hi  = metal::mulhi(a, b);
    ulong c_hi_lo = c_hi & GL_CQ;
    ulong c_hi_hi = c_hi >> 32;

    ulong s  = c_lo - c_hi;
    int borrow = (s > c_lo) ? 1 : 0;

    ulong s2 = s + (c_hi_lo << 32);
    int carry2 = (s2 < s) ? 1 : 0;

    ulong s3 = s2 + c_hi_hi * GL_CQ;
    int carry3 = (s3 < s2) ? 1 : 0;

    // adj ∈ {-1, 0, 1, 2}. Branchless reduction via signed 64-bit mul:
    // delta = adj * GL_CQ as a signed 64-bit value. s3 + delta wraps to
    // the correct modular result (s3 - GL_CQ when adj = -1, etc.).
    int adj = carry2 + carry3 - borrow;
    long delta = (long)adj * (long)GL_CQ;
    ulong r = s3 + (ulong)delta;
    // INTENTIONALLY no: if (r >= GL_PRIME) r -= GL_PRIME;
    return r;
}

// gl_canonicalize: [0, 2p) -> [0, p)
// Call only at kernel boundaries / MTLBuffer write.
inline ulong gl_canonicalize(ulong a) {
    return (a >= GL_PRIME) ? (a - GL_PRIME) : a;
}

// gl_mul_small: multiply 64-bit x by a small constant k (k < 2^32),
// result in [0, 2p) (same lazy contract as gl_mul).
//
// Since k < 2^32 and x < 2^64, the product x*k < 2^96 — no full 128-bit
// reduction needed. Saves ~half the ops vs gl_mul for the common case of
// multiplying by an MDS matrix entry (the Goldilocks Poseidon12 M and P
// matrices contain only values < 0x30, well within 8 bits).
//
// Structure: decompose x into 32-bit halves (x_hi, x_lo), compute two
// uint×uint → ulong widening products, assemble into 96-bit form as
// (hi_32 * 2^64 + lo_64), then reduce using 2^64 ≡ CQ (mod p).
inline ulong gl_mul_small(ulong x, uint k) {
    uint  x_lo = (uint)(x & 0xFFFFFFFFu);
    uint  x_hi = (uint)(x >> 32);
    ulong p_lo = (ulong)x_lo * (ulong)k;   // native widening 32×32 → 64
    ulong p_hi = (ulong)x_hi * (ulong)k;

    // 96-bit product = (p_hi << 32) + p_lo, split into (hi_32, lo_64).
    ulong lo_64 = p_lo + (p_hi << 32);
    uint  carry = (lo_64 < p_lo) ? 1u : 0u;
    uint  hi_32 = (uint)(p_hi >> 32) + carry;

    // Reduce: 2^64 ≡ CQ (mod p). product mod p ≡ lo_64 + hi_32 * CQ.
    // hi_32 * CQ fits in 64 bits (both < 2^32).
    ulong hi_cq  = (ulong)hi_32 * GL_CQ;
    ulong result = lo_64 + hi_cq;
    if (result < lo_64) result += GL_CQ;    // single-step overflow fixup
    return result;                          // in [0, 2p)
}
