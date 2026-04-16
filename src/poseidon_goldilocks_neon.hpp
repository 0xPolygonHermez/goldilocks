#ifndef POSEIDON_GOLDILOCKS_NEON_HPP
#define POSEIDON_GOLDILOCKS_NEON_HPP

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include "simd/goldilocks_simd.hpp"

#ifdef GOLDILOCKS_HAS_NEON
#include <arm_neon.h>

inline void PoseidonGoldilocks::hash_neon(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result_neon(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

inline void PoseidonGoldilocks::pow7_neon(uint64x2_t st[6])
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    // Phase 1j: unrolled in pairs so the compiler / CPU sees two independent
    // mul chains per iteration. Apple Silicon has 2 integer-mul pipes;
    // interleaving exposes them both every cycle instead of stalling on the
    // pw2 -> pw4 dependency.
    for (int i = 0; i < 6; i += 2) {
        auto a0 = st[i];
        auto a1 = st[i + 1];
        auto pw2_0 = N::square_reduced(a0);
        auto pw2_1 = N::square_reduced(a1);
        auto pw4_0 = N::square_reduced(pw2_0);
        auto pw4_1 = N::square_reduced(pw2_1);
        auto pw3_0 = N::mul_reduced(pw2_0, a0);
        auto pw3_1 = N::mul_reduced(pw2_1, a1);
        st[i]     = N::mul_reduced(pw3_0, pw4_0);
        st[i + 1] = N::mul_reduced(pw3_1, pw4_1);
    }
}

inline void PoseidonGoldilocks::add_neon(uint64x2_t st[6], const Goldilocks::Element C[SPONGE_WIDTH])
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    for (int i = 0; i < 6; ++i) {
        auto c = N::load(&C[i * 2]);
        st[i] = N::add(st[i], c);
    }
}

// NEON matrix-vector product: state = M^T * old_state (shape SPONGE_WIDTH x SPONGE_WIDTH).
// Operates entirely in NEON registers; caller provides state both as input (old) and output.
//
// Phase 1l: uses mul_small_reduced for MDS matrix entries (Poseidon M matrix has
// all entries ≤ 41). P matrix (used once per hash) has full 64-bit values and
// still uses mul_reduced.
inline void PoseidonGoldilocks::mvp_neon(Goldilocks::Element *state,
    const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH],
    bool mat_is_small)
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    Goldilocks::Element old_state[SPONGE_WIDTH];
    std::memcpy(old_state, state, SPONGE_WIDTH * sizeof(Goldilocks::Element));

    uint64x2_t out[6];
    for (int i = 0; i < 6; ++i) out[i] = N::splat(0);

    if (mat_is_small) {
        // Phase 1m single-hash path: __uint128_t in C (compiler schedules mul+umulh).
        // Tried asm variant in Phase 1n; regressed — clang's own scheduling is
        // better here because the 3-operand (a, m0, m1) structure doesn't benefit
        // from forcing asm.
        for (int i = 0; i < 6; ++i) {
            uint64_t sum_lo0 = 0, sum_hi0 = 0, carries0 = 0;
            uint64_t sum_lo1 = 0, sum_hi1 = 0, carries1 = 0;
            for (int j = 0; j < SPONGE_WIDTH; ++j) {
                uint64_t a = old_state[j].fe;
                uint64_t m0 = mat[j][i * 2].fe;
                uint64_t m1 = mat[j][i * 2 + 1].fe;
                uint64_t lo0 = a * m0;
                uint64_t lo1 = a * m1;
                uint64_t hi0 = (uint64_t)(((__uint128_t)a * m0) >> 64);
                uint64_t hi1 = (uint64_t)(((__uint128_t)a * m1) >> 64);
                uint64_t new_lo0 = sum_lo0 + lo0;
                carries0 += (new_lo0 < sum_lo0) ? 1 : 0;
                sum_lo0 = new_lo0;
                sum_hi0 += hi0;
                uint64_t new_lo1 = sum_lo1 + lo1;
                carries1 += (new_lo1 < sum_lo1) ? 1 : 0;
                sum_lo1 = new_lo1;
                sum_hi1 += hi1;
            }
            uint64_t r0 = N::finalize_small_sum(sum_lo0, sum_hi0, carries0);
            uint64_t r1 = N::finalize_small_sum(sum_lo1, sum_hi1, carries1);
            uint64_t tmp[2] = {r0, r1};
            out[i] = vld1q_u64(tmp);
        }
    } else {
        // General path: mat entries can be full 64-bit (Poseidon P).
        for (int j = 0; j < SPONGE_WIDTH; ++j) {
            uint64x2_t bc = N::splat(old_state[j].fe);
            for (int i = 0; i < 6; ++i) {
                uint64x2_t m_ji = N::load(&mat[j][i * 2]);
                out[i] = N::add(out[i], N::mul_reduced(bc, m_ji));
            }
        }
    }
    for (int i = 0; i < 6; ++i) N::store(&state[i * 2], out[i]);
}

// Phase 1c: 2-hash-parallel NEON matrix-vector product.
// Layout: st[k] = {stA[k], stB[k]} for k=0..11. Processes 2 independent
// Poseidon hashes per NEON register; doubles throughput when merkle rows pair up.
inline void PoseidonGoldilocks::mvp_neon_2(uint64x2_t st[12],
    const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH],
    bool mat_is_small)
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    uint64x2_t old[12];
    for (int k = 0; k < 12; ++k) old[k] = st[k];

    if (mat_is_small) {
        // Phase 1m: bulk-reduce — accumulate raw partial products, reduce once
        // per output. Saves ~2 ops per mul_small-equivalent iteration.
        uint64_t s0[12], s1[12];
        for (int k = 0; k < 12; ++k) {
            s0[k] = vgetq_lane_u64(old[k], 0);
            s1[k] = vgetq_lane_u64(old[k], 1);
        }
        for (int k = 0; k < 12; ++k) {
            uint64_t sum_lo0 = 0, sum_hi0 = 0, carries0 = 0;
            uint64_t sum_lo1 = 0, sum_hi1 = 0, carries1 = 0;
            for (int j = 0; j < 12; ++j) {
                uint64_t m = mat[j][k].fe;
                N::accumulate_small_pair(s0[j], s1[j], m,
                    sum_lo0, sum_hi0, carries0,
                    sum_lo1, sum_hi1, carries1);
            }
            uint64_t r0 = N::finalize_small_sum(sum_lo0, sum_hi0, carries0);
            uint64_t r1 = N::finalize_small_sum(sum_lo1, sum_hi1, carries1);
            uint64_t tmp[2] = {r0, r1};
            st[k] = vld1q_u64(tmp);
        }
    } else {
        // General path (used once per hash for P matrix).
        for (int k = 0; k < 12; ++k) {
            uint64x2_t acc = N::splat(0);
            for (int j = 0; j < 12; ++j) {
                uint64x2_t m_jk = N::splat(mat[j][k].fe);
                acc = N::add(acc, N::mul_reduced(m_jk, old[j]));
            }
            st[k] = acc;
        }
    }
}

// NEON dot product: sum_{i=0..11} x[i] * C[i].
// Phase 1p: bulk-reduce — accumulate all 12 raw 128-bit products into one
// (sum_lo, sum_hi, cl, ch) state per output lane, reduce once. Pairs up muls
// so both integer-mul pipes stay busy.
inline Goldilocks::Element PoseidonGoldilocks::dot_neon(const Goldilocks::Element *x,
    const Goldilocks::Element C[SPONGE_WIDTH])
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    uint64_t slo = 0, shi = 0, cl = 0, ch = 0;
    for (int k = 0; k < SPONGE_WIDTH; k += 2) {
        N::accumulate_full_single_pair(
            x[k].fe, C[k].fe, x[k + 1].fe, C[k + 1].fe,
            slo, shi, cl, ch);
    }
    uint64_t r = N::finalize_full_sum(slo, shi, cl, ch);
    // r is non-canonical [0, 2^64); canonicalize for return value.
    if (r >= GOLDILOCKS_PRIME) r -= GOLDILOCKS_PRIME;
    return Goldilocks::Element{r};
}

#endif // GOLDILOCKS_HAS_NEON
#endif // POSEIDON_GOLDILOCKS_NEON_HPP
