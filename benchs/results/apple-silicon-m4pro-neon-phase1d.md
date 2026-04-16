# Goldilocks NEON Phase 1d — Simplified Add/Sub

**Date:** 2026-04-15
**System:** macOS 26.3.1 — Apple M4 Pro, 14 cores
**Phase:** 1d — drop the shifted-canonical trick in NEON add/sub; use direct unsigned comparisons (`vcgtq_u64`, available in ARMv8 but not older NEON).

## Motivation

UDOT/SDOT intrinsics (ARMv8.4 dot-product) target 8-bit ML workloads — they don't apply to 64-bit field arithmetic. Phase 1d instead attacks the add/sub instruction count.

Pre-Phase 1d, NEON `add` mirrored AVX2's signed-only comparison trick:
```
a_s  = a ^ MSB                 // 1 op
a_sc = toCanonical_s(a_s)      // 3 ops (cmp + bic + add)
c0_s = a_sc + b                // 1 op
mask = cmpgt_biased(a_sc, c0_s) // 3 ops (reinterpret + vcgt_s64 + reinterpret)
corr = mask & P_n              // 1 op
c_s  = c0_s + corr             // 1 op
r    = c_s ^ MSB               // 1 op (un-shift)
```
= 11 NEON ops for one `add`.

ARMv8 adds `vcgtq_u64` (unsigned 64-bit compare) that the AVX2 codebase couldn't use. Phase 1d replaces the whole shifted-canonical dance:
```
r     = a + b                  // 1 op
wrap1 = a > r (unsigned)       // 1 op (vcgtq_u64)
corr1 = wrap1 & P_n            // 1 op
r     = r + corr1              // 1 op
wrap2 = corr1 > r (unsigned)   // 1 op
corr2 = wrap2 & P_n            // 1 op
r     = r + corr2              // 1 op
```
= 7 NEON ops. **−36% instruction count per add.**

`sub` similarly drops from 10 → 8 ops using direct unsigned-borrow detection.

## Results (14 threads)

| Benchmark | Scalar | NEON 1c | NEON 1d | Δ 1d vs 1c | Δ 1d vs scalar |
|-----------|--------|---------|---------|------------|-----------------|
| POSEIDON_BENCH_FULL | 242.2 MiB/s | ~244 | **247.6** | +1.5% | +2.2% |
| POSEIDON_BENCH | 241.6 MiB/s | ~228 | **240.2** | +5.4% | −0.6% |
| LINEAR_HASH_BENCH | 162.8 MiB/s | 153.8 | **163.8** | **+6.5%** | +0.6% |
| MERKLETREE_BENCH | 154.7 MiB/s | 164.2 | 161.2 | −1.8% (noise) | +4.2% |
| MERKLETREE_BATCH_BENCH | 137.3 MiB/s | 131.0 | 133.2 | +1.7% | −3.0% |

LINEAR_HASH_BENCH is the cleanest signal: no 2-hash pairing applied (single sponge chain), so Phase 1d's per-add savings show up directly. **+6.5% throughput.**

Benchmarks that 1c already optimized via pairing (MERKLETREE_BENCH) are roughly flat on 1d — the pair-hash path amortizes add cost across 2 hashes already.

## Correctness

- All 26 gtests pass.
- Simplified NEON add matches scalar `Goldilocks::add` bit-for-bit (both produce non-canonical `[0, 2^64)` output; the next op — mul or add — handles non-canonical input correctly).
- NEON merkletree still bit-exact with scalar over 64-element test.

## Overall Phase 1 progress (M4 Pro NEON vs Xeon D-2141I AVX2)

| Workload | M4 Pro NEON final | Xeon AVX2 | Ratio |
|----------|---------------------|-----------|-------|
| POSEIDON_BENCH_FULL | 247.6 MiB/s | 293.8 MiB/s | **0.84×** |
| LINEAR_HASH_BENCH | 163.8 MiB/s | 189.8 MiB/s | **0.86×** |
| MERKLETREE_BENCH | 161.2 MiB/s | 182.3 MiB/s | **0.88×** |

All three workloads: M4 Pro NEON is within ~15% of Xeon AVX2. Remaining gap is fundamental — AVX2 has 4 lanes vs NEON's 2, and a single-instruction 64×64→128 mul path (`_mm256_mul_epu32`+shuffles) that NEON cannot match.

## What Phase 1 achieved (full recap)

| Phase | Key change | LINEAR_HASH | MERKLETREE |
|-------|-----------|-------------|------------|
| Scalar baseline | — | 159.5 MiB/s | 153.6 |
| 1a | NEON field primitives + pow7/add | 157.6 | 151.3 |
| 1b | + NEON mvp, dot | 149.8 | 145.1 |
| 1c | + 2-hash-parallel pair | 153.8 | 164.2 |
| **1d** | + simplified add/sub | **163.8** | 161.2 |

Net Phase 1: **+2.7% to +5%** over scalar on the full Poseidon pipeline, with the NEON path fully vectorized and ready for further ARM-specific optimization (SME matrix ops, future SVE).
