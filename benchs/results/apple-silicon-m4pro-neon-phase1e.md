# Goldilocks NEON Phase 1e — Scalar mul+umulh per lane (Plonky3 approach)

**Date:** 2026-04-15
**System:** macOS 26.3.1 — Apple M4 Pro, 14 cores
**Phase:** 1e — replace 4× `vmull_u32` reduction with two scalar `mul+umulh` ops per lane. Matches Plonky3's `aarch64_neon/packing.rs:295-395` approach. Also uses Plonky3's `(hi_hi << 32) - hi_hi` shift-sub to replace one multiply-by-EPSILON per mul.

## Key insight (from research into Plonky3 / plonky2)

Both repos abandoned NEON `vmull_u32` for Goldilocks multiplication:

- Plonky3 extracts both lanes to GPRs (`vgetq_lane_u64`), does two scalar `mul`+`umulh` in interleaved inline ASM, repacks via NEON.
- plonky2 does the same for Poseidon S-box multiplies.

Reason: **Apple Silicon has 2 integer-multiply pipes that sit idle while NEON `vmull_u32` chains bottleneck on the NEON ALU.** Dispatching scalar mul lets the hardware parallelize mul (integer pipes) with add/shift (NEON pipes).

## Implementation

`GLSimd<Neon>::mul_reduced` now:
1. Extracts lane 0 and lane 1 to scalar `uint64_t`.
2. For each lane, does `__uint128_t` mul (compiles to `mul`+`umulh`) plus scalar Goldilocks reduction with Plonky3's shift-sub EPSILON.
3. Repacks two `uint64_t` back to `uint64x2_t` via `vld1q_u64`.

Compiler auto-interleaves the two lane muls since they have no dependency.

## Results (14 threads, single-iter real-time)

| Benchmark | Scalar | **NEON 1e** | Phase 1d | Δ 1e vs 1d | Δ 1e vs scalar |
|-----------|--------|--------------|----------|------------|-----------------|
| POSEIDON_BENCH_FULL | 232.8 MiB/s | **276.1** | 247.6 | **+11.5%** | **+18.6%** |
| POSEIDON_BENCH | 247.6 | **268.5** | 240.2 | +11.8% | +8.4% |
| LINEAR_HASH_BENCH | 166.8 | **176.7** | 163.8 | +7.9% | +6.0% |
| MERKLETREE_BENCH | 151.6 | **172.8** | 161.2 | +7.2% | **+14.0%** |

### Mul microbenchmark (isolated)

| Variant | Throughput | Per-pair ns |
|---------|-----------|-------------|
| Scalar `__uint128_t` | 160.8 M pairs/s | 6.22 |
| NEON 1e (scalar-per-lane) | 148.9 M pairs/s | 13.43 |

The isolated microbench shows NEON SLIGHTLY SLOWER because there's no other NEON work to hide latency. But in the real pipeline (Poseidon with its mix of add/mul/load/store), the scalar-mul path wins big because it parallelizes with NEON ops on different execution units.

## Cross-platform comparison

| Benchmark | M4 Pro NEON 1e | Xeon D-2141I AVX2 | Ratio |
|-----------|-----------------|---------------------|-------|
| POSEIDON_BENCH_FULL | **276.1 MiB/s** | 293.8 | **0.94×** |
| POSEIDON_BENCH | 268.5 | 292.0 | **0.92×** |
| LINEAR_HASH_BENCH | 176.7 | 189.8 | 0.93× |
| MERKLETREE_BENCH | 172.8 | 182.3 | **0.95×** |

**M4 Pro NEON is now within 5-8% of Xeon AVX2** on every Poseidon/merkle workload, closed from the ~15% gap in Phase 1d. MERKLETREE especially: 0.95× of AVX2 on a CPU with half the SIMD lane count and no 64×64→128 single-instruction multiply.

## Why this works on Apple Silicon specifically

- **Dual integer-mul pipes** with low latency on `mul`/`umulh`.
- **Out-of-order execution** interleaves the two lane muls without hint.
- **NEON ALU** stays free for add/shift/load/store in parallel.

On x86-64 AVX2, the `_mm256_mul_epu32` path is competitive per-lane but only 32-bit wide (needs shuffles for 64×64). On ARM64, the scalar 64×64 mul exists natively, so it's strictly better to use it.

## Evolution of Phase 1 (M4 Pro POSEIDON_BENCH_FULL @ 14 threads)

| Phase | Throughput | Delta |
|-------|-----------|-------|
| Scalar baseline | ~232 MiB/s | — |
| 1a (NEON pow7+add) | ~237 | +2% |
| 1b (+ NEON mvp+dot) | ~244 | +5% |
| 1c (+ 2-hash pair) | ~244 | +5% |
| 1d (+ simplified add/sub) | ~247 | +6% |
| **1e (scalar mul per lane)** | **276** | **+19%** |

Phase 1e is the single largest win in Phase 1, and required abandoning the "use NEON everywhere" instinct in favor of the hardware's actual preferences.

## Remaining gap to Xeon AVX2

~5-8%. Closing it needs:
- Inline ASM (force exact instruction sequence; LLVM sometimes re-orders across `asm!` blocks poorly).
- Dual-lane interleaved `asm!` block (Plonky3 does this; explicit control over which mul pipe each op goes to).
- Canonical-form deferral across multiple ops (large refactor; Plonky3 does this too).

Or accept ~95% of AVX2 as excellent for a scalar-port effort and focus on Metal GPU.

## Correctness

- 26/26 gtests pass.
- 14,610 mul/square fuzz pairs: 0 failures.
- Merkletree NEON output bit-exact vs scalar (64-element test).
