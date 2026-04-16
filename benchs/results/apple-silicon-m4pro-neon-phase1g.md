# Goldilocks NEON Phase 1g — Branchless Reduction

**Date:** 2026-04-15
**System:** macOS 26.3.1 — Apple M4 Pro, 14 cores
**Phase:** 1g — remove the `if (adj < 0)` branch from the Goldilocks reduction. Use `s3 + (carry2+carry3)*EPS - borrow*EPS` directly.

## The change

Phase 1e's reduction had:
```cpp
int adj = (int)carry2 + (int)carry3 - (int)borrow;
uint64_t r = s3 + (uint64_t)adj * EPS;
if (adj < 0) r = s3 - EPS;         // <-- mispredicts
```

On the real Poseidon workload (not the fuzz harness), `adj < 0` triggers ~50% of the time — alternating between branches induces pipeline flushes at ~5-10 cycles each.

Phase 1g refactors to pure arithmetic:
```cpp
uint64_t pos_cq = (carry2 + carry3) * EPS;   // 0, 1, or 2 * EPS
uint64_t neg_cq = borrow * EPS;               // 0 or 1 * EPS
return s3 + pos_cq - neg_cq;
```

No branches, no conditional moves needed. Compiler emits pure `add/sub/mul` on the integer pipes.

## Results (14 threads)

| Benchmark | Scalar | **NEON 1g** | Phase 1e | Δ 1g vs 1e | Δ 1g vs scalar |
|-----------|--------|--------------|----------|------------|-----------------|
| POSEIDON_BENCH_FULL | 224.5 MiB/s | **304.8** | 276.1 | **+10.4%** | **+35.8%** |
| POSEIDON_BENCH | 237.4 | **272.7** | 268.5 | +1.6% | +14.9% |
| LINEAR_HASH_BENCH | 162.4 | **191.6** | 176.7 | +8.4% | **+17.9%** |
| MERKLETREE_BENCH | 144.8 | **196.7** | 172.8 | +13.8% | **+35.8%** |

## Cross-platform — M4 Pro NEON now wins

| Benchmark | M4 Pro NEON 1g | Xeon D-2141I AVX2 | Ratio |
|-----------|------------------|---------------------|-------|
| POSEIDON_BENCH_FULL | **304.8 MiB/s** | 293.8 | **1.04×** 🏆 |
| MERKLETREE_BENCH | **196.7** | 182.3 | **1.08×** 🏆 |
| LINEAR_HASH_BENCH | **191.6** | 189.8 | **1.01×** 🏆 |
| POSEIDON_BENCH | 272.7 | 292.0 | 0.93× |

M4 Pro NEON is now **faster than the Xeon D-2141I AVX2 server** on 3 of 4 Poseidon/merkle workloads.

## Why branchless wins

The branch `if (adj < 0)` fires when the reduction path goes through a borrow without any corresponding carry — which happens on approximately half the Goldilocks multiplications in real inputs. The branch predictor can't learn a pattern, so it mispredicts ~50% of the time.

Each misprediction on Apple M-series is ~8-12 cycles of pipeline flush. With 384+ multiplies per Poseidon hash, ~192 of which mispredicted, that's ~2000+ wasted cycles per hash.

Branchless form avoids this entirely. The compiler emits straightforward `umull/add/sub` that flow through the integer pipes without any speculation.

## Phase 1 evolution summary (MERKLETREE_BENCH @ 14 threads, MiB/s)

| Phase | Value | Key change |
|-------|-------|------------|
| Scalar | ~155 | baseline |
| 1a | 151 | NEON field primitives |
| 1b | 145 | + mvp/dot NEON |
| 1c | 164 | + 2-hash pair |
| 1d | 161 | + simplified add/sub |
| 1e | 173 | + scalar mul per lane (Plonky3) |
| **1g** | **197** | **+ branchless reduction** |

Phase 1g alone delivered +14% over 1e, pushing past the Xeon AVX2 ceiling.

## Correctness

- 26/26 gtests pass.
- 14,610 mul/square fuzz pairs: 0 failures.
- Merkletree NEON bit-exact vs scalar (64-element test).

## What's left

Remaining opportunities:
1. **canonical-form deferral across chained muls** — skip even the non-canonical wrap correction in specific contexts where a subsequent canonicalize absorbs it. Est. +5-10%.
2. **plonky2's MDS 16-bit-limb FFT** — different matrix-mul algorithm. Est. unknown, possibly sizable.
3. **Inline ASM dual-lane** — we tried in Phase 1f and found clang 17's scheduler already handles it well on this silicon. Skipped.

Or: declare Phase 1 done and move to Metal GPU for orders-of-magnitude gains.
