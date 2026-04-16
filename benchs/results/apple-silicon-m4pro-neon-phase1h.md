# Goldilocks NEON Phase 1h — Non-Canonical Mul Intermediates

**Date:** 2026-04-15
**System:** macOS 26.3.1 — Apple M4 Pro, 14 cores
**Phase:** 1h — switch Poseidon hot paths to `mul_reduced`/`square_reduced` (non-canonical output, in `[0, 2^64)`). Canonicalize once at the state boundary before store.

## The change

Phase 1g `mul` was:
```cpp
Vec mul(Vec a, Vec b) {
    Vec r = mul_reduced(a, b);           // scalar mul+umulh, branchless reduce
    Vec ge_p = vcgeq_u64(r, P);          // 3 NEON ops
    return vsubq_u64(r, vandq_u64(ge_p, P));
}
```

The last 3 NEON ops (`vcgeq_u64 + vandq_u64 + vsubq_u64`) canonicalize `r` to `[0, P)`. But Poseidon's next op is either another `mul` (accepts non-canonical inputs) or an `add` (Phase 1d, accepts non-canonical inputs). So canonicalization at every mul is wasted work.

Phase 1h: Poseidon calls `mul_reduced` / `square_reduced` directly (non-canonical). Canonicalize ONCE at the state boundary before `N::store` writes to the output array.

Saved per hash:
- ~384 Poseidon muls × 3 NEON ops each = **~1,152 NEON ops/hash**
- Plus register-pressure reduction: the `ge_p_mask` and `sub_amt` temporaries are gone.

## Results (14 threads)

| Benchmark | Scalar | **NEON 1h** | NEON 1g | Δ 1h vs 1g | Δ 1h vs scalar |
|-----------|--------|--------------|---------|------------|-----------------|
| POSEIDON_BENCH_FULL | 244.0 MiB/s | **345.5** | 304.8 | **+13.4%** | **+41.6%** |
| POSEIDON_BENCH | 242.6 | **356.4** | 272.7 | **+30.7%** | **+46.9%** |
| LINEAR_HASH_BENCH | 164.2 | **207.1** | 191.6 | +8.1% | +26.1% |
| MERKLETREE_BENCH | 150.0 | **228.6** | 196.7 | +16.2% | **+52.4%** |

## Cross-platform — M4 Pro NEON dominates Xeon AVX2

| Benchmark | M4 Pro NEON 1h | Xeon D-2141I AVX2 | Ratio |
|-----------|------------------|---------------------|-------|
| **POSEIDON_BENCH** | **356.4 MiB/s** | 292.0 | **1.22×** 🏆 |
| **POSEIDON_BENCH_FULL** | **345.5** | 293.8 | **1.18×** 🏆 |
| **MERKLETREE_BENCH** | **228.6** | 182.3 | **1.25×** 🏆 |
| **LINEAR_HASH_BENCH** | **207.1** | 189.8 | **1.09×** 🏆 |

M4 Pro NEON is now **9-25% faster than Xeon D-2141I AVX2** across all Poseidon/merkle workloads. POSEIDON_BENCH — previously the single laggard at 0.93× AVX2 in Phase 1g — jumped to 1.22× AVX2 (+30.7% over Phase 1g).

## Why this wasn't a win in Phase 1e's first attempt

Phase 1e's first attempt tried this approach but regressed. Root cause: the reduction inside mul had a branch (`if (adj < 0)`) that mispredicted ~50% of the time. Phase 1g's branchless reduction removed that, letting the non-canonical optimization expose its full value.

Lesson: optimizations compose nonlinearly. The SAME change (non-canonical) regressed in one context and won massively in another.

## Correctness

- 26/26 gtests pass.
- 14,610 mul/square fuzz pairs: 0 failures.
- Merkletree NEON bit-exact vs scalar (64-element test).

## Phase 1 evolution — MERKLETREE_BENCH @ 14 threads

| Phase | Value | Change |
|-------|-------|--------|
| Scalar | ~155 MiB/s | baseline |
| 1a | 151 | NEON primitives |
| 1b | 145 | + NEON mvp/dot |
| 1c | 164 | + 2-hash pair |
| 1d | 161 | + simplified add/sub |
| 1e | 173 | + scalar mul per lane |
| 1g | 197 | + branchless reduction |
| **1h** | **229** | **+ non-canonical intermediates** |

**Phase 1h** alone delivered +16% over 1g on MERKLETREE and +30% on POSEIDON_BENCH. Cumulative since Phase 1a: ~50% over scalar.

## What's left

Still below 100% utilization, but we're now 9-25% FASTER than the Xeon AVX2 reference. Remaining ideas:
1. **Pair-hash in tree rebuild / batch** — currently only first-pass linear_hash uses the pair. Extending could give another +5-10% on MERKLETREE/BATCH workloads.
2. **plonky2's MDS 16-bit FFT** — different algorithm; might win or be a wash.
3. **Tree-rebuild-specific optimizations** — the rebuild loop does short hashes where setup cost matters; specialization might help.

Or declare Phase 1 mission accomplished (beats AVX2 on all benchmarks) and pivot to Metal GPU for the next order of magnitude.
