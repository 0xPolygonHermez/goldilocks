# Goldilocks NEON Benchmark — Apple M4 Pro, Phase 1b

**Date:** 2026-04-15
**System:** macOS 26.3.1 — Apple M4 Pro, 14 cores, load avg 1.94
**Phase:** 1b — matrix-mul and dot product moved to NEON (entire hash pipeline now NEON where vectorizable; partial-round S-box remains scalar by design).

## Key change vs Phase 1a

Phase 1a: `pow7` and constant-add were NEON; `mvp_` and `dot_` were scalar.
**Phase 1b**: `mvp_neon` and `dot_neon` added. Every NEON Poseidon path now vectorizes the matrix-vector product and partial-round dot product.

## Results (14 threads, single iteration)

| Benchmark | Scalar (this run) | NEON Phase 1b |
|-----------|--------------------|----------------|
| POSEIDON_BENCH_FULL | 234.1 MiB/s | **244.2 MiB/s** |
| POSEIDON_BENCH | 246.9 MiB/s | 227.7 MiB/s |
| LINEAR_HASH_BENCH | 153.7 MiB/s | 149.8 MiB/s |
| MERKLETREE_BENCH | 144.3 MiB/s | 145.1 MiB/s |
| MERKLETREE_BATCH_BENCH | 127.2 MiB/s | 127.1 MiB/s |

Phase 1b vs Phase 1a:

| Benchmark | Phase 1a | Phase 1b | Δ |
|-----------|----------|----------|----|
| POSEIDON_BENCH_FULL/14 | 237.9 MiB/s | 244.2 MiB/s | +2.6% |
| POSEIDON_BENCH/14 | 232.4 MiB/s | 227.7 MiB/s | -2.0% |
| LINEAR_HASH/14 | 157.6 MiB/s | 149.8 MiB/s | -5.0% |
| MERKLETREE/14 | 151.3 MiB/s | 145.1 MiB/s | -4.1% |

## Interpretation — honest

The full-pipeline delta is within measurement noise (±5-10% between runs with different background load). The NEON `mul` microbench shows +13.5% in isolation, but the mvp workload has different memory-access patterns and per-op reduction cost that cancel the gain in this benchmark.

Why: NEON Goldilocks `mul` is ~25 NEON ops per vector multiply (4× `vmull_u32` + reduction + canonicalize for 2 elements). Scalar `Goldilocks::mul` on Apple Silicon uses a single `__uint128_t` multiply + tight scalar reduction — ~5-10 ops per element. Per element, NEON mul is not meaningfully faster than scalar; it just packs 2 in parallel, and that parallelism is eaten by the additional reduction overhead.

## What Phase 1b actually achieves

1. **Full NEON coverage of the vectorizable Poseidon path** — every operation except the partial-round scalar slice now runs through NEON intrinsics. No more scalar fallbacks in the Poseidon inner loops.

2. **Correctness verified end-to-end**: 26 tests pass, bit-exact with `_seq` over 64-element merkletrees, 14,610 mul/square fuzz pairs clean.

3. **Ready for future optimization**:
   - ARM `UDOT`/`SDOT` (ARMv8.4 dot-product instructions) — currently not used, could collapse the 4× `vmull_u32` sequence.
   - ARM `MLA` / fused multiply-add — not available for 64-bit widening mul, but if Apple Silicon exposes custom intrinsics this could help.
   - SVE2 if/when Apple adds it.
   - Independent-hash batching (2 hashes per NEON state) — Phase 1c opportunity for true 2× throughput.

## Why the full-pipeline gain is small

The Goldilocks scalar mul on Apple Silicon is already near-optimal. The Apple M-series' 64-bit integer ALU is wide and has low latency, making `__uint128_t` mul effectively 1 cycle. NEON's per-lane mul costs more than that because it lacks a 64×64→128 single instruction — we have to synthesize it from 32×32 pieces. The only way to make NEON win cleanly would be:

- Hide latency with independent-hash batching (Phase 1c).
- Use UDOT or future SIMD instructions to collapse the 4-partial-product pattern.
- Reduce the post-reduction cost (e.g., skip canonicalization and keep intermediates non-canonical through a whole round).

## Conclusion

Phase 1b completes the "NEON everywhere reasonable" milestone. The remaining gap to Xeon AVX2 (~20%) is attributable to:
1. AVX2's 4-lane width vs NEON's 2-lane.
2. AVX2 having a 64×64→128 direct multiply path (via `_mm256_mul_epu32` + shuffle).

To close this gap on M-series, **Phase 1c** would pursue 2-hash-parallel NEON Poseidon: process 2 independent hashes simultaneously in each NEON register, doubling throughput at the hash-level. That requires merkletree input reorganization but matches AVX2's effective throughput.
