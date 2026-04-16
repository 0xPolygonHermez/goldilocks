# Goldilocks NEON Phase 1c — 2-Hash-Parallel Poseidon

**Date:** 2026-04-15
**System:** macOS 26.3.1 — Apple M4 Pro, 14 cores
**Phase:** 1c — 2 independent Poseidon hashes per NEON state (lane 0 = hash A, lane 1 = hash B). `merkletree_neon` now pairs adjacent rows in its first pass, processing 2 hashes per NEON iteration.

## What's new

- `hash_full_result_neon_2(state_A, input_A, state_B, input_B)` — full Poseidon across 12 NEON registers, lane 0 is hash A, lane 1 is hash B. All `pow7`, `add`, `mvp`, `dot` naturally process both hashes in parallel.
- `mvp_neon_2` — 12×12 matrix-vector for the paired layout.
- `linear_hash_neon_pair(out_A, in_A, out_B, in_B, size)` — sponge chain for 2 paired rows.
- `merkletree_neon`'s first-pass loop iterates `i += 2` and calls `linear_hash_neon_pair`; odd-row fallback handles the tail.

## Microbench (single-thread, isolated)

Correctness verified first: `hash_full_result_neon_2` output matches 2x `hash_full_result_neon` bit-for-bit.

| Variant | Hashes/s | Per-hash µs |
|---------|----------|-------------|
| Single-hash NEON | 0.26 M | 3.85 |
| **2-hash NEON** | **0.28 M** | **3.57** |

**+7.7% throughput** in isolation. Less than 2× because Apple Silicon OOO cores already extract ILP from the single-hash version; the 2-hash version mostly reduces per-hash instruction count rather than hiding latency.

## Full pipeline benchmark (14 threads)

| Benchmark | Scalar | NEON Phase 1b | **NEON Phase 1c** | 1c vs 1b | 1c vs scalar |
|-----------|--------|---------------|-------------------|----------|--------------|
| MERKLETREE_BENCH | 155.4 MiB/s | 145.1 MiB/s | **164.2 MiB/s** | **+13.1%** | **+5.7%** |
| LINEAR_HASH_BENCH | 155.0 MiB/s | 149.8 MiB/s | 153.8 MiB/s | +2.7% | −0.8% |
| MERKLETREE_BATCH_BENCH | 133.7 MiB/s | 127.1 MiB/s | 131.0 MiB/s | +3.1% | −2.0% |

**MERKLETREE_BENCH is the headline**: NEON now wins against scalar by a real margin (+5.7%). Previous phases (1a, 1b) were within noise. The first-pass linear-hash on merkle input is the dominant cost, and row-pairing maps to it naturally.

LINEAR_HASH_BENCH doesn't benefit because it's a single sponge chain — no natural pairing at the call site (would need external batching).

MERKLETREE_BATCH_BENCH shows modest gain — batching already groups inputs; pair integration in the first-pass is less impactful.

## Cross-platform comparison (14 threads M4 Pro vs 16 threads Xeon D-2141I)

| Benchmark | M4 Pro NEON 1c | Xeon AVX2 | Ratio |
|-----------|-----------------|-----------|-------|
| MERKLETREE_BENCH | **164.2 MiB/s** | 182.3 MiB/s | **0.90×** |
| LINEAR_HASH_BENCH | 153.8 MiB/s | 189.8 MiB/s | 0.81× |
| POSEIDON_BENCH_FULL (earlier run) | ~244 MiB/s | 293.8 MiB/s | 0.83× |

M4 Pro NEON is now within **10% of Xeon AVX2** on MERKLETREE_BENCH — the headline workload. AVX2 retains an edge because its 4-lane width + direct 64×64→128 multiply path has no single-instruction NEON equivalent.

## Path to closing the remaining gap

1. **ARM `UDOT` / `SDOT`** (ARMv8.4-A dot-product instructions) — Apple M4 supports these. Could collapse the 4× `vmull_u32` multiply pattern into 2 dot-products, likely closing the rest of the gap.
2. **Matrix layout for wider vectorization** — current `mvp_neon_2` processes the matrix row-by-row; a blocked layout that hits L1 cache better could help on larger merkletrees.
3. **Canonical-form deferral** — accumulate multiple muls before canonicalizing. Requires algorithmic redesign but can amortize the reduction cost.

## Test correctness

- 26 gtest cases pass (17 scalar + 9 NEON) on Apple Silicon.
- NEON merkletree bit-exact with scalar over 64-element test.
- 14,610 mul/square fuzz pairs, 0 failures.
