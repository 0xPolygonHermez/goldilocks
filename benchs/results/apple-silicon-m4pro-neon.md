# Goldilocks NEON Benchmark — Apple Silicon M4 Pro (Phase 1)

**Date:** 2026-04-15
**System:** macOS 26.3.1 — Apple M4 Pro (14 cores)
**Compiler:** Apple clang 17.0.0
**Build:** `-O3 -std=c++17` (NEON auto-enabled on AArch64)
**Phase:** 1 — NEON vectorization of field ops + pow7/add in Poseidon. Matrix multiplies remain scalar.

## Results (selected NEON-vs-scalar comparison, 14 threads)

| Benchmark | Scalar | NEON | Delta |
|-----------|--------|------|-------|
| POSEIDON_BENCH_FULL | 236.4 MiB/s | 237.9 MiB/s | **+0.6%** |
| POSEIDON_BENCH | 222.6 MiB/s | 232.4 MiB/s | **+4.4%** |
| LINEAR_HASH_BENCH | 152.2 MiB/s | 157.6 MiB/s | **+3.5%** |
| MERKLETREE_BENCH | 143.5 MiB/s | 151.3 MiB/s | **+5.4%** |
| MERKLETREE_BATCH_BENCH | 88.1 MiB/s | 95.3 MiB/s | **+8.2%** |

Per-op rates (µs/hash, 14 threads):

| Benchmark | Scalar | NEON |
|-----------|--------|------|
| POSEIDON_BENCH_FULL | 2.71 µs | 2.69 µs |
| POSEIDON_BENCH | 2.88 µs | 2.76 µs |
| LINEAR_HASH_BENCH | 2.81 µs | 2.71 µs |
| MERKLETREE_BENCH | 2.98 µs | 2.82 µs |
| MERKLETREE_BATCH_BENCH | 4.85 µs | 4.48 µs |

### Standalone NEON `mul` microbenchmark

| | Throughput | Rate |
|---|-----------|------|
| Scalar 64×64 mul | 166.2 M pairs/s | 6.02 ns/pair |
| NEON 64×64 mul (2 pairs) | 188.6 M pairs/s | 5.30 ns/pair |

**+13.5%** on the mul operation in isolation.

---

## Interpretation

NEON vectorizes `pow7` (6 `vmull_u32`-based squares and 6 muls per full round across 6 NEON 128-bit registers for a 12-element Poseidon state) and constant-add (6 NEON adds per round). But the 12×12 MDS matrix-vector product remains scalar — it dominates the Poseidon runtime. With ~40% of time in NEON-vectorizable `pow7`+add and ~50% in scalar `mvp_`, the observed ~5% speedup tracks.

### Phase 1b follow-up opportunities

1. **NEON `mmult_4x12`**: vectorize the 12-element matrix-vector product via `permute_lanes` + `vmlal_u32` dot-products. Expected uplift: 3-5× on the remaining 50%.
2. **NEON `dot_avx` equivalent**: vectorize the partial-round dot product. Small win but closes the gap.
3. **Batch merkletree entries**: process 2 independent hashes per NEON state, mirroring AVX2's single-hash-4-wide. Requires merkletree input re-organization.

---

## Cross-platform comparison (14 threads, selected)

| Benchmark | M4 Pro scalar | M4 Pro NEON | Xeon D-2141I scalar | Xeon D-2141I AVX2 |
|-----------|---------------|-------------|---------------------|-------------------|
| POSEIDON_BENCH_FULL | 236.4 MiB/s | 237.9 MiB/s | 118.6 MiB/s | 293.8 MiB/s |
| LINEAR_HASH_BENCH | 152.2 MiB/s | 157.6 MiB/s | 75.0 MiB/s | 189.8 MiB/s |
| MERKLETREE_BENCH | 143.5 MiB/s | 151.3 MiB/s | 73.2 MiB/s | 182.3 MiB/s |

M4 Pro NEON is **~2.0× Xeon scalar** and **~0.81× Xeon AVX2**. Closing the remaining gap requires Phase 1b (matrix-mul NEON).

---

## Test count

| Platform | Tests pass |
|----------|-----------|
| x86-64 + AVX2 | 30 tests |
| ARM64 (Apple Silicon, NEON) | **26 tests** (17 scalar + 9 NEON) |
| Pure scalar ARM64 | 17 tests |

NEON-qualified tests: `add_neon`, `sub_neon`, `mul_neon`, `square_neon`, `poseidon_neon`, `poseidon_full_neon`, `linear_hash_neon`, `merkletree_neon`, `merkletree_batch_neon`. All bit-exact against scalar reference.

## Fuzz verification

- `tests/check_neon_mul.cpp`: 12,100 mul + 2,510 square pairs (edge × edge, random, canonical × canonical). **0 failures.**
- `tests/check_simd_traits_compile.cpp`: all 15 SIMD primitives pass smoke test.
- Merkletree parity: 64-element tree, NEON vs scalar, 0 mismatches.

## How to reproduce

```bash
# Build
make testcpu benchcpu check_neon_mul check_neon_mul_bench

# Regression
./testcpu                                           # 26 tests
./check_neon_mul                                    # 14 610 fuzz pairs
./check_neon_mul_bench                              # microbench

# Full benchmark
./benchcpu --benchmark_min_time=0.1s 2>&1 | \
    tee benchs/results/apple-silicon-m4pro-neon-raw.log
```
