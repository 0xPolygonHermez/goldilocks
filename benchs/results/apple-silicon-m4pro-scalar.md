# Goldilocks CPU Benchmark — Apple Silicon Scalar Path

**Date:** 2026-04-14
**System:** macOS 26.3.1 (Darwin 25.3.0) — Apple M4 Pro (14 cores)
**Memory:** 48 GB
**Compiler:** Apple clang 17.0.0 (clang-1700.6.4.2)
**Build:** `-O3 -std=c++17` (no SIMD; `SIMD_FLAGS` empty on ARM64)
**Path exercised:** Portable `__uint128_t` scalar arithmetic + `_seq` merkle/poseidon
**Commit:** baseline after Parts 1–7 of the Apple Silicon port

**Notes:**
- NEON intrinsics are *not* used — this is the pure scalar baseline.
- AVX2-specific benchmarks are compiled out via `#ifdef GOLDILOCKS_HAS_AVX2`.
- `hw.ncpu = 14` performance+efficiency cores; OpenMP uses up to 14.
- Benchmark runner: `--benchmark_min_time=0.1s` (single iteration of each large workload).
- `FFT_SIZE = 1 << 23` (8,388,608) for NTT/LDE/ExtendedPol.
- `NCOLS_HASH = 128`, `NROWS_HASH = 1 << 6` (64) for Merkle/linear-hash benchmarks.

---

## Results

```
Running ./benchcpu
Run on (14 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x14)
Load Average: 2.53, 2.47, 2.52
----------------------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------
POSEIDON_BENCH_FULL/7/real_time        1154206 us      1153997 us            1 BytesProcessed=166.348Mi/s Rate=3.85258us
POSEIDON_BENCH_FULL/14/real_time        780737 us       678197 us            1 BytesProcessed=245.921Mi/s Rate=2.60599us
POSEIDON_BENCH/7/real_time             1174082 us      1160128 us            1 BytesProcessed=163.532Mi/s Rate=3.91892us
POSEIDON_BENCH/14/real_time             837159 us       647486 us            1 BytesProcessed=229.347Mi/s Rate=2.79432us
LINEAR_HASH_BENCH/7/real_time         74896757 us     74698948 us            1 BytesProcessed=109.377Mi/s Rate=3.90617us
LINEAR_HASH_BENCH/14/real_time        51352085 us     46310365 us            1 BytesProcessed=159.526Mi/s Rate=2.67822us
MERKLETREE_BENCH/7/real_time          79154719 us     79067722 us            1 BytesProcessed=103.494Mi/s Rate=4.12824us
MERKLETREE_BENCH/14/real_time         53348654 us     48326708 us            1 BytesProcessed=153.556Mi/s Rate=2.78235us
MERKLETREE_BATCH_BENCH/7/real_time    88428586 us     88293470 us            1 BytesProcessed=92.6397Mi/s Rate=4.61191us
MERKLETREE_BATCH_BENCH/14/real_time   59538413 us     54328801 us            1 BytesProcessed=137.592Mi/s Rate=3.10517us
NTT_BENCH/7/real_time                     19.0 s          19.0 s             1
NTT_BENCH/14/real_time                    12.5 s          11.8 s             1
NTT_BLOCK_BENCH/7/real_time               8.42 s          7.48 s             1
NTT_BLOCK_BENCH/14/real_time              5.00 s          4.21 s             1
LDE_BENCH/7/real_time                     44.7 s          44.7 s             1
LDE_BENCH/14/real_time                    26.9 s          24.8 s             1
LDE_BLOCK_BENCH/7/real_time               23.0 s          17.8 s             1
LDE_BLOCK_BENCH/14/real_time              15.4 s          13.6 s             1
EXTENDEDPOL_BENCH/7/real_time             28.8 s          24.4 s             1
EXTENDEDPOL_BENCH/14/real_time            17.6 s          13.7 s             1
```

---

## Summary

### Throughput (real time, 14 threads)

| Benchmark | Throughput | Per-op rate |
|-----------|-----------|-------------|
| POSEIDON_BENCH_FULL | 245.9 MiB/s | 2.606 µs/hash |
| POSEIDON_BENCH | 229.3 MiB/s | 2.794 µs/hash |
| LINEAR_HASH_BENCH | 159.5 MiB/s | 2.678 µs/hash |
| MERKLETREE_BENCH | 153.6 MiB/s | 2.782 µs/hash |
| MERKLETREE_BATCH_BENCH | 137.6 MiB/s | 3.105 µs/hash |

### Wall-time (14 threads, FFT_SIZE = 2²³)

| Benchmark | Time |
|-----------|------|
| NTT_BENCH | 12.5 s |
| NTT_BLOCK_BENCH | 5.00 s |
| LDE_BENCH | 26.9 s |
| LDE_BLOCK_BENCH | 15.4 s |
| EXTENDEDPOL_BENCH | 17.6 s |

### Scaling from 7 → 14 threads

| Benchmark | Speedup |
|-----------|---------|
| POSEIDON_BENCH_FULL | 1.48× |
| LINEAR_HASH_BENCH | 1.46× |
| MERKLETREE_BENCH | 1.48× |
| NTT_BENCH | 1.52× |
| LDE_BENCH | 1.66× |

Near-linear scaling is capped by memory bandwidth on NTT/LDE (FFT-bound) and by Apple Silicon's 10 P-cores + 4 E-cores layout.
