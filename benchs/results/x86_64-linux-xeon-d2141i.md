# Goldilocks CPU Benchmark — Xeon D-2141I (x86-64)

**Date:** 2026-04-14 (20:05 UTC)
**System:** Ubuntu Linux 6.8.0-101-generic — Intel Xeon D-2141I (8 cores / 16 threads, 2.20 GHz base, 3.00 GHz max)
**Memory:** 64 GB
**Cache:** L1d 32 KiB × 8, L1i 32 KiB × 8, L2 1 MiB × 8, L3 11 MiB
**ISA flags present:** AVX, AVX2, AVX-512F/DQ/CD/BW/VL, BMI1/2, AES
**Compiler:** g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1)
**Build:** `-O3 -std=c++17 -mavx2 -fopenmp`
**Path exercised:** Scalar (portable `__uint128_t`) AND AVX2 variants
**Commit:** `apple-silicon-port` branch on `eduadiez/goldilocks`

**Caveats:**
- ⚠️ `CPU scaling is enabled` — real-time measurements are noisy and CPU may throttle.
- ⚠️ `Library was built as DEBUG` — this is the distro-packaged libbenchmark's framework overhead; measured workload code is still `-O3`.
- ⚠️ Hyperthreading is on (16 threads on 8 physical cores). `/8` uses 1 thread per physical core; `/16` oversubscribes.
- AVX-512 benchmarks are NOT registered because the build was not compiled with `-D__AVX512__`.

---

## Results

```
Benchmark                                        Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------
POSEIDON_BENCH_FULL/8/real_time            1898394 us      1807880 us            1 BytesProcessed=101.138Mi/s Rate=7.2418us
POSEIDON_BENCH_FULL/16/real_time           1619216 us      1529634 us            1 BytesProcessed=118.576Mi/s Rate=6.17682us
POSEIDON_BENCH_FULL_AVX/8/real_time         875236 us       875199 us            1 BytesProcessed=219.369Mi/s Rate=3.33876us
POSEIDON_BENCH_FULL_AVX/16/real_time        653524 us       632809 us            1 BytesProcessed=293.792Mi/s Rate=2.493us
POSEIDON_BENCH/8/real_time                 1628304 us      1628096 us            1 BytesProcessed=117.914Mi/s Rate=6.21149us
POSEIDON_BENCH/16/real_time                1721879 us      1573956 us            1 BytesProcessed=111.506Mi/s Rate=6.56845us
POSEIDON_BENCH_AVX/8/real_time              848251 us       846451 us            1 BytesProcessed=226.348Mi/s Rate=3.23582us
POSEIDON_BENCH_AVX/16/real_time             657628 us       625946 us            1 BytesProcessed=291.958Mi/s Rate=2.50865us
LINEAR_HASH_BENCH/8/real_time            106210173 us    106123181 us            1 BytesProcessed=77.1301Mi/s Rate=6.33062us
LINEAR_HASH_BENCH/16/real_time           109209856 us     97431065 us            1 BytesProcessed=75.0115Mi/s Rate=6.50941us
LINEAR_HASH_BENCH_AVX/8/real_time         47688525 us     47564695 us            1 BytesProcessed=171.781Mi/s Rate=2.84246us
LINEAR_HASH_BENCH_AVX/16/real_time        43156679 us     39231190 us            1 BytesProcessed=189.82Mi/s Rate=2.57234us
MERKLETREE_BENCH/8/real_time             113631897 us    113110830 us            1 BytesProcessed=72.0924Mi/s Rate=6.77299us
MERKLETREE_BENCH/16/real_time            111873427 us    104445425 us            1 BytesProcessed=73.2256Mi/s Rate=6.66817us
MERKLETREE_BENCH_AVX/8/real_time          49346802 us     48859829 us            1 BytesProcessed=166.009Mi/s Rate=2.9413us
MERKLETREE_BENCH_AVX/16/real_time         44947663 us     41742311 us            1 BytesProcessed=182.256Mi/s Rate=2.67909us
MERKLETREE_BATCH_BENCH/8/real_time       124302223 us    122930476 us            1 BytesProcessed=65.9039Mi/s Rate=7.40899us
MERKLETREE_BATCH_BENCH/16/real_time      124755674 us    116676270 us            1 BytesProcessed=65.6643Mi/s Rate=7.43602us
MERKLETREE_BATCH_BENCH_AVX/8/real_time    56454414 us     56024984 us            1 BytesProcessed=145.108Mi/s Rate=3.36494us
MERKLETREE_BATCH_BENCH_AVX/16/real_time   50338709 us     46901209 us            1 BytesProcessed=162.738Mi/s Rate=3.00042us
NTT_BENCH/8/real_time                         45.9 s          45.8 s             1
NTT_BENCH/16/real_time                        34.4 s          32.4 s             1
NTT_BLOCK_BENCH/8/real_time                   8.94 s          8.92 s             1
NTT_BLOCK_BENCH/16/real_time                  8.74 s          8.07 s             1
LDE_BENCH/8/real_time                         94.5 s          94.4 s             1
LDE_BENCH/16/real_time                        71.3 s          66.7 s             1
LDE_BLOCK_BENCH/8/real_time                   29.5 s          26.4 s             1
LDE_BLOCK_BENCH/16/real_time                  29.1 s          25.4 s             1
```

Note: `EXTENDEDPOL_BENCH` rows are missing from the captured log (run may have been interrupted).

---

## Summary — Xeon D-2141I (16 threads, real time)

| Benchmark | Scalar | AVX2 | AVX2 speedup |
|-----------|--------|------|--------------|
| POSEIDON_BENCH_FULL | 118.6 MiB/s (6.18 µs) | 293.8 MiB/s (2.49 µs) | **2.48×** |
| POSEIDON_BENCH | 111.5 MiB/s (6.57 µs) | 292.0 MiB/s (2.51 µs) | **2.62×** |
| LINEAR_HASH_BENCH | 75.0 MiB/s (6.51 µs) | 189.8 MiB/s (2.57 µs) | **2.53×** |
| MERKLETREE_BENCH | 73.2 MiB/s (6.67 µs) | 182.3 MiB/s (2.68 µs) | **2.49×** |
| MERKLETREE_BATCH_BENCH | 65.7 MiB/s (7.44 µs) | 162.7 MiB/s (3.00 µs) | **2.48×** |

AVX2 gives a consistent ~2.5× throughput gain on this CPU, matching the 4-lane × single-issue theoretical ceiling minus overhead.

---

## Cross-Platform Comparison — Apple M4 Pro vs Xeon D-2141I

Using the highest thread count measured on each platform (M4 Pro: 14 threads = all cores; Xeon: 16 threads = all logical CPUs, 8 physical + HT).

### Poseidon hash throughput (MiB/s, higher is better)

| Benchmark | M4 Pro scalar | Xeon scalar | Xeon AVX2 | M4 Pro vs Xeon scalar | M4 Pro vs Xeon AVX2 |
|-----------|---------------|-------------|-----------|------------------------|----------------------|
| POSEIDON_BENCH_FULL | **245.9** | 118.6 | 293.8 | 2.07× faster | 0.84× |
| POSEIDON_BENCH | **229.3** | 111.5 | 292.0 | 2.06× faster | 0.79× |
| LINEAR_HASH_BENCH | **159.5** | 75.0 | 189.8 | 2.13× faster | 0.84× |
| MERKLETREE_BENCH | **153.6** | 73.2 | 182.3 | 2.10× faster | 0.84× |
| MERKLETREE_BATCH_BENCH | **137.6** | 65.7 | 162.7 | 2.09× faster | 0.85× |

### Per-hash latency (µs, lower is better)

| Benchmark | M4 Pro | Xeon scalar | Xeon AVX2 |
|-----------|--------|-------------|-----------|
| POSEIDON_BENCH_FULL | 2.61 | 6.18 | 2.49 |
| POSEIDON_BENCH | 2.79 | 6.57 | 2.51 |
| LINEAR_HASH_BENCH | 2.68 | 6.51 | 2.57 |
| MERKLETREE_BENCH | 2.78 | 6.67 | 2.68 |
| MERKLETREE_BATCH_BENCH | 3.11 | 7.44 | 3.00 |

### NTT / LDE wall time (seconds, lower is better)

| Benchmark | M4 Pro scalar | Xeon scalar (no AVX variant exists) | Ratio |
|-----------|---------------|--------------------------------------|-------|
| NTT_BENCH | 12.5 s | 34.4 s | **2.75× faster** |
| NTT_BLOCK_BENCH | 5.00 s | 8.74 s | **1.75× faster** |
| LDE_BENCH | 26.9 s | 71.3 s | **2.65× faster** |
| LDE_BLOCK_BENCH | 15.4 s | 29.1 s | **1.89× faster** |

Note: NTT/LDE paths only have scalar implementations in both builds, so these are apples-to-apples scalar-vs-scalar comparisons.

---

## Takeaways

1. **Apple M4 Pro's scalar path beats Xeon D-2141I's scalar path by ~2.1×** across all Poseidon/Merkle benchmarks. The portable `__uint128_t` code path is highly efficient on ARM64 with wide integer ALUs.
2. **Apple M4 Pro scalar reaches ~84% of Xeon D-2141I AVX2 throughput** with no SIMD at all — strong motivation for implementing Phase 1 (NEON) from the roadmap.
3. **AVX2 gives ~2.5× speedup on Xeon** — in line with 4-lane SIMD minus overhead. If NEON (2-lane on ARM64) delivers the expected ~1.8×, M4 Pro with NEON should comfortably outpace Xeon AVX2 on this workload.
4. **NTT/LDE are FFT-bound** — M4 Pro wins by ~2.6× on scalar, reflecting its larger memory bandwidth and cache hierarchy (14 cores vs. 8 physical + HT on Xeon D-2141I).
5. **Xeon caveats:** the measured run has CPU scaling enabled (throttles mid-run), debug-framework libbenchmark overhead, and hyperthreading contention on `/16`. A pinned, `governor=performance` run on the Xeon would likely tighten the scalar numbers by 10-20% but not close the gap with M4 Pro scalar.

---

## How to re-run

See `benchs/results/REPRODUCE.md`. The exact commands used to produce this log:

```bash
make benchcpu
./benchcpu --benchmark_min_time=0.1s 2>&1 | tee benchs/results/x86_64-linux-raw.log
```

To reduce noise on the Xeon, consider:
```bash
sudo cpupower frequency-set --governor performance
taskset -c 0-7 ./benchcpu --benchmark_min_time=0.1s ...
```
