# Reproducing the Goldilocks CPU Benchmarks

This document gives step-by-step instructions to build and run `benchcpu` and save results under `benchs/results/`. Both macOS ARM64 and Linux x86-64 paths are covered; the build auto-detects the platform.

---

## 0. Prerequisites

You need:
- A C++17 compiler (Apple clang 14+ on macOS, g++ 10+ or clang 14+ on Linux)
- GNU Make
- [Google Benchmark](https://github.com/google/benchmark), [GoogleTest](https://github.com/google/googletest), [GMP](https://gmplib.org/), OpenMP

### 0.1 macOS (Apple Silicon or Intel)

```bash
# Homebrew must be installed: https://brew.sh
brew install gmp libomp googletest google-benchmark
```

`libomp` is keg-only; the Makefile picks it up via `brew --prefix libomp`.

### 0.2 Linux (x86-64)

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential libgmp-dev libgmpxx4ldbl \
    libomp-dev libgtest-dev libbenchmark-dev
```

Fedora/RHEL:
```bash
sudo dnf install -y gcc-c++ make gmp-devel libomp-devel gtest-devel \
    benchmark-devel
```

---

## 1. Clone and prepare

```bash
git clone <this-repo> goldilocks && cd goldilocks
git log --oneline -1    # sanity-check you're on the expected commit
mkdir -p benchs/results
```

---

## 2. Build `benchcpu`

```bash
make benchcpu
```

**Expected output** (trimmed):
- 3 warnings about C++ VLAs (pre-existing, harmless)
- Binary `./benchcpu` produced (roughly 350-500 KB depending on platform)

**If build fails with missing headers** (e.g. `'benchmark/benchmark.h' not found`):
- **macOS**: verify `brew --prefix google-benchmark` returns a path; re-run `brew install google-benchmark`.
- **Linux**: verify `dpkg -l | grep libbenchmark-dev` (Ubuntu) or equivalent.

**To confirm which SIMD path was compiled**:
```bash
# On x86-64 (AVX2 path enabled):
make -n benchcpu | grep -o -- '-mavx2'
# Expected: -mavx2

# On ARM64 (no SIMD, scalar path only):
make -n benchcpu | grep -o -- '-mavx2'
# Expected: empty (SIMD_FLAGS is unset)
```

---

## 3. Run the scalar/seq benchmarks (always available)

Use `--benchmark_min_time=0.1s` to get a single-iteration baseline without long multi-iter averaging:

```bash
./benchcpu --benchmark_min_time=0.1s 2>&1 | tee benchs/results/$(uname -m)-$(uname -s | tr '[:upper:]' '[:lower:]')-raw.log
```

This runs **all** registered benchmarks. On x86-64 that includes `*_AVX` variants; on ARM64 those are compiled out.

To run only scalar benchmarks (skipping AVX on x86-64):
```bash
./benchcpu --benchmark_min_time=0.1s \
    --benchmark_filter='(POSEIDON_BENCH_FULL|POSEIDON_BENCH|LINEAR_HASH_BENCH|MERKLETREE_BENCH|MERKLETREE_BATCH_BENCH|NTT_BENCH|NTT_BLOCK_BENCH|LDE_BENCH|LDE_BLOCK_BENCH|EXTENDEDPOL_BENCH)/' \
    2>&1 | tee benchs/results/$(uname -m)-scalar.log
```

To run only AVX benchmarks on x86-64:
```bash
./benchcpu --benchmark_min_time=0.1s \
    --benchmark_filter='.*_AVX' \
    2>&1 | tee benchs/results/$(uname -m)-avx2.log
```

---

## 4. Collect system metadata

Append system info to your result log so the numbers are comparable across machines:

### macOS

```bash
{
  echo "=== System ==="
  sw_vers
  sysctl -n machdep.cpu.brand_string
  sysctl -n hw.ncpu hw.memsize
  clang++ --version | head -1
} >> benchs/results/$(uname -m)-$(uname -s | tr '[:upper:]' '[:lower:]')-raw.log
```

### Linux

```bash
{
  echo "=== System ==="
  uname -a
  lscpu | head -20
  grep MemTotal /proc/meminfo
  g++ --version | head -1
} >> benchs/results/$(uname -m)-linux-raw.log
```

---

## 5. Save a structured summary

After running the raw benchmarks, write a Markdown summary that captures the table of throughputs/times alongside the system info. Use `benchs/results/apple-silicon-m4pro-scalar.md` as a template.

Include, at minimum:
- Date, OS, CPU model, core count, memory
- Compiler version
- Whether AVX2/AVX-512 was enabled (and which variants were compiled in)
- The raw `benchcpu` stdout
- A "Summary" table of throughputs and per-op rates

---

## 6. Quick sanity check (optional)

Before a long benchmark run, confirm the binary works:

```bash
./benchcpu --benchmark_list_tests
# Lists roughly 20-30 benchmarks (fewer on ARM64 where AVX2 variants are guarded out)

./benchcpu --benchmark_filter=POSEIDON_BENCH/14 --benchmark_min_time=0.05s
# Single short run (~1-2 seconds)
```

---

## 7. Expected runtime

Full `--benchmark_min_time=0.1s` run:
- Apple M4 Pro (14 cores, scalar only): ~3 min 30 s
- x86-64 with AVX2 (12-core Xeon/Ryzen, scalar + AVX): ~5-8 min (more benchmarks registered)

Most time is spent in `NTT_BENCH`, `LDE_BENCH`, `EXTENDEDPOL_BENCH` (FFT_SIZE = 2²³ = 8M).

---

## 8. Where to share the result

Commit your `.log` and `.md` files under `benchs/results/` named like:

- `<arch>-<os>-<variant>.log` — raw benchmark stdout
- `<arch>-<os>-<variant>.md` — structured summary with system info

Examples:
- `apple-silicon-m4pro-scalar.md`
- `x86_64-linux-avx2.md`
- `x86_64-linux-avx512.md`

---

## 9. Known issues on Linux x86-64

- If `libgtest.a` is not prebuilt on Ubuntu (only sources under `/usr/src/gtest`), build and install it first:
  ```bash
  cd /usr/src/gtest && sudo cmake . && sudo make && sudo cp lib/*.a /usr/lib
  ```
- If you see `CudaArch.mk: No such file or directory`: this is harmless — the Makefile uses `-include` so missing CUDA config is silent. Upgrade Make if you see a hard error here.
- If `-mavx2` causes a compile error: your CPU does not support AVX2. Set `SIMD_FLAGS=` on the command line to force the scalar path:
  ```bash
  make benchcpu SIMD_FLAGS=
  ```
