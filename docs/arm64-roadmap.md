# ARM64 / Apple Silicon — Future-Work Roadmap

This document catalogues all optimization and feature phases that are explicitly **out of scope** for the initial ARM64 port (Parts 1–6). The initial port makes the library compile and pass all scalar/sequential tests on Apple Silicon; everything listed here is subsequent work.

---

## Context: What Parts 1–6 Delivered

| Part | Deliverable |
|------|-------------|
| 1 | `src/platform.hpp` — canonical macros (`GOLDILOCKS_ARCH_X86_64`, `GOLDILOCKS_ARCH_AARCH64`, `GOLDILOCKS_HAS_AVX2`, `GOLDILOCKS_HAS_AVX512`, `GOLDILOCKS_HAS_NEON`) |
| 2 | All `#include <immintrin.h>`, AVX2/AVX-512 type declarations, and file-scope SIMD constants guarded behind `platform.hpp` macros |
| 3 | `goldilocks_base_field_scalar.hpp` — x86 inline ASM replaced with `__uint128_t` portable C++ behind `#else` branches; x86 path preserved |
| 4 | `poseidon_goldilocks.hpp` merkle wrapper — `#elif !GOLDILOCKS_HAS_AVX2` branch routing to `_seq` variants on ARM64 |
| 5 | `Makefile` — platform/arch detection, macOS/ARM64 support, Homebrew OpenMP/GMP/GTest discovery |
| 6 | `tests/tests.cpp` — 13 AVX2 tests guarded with `#ifdef GOLDILOCKS_HAS_AVX2`; 17 scalar tests verified passing on ARM64 |

No performance optimization, no SIMD vectorization, no GPU work was done.

---

## Phase 1 — NEON Intrinsics (AVX2-Equivalent Throughput)

**Goal:** Reach AVX2-level throughput on Apple Silicon for field arithmetic and Poseidon hashing using ARM NEON/AdvSIMD 128-bit intrinsics.

**Motivation:** The scalar `__uint128_t` path introduced in Part 3 is correct but single-lane. Apple M-series chips have 128-bit NEON units capable of operating on two `uint64_t` elements in parallel, providing roughly 2× throughput versus scalar.

**Affected files:**
- `src/goldilocks_base_field_scalar.hpp` — add `#elif GOLDILOCKS_HAS_NEON` branch for `add`, `sub`, `mul`, `mul2`
- `src/goldilocks_base_field_tools.hpp` — `to_montgomery`, `from_montgomery` NEON paths
- `src/goldilocks_base_field_avx.hpp` — NEON equivalent (e.g., `goldilocks_base_field_neon.hpp`) providing 2-lane equivalents of current 4-lane AVX2 functions
- `src/poseidon_goldilocks.hpp` — dispatch to `_neon` variant in `merkletree`/`merkletree_batch` wrappers

**Tasks:**

- [ ] **Task 20** — Audit all AVX2 intrinsic calls in `goldilocks_base_field_avx.hpp` and produce a mapping table (`_mm256_*` → `v*q_u64` or `v*q_u32` NEON equivalent); document cases where NEON width (128 bits, 2 lanes) differs from AVX2 (256 bits, 4 lanes) and decide whether to use 2-lane-wide or loop-unrolled 2×2-lane approach.

- [ ] **Task 21** — Create `src/goldilocks_base_field_neon.hpp` under `#ifdef GOLDILOCKS_HAS_NEON`, implementing 2-lane Goldilocks field arithmetic using `<arm_neon.h>`. Must pass the same unit tests as the AVX2 path, with identical numerical output (bit-for-bit match with scalar reference).

- [ ] **Task 22** — Add `GOLDILOCKS_HAS_NEON` branch in `goldilocks_base_field.hpp` include chain (currently `goldilocks_base_field_avx.hpp` is the only non-scalar non-AVX512 inclusion); add guard to `Makefile` that passes `-DGOLDILOCKS_HAS_NEON` when building on AArch64.

- [ ] **Task 23** — Create `src/poseidon_goldilocks_neon.hpp` with a 2-lane Poseidon permutation, following the structure of `poseidon_goldilocks_avx.hpp`. Verify S-box (`x^7`) and linear layer produce correct hash output against `_seq` reference.

- [ ] **Task 24** — Update the merkle dispatch in `poseidon_goldilocks.hpp` (lines 82–97) to prefer `_neon` over `_seq` when `GOLDILOCKS_HAS_NEON` is defined and `GOLDILOCKS_HAS_AVX2` is not. Add corresponding GTest cases gated with `#ifdef GOLDILOCKS_HAS_NEON`.

- [ ] **Task 25** — Run benchmarks (Google Benchmark) comparing NEON vs scalar on M1/M2/M3 class hardware for: field `mul`, Poseidon `hash_full_result`, `merkletree`, NTT size 2^20. Document speedup relative to Part 3 scalar baseline.

---

## Phase 2 — SVE/SVE2 Exploration (AVX-512 Equivalent)

**Goal:** Evaluate SVE/SVE2 (Scalable Vector Extension) as an ARM64 replacement for AVX-512 8-lane field arithmetic, for use on server-class ARM hardware (AWS Graviton3, Ampere Altra).

**Motivation:** Apple Silicon does not implement SVE/SVE2 as of M3 (2024). However, server ARM platforms do, and those are the likely deployment targets for prover workloads. SVE vectors are runtime-sized (128–2048 bits), offering potential for 8-lane or wider Goldilocks arithmetic.

**Tasks:**

- [ ] **Task 26** — Research SVE availability: check which Graviton generation (Graviton3 = SVE 256-bit, Graviton4 = SVE2 512-bit) is available in CI/CD or production. Determine minimum vector length guarantee for Goldilocks 8-lane operations.

- [ ] **Task 27** — Add `GOLDILOCKS_HAS_SVE` detection to `src/platform.hpp` using `__ARM_FEATURE_SVE` compiler predefined macro; add corresponding `-march=armv8-a+sve` flag logic in `Makefile`.

- [ ] **Task 28** — Prototype SVE 8-lane Goldilocks `mul` in `src/goldilocks_base_field_sve.hpp` using `<arm_sve.h>` intrinsics. Validate bit-identical output against scalar reference. Benchmark vs AVX-512 on same input sizes.

- [ ] **Task 29** — If Task 28 shows competitive throughput, extend SVE path to Poseidon (8-lane permutation) following `poseidon_goldilocks_avx512.hpp` structure. Update dispatch macros and tests.

---

## Phase 3 — Metal GPU Compute Shaders (NTT, Poseidon, Merkle)

**Goal:** Replace the CUDA GPU kernels with Metal compute shaders for Apple Silicon, enabling GPU-accelerated NTT, Poseidon hashing, and Merkle tree construction on macOS.

**Motivation:** CUDA is unavailable on Apple Silicon. The current codebase has three GPU-accelerated workloads (`ntt_goldilocks.cu`, `poseidon_goldilocks.cu`, `merkletree` within Poseidon), all written in CUDA/PTX. Metal Shading Language (MSL) and the Metal Performance Shaders (MPS) framework are the native GPU compute path on Apple Silicon.

**Affected files (new):**
- `src/metal/gl64_t.metal` — Goldilocks field arithmetic in MSL (port of `src/gl64_t.cuh`)
- `src/metal/ntt_goldilocks.metal` — NTT butterfly kernel
- `src/metal/poseidon_goldilocks.metal` — Poseidon permutation kernel
- `src/metal/metal_utils.mm` / `metal_utils.hpp` — Objective-C++ Metal device/command queue management

**Tasks:**

- [ ] **Task 30** — Study `src/gl64_t.cuh`: enumerate every PTX instruction used (`mad.lo`, `mad.hi`, `add.cc`, `addc`, `sub.cc`, `subc`, `mul.lo`, `mul.hi`, `shl`, `shr`). Produce a mapping table to MSL equivalents; identify instructions with no direct MSL equivalent (e.g., carry-chain arithmetic) and document workaround strategy using `uint2` or `ulong2` decomposition.

- [ ] **Task 31** — Implement `gl64_t` in Metal Shading Language (`src/metal/gl64_t.metal`): struct holding a single `ulong` with device-function operators `+`, `-`, `*`, `^7` (S-box). Validate correctness with a CPU-side Metal unit test using `MTLCommandBuffer` and `MTLComputeCommandEncoder`.

- [ ] **Task 32** — Port NTT butterfly kernel (`ntt_goldilocks.cuh` → `src/metal/ntt_goldilocks.metal`). Map CUDA `blockDim`/`gridDim`/`threadIdx` to Metal `threads_per_threadgroup`/`threadgroup_position_in_grid`/`thread_position_in_threadgroup`. Pay special attention to `ncols`-as-block-dimension pattern.

- [ ] **Task 33** — Port Poseidon permutation kernel (`poseidon_goldilocks.cu` → `src/metal/poseidon_goldilocks.metal`). Poseidon uses 913 `uint64_t` per threadgroup (7.3 KB) of shared memory; Metal threadgroup memory limit is 32 KB — fits comfortably. Validate hash output against `_seq` reference.

- [ ] **Task 34** — Port Merkle tree construction kernel. Determine whether the current CUDA Merkle kernel is standalone or calls the Poseidon kernel; replicate the calling convention in Metal using `MTLIndirectCommandBuffer` if needed.

- [ ] **Task 35** — Implement `src/metal/metal_utils.mm`: device selection, command queue creation, pipeline state caching, buffer allocation helpers. Expose a C++ header `src/metal/metal_utils.hpp` that hides Objective-C++ from callers.

- [ ] **Task 36** — Integrate Metal path into build: add Makefile target `testmetal` that compiles `.metal` shaders with `xcrun metal`, links `.mm` files with `clang++ -framework Metal -framework Foundation`, and runs Metal GTest cases. Gate entirely behind `GOLDILOCKS_HAS_METAL` macro set when `$(shell uname -s)` is `Darwin`.

- [ ] **Task 37** — Add Metal GTest cases in `tests/tests.cpp` gated with `#ifdef GOLDILOCKS_HAS_METAL`: `ntt_metal`, `poseidon_metal`, `merkletree_metal`. Each test must validate output against the `_seq` reference implementation.

---

## Phase 4 — Unified Memory Optimization for Metal

**Goal:** Exploit Apple Silicon's CPU–GPU unified memory to eliminate explicit copy operations between CPU and GPU buffers in the Metal path.

**Motivation:** The CUDA path allocates device memory separately and uses `cudaMemcpy`/`cudaMemcpyPeerAsync` for transfers. Apple Silicon has a single physical memory pool accessible by both CPU and GPU; Metal buffers created with `MTLResourceStorageModeShared` require zero explicit copies.

**Tasks:**

- [ ] **Task 38** — Audit all `cudaMalloc`/`cudaMemcpy`/`cudaFree` call sites in `src/ntt_goldilocks.cu` and `src/poseidon_goldilocks.cu`; for each, identify the Metal equivalent (`MTLBuffer` with shared storage mode, direct CPU pointer via `[buffer contents]`).

- [ ] **Task 39** — Refactor `src/metal/metal_utils.mm` buffer allocation to default to `MTLResourceStorageModeShared`. Expose a typed `GoldilocksBuffer` C++ wrapper that returns a raw `uint64_t*` pointer valid for both CPU reads and GPU dispatches with no synchronization step.

- [ ] **Task 40** — Remove all intermediate staging buffers from the Metal NTT and Poseidon dispatch paths (Tasks 32–34). Measure latency reduction on M1/M2 Pro vs the naive copy-then-compute approach on an equivalent NVIDIA GPU.

- [ ] **Task 41** — Handle `cudaMallocHost` (pinned host memory) — identify call sites and confirm they are unnecessary in the Metal path (unified memory is always host-accessible); document this simplification in code comments.

---

## Phase 5 — Performance Benchmarking Baseline

**Goal:** Establish a documented, reproducible performance baseline comparing scalar ARM64 (Part 3 output), NEON ARM64 (Phase 1 output), and x86 AVX2 (existing) across all key operations.

**Motivation:** Without a baseline, future optimizations have no reference point. The benchmark suite (`benchs/bench.cpp`) currently only covers AVX paths.

**Tasks:**

- [ ] **Task 42** — Extend `benchs/bench.cpp` with scalar ARM64 benchmarks for: `Goldilocks::mul` (1M iterations), `Goldilocks::add` (1M iterations), `PoseidonGoldilocks::hash_full_result_seq` (10K hashes), `NTT_Goldilocks::NTT` (size 2^20, 2^24). Gate with `#ifndef GOLDILOCKS_HAS_AVX2` to run automatically on ARM64.

- [ ] **Task 43** — Add Makefile target `bencharm64` that builds `benchs/bench.cpp` without `-mavx2`, links Google Benchmark, and runs on the current platform. Output CSV-compatible results.

- [ ] **Task 44** — Run benchmarks on: Apple M1 Pro (or equivalent), Apple M2 Max (or equivalent), AWS Graviton3 (c7g instance), and a reference x86 machine with AVX2 (e.g., AWS c5 or c6i). Capture results in `docs/benchmarks/arm64-baseline.csv`.

- [ ] **Task 45** — After Phase 1 (NEON), re-run the same benchmark suite on ARM64 hardware and record NEON speedup vs scalar baseline. Add a `docs/benchmarks/neon-vs-scalar.csv`. After Phase 3 (Metal), add `docs/benchmarks/metal-vs-cuda.csv`.

---

## Phase 6 — CI Matrix for Multi-Platform Testing

**Goal:** Establish automated CI that runs tests on both Linux x86-64 and macOS ARM64 on every pull request.

**Motivation:** Without CI, regressions on either platform will not be caught. The current repo has no CI configuration at all.

**Tasks:**

- [ ] **Task 46** — Create `.github/workflows/ci.yml` with two jobs: `test-linux-x86` (runs on `ubuntu-latest`, installs `libomp-dev libgmp-dev libgtest-dev`, builds with `make testcpu`, runs all tests) and `test-macos-arm64` (runs on `macos-latest` which is ARM64 as of 2024, installs Homebrew `libomp gmp googletest`, builds with `make testcpu`, runs scalar tests only).

- [ ] **Task 47** — Add a test filter mechanism to `Makefile`: `make testcpu FILTER=scalar` runs only `--gtest_filter=GOLDILOCKS_TEST.*_seq:GOLDILOCKS_TEST.one:GOLDILOCKS_TEST.add:GOLDILOCKS_TEST.sub:GOLDILOCKS_TEST.mul:GOLDILOCKS_TEST.div:GOLDILOCKS_TEST.inv:GOLDILOCKS_TEST.ntt*:GOLDILOCKS_TEST.LDE*:GOLDILOCKS_TEST.extendePol`. The macOS CI job uses this filter.

- [ ] **Task 48** — Pin Homebrew formula versions in CI (use `brew install gmp@<version>`) to prevent dependency drift from breaking macOS builds silently. Add a `docs/dev-setup-macos.md` with the exact `brew install` commands for local development.

- [ ] **Task 49** — Add a third CI job `test-macos-neon` (conditional on Phase 1 completion): same as `test-macos-arm64` but builds with `-DGOLDILOCKS_HAS_NEON` and runs all tests including `_neon` variants. Activate once `goldilocks_base_field_neon.hpp` is merged.

- [ ] **Task 50** — Configure CI to post test result summaries as PR comments using `actions/upload-artifact` and a summary step. Include build time and test pass/fail counts for each platform.

---

## Phase 7 — Expressions GPU Evaluation for Metal

**Goal:** Port the `op_gpu` / `copy_gpu` expressions evaluation path (added in commit `ac41bcc` / `f5a8ebd`) from CUDA to Metal.

**Motivation:** The expressions GPU path was added after the initial scalar port scope was fixed. It adds new CUDA kernels for bulk expression evaluation (`op_gpu`, `copy_gpu`). If the Metal GPU path (Phase 3) is adopted, expressions evaluation must follow.

**Affected files (CUDA side, for reference):**
- Any `.cu` files modified in commits `ac41bcc` and `f5a8ebd` (expressions_gpu branch)
- New Metal files would mirror the CUDA expressions kernel structure

**Tasks:**

- [ ] **Task 51** — Audit commits `ac41bcc` and `f5a8ebd`: enumerate every new CUDA kernel, host-side dispatch function, and data structure introduced for expressions GPU evaluation. Produce a table of kernel names, input/output buffer shapes, and thread block configurations.

- [ ] **Task 52** — Design the Metal equivalents for each expressions kernel. Determine whether `op_gpu` and `copy_gpu` semantics map directly to compute shaders or require restructuring for Metal's dispatch model (which lacks CUDA-style dynamic parallelism).

- [ ] **Task 53** — Implement Metal expressions kernels in `src/metal/expressions_goldilocks.metal`. Validate output against CPU `_seq` reference. Add GTest cases gated with `#ifdef GOLDILOCKS_HAS_METAL`.

- [ ] **Task 54** — Benchmark Metal expressions evaluation vs CUDA on representative expression sizes used by the prover. If Metal throughput is within 2× of CUDA on the same problem size, consider Metal the primary GPU path for Apple Silicon deployments.

---

## Phase 8 — Multi-Architecture Binary / Fat Library Support

**Goal:** Produce a single distributable static library (`libgoldilocks.a`) or framework that runs on both x86-64 and ARM64 macOS without requiring the consumer to pick an architecture.

**Motivation:** Downstream tools (provers, zkEVM backends) are often distributed as pre-built binaries. A fat (universal) library avoids requiring consumers to build from source for each architecture.

**Tasks:**

- [ ] **Task 55** — Add Makefile targets `libgoldilocks-x86_64.a` and `libgoldilocks-arm64.a` that cross-compile (or natively compile on each arch) the library sources into architecture-specific static archives.

- [ ] **Task 56** — Add Makefile target `libgoldilocks-universal.a` that invokes `lipo -create libgoldilocks-x86_64.a libgoldilocks-arm64.a -output libgoldilocks-universal.a`. Verify with `lipo -info` that both slices are present.

- [ ] **Task 57** — Evaluate CMake as an alternative build system for cross-compilation support. If adopted, create `CMakeLists.txt` that detects `CMAKE_SYSTEM_PROCESSOR`, sets appropriate `-march` flags, finds Homebrew or system OpenMP/GMP, and exports a CMake package config for `find_package(goldilocks)`.

- [ ] **Task 58** — Test fat library linkage from a sample downstream project on both macOS ARM64 and macOS x86-64 (via Rosetta or a native Intel Mac in CI). Confirm that the correct slice is loaded and that scalar tests pass on each.

- [ ] **Task 59** — Publish release artifacts to GitHub Releases using `gh release create` in CI: include `libgoldilocks-universal.a`, `libgoldilocks-arm64.a`, and `libgoldilocks-x86_64.a` with SHA-256 checksums. Trigger on version tags (`v*.*.*`).

---

## Phase Dependency Graph

```
Parts 1–6 (ARM64 scalar port — DONE)
    │
    ├─── Phase 1: NEON intrinsics  ──────────────────────────────────────────┐
    │         │                                                               │
    │         └─── Phase 2: SVE/SVE2 (server ARM64)                         │
    │                                                                         │
    ├─── Phase 3: Metal GPU compute  ────────────────────────────────────────┤
    │         │                                                               │
    │         └─── Phase 4: Unified memory optimization                      │
    │         │                                                               │
    │         └─── Phase 7: Expressions GPU → Metal                          │
    │                                                                         │
    ├─── Phase 5: Benchmarking baseline  (requires Phase 1 for NEON data)   │
    │                                                                         │
    ├─── Phase 6: CI matrix  (no hard dependencies, can start now)          │
    │                                                                         │
    └─── Phase 8: Fat library / multi-arch binary  ──────────────────────────┘
              (requires Phases 1, 3 to have useful content in all slices)
```

**Recommended sequencing:**
1. Phase 6 (CI) — start immediately, no dependencies, prevents regressions
2. Phase 5 (Benchmarks) — establish baseline before optimization
3. Phase 1 (NEON) — highest ROI CPU optimization for Apple Silicon
4. Phase 3 (Metal) — GPU path, depends on Phase 1 for CPU fallback
5. Phase 4 (Unified memory) — follow-on to Phase 3
6. Phase 7 (Expressions Metal) — follow-on to Phase 3
7. Phase 2 (SVE) — server ARM64, parallel track to Phase 1
8. Phase 8 (Fat library) — packaging, do last

---

## Reference: 17 Scalar Tests Validated in Parts 1–6

The following tests pass on ARM64 after Parts 1–6 and serve as the correctness baseline for all future phases:

| Test name | Subsystem |
|-----------|-----------|
| `GOLDILOCKS_TEST.one` | Field constant |
| `GOLDILOCKS_TEST.add` | Field arithmetic |
| `GOLDILOCKS_TEST.sub` | Field arithmetic |
| `GOLDILOCKS_TEST.mul` | Field arithmetic |
| `GOLDILOCKS_TEST.div` | Field arithmetic |
| `GOLDILOCKS_TEST.inv` | Field arithmetic |
| `GOLDILOCKS_TEST.poseidon_avx_seq` | Poseidon (scalar path) |
| `GOLDILOCKS_TEST.poseidon_full_seq` | Poseidon (scalar path) |
| `GOLDILOCKS_TEST.linear_hash_seq` | Poseidon (scalar path) |
| `GOLDILOCKS_TEST.merkletree_seq` | Merkle tree (scalar path) |
| `GOLDILOCKS_TEST.merkletree_batch_seq` | Merkle tree (scalar path) |
| `GOLDILOCKS_TEST.ntt` | NTT |
| `GOLDILOCKS_TEST.ntt_block` | NTT |
| `GOLDILOCKS_TEST.LDE` | NTT / LDE |
| `GOLDILOCKS_TEST.LDE_block` | NTT / LDE |
| `GOLDILOCKS_TEST.extendePol` | NTT / polynomial extension |
| `GOLDILOCKS_CUBIC_TEST.one` | Cubic extension constant |

Any future phase that introduces a regression in these 16 tests on ARM64 must be treated as a blocking defect.
