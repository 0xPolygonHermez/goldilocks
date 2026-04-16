# Goldilocks Apple Metal GPU Backend

Metal-backed implementations of Goldilocks Poseidon12, Merkle tree, and NTT
for Apple Silicon (M1–M4+). Bit-exact output vs the scalar/NEON CPU paths,
plugged in as explicit entry points behind a compile-time gate
(`GOLDILOCKS_HAS_METAL`), mirroring the existing CUDA `__USE_CUDA__`
precedent.

## Status

- Tested on Apple M4 Pro.
- Merkle tree at STARK-prover scale (8M rows × 128 cols): **~2.8× NEON, ~9.6× scalar**.
- NTT forward at production shapes (2¹⁸ × 128): **~1.9× NEON**.
- All paths bit-exact vs the CPU reference; `INTT(NTT(x)) == x` verified.
- Requires macOS 13+, Xcode 15+, Metal Toolchain component installed.

## Quick start

```cpp
#include "metal/goldilocks_metal.hpp"

// Merkle tree (mirror of PoseidonGoldilocks::merkletree_seq)
PoseidonGoldilocks::merkletree_metal(tree_ptr, input_ptr, num_cols, num_rows);

// Forward NTT
NTT_Goldilocks ntt(N);
ntt.NTT_Metal(dst, src, N, ncols, /*inverse=*/false);

// Inverse NTT (in-place or out-of-place)
ntt.NTT_Metal(dst, src, N, ncols, /*inverse=*/true);

// Zero-copy aligned allocator (optional, avoids memcpy on large inputs)
auto* aligned_cols = goldilocks_metal::allocate_aligned_elements(n_elem);
// … use it …
goldilocks_metal::free_aligned(aligned_cols);

// Batched Merkle API (for repeated-call workloads)
goldilocks_metal::merkletree_metal_batch(
    trees, inputs, count, num_cols, num_rows);
```

## Build

The root `Makefile` grows a Darwin-gated Metal section below the existing
GPU targets. Build via:

```bash
make testmetal       # links the metallib + all .mm into a standalone gtest binary
make runtestmetal    # runs the bit-exact oracle test
```

The build generates MSL shaders via `xcrun metal`:

```
.metal sources       →  xcrun metal -std=metal3.0 -c  →  .air objects
.air objects         →  xcrun metallib                →  goldilocks.metallib
.mm files            →  clang++ -fobjc-arc -DGOLDILOCKS_HAS_METAL
linked with:         →  clang++ -framework Metal -framework Foundation
                        -framework QuartzCore  (per-target; never in
                        global LDFLAGS)
```

The runtime Metal library is loaded by the `MetalEnvironment` test fixture
at gtest `SetUp` time — it tries `./goldilocks.metallib`, then
`argv[0]`-adjacent, then falls back to runtime source compile.

### Prerequisites
- **Xcode.app** installed (Command Line Tools alone is insufficient — the
  `metal` shader compiler ships only with the full Xcode).
- **Metal Toolchain component** — may need
  `xcodebuild -downloadComponent MetalToolchain` after the first Xcode
  install.
- **Xcode license accepted**: `sudo xcodebuild -license`.

### Apple Silicon only
`GOLDILOCKS_HAS_METAL` is auto-defined in `src/platform.hpp` when
`__APPLE__ && __aarch64__` and `GOLDILOCKS_NO_METAL` is not defined.
Builds on Linux / Intel Mac / any non-Apple-arm64 host are bit-identical
pre- and post-this-module (every edit to a non-Metal file is inside
`#ifdef GOLDILOCKS_HAS_METAL`).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Callers:                                                        │
│   PoseidonGoldilocks::merkletree_metal(tree, input, nc, nr)     │
│   NTT_Goldilocks::NTT_Metal(dst, src, size, ncols, inverse)     │
│   goldilocks_metal::merkletree_metal_batch(trees, ins, cnt...)  │
├─────────────────────────────────────────────────────────────────┤
│ Facade (pure C++, no Obj-C types):                              │
│   src/metal/goldilocks_metal.hpp                                │
│   namespace goldilocks_metal { … }                              │
├─────────────────────────────────────────────────────────────────┤
│ Obj-C++ bridge (.mm, ARC + @autoreleasepool per entry):         │
│   metal_context.{hpp,mm}  — singleton (device, queue, library,  │
│                             pipeline & twiddle caches)          │
│   poseidon_metal.mm       — merkletree_metal + batched version  │
│   ntt_metal.mm            — NTT_Metal (forward & inverse)       │
├─────────────────────────────────────────────────────────────────┤
│ MSL kernels (src/metal/kernels/*.metal → metallib):             │
│   field.metal    — gl_add/sub/mul (lazy-reduce), gl_mul_small,  │
│                    gl_canonicalize                              │
│   poseidon.metal — pod12, linear_hash, merkle_leaves[_simd|_x2| │
│                    _cm|_tg], merkle_parents                     │
│   ntt.metal      — rev_butterfly_{s1,s1s2}, butterfly_phase,    │
│                    radix4_phase, intt_reorder_scale             │
│   constants.metal.inc  (generated at build time from            │
│                        poseidon_goldilocks_constants.hpp)       │
└─────────────────────────────────────────────────────────────────┘
```

### Key design choices

1. **Lazy-reduce field ops** — `gl_mul` returns in `[0, 2p)`, matching the
   CPU NEON `mul_reduced` contract. Canonicalization happens only at
   kernel output boundaries. This is load-bearing: violating it by
   canonicalizing inside kernels silently regresses perf because the
   compiler loses visibility into the reduction chain.

2. **One-thread-per-row Poseidon** — each thread hashes one Merkle row.
   The cooperative SIMD variant (12 threads per hash via `simd_shuffle`)
   is kept in tree but loses at large N due to shuffle overhead.

3. **`gl_mul_small`** — specialized multiply for small-constant operands.
   The Goldilocks Poseidon12 M matrix has all 144 entries `< 2^32`, so
   `mvp_M` uses a 96-bit-product reduction instead of 128-bit. P and S
   matrices keep the full `gl_mul` because they have 64-bit entries.

4. **NTT radix-4** — paired radix-2 stages into one kernel pass, halving
   memory traffic for the butterfly loop. Gated on `(domain_size/4) *
   ncols >= 8192` so small workloads stay on radix-2 where per-thread
   register pressure matters more than memory saved.

5. **Fused rev + s=1 + s=2** — first three NTT stages collapse into one
   kernel pass when src ≠ dst. Saves two full read+write cycles at the
   start of every forward NTT.

6. **Fused intt_reorder + scale** — INTT's final permutation (N-i mod N)
   and multiply-by-1/N are combined into one kernel.

7. **Twiddle cache** — the full `roots[]` array is staged once per
   `NTT_Goldilocks` instance into an `MTLBuffer` owned by the singleton
   `metal_context`. Subsequent NTT calls at the same size hit the cache.

8. **Pipeline state cache** — compiled `MTLComputePipelineState`s cached
   by kernel name. First call to each kernel pays compile cost once.

### Threading contract

Metal bridge entries **must not** be called from inside a `#pragma omp
parallel` region. `@autoreleasepool` wraps each entry but only protects
the calling thread; OpenMP worker threads don't have an autorelease pool
and would leak Obj-C objects. The CPU `merkletree_seq` uses OMP for leaf
hashing — the Metal version does not need it (GPU parallelizes
internally). Callers should submit from a single driver thread.

### Canonicalization invariant

Intermediate sponge state carries lazy-reduce values in `[0, 2^64)`, not
`[0, p)`. `pod12` is mod-p correct on unreduced inputs. `gl_canonicalize`
is called exactly once — on the final 4-element hash before writing to
the output tree. Don't add canonicalization inside the permutation loop;
we measured this and it regresses.

## Performance

**Hardware**: Apple M4 Pro (20 GPU cores, 14 CPU cores @ 4.4 GHz).

### Merkle tree (MERKLETREE_BENCH workload: 8M rows × 128 cols, ~8 GB)

| Backend | Time | MiB/s | µs/hash | vs scalar | vs NEON |
|---|---:|---:|---:|---:|---:|
| Scalar (M4 Pro 14 threads) | 54.12 s | 152 | 2.80 | 1.00× | 0.29× |
| NEON (M4 Pro 14 threads) | 15.77 s | 519 | 0.83 | 3.43× | 1.00× |
| AVX2 (Xeon D-2141I 16 threads) | 44.95 s | 182 | 2.68 | — | — |
| **Metal (M4 Pro GPU)** | **5.64 s** | **1452** | **0.67** | **9.60×** | **2.80×** |

### NTT forward (bench_ntt measured)

| Shape | NEON M4 | **Metal M4** | Metal/NEON |
|---|---:|---:|---:|
| N=2¹⁴ × 64 | 1.31 ms | 1.24 ms | 1.08× |
| N=2¹⁶ × 64 | 6.28 ms | 4.09 ms | **1.54×** |
| N=2¹⁶ × 256 | 23.10 ms | 11.94 ms | **1.93×** |
| N=2¹⁸ × 64 | 27.33 ms | 13.28 ms | **2.06×** |
| N=2¹⁸ × 128 (STARK) | 51.06 ms | 26.30 ms | **1.94×** |
| N=2²⁰ × 1 | 6.20 ms | 2.88 ms | **2.15×** |

### Crossover thresholds

| Workload | Metal wins above | Metal loses below | Notes |
|---|---|---|---|
| Merkle (128 cols) | ≥ ~4k rows | < 4k rows | Launch overhead dominates below |
| NTT | ≥ 2¹⁶ × ≥64 cols | tiny single-col | Gate: `(N/4) * ncols >= 8192` |

Don't call Metal for sub-crossover workloads — the `merkletree_seq` /
`NTT()` CPU paths are faster there due to kernel launch latency.

## API reference

All public symbols in `goldilocks_metal.hpp` (pure C++; no Obj-C types
exposed). All marked with `#ifdef GOLDILOCKS_HAS_METAL` — calling on
non-Apple-Silicon builds is a link-time error by design.

### `PoseidonGoldilocks::merkletree_metal`
```cpp
static void merkletree_metal(
    Goldilocks::Element* tree,    // out: size (2n-1)*4 elements
    Goldilocks::Element* input,   // in:  size n*ncols elements (row-major)
    uint64_t num_cols,
    uint64_t num_rows);
```
Synchronous; blocks on `waitUntilCompleted` before returning. Output bit-
exact with `merkletree_seq`. `num_rows` must be a positive integer (not
necessarily a power of two — odd tails at each level copy the last child).

### `NTT_Goldilocks::NTT_Metal`
```cpp
void NTT_Metal(
    Goldilocks::Element* dst,
    Goldilocks::Element* src,
    uint64_t size,
    uint64_t ncols,
    bool inverse = false);
```
In-place (`src == dst`) and out-of-place supported. Out-of-place hits the
fused-kernel fast path. `size` must be a power of two.

**Performance note — prefer out-of-place for INTT.** The fused rev+s1+s2+s3
kernel requires `src != dst`. When the caller passes in-place, the path
falls back to a separate `reverse_permutation` pass plus the full phase
loop — one extra memory pass. Measured deltas on M4 Pro at production
shapes:

| Shape           | in-place INTT | out-of-place INTT | Δ      |
|-----------------|---------------|-------------------|--------|
| N=2²⁰ × 1       | 3.1 ms (2.2× cpu) | 1.2 ms (5.6× cpu)  | 2.60×  |
| N=2¹⁶ × 64      | 4.6 ms (1.5× cpu) | 2.8 ms (2.5× cpu)  | 1.68×  |
| N=2¹⁸ × 128     | 21 ms (2.5× cpu)  | 20 ms (2.6× cpu)   | 1.06×  |

The win is shape-dependent (largest at small-ncols) but never negative —
callers who can spare a scratch buffer should always pass distinct pointers.

### `NTT_Goldilocks::extendPol_Metal`
```cpp
void extendPol_Metal(
    Goldilocks::Element* output,
    Goldilocks::Element* input,
    uint64_t N_Extended,
    uint64_t N,
    uint64_t ncols);
```
Low-Degree Extension on the GPU. Mirrors `extendPol` semantics:
`INTT(input, N, ncols)` with coset shift → zero-pad to `N_Extended` →
forward `NTT(output, N_Extended, ncols)`. The coset reorder+scale uses
a dedicated `intt_reorder_coset_scale` kernel that fuses the index
permutation with the per-element `r_[i] = shift^i / N` multiplication.

Calls `computeR(N)` on the caller instance the first time through so
`r_[]` is populated. `N_Extended` must be a power of two ≥ `N`; both
`N` and `N_Extended` must be ≥ 2. Does NOT require `NTT_Metal` to have
run previously.

Bit-exact with CPU `extendPol` on every shape measured. See
`benchs/metal_probes/bench_lde.mm` for validation and timings.

### `goldilocks_metal::allocate_aligned_elements`
```cpp
Goldilocks::Element* allocate_aligned_elements(uint64_t n);
void                 free_aligned(void* ptr);
```
Returns memory aligned to the 16 KB Apple Silicon page boundary. Buffers
from this allocator hit Metal's `newBufferWithBytesNoCopy` zero-copy
path — no memcpy in, no readback out. Matters most at multi-hundred-MB
scale; ~1% improvement at typical 256k-row Merkle sizes.

### `goldilocks_metal::merkletree_metal_batch`
```cpp
void merkletree_metal_batch(
    Goldilocks::Element** trees,
    Goldilocks::Element** inputs,
    uint64_t count,
    uint64_t num_cols,
    uint64_t num_rows);
```
Processes `count` independent trees in ONE GPU submission. Uses separate
command-encoders per tree within one command buffer so Metal's per-buffer
hazard tracking allows the GPU to overlap them when there's slack.
Current per-tree kernels are already compute-saturating, so the batched
win is only ~1-5% in practice; the API is still useful for reducing
per-call sync latency in caller loops.

## Probes and benchmarks

`benchs/metal_probes/` is a self-contained sandbox (mirrors the
`benchs/sme_probes/` convention). Own Makefile, no dependency on the
main library build.

```bash
cd benchs/metal_probes

make run                 # runs all probes (device, field, poseidon, merkle)
make bench_merkle        # build merkle throughput comparison
./bench_merkle           # small sizes
./bench_merkle --big     # adds 1 GB and 8 GB shapes
make bench_ntt           # build NTT throughput comparison
./bench_ntt
make bench_merkle_batch  # sequential vs batched N trees
./bench_merkle_batch
make probe_profile       # GPU timing / PSO introspection
```

Each probe is a bare `main()` with `assert()`-style pass/fail; the
correctness probes cross-check against the CPU scalar reference on
Fibonacci-seeded inputs.

## Kernel variants (selectable via `g_merkle_use_simd_coop`)

`poseidon_metal.mm` has a weak `extern "C" int g_merkle_use_simd_coop`
selector used by the bench harness. Default value 0 (row-per-thread)
is the shipped path.

| Value | Variant | When to use |
|---|---|---|
| **0** (default) | row-per-thread | Always. The shipped path. Wins at ≥ 4k rows. |
| 1 | SIMD-cooperative (12 threads/hash) | Research only. Faster at < 1k rows, 3× slower at 262k+ |
| 2 | 2-rows-per-thread ILP | Research only. Register-pressure regression at all sizes |
| 3 | Column-major (transpose + coalesced reads) | Research only. Transpose overhead > coalescing gain |
| 4 | Threadgroup-tile cooperative load | Research only. Barrier overhead ≈ coalescing gain |

## Bit-exact oracle

The canonical correctness gate is the Fibonacci 128 × 64 Merkle root
from `tests/tests.cpp:2232-2235`:

```
root[0] = 0x918F7CD0C3E8701F
root[1] = 0x83A130E00F961B02
root[2] = 0x6921497B364123F8
root[3] = 0xBD2B98A57B748BF4
```

The `testmetal` gtest target and `probe_merkle` probe both verify
against these values.

## Maintenance

- **MSL compile errors**: the `metal` shader compiler lives only in
  Xcode.app, not CLT. First-time setup may need
  `xcodebuild -downloadComponent MetalToolchain`.
- **Runtime library load fails**: `metal_context_load_library` tries
  CWD then `argv[0]`-adjacent. The bench harness also falls back to
  runtime source compile via `metal_context_load_source` which doesn't
  need the metallib — useful for development.
- **Changing the Goldilocks Poseidon constants**: `tools/gen_metal_constants.cpp`
  reads `src/poseidon_goldilocks_constants.hpp` and writes the MSL
  `constant ulong C[118]`, `M_[144]`, `P_[144]`, `S[507]` tables
  consumed by `poseidon.metal`. Regenerates at every build. If you
  change the `USE_MONTGOMERY` flag (currently hardcoded to 0), this
  codegen picks up the active `#else` canonical block automatically.
- **Adding a kernel**: new `.metal` files go under
  `src/metal/kernels/`. Add the source to `METAL_SHADERS` in the
  root `Makefile`. Expose via a dispatch helper in
  `metal_context.{hpp,mm}` and a public facade entry in
  `goldilocks_metal.hpp` if caller-facing.

## Known limitations

- **Metal submission from OMP worker threads is unsafe.** Bridge entries
  must be called from the main/submitting thread. `@autoreleasepool`
  wrapping only protects the calling thread.
- **No graceful CPU fallback.** If `goldilocks.metallib` is missing at
  runtime, the bridge calls `NSLog` + `abort()` on first Metal dispatch.
  Callers that need a fallback must check availability themselves (e.g.,
  `MTLCreateSystemDefaultDevice() != nil`) and choose
  `merkletree_seq` / `NTT()` explicitly.
- **Small-N performance** (< 1k Merkle rows, < 2¹⁴ NTT) is worse than
  NEON due to kernel launch overhead. Callers should pick CPU paths for
  these sizes — the Metal entries remain callable but are slower.
- **Inverse NTT in-place** (`src == dst`) uses the legacy rev+full-phase
  path (fused kernels require distinct source and destination buffers).
  Callers that want the fused fast path should pass a separate output
  buffer.
- **`extendPol`-on-Metal** (coset LDE) is not implemented. The existing
  CPU `NTT_Goldilocks::extendPol` remains the only path for
  coset-extended NTTs.

## See also

- `benchs/results/apple-silicon-m4pro-metal.md` — full benchmark trace
  with cross-platform comparison
- `benchs/results/x86_64-linux-xeon-d2141i.md` — x86-64 AVX2 baseline
- `benchs/metal_probes/` — sandbox for probing and benchmarking
- `src/poseidon_goldilocks.cpp:5` — CPU scalar reference for the
  permutation structure
- `src/ntt_goldilocks.cpp:15` — CPU scalar reference for NTT
