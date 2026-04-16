// metal_context.hpp — C++ API for the Metal singleton context (no Obj-C types exposed).
//
// Exposes only plain C++ types so this header can be included from pure C++ TUs.
// The implementation in metal_context.mm uses Obj-C++ internally.
//
// Usage:
//   #include "metal_context.hpp"
//   MetalCtxHandle h = metal_context_get();
//   metal_context_compile_library(h, "/path/to/goldilocks.metallib");
//   MetalPipelineHandle p = metal_context_pipeline(h, "merkle_leaves");
//
// Buffer helpers:
//   MetalBufHandle metal_buf_alias(MetalCtxHandle, void* ptr, size_t bytes, bool* is_copy)
//   MetalBufHandle metal_buf_alloc(MetalCtxHandle, size_t bytes)
//   void*          metal_buf_contents(MetalBufHandle)
//   void           metal_buf_release(MetalBufHandle)
//
// Dispatch:
//   void metal_dispatch_merkle_leaves(...)
//   void metal_dispatch_merkle_parents(...)
//   void metal_dispatch_ntt_reverse_permutation(...)
//   void metal_dispatch_ntt_butterfly_phase(...)
//   void metal_dispatch_intt_scale(...)
//
// All functions are thread-safe (internal mutex in the singleton).

#pragma once
#ifdef GOLDILOCKS_HAS_METAL

#include <stddef.h>
#include <stdint.h>

// Opaque handles — the implementation holds ARC-retained Obj-C objects behind these.
typedef void* MetalCtxHandle;
typedef void* MetalPipelineHandle;
typedef void* MetalBufHandle;

#ifdef __cplusplus
extern "C" {
#endif

// ---------- Context lifecycle -----------------------------------------------

// Returns the singleton context (creating it on first call).
// Calls NSLog + abort() if no Metal device is available.
MetalCtxHandle metal_context_get(void);

// Load a .metallib from a file path. Call once after metal_context_get().
// Returns 0 on success, -1 on error (error printed to NSLog).
int metal_context_load_library(MetalCtxHandle ctx, const char* metallib_path);

// Compile MSL source at runtime via newLibraryWithSource:options:error:.
// `source` must be a NULL-terminated C string of concatenated MSL source.
// Returns 0 on success, -1 on compile error (details printed to NSLog).
int metal_context_load_source(MetalCtxHandle ctx, const char* source);

// ---------- Pipeline cache --------------------------------------------------

// Returns a cached MTLComputePipelineState for the named kernel function.
// Compiles on first access; aborts if compilation fails.
MetalPipelineHandle metal_context_pipeline(MetalCtxHandle ctx, const char* kernel_name);

// ---------- Buffer helpers --------------------------------------------------

// Try to create a zero-copy alias buffer (requires 16 KB page alignment on the
// ptr). If the ptr is not page-aligned, copies the data into a new shared buffer.
// *is_copy is set to true if a copy was made (caller must readback after GPU work).
MetalBufHandle metal_buf_alias(MetalCtxHandle ctx, void* ptr, size_t bytes, int* is_copy);

// Allocate a new MTLStorageModeShared buffer of `bytes`.
MetalBufHandle metal_buf_alloc(MetalCtxHandle ctx, size_t bytes);

// Return the CPU-accessible contents pointer of a shared-mode buffer.
void* metal_buf_contents(MetalBufHandle buf);

// Release a buffer obtained from metal_buf_alias or metal_buf_alloc.
void metal_buf_release(MetalBufHandle buf);

// ---------- Command submission ----------------------------------------------
// Each dispatch_* function:
//   1. Creates a command buffer + compute command encoder.
//   2. Sets the pipeline state for the named kernel.
//   3. Binds the provided buffers and scalar constants.
//   4. Dispatches with a 1-D threadgroup grid.
//   5. Commits and calls waitUntilCompleted.
//
// Callers that need to batch multiple dispatches should use the raw Obj-C
// layer directly; these helpers cover the standard per-level dispatch pattern.

// merkle_leaves: one thread per leaf row.
//   in_buf   = input matrix [num_rows * ncols]  (buffer(0))
//   tree_buf = output tree  [num_rows * 4]       (buffer(1))
//   ncols    = element count per row             (buffer(2) constant)
//   dim      = element stride (always 1 for this use)  (buffer(3) constant)
//   num_rows = dispatch thread count
void metal_dispatch_merkle_leaves(MetalCtxHandle ctx,
                                   MetalBufHandle in_buf,
                                   MetalBufHandle tree_buf,
                                   uint32_t ncols,
                                   uint32_t dim,
                                   uint32_t num_rows);

// merkle_leaves_simd: SIMD-group-cooperative variant. One simdgroup (32 lanes)
// hashes ONE row; lanes 0..11 each own one Poseidon state element; MDS uses
// `simd_shuffle`. Drastically lower per-thread register pressure than the
// single-threaded version → higher occupancy on M-series GPUs.
// Dispatch: num_rows threadgroups × 32 threads each.
void metal_dispatch_merkle_leaves_simd(MetalCtxHandle ctx,
                                        MetalBufHandle in_buf,
                                        MetalBufHandle tree_buf,
                                        uint32_t ncols,
                                        uint32_t dim,
                                        uint32_t num_rows);

// merkle_leaves_x2: one thread hashes TWO consecutive rows in lockstep.
// Doubles per-thread register pressure but exposes 2-way ILP to the compiler
// (non-data-dependent ops across the two hashes can issue interleaved).
// Mirrors the CPU NEON `hash_full_result_neon_2` strategy.
// Dispatch: ceil(num_rows / 2) threads.
void metal_dispatch_merkle_leaves_x2(MetalCtxHandle ctx,
                                       MetalBufHandle in_buf,
                                       MetalBufHandle tree_buf,
                                       uint32_t ncols,
                                       uint32_t dim,
                                       uint32_t num_rows);

// transpose_rowmajor: row-major [num_rows][ncols] → column-major
// [ncols][num_rows]. Prep kernel for the coalesced-read leaf variant.
// Uses 32x32 threadgroup tiles.
void metal_dispatch_transpose_rowmajor(MetalCtxHandle ctx,
                                        MetalBufHandle src_buf,
                                        MetalBufHandle dst_buf,
                                        uint32_t num_rows,
                                        uint32_t ncols);

// merkle_leaves_cm: coalesced-read variant. Reads column-major input so
// adjacent threads in a simdgroup fetch adjacent memory addresses. Much
// better bandwidth utilization than the row-major kernel when the input
// is large enough for L1 cache to matter.
void metal_dispatch_merkle_leaves_cm(MetalCtxHandle ctx,
                                      MetalBufHandle inp_cm_buf,
                                      MetalBufHandle tree_buf,
                                      uint32_t ncols,
                                      uint32_t dim,
                                      uint32_t num_rows);

// merkle_leaves_tg: fused-tile variant — each threadgroup cooperatively loads
// a 32-row × 8-col tile into threadgroup memory (coalesced), then each
// thread consumes its own row. No separate transpose buffer. Expected to
// beat the plain row-major kernel when memory bandwidth matters.
void metal_dispatch_merkle_leaves_tg(MetalCtxHandle ctx,
                                      MetalBufHandle in_buf,
                                      MetalBufHandle tree_buf,
                                      uint32_t ncols,
                                      uint32_t dim,
                                      uint32_t num_rows);

// Batched Merkle tree build: processes `count` trees in ONE command buffer,
// using separate compute encoders per tree. Since distinct trees touch
// distinct buffers, Metal's automatic hazard tracking across encoders
// allows tree N+1's leaf work to overlap with tree N's parent loops on the
// GPU scheduler. One waitUntilCompleted at the very end instead of per
// tree. Expected win for STARK-prover-style workloads that build many
// Merkle trees in sequence.
//
// in_bufs  = array of `count` input buffer handles (each [num_rows * ncols])
// tree_bufs = array of `count` tree buffer handles (each tree_bytes)
void metal_dispatch_merkletree_batch(MetalCtxHandle ctx,
                                      const MetalBufHandle* in_bufs,
                                      const MetalBufHandle* tree_bufs,
                                      uint32_t count,
                                      uint32_t ncols,
                                      uint32_t dim,
                                      uint32_t num_rows);

// merkle_parents: one thread per parent output node.
//   buf       = full tree buffer  (buffer(0))
//   nextIndex = flat element offset to start of children level  (buffer(1) constant)
//   pending   = number of child nodes at this level             (buffer(2) constant)
//   nextN     = ceil(pending/2) = dispatch thread count
void metal_dispatch_merkle_parents(MetalCtxHandle ctx,
                                    MetalBufHandle buf,
                                    uint32_t nextIndex,
                                    uint32_t pending,
                                    uint32_t nextN);

// Batched: encode all tree-level parent reductions into one command buffer
// with memoryBarrierWithScope:MTLBarrierScopeBuffers between levels and one
// waitUntilCompleted at the end. Kills log2(N) × ~1.5ms commit overhead.
void metal_dispatch_merkle_parents_all_levels(MetalCtxHandle ctx,
                                                MetalBufHandle buf,
                                                uint32_t initial_pending);

// ntt_reverse_permutation: one thread per row index in [0, domain_size).
//   buf        = data buffer [domain_size * ncols]  (buffer(0))
//   domainPow  = log2(domain_size)                  (buffer(1) constant)
//   ncols      = column count                       (buffer(2) constant)
void metal_dispatch_ntt_reverse_permutation(MetalCtxHandle ctx,
                                             MetalBufHandle buf,
                                             uint32_t domainPow,
                                             uint32_t ncols);

// ntt_rev_butterfly_s1: fused kernel that reads `src` in natural order and
// writes `dst` = output of (reverse-permutation ∘ phase-1 butterfly).
// Saves one full pass vs the separate reverse + s=1 butterfly dispatches.
// Must have src ≠ dst. Dispatch: (domain_size/2 * ncols) threads.
void metal_dispatch_ntt_rev_butterfly_s1(MetalCtxHandle ctx,
                                          MetalBufHandle src,
                                          MetalBufHandle dst,
                                          uint32_t domain_pow,
                                          uint32_t ncols);

// ntt_rev_butterfly_s1s2: fuses rev-perm + s=1 + s=2 butterflies into ONE
// pass. Saves two full read+write passes vs the unfused trio. Requires
// src ≠ dst and domain_pow ≥ 2. I_val = primitive 4th root of unity
// (roots[1 << (s_global - 2)] = ω_4).
// Dispatch: (domain_size/4 * ncols) threads.
void metal_dispatch_ntt_rev_butterfly_s1s2(MetalCtxHandle ctx,
                                             MetalBufHandle src,
                                             MetalBufHandle dst,
                                             uint32_t domain_pow,
                                             uint32_t ncols,
                                             uint64_t I_val);

// ntt_rev_butterfly_s1s2s3: fuses rev-perm + s=1 + s=2 + s=3 butterflies
// into ONE pass. Saves THREE full read+write passes vs the unfused
// sequence. Requires src ≠ dst and domain_pow ≥ 3.
//   I_val  = ω_4   = roots[1 << (s_global - 2)]
//   W8_val = ω_8   = roots[1 << (s_global - 3)]
//   W8c_val = ω_8^3 = Goldilocks::mul(W8, Goldilocks::mul(W8, W8))
// Dispatch: (domain_size/8 * ncols) threads.
void metal_dispatch_ntt_rev_butterfly_s1s2s3(MetalCtxHandle ctx,
                                              MetalBufHandle src,
                                              MetalBufHandle dst,
                                              uint32_t domain_pow,
                                              uint32_t ncols,
                                              uint64_t I_val,
                                              uint64_t W8_val,
                                              uint64_t W8c_val);

// ntt_butterfly_phase: one thread per (butterfly pair, col) = domain_size/2 * ncols.
//   buf                 = data buffer [domain_size * ncols]  (buffer(0))
//   twiddles            = full roots array [1 << s_global]    (buffer(1))
//                         Reused across all phases of a call; staged ONCE
//                         per NTT_Metal invocation via the twiddle cache.
//   ncols               = column count                        (buffer(2) constant)
//   domain_size         = 2^domainPow                         (buffer(3) constant)
//   s                   = current phase level                 (buffer(4) constant)
//   roots_stride_shift  = s_global - s                        (buffer(5) constant)
//                         Kernel reads twiddles[j << roots_stride_shift]
//                         to get root(s, j); matches CPU `root()` accessor.
void metal_dispatch_ntt_butterfly_phase(MetalCtxHandle ctx,
                                         MetalBufHandle buf,
                                         MetalBufHandle twiddles,
                                         uint32_t ncols,
                                         uint32_t domain_size,
                                         uint32_t s,
                                         uint32_t roots_stride_shift);

// Batched variant: encode phases s = start_s..domain_pow in one command
// buffer with a single waitUntilCompleted. Pass start_s=1 for a full NTT,
// or start_s=2 if the s=1 phase was already consumed by the fused
// ntt_rev_butterfly_s1 kernel.
void metal_dispatch_ntt_butterfly_all_phases(MetalCtxHandle ctx,
                                              MetalBufHandle buf,
                                              MetalBufHandle twiddles,
                                              uint32_t ncols,
                                              uint32_t domain_size,
                                              uint32_t start_s,
                                              uint32_t domain_pow,
                                              uint32_t s_global);

// intt_reorder: inverse-NTT index permutation out[(N-i) % N] = in[i], in-place.
//   buf          = data buffer  (buffer(0))
//   domain_size  = N             (buffer(1) constant)
//   ncols        = column count  (buffer(2) constant)
// Dispatch (domain_size / 2) threads.
void metal_dispatch_intt_reorder(MetalCtxHandle ctx,
                                  MetalBufHandle buf,
                                  uint32_t domain_size,
                                  uint32_t ncols);

// intt_reorder_scale: fused reorder + 1/N scale in one kernel (one pass,
// one commit+wait). Replaces the separate intt_reorder + intt_scale
// dispatches.
void metal_dispatch_intt_reorder_scale(MetalCtxHandle ctx,
                                        MetalBufHandle buf,
                                        uint32_t domain_size,
                                        uint32_t ncols,
                                        uint64_t inv_n);

// intt_reorder_coset_scale: variant of intt_reorder_scale that applies a
// PER-ELEMENT coset multiplier r_inv[dsty] instead of the scalar 1/N. This
// is the fused reorder+coset-shift step at the tail of an LDE INTT.
//
// `r_inv` MUST point to a buffer of at least `domain_size` ulongs matching
// NTT_Goldilocks::r_ (shift^i / N, computed by `computeR`).
void metal_dispatch_intt_reorder_coset_scale(MetalCtxHandle ctx,
                                              MetalBufHandle buf,
                                              MetalBufHandle r_inv,
                                              uint32_t domain_size,
                                              uint32_t ncols);

// intt_scale: one thread per flat element [0, domain_size * ncols).
//   buf    = data buffer  (buffer(0))
//   inv_n  = 1/domain_size mod p  (buffer(1) constant ulong)
//   count  = domain_size * ncols  (buffer(2) constant)
void metal_dispatch_intt_scale(MetalCtxHandle ctx,
                                MetalBufHandle buf,
                                uint64_t inv_n,
                                uint32_t count);

// ---------- Twiddle cache ---------------------------------------------------
// Returns (or creates) a shared MTLBuffer containing the NTT roots for
// the given array length (which must be a power of 2).
// roots_ptr points to host-side roots[0..roots_len-1] (each a uint64_t).
MetalBufHandle metal_twiddle_buffer(MetalCtxHandle ctx,
                                     const uint64_t* roots_ptr,
                                     uint64_t roots_len);

#ifdef __cplusplus
}
#endif

#endif // GOLDILOCKS_HAS_METAL
