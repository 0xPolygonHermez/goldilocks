#pragma once
#ifdef GOLDILOCKS_HAS_METAL
#include <cstdint>
#include "../goldilocks_base_field.hpp"
class NTT_Goldilocks;  // forward declaration — avoids pulling in ntt_goldilocks.hpp

// THREADING CONTRACT: the bridge entries below MUST be invoked from the
// main/submitting thread — NEVER from inside a `#pragma omp parallel` region.
// Metal framework objects require an @autoreleasepool on the calling thread;
// OMP worker threads do not have one and would leak Obj-C objects per iter.
// The bridge already wraps each entry in @autoreleasepool, but only the
// single thread that called the entry is protected.
namespace goldilocks_metal {
    void merkletree_metal(Goldilocks::Element* tree, Goldilocks::Element* input,
                          uint64_t num_cols, uint64_t num_rows);
    void NTT_Metal(Goldilocks::Element* dst, Goldilocks::Element* src,
                   uint64_t size, uint64_t ncols, NTT_Goldilocks* ntt_ctx, bool inverse);

    // Low-Degree Extension. Mirrors NTT_Goldilocks::extendPol semantics:
    //   1. Copy `input` (N × ncols) into the first N×ncols of `output`.
    //   2. INTT butterflies on that region; apply coset reorder+scale using
    //      r_[] (computed per NTT_Goldilocks::computeR on the caller's ctx).
    //   3. Zero the tail [N×ncols, N_Extended×ncols).
    //   4. Forward NTT on the full N_Extended × ncols buffer.
    // `ntt_ctx` must be sized for N (caller calls `computeR(N)` if needed).
    // `r_inv` points to ntt_ctx->r_ (shift^i / N, length N).
    void extendPol_Metal(Goldilocks::Element* output,
                         Goldilocks::Element* input,
                         uint64_t N_Extended,
                         uint64_t N,
                         uint64_t ncols,
                         NTT_Goldilocks* ntt_ctx,
                         Goldilocks::Element* r_inv);

    // Allocate an array of `n` Goldilocks::Element on a 16KB-page-aligned
    // boundary. Buffers passed to merkletree_metal/NTT_Metal that come from
    // this allocator hit the `newBufferWithBytesNoCopy` zero-copy path inside
    // the Metal bridge — no memcpy in or memcpy-readback out, which matters
    // for large inputs (e.g., a 262k-row × 128-col tree is 256 MB of input
    // and ~8 MB of tree readback avoided on the `tree` side alone).
    //
    // Returns nullptr on allocation failure. Must be freed with free_aligned.
    Goldilocks::Element* allocate_aligned_elements(uint64_t n);
    void                 free_aligned(void* ptr);

    // Batched Merkle-tree build: processes `count` trees in ONE Metal command
    // buffer using separate compute encoders per tree so the GPU scheduler
    // can overlap tree N+1's leaf work with tree N's parent loops. Expected
    // ~1.3-1.5× throughput improvement over sequential merkletree_metal calls
    // for STARK-prover-style workloads that build many Merkle trees in a row
    // (e.g., one per column group, 10-40 trees per proof).
    //
    // All trees MUST share the same (num_cols, num_rows) shape. Each
    // tree[i] / input[i] pair is treated as it would be by merkletree_metal
    // (same layout, same bit-exact output guarantee). Returns when all
    // trees are fully written back.
    void merkletree_metal_batch(Goldilocks::Element** trees,
                                Goldilocks::Element** inputs,
                                uint64_t count,
                                uint64_t num_cols,
                                uint64_t num_rows);
}
#endif
