// poseidon_metal.mm — Metal implementation of merkletree_metal.
//
// Mirrors the structure of PoseidonGoldilocks::merkletree_seq:
//   1. Dispatch merkle_leaves to compute leaf hashes (one thread per row).
//   2. Level-by-level dispatch of merkle_parents until root is reached.
//
// Buffer strategy:
//   - For `input`: try zero-copy alias (newBufferWithBytesNoCopy) if ptr is
//     16 KB page-aligned. Otherwise copy in. No readback needed for input.
//   - For `tree`:  same alias attempt; if a copy was made, readback via
//     memcpy from buffer.contents into caller's tree pointer after all work.
//
// OMP constraint: do NOT call this from inside an OMP parallel region.
//   The GPU parallelises all leaf hashing internally; the host loop over
//   tree levels is sequential on the calling thread.
//
// Build: clang++ -std=c++17 -fobjc-arc -ObjC++ -DGOLDILOCKS_HAS_METAL \
//               -framework Metal -framework Foundation -I../../src

#ifdef GOLDILOCKS_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <vector>

#include "../goldilocks_base_field.hpp"
#include "metal_context.hpp"
#include "goldilocks_metal.hpp"  // declares goldilocks_metal::merkletree_metal

// A/B switch for leaf kernel. 0 = one-thread-per-row (default; shipped).
// 1 = SIMD-cooperative (experimental, being A/B'd). Set via env var
// GOLDILOCKS_METAL_COOP by `bench_merkle`, or left at 0 for normal callers.
// Weak symbol so test harnesses can override without linker games.
// A/B selector for leaf kernel:
//   0 = row-per-thread (default, shipped)
//   1 = SIMD-cooperative
//   2 = 2-rows-per-thread (ILP variant)
//   3 = transpose + column-major (coalesced memory reads)
extern "C" { __attribute__((weak)) int g_merkle_use_simd_coop = 0; }

// Constants mirroring poseidon_goldilocks.hpp / merklehash_goldilocks.hpp
static constexpr uint32_t HASH_SIZE_ = 4;   // CAPACITY = HASH_SIZE = 4 elements
static constexpr uint32_t RATE_      = 8;   // RATE = 8 elements

namespace goldilocks_metal {

void merkletree_metal(Goldilocks::Element* tree,
                      Goldilocks::Element* input,
                      uint64_t num_cols,
                      uint64_t num_rows)
{
    @autoreleasepool {
        if (num_rows == 0) return;

        MetalCtxHandle ctx = metal_context_get();

        // -------------------------------------------------------------------
        // numElementsTree = (2*num_rows - 1) * HASH_SIZE
        // leaf hashes occupy tree[0 .. num_rows*4)
        // parent levels are stacked above
        // -------------------------------------------------------------------
        uint64_t numElementsTree = (2 * num_rows - 1) * HASH_SIZE_;
        size_t tree_bytes  = numElementsTree * sizeof(Goldilocks::Element);
        size_t input_bytes = num_rows * num_cols * sizeof(Goldilocks::Element);

        // -------------------------------------------------------------------
        // Allocate / alias buffers
        // -------------------------------------------------------------------
        int input_is_copy = 0;
        int tree_is_copy  = 0;
        MetalBufHandle input_buf = NULL;
        MetalBufHandle tree_buf  = NULL;

        // Input buffer (read-only by GPU): try zero-copy alias if num_cols > 0.
        if (num_cols > 0 && input != NULL) {
            input_buf = metal_buf_alias(ctx, (void*)input, input_bytes, &input_is_copy);
        } else {
            // num_cols == 0: merkle_leaves kernel still runs but reads nothing.
            // Allocate a tiny dummy buffer (1 byte minimum).
            input_buf = metal_buf_alloc(ctx, sizeof(uint64_t));
        }

        // Tree buffer (read-write by GPU and CPU): alias or alloc.
        tree_buf = metal_buf_alias(ctx, (void*)tree, tree_bytes, &tree_is_copy);

        // -------------------------------------------------------------------
        // Phase 1: merkle_leaves — one thread per leaf row.
        // kernel signature: (inp, tree, ncols, dim)
        // dim is always 1 for merkletree (no cubic extension here).
        // -------------------------------------------------------------------
        uint32_t ncols32 = (uint32_t)num_cols;
        uint32_t dim32   = 1;
        uint32_t nrows32 = (uint32_t)num_rows;

        switch (g_merkle_use_simd_coop) {
            case 1:
                metal_dispatch_merkle_leaves_simd(ctx, input_buf, tree_buf,
                                                  ncols32, dim32, nrows32);
                break;
            case 2:
                metal_dispatch_merkle_leaves_x2(ctx, input_buf, tree_buf,
                                                 ncols32, dim32, nrows32);
                break;
            case 3: {
                // Transpose input to column-major, then run coalesced-read
                // kernel. Extra pass but unlocks simdgroup memory coalescing
                // for the pod12 absorb loop (8 reads per thread × 16
                // iterations per row, adjacent threads → adjacent addresses).
                MetalBufHandle inp_cm = metal_buf_alloc(ctx, input_bytes);
                metal_dispatch_transpose_rowmajor(ctx, input_buf, inp_cm,
                                                   nrows32, ncols32);
                metal_dispatch_merkle_leaves_cm(ctx, inp_cm, tree_buf,
                                                 ncols32, dim32, nrows32);
                metal_buf_release(inp_cm);
                break;
            }
            case 4:
                // Fused-tile: cooperative threadgroup load + pod12 in one
                // kernel. Avoids the separate transpose pass's alloc and
                // round-trip through global memory.
                metal_dispatch_merkle_leaves_tg(ctx, input_buf, tree_buf,
                                                 ncols32, dim32, nrows32);
                break;
            default:
                metal_dispatch_merkle_leaves(ctx, input_buf, tree_buf,
                                              ncols32, dim32, nrows32);
                break;
        }

        // -------------------------------------------------------------------
        // Phase 2: merkle_parents — one GPU dispatch per tree level.
        // Mirrors the CPU loop in merkletree_seq:
        //   pending     = num_rows (children at current level)
        //   nextN       = ceil(pending / 2) = number of parent nodes to write
        //   nextIndex   = flat element offset to START of children at this level
        // -------------------------------------------------------------------
        // All tree levels in one command buffer (one waitUntilCompleted).
        metal_dispatch_merkle_parents_all_levels(ctx,
                                                  tree_buf,
                                                  (uint32_t)num_rows);

        // -------------------------------------------------------------------
        // Readback if tree buffer was a copy (not a zero-copy alias).
        // -------------------------------------------------------------------
        if (tree_is_copy) {
            void* gpu_ptr = metal_buf_contents(tree_buf);
            std::memcpy(tree, gpu_ptr, tree_bytes);
        }

        // Release buffer handles (decrements ARC-retained count).
        metal_buf_release(input_buf);
        metal_buf_release(tree_buf);
    }  // @autoreleasepool
}

// ---------------------------------------------------------------------------
// merkletree_metal_batch — build `count` Merkle trees in one GPU submission.
//
// Why this helps: the GPU queue runs command buffers serially, but WITHIN
// one command buffer Metal's hazard tracker allows dispatches from different
// compute encoders to overlap if they touch different resources. Putting
// each tree in its OWN encoder inside one big command buffer lets the GPU
// start tree N+1's leaf kernel while tree N's parent loop is still running.
//
// All trees must share (num_cols, num_rows). Same bit-exact contract as
// merkletree_metal.
void merkletree_metal_batch(Goldilocks::Element** trees,
                            Goldilocks::Element** inputs,
                            uint64_t count,
                            uint64_t num_cols,
                            uint64_t num_rows)
{
    if (count == 0 || num_rows == 0) return;
    @autoreleasepool {
        MetalCtxHandle ctx = metal_context_get();

        uint64_t numElementsTree = (2 * num_rows - 1) * HASH_SIZE_;
        size_t   tree_bytes      = numElementsTree * sizeof(Goldilocks::Element);
        size_t   input_bytes     = num_rows * num_cols * sizeof(Goldilocks::Element);

        std::vector<MetalBufHandle> in_bufs(count, nullptr);
        std::vector<MetalBufHandle> tree_bufs(count, nullptr);
        std::vector<int>            in_is_copy(count, 0);
        std::vector<int>            tree_is_copy(count, 0);

        for (uint64_t i = 0; i < count; i++) {
            if (num_cols > 0 && inputs[i] != nullptr) {
                in_bufs[i] = metal_buf_alias(ctx, (void*)inputs[i],
                                              input_bytes, &in_is_copy[i]);
            } else {
                in_bufs[i] = metal_buf_alloc(ctx, sizeof(uint64_t));
            }
            tree_bufs[i] = metal_buf_alias(ctx, (void*)trees[i],
                                            tree_bytes, &tree_is_copy[i]);
        }

        uint32_t ncols32 = (uint32_t)num_cols;
        uint32_t dim32   = 1;
        uint32_t nrows32 = (uint32_t)num_rows;

        metal_dispatch_merkletree_batch(ctx,
                                         in_bufs.data(),
                                         tree_bufs.data(),
                                         (uint32_t)count,
                                         ncols32, dim32, nrows32);

        // Readback copy-fallback trees and release handles.
        for (uint64_t i = 0; i < count; i++) {
            if (tree_is_copy[i]) {
                void* gpu_ptr = metal_buf_contents(tree_bufs[i]);
                std::memcpy(trees[i], gpu_ptr, tree_bytes);
            }
            metal_buf_release(in_bufs[i]);
            metal_buf_release(tree_bufs[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Page-aligned allocator for zero-copy MTLBuffer aliasing.
//
// MTLDevice `newBufferWithBytesNoCopy` requires the backing pointer to be
// aligned to a VM page boundary (16 KB on Apple Silicon). When callers
// allocate via `new Element[]` or `malloc`, alignment is typically 16 B
// — far too small — so the Metal bridge falls back to a `newBufferWithBytes`
// copy + readback. For large inputs/outputs the memcpy dominates.
//
// allocate_aligned_elements returns page-aligned memory so the bridge takes
// the zero-copy path automatically.
//
// Implementation: posix_memalign with 16 KB alignment, rounded up to the
// nearest page (posix_memalign doesn't require size to be a multiple of
// alignment on modern libc, but some older libs do).
Goldilocks::Element* allocate_aligned_elements(uint64_t n) {
    if (n == 0) return nullptr;
    constexpr size_t PAGE = 16384;
    size_t bytes = n * sizeof(Goldilocks::Element);
    size_t aligned = (bytes + PAGE - 1) & ~(PAGE - 1);
    void* p = nullptr;
    if (posix_memalign(&p, PAGE, aligned) != 0) return nullptr;
    return reinterpret_cast<Goldilocks::Element*>(p);
}

void free_aligned(void* ptr) {
    free(ptr);
}

}  // namespace goldilocks_metal

#endif // GOLDILOCKS_HAS_METAL
