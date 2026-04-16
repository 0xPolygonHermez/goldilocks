// ntt_metal.mm — Metal implementation of NTT_Metal.
//
// Implements the Cooley-Tukey DIT NTT on the GPU using three kernel types:
//   1. ntt_reverse_permutation  — bit-reverse permute the source into dst
//   2. ntt_butterfly_phase      — one level s of butterfly per dispatch
//   3. intt_scale               — multiply all elements by 1/N (inverse only)
//
// Algorithm mirrors NTT_Goldilocks::NTT_iters (ntt_goldilocks.cpp), plain path
// (no coset/extend — deferred per SDD).
//
// Twiddle factors:
//   The GPU butterfly kernel for level s needs mdiv2 = 2^(s-1) twiddle factors.
//   These are roots[j << (ctx_s - s)] for j in [0, mdiv2), which are exactly
//   the first mdiv2 entries of ntt_ctx->root(s, j).
//   We stage the FULL roots array (length 1 << ctx_s) into a single MTLBuffer
//   keyed by roots_len, then the kernel indexes into it the same way root() does
//   by passing the full roots buffer and having the kernel use j directly
//   (kernel uses twiddles[j] where twiddles is roots shifted by (ctx_s-s) stride).
//
//   Simpler approach (used here): for each level s, extract the relevant slice
//   into the full roots buffer and pass the full buffer — the kernel reads
//   twiddles[j] and we set the twiddles buffer pointer to roots + offset where
//   offset = 0 and the kernel's stride into the buffer is j << (s - domainPow)...
//
//   Actually: ntt_butterfly_phase kernel uses twiddles[j] directly for j in
//   [0, mdiv2). root(domainPow, j) = roots[j << (ctx_s - domainPow)].
//   So for level s: twiddle[j] = roots[j << (ctx_s - s)].
//   The simplest correct approach: for each phase s, pass an offset into the
//   full roots buffer equal to byte_offset = 0, and let the kernel index:
//     twiddles[j * (1 << (ctx_s - s))]
//   BUT the kernel as written uses twiddles[j] without a stride parameter.
//
//   Resolution: pre-build a per-level twiddle slice using metal_twiddle_buffer
//   keyed by (domainPow << 32 | s), or — simpler — compute the slice on the CPU
//   and upload as a temporary buffer. We choose: for each call, build a full
//   flattened twiddles array of length (domain_size/2) that is EXACTLY the roots
//   indexed as root(domainPow, j) = roots[j << (s_global - domainPow)] for
//   j in [0, domain_size/2). This is a one-time upload at NTT_Metal call time.
//
//   Then for each phase s, the kernel needs mdiv2 = 2^(s-1) twiddle values:
//     w[j] = roots[j << (s_global - s)]  for j in [0, 2^(s-1))
//   We can represent this as a stride-based read into the full roots buffer:
//     offset_elems = 0, stride = 1 << (s_global - s)
//   But the kernel doesn't support stride. So we pack per-level slice buffers
//   at buffer creation time (one per level) into a single staging buffer, and
//   pass the slice as a buffer with byte_offset.
//
//   FINAL CHOICE (simplest, provably correct):
//   Allocate one buffer of length (domain_size/2) from the full roots array.
//   For level s, the twiddle for index j is:
//       twiddle_j = roots[j << (s_global - s)]
//   We allocate a single "all-twiddles" buffer of domain_size/2 elements where
//   element[j] = roots[j << (s_global - domainPow)] — i.e. root(domainPow, j).
//   For level s < domainPow, the needed twiddles are a SUBSET: indices 0,2,...
//   strided by 2^(domainPow-s). The kernel is dispatched with mdiv2 threads and
//   uses twiddles[j]. We need twiddles[j] = roots[j << (s_global - s)].
//
//   We do per-phase: build a CPU-side array of mdiv2 uint64_t values from
//   ntt_ctx->roots_ptr() and upload as a fresh buffer per phase.
//   This is O(N/2) copies total across all phases = O(N log N) ops — acceptable.
//
// OMP constraint: NTT_Metal must not be called from within an OMP parallel region.
//
// Build: clang++ -std=c++17 -fobjc-arc -ObjC++ -DGOLDILOCKS_HAS_METAL \
//               -framework Metal -framework Foundation -I../../src

#ifdef GOLDILOCKS_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "../goldilocks_base_field.hpp"
#include "../ntt_goldilocks.hpp"
#include "metal_context.hpp"
#include "goldilocks_metal.hpp"  // declares goldilocks_metal::NTT_Metal

namespace goldilocks_metal {

void NTT_Metal(Goldilocks::Element* dst,
               Goldilocks::Element* src,
               uint64_t size,
               uint64_t ncols,
               NTT_Goldilocks* ntt_ctx,
               bool inverse)
{
    @autoreleasepool {
        if (size == 0 || ncols == 0) return;

        MetalCtxHandle ctx = metal_context_get();

        uint32_t domainPow = 0;
        {
            uint64_t s = size;
            while (s > 1) { s >>= 1; domainPow++; }
        }

        // Globals from ntt_ctx
        uint32_t s_global = ntt_ctx->get_s();
        const Goldilocks::Element* roots_raw = ntt_ctx->roots_ptr();
        uint64_t rlen = ntt_ctx->roots_len();  // = 1 << s_global
        (void)rlen;

        size_t data_bytes = size * ncols * sizeof(Goldilocks::Element);
        uint32_t ncols32  = (uint32_t)ncols;
        uint32_t domain32 = (uint32_t)size;

        int dst_is_copy = 0;
        MetalBufHandle dst_buf = metal_buf_alias(ctx, (void*)dst, data_bytes, &dst_is_copy);

        // -------------------------------------------------------------------
        // Fused-path (preferred): src ≠ dst. Skip the memcpy + reverse +
        // phase-1 butterfly (three passes over the data) and do all three
        // in one pass via ntt_rev_butterfly_s1.
        //
        // Fallback: src == dst (in-place). memcpy is a no-op, we must
        // run the legacy reverse-permutation + full phase loop.
        // -------------------------------------------------------------------
        MetalBufHandle tw_buf = metal_twiddle_buffer(
            ctx,
            reinterpret_cast<const uint64_t*>(roots_raw),
            rlen);

        if (dst != src && domainPow >= 3) {
            // Fused rev + s=1 + s=2 + s=3 in one pass (four passes collapsed).
            int src_is_copy = 0;
            MetalBufHandle src_buf = metal_buf_alias(ctx, (void*)src,
                                                      data_bytes, &src_is_copy);

            uint64_t I_val  = roots_raw[1ULL << (s_global - 2)].fe;  // ω_4
            uint64_t W8_val = roots_raw[1ULL << (s_global - 3)].fe;  // ω_8
            // ω_8^3 via Goldilocks field mul. Cheap: two mods, done once.
            Goldilocks::Element w8_el  = Goldilocks::fromU64(W8_val);
            Goldilocks::Element w8sq   = w8_el  * w8_el;
            Goldilocks::Element w8_cub = w8sq   * w8_el;
            uint64_t W8c_val = Goldilocks::toU64(w8_cub);

            metal_dispatch_ntt_rev_butterfly_s1s2s3(
                ctx, src_buf, dst_buf, domainPow, ncols32,
                I_val, W8_val, W8c_val);

            // Continue from phase s=4.
            metal_dispatch_ntt_butterfly_all_phases(
                ctx, dst_buf, tw_buf,
                ncols32, domain32,
                /*start_s=*/4, domainPow, s_global);

            metal_buf_release(src_buf);
        } else if (dst != src && domainPow == 2) {
            // domainPow < 3: fall back to the 2-stage fused kernel.
            int src_is_copy = 0;
            MetalBufHandle src_buf = metal_buf_alias(ctx, (void*)src,
                                                      data_bytes, &src_is_copy);
            uint64_t I_val = roots_raw[1ULL << (s_global - 2)].fe;
            metal_dispatch_ntt_rev_butterfly_s1s2(
                ctx, src_buf, dst_buf, domainPow, ncols32, I_val);
            metal_dispatch_ntt_butterfly_all_phases(
                ctx, dst_buf, tw_buf,
                ncols32, domain32,
                /*start_s=*/3, domainPow, s_global);
            metal_buf_release(src_buf);
        } else if (dst != src && domainPow == 1) {
            // Fallback to s=1-only fused kernel for N=2.
            int src_is_copy = 0;
            MetalBufHandle src_buf = metal_buf_alias(ctx, (void*)src,
                                                      data_bytes, &src_is_copy);
            metal_dispatch_ntt_rev_butterfly_s1(
                ctx, src_buf, dst_buf, domainPow, ncols32);
            metal_buf_release(src_buf);
        } else {
            // In-place path (src == dst) or trivial N=1.
            if (dst != src) std::memcpy(dst, src, data_bytes);
            metal_dispatch_ntt_reverse_permutation(ctx, dst_buf, domainPow, ncols32);
            metal_dispatch_ntt_butterfly_all_phases(
                ctx, dst_buf, tw_buf,
                ncols32, domain32,
                /*start_s=*/1, domainPow, s_global);
        }

        // -------------------------------------------------------------------
        // 5. Inverse NTT: scale each element by 1/N mod p.
        //    powTwoInv[domainPow] holds the precomputed value; access via
        //    root accessor can't get it — use the public root() interface:
        //    Actually powTwoInv is not exposed. Compute 1/N using Goldilocks::inv.
        // -------------------------------------------------------------------
        if (inverse) {
            // INTT = forward-NTT butterflies + index permutation (N-i) % N + scale 1/N.
            // Fused reorder+scale kernel does both in one pass/commit vs the
            // previous two-dispatch approach (see ntt_reorder_scale in
            // ntt.metal).
            Goldilocks::Element inv_n = Goldilocks::inv(Goldilocks::fromU64(size));
            uint64_t inv_n_u64 = Goldilocks::toU64(inv_n);
            metal_dispatch_intt_reorder_scale(ctx, dst_buf,
                                                domain32, ncols32, inv_n_u64);
        }

        // -------------------------------------------------------------------
        // 6. Readback if buffer was a copy.
        // -------------------------------------------------------------------
        if (dst_is_copy) {
            void* gpu_ptr = metal_buf_contents(dst_buf);
            std::memcpy(dst, gpu_ptr, data_bytes);
        }

        metal_buf_release(dst_buf);
    }  // @autoreleasepool
}

// ---------------------------------------------------------------------------
// extendPol_Metal — Low-Degree Extension on the GPU.
//
// Pipeline (mirrors NTT_Goldilocks::extendPol):
//   1. memcpy input (N × ncols) into the first N×ncols of `output`.
//   2. memset the tail [N×ncols, N_Extended×ncols) of `output` to zero.
//   3. In-place INTT butterflies on the first N×ncols (domain_size = N).
//   4. Coset reorder+scale: intt_reorder_coset_scale(buf, r_inv, N, ncols).
//      After this step the first N×ncols of output holds the CPU extendPol
//      intermediate (INTT-with-coset), and the tail is still zero.
//   5. In-place forward NTT on the full N_Extended × ncols buffer.
//
// The INTT step runs with `src == dst` (both the user's output buffer) so
// the legacy in-place path is used. The subsequent forward NTT is also
// in-place, same reason. Using the fused rev+s1+s2+s3 kernel would require
// an N_Extended × ncols scratch buffer, which doubles memory use at the
// 2²³ × 100 benchmark scale — not worth the ~15% speedup.
//
// Twiddles for the N-sized INTT come from ntt_ctx (already sized for N).
// Twiddles for the N_Extended NTT come from a second NTT_Goldilocks instance
// allocated here; this matches how CPU extendPol builds its
// ntt_extension context on the fly.
//
// Bit-exact vs CPU reference for every shape measured (see bench_lde.mm).
void extendPol_Metal(Goldilocks::Element* output,
                     Goldilocks::Element* input,
                     uint64_t N_Extended,
                     uint64_t N,
                     uint64_t ncols,
                     NTT_Goldilocks* ntt_ctx,
                     Goldilocks::Element* r_inv) {
    @autoreleasepool {
        if (N_Extended == 0 || N == 0 || ncols == 0) return;

        MetalCtxHandle ctx = metal_context_get();

        uint32_t domainPow_N = 0;
        { uint64_t s = N; while (s > 1) { s >>= 1; domainPow_N++; } }
        uint32_t domainPow_Ext = 0;
        { uint64_t s = N_Extended; while (s > 1) { s >>= 1; domainPow_Ext++; } }

        uint32_t ncols32 = (uint32_t)ncols;
        uint32_t N32     = (uint32_t)N;
        uint32_t NExt32  = (uint32_t)N_Extended;

        // Step 1 & 2: stage output = [input | zeros]
        std::memcpy(output, input,
                    N * ncols * sizeof(Goldilocks::Element));
        std::memset(output + N * ncols, 0,
                    (N_Extended - N) * ncols * sizeof(Goldilocks::Element));

        // Alias the full N_Extended × ncols buffer for all GPU work.
        size_t bytes_ext = N_Extended * ncols * sizeof(Goldilocks::Element);
        int out_is_copy = 0;
        MetalBufHandle out_buf = metal_buf_alias(
            ctx, (void*)output, bytes_ext, &out_is_copy);

        // Step 3: in-place INTT butterflies on the first N rows.
        //   INTT = (forward NTT butterflies) + (reorder + scale) — here we
        //   run the butterflies then replace the reorder+scale with the
        //   coset variant.
        uint32_t s_global_N = ntt_ctx->get_s();
        const uint64_t* roots_N = reinterpret_cast<const uint64_t*>(
            ntt_ctx->roots_ptr());
        uint64_t rlen_N = ntt_ctx->roots_len();
        MetalBufHandle tw_N = metal_twiddle_buffer(ctx, roots_N, rlen_N);

        metal_dispatch_ntt_reverse_permutation(
            ctx, out_buf, domainPow_N, ncols32);
        metal_dispatch_ntt_butterfly_all_phases(
            ctx, out_buf, tw_N, ncols32, N32,
            /*start_s=*/1, domainPow_N, s_global_N);

        // Step 4: coset reorder+scale using r_[] = shift^i / N.
        //   r_inv points to ntt_ctx->r_, which is length N.
        int r_is_copy = 0;
        MetalBufHandle r_buf = metal_buf_alias(
            ctx, (void*)r_inv,
            N * sizeof(Goldilocks::Element), &r_is_copy);
        metal_dispatch_intt_reorder_coset_scale(
            ctx, out_buf, r_buf, N32, ncols32);
        metal_buf_release(r_buf);

        // Step 5: in-place forward NTT on the full extended domain.
        //   Need roots for the extended size. Build a second NTT context.
        //   (Matches CPU extendPol at ntt_goldilocks.cpp:375.)
        NTT_Goldilocks ntt_extension(N_Extended, 1, N_Extended / N);
        uint32_t s_global_Ext = ntt_extension.get_s();
        const uint64_t* roots_Ext = reinterpret_cast<const uint64_t*>(
            ntt_extension.roots_ptr());
        uint64_t rlen_Ext = ntt_extension.roots_len();
        MetalBufHandle tw_Ext = metal_twiddle_buffer(ctx, roots_Ext, rlen_Ext);

        metal_dispatch_ntt_reverse_permutation(
            ctx, out_buf, domainPow_Ext, ncols32);
        metal_dispatch_ntt_butterfly_all_phases(
            ctx, out_buf, tw_Ext, ncols32, NExt32,
            /*start_s=*/1, domainPow_Ext, s_global_Ext);

        // Readback if aliasing had to copy.
        if (out_is_copy) {
            void* gpu_ptr = metal_buf_contents(out_buf);
            std::memcpy(output, gpu_ptr, bytes_ext);
        }
        metal_buf_release(out_buf);
    }  // @autoreleasepool
}

}  // namespace goldilocks_metal

#endif // GOLDILOCKS_HAS_METAL
