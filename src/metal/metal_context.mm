// metal_context.mm — Metal singleton context: device, queue, library, pipeline
// and twiddle caches. ARC enabled; all Obj-C objects retained by the singleton.
//
// Build: clang++ -std=c++17 -fobjc-arc -ObjC++ -DGOLDILOCKS_HAS_METAL \
//               -framework Metal -framework Foundation -I../../src

#ifdef GOLDILOCKS_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#include "metal_context.hpp"

// ---------------------------------------------------------------------------
// Internal Obj-C class holding all Metal state
// ---------------------------------------------------------------------------
@interface GoldilocksMetalContext : NSObject {
@public
    id<MTLDevice>       _device;
    id<MTLCommandQueue> _queue;
    id<MTLLibrary>      _library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> _pipelineCache;
    // key = roots_len (number of uint64_t elements)
    std::unordered_map<uint64_t, id<MTLBuffer>> _twiddleCache;
    std::mutex _mutex;
}
@end

@implementation GoldilocksMetalContext
- (instancetype)init {
    self = [super init];
    if (self) {
        _device  = MTLCreateSystemDefaultDevice();
        if (_device == nil) {
            NSLog(@"GoldilocksMetalContext: MTLCreateSystemDefaultDevice() returned nil. "
                  @"No Metal-capable GPU found.");
            abort();
        }
        NSLog(@"GoldilocksMetalContext: Metal device = %@", [_device name]);
        _queue = [_device newCommandQueue];
        if (_queue == nil) {
            NSLog(@"GoldilocksMetalContext: newCommandQueue() failed.");
            abort();
        }
        _library = nil;
    }
    return self;
}
@end

// ---------------------------------------------------------------------------
// Singleton via dispatch_once
// ---------------------------------------------------------------------------
static GoldilocksMetalContext* g_ctx = nil;
static dispatch_once_t g_once;

static GoldilocksMetalContext* get_impl(MetalCtxHandle h) {
    return (__bridge GoldilocksMetalContext*)(h);
}

// ---------------------------------------------------------------------------
// Public C API implementation
// ---------------------------------------------------------------------------

MetalCtxHandle metal_context_get(void) {
    dispatch_once(&g_once, ^{
        @autoreleasepool {
            g_ctx = [[GoldilocksMetalContext alloc] init];
        }
    });
    return (__bridge void*)(g_ctx);
}

int metal_context_load_library(MetalCtxHandle ctx, const char* metallib_path) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        std::lock_guard<std::mutex> lock(impl->_mutex);

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSURL*    url  = [NSURL fileURLWithPath:path];
        NSError*  err  = nil;
        id<MTLLibrary> lib = [impl->_device newLibraryWithURL:url error:&err];
        if (lib == nil) {
            NSLog(@"metal_context_load_library: failed to load '%@': %@", path, err);
            return -1;
        }
        impl->_library = lib;
        NSLog(@"metal_context_load_library: loaded '%@'", path);
        return 0;
    }
}

int metal_context_load_source(MetalCtxHandle ctx, const char* source) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        std::lock_guard<std::mutex> lock(impl->_mutex);

        NSString* src = [NSString stringWithUTF8String:source];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        NSError* err = nil;
        id<MTLLibrary> lib = [impl->_device newLibraryWithSource:src
                                                         options:opts
                                                           error:&err];
        if (lib == nil) {
            NSLog(@"metal_context_load_source: compile failed: %@", err);
            return -1;
        }
        if (err != nil) {
            // Warnings — not fatal.
            NSLog(@"metal_context_load_source: compile warnings: %@", err);
        }
        impl->_library = lib;
        return 0;
    }
}

MetalPipelineHandle metal_context_pipeline(MetalCtxHandle ctx, const char* kernel_name) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        std::lock_guard<std::mutex> lock(impl->_mutex);

        std::string key(kernel_name);
        auto it = impl->_pipelineCache.find(key);
        if (it != impl->_pipelineCache.end()) {
            // Return raw retained pointer; ARC keeps the object alive in _pipelineCache.
            return (__bridge void*)(it->second);
        }

        if (impl->_library == nil) {
            // Attempt default library fallback (works when linked into an app bundle).
            impl->_library = [impl->_device newDefaultLibrary];
            if (impl->_library == nil) {
                NSLog(@"metal_context_pipeline: no library loaded and newDefaultLibrary "
                      @"returned nil. Call metal_context_load_library() first.");
                abort();
            }
        }

        NSString* name = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> fn = [impl->_library newFunctionWithName:name];
        if (fn == nil) {
            NSLog(@"metal_context_pipeline: kernel '%@' not found in library.", name);
            abort();
        }

        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [impl->_device newComputePipelineStateWithFunction:fn error:&err];
        if (pso == nil) {
            NSLog(@"metal_context_pipeline: PSO creation failed for '%@': %@", name, err);
            abort();
        }

        impl->_pipelineCache[key] = pso;
        return (__bridge void*)(pso);
    }
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

// vm_page_size is declared in <mach/mach.h>; on Apple Silicon it is 16384.
#include <mach/mach.h>

MetalBufHandle metal_buf_alias(MetalCtxHandle ctx, void* ptr, size_t bytes, int* is_copy) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        *is_copy = 0;

        uintptr_t addr = (uintptr_t)ptr;
        if (ptr != NULL && (addr & (vm_page_size - 1)) == 0) {
            // Page-aligned: create zero-copy alias buffer.
            // Align size up to page boundary as required by the API.
            size_t aligned_len = (bytes + vm_page_size - 1) & ~(vm_page_size - 1);
            id<MTLBuffer> buf =
                [impl->_device newBufferWithBytesNoCopy:ptr
                                                 length:aligned_len
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
            if (buf != nil) {
                return (__bridge_retained void*)(buf);
            }
            // Fall through to copy path if alias fails.
        }

        // Not page-aligned or alias failed: copy in.
        *is_copy = 1;
        id<MTLBuffer> buf =
            [impl->_device newBufferWithBytes:ptr
                                       length:bytes
                                      options:MTLResourceStorageModeShared];
        if (buf == nil) {
            NSLog(@"metal_buf_alias: newBufferWithBytes failed (size=%zu)", bytes);
            abort();
        }
        return (__bridge_retained void*)(buf);
    }
}

MetalBufHandle metal_buf_alloc(MetalCtxHandle ctx, size_t bytes) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLBuffer> buf =
            [impl->_device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
        if (buf == nil) {
            NSLog(@"metal_buf_alloc: newBufferWithLength failed (size=%zu)", bytes);
            abort();
        }
        return (__bridge_retained void*)(buf);
    }
}

void* metal_buf_contents(MetalBufHandle buf) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)(buf);
    return [b contents];
}

void metal_buf_release(MetalBufHandle buf) {
    // Release the +1 retain count taken by __bridge_retained in metal_buf_alias/alloc.
    id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)(buf);
    (void)b;  // ARC releases when b goes out of scope.
}

// ---------------------------------------------------------------------------
// Twiddle cache
// ---------------------------------------------------------------------------

MetalBufHandle metal_twiddle_buffer(MetalCtxHandle ctx,
                                     const uint64_t* roots_ptr,
                                     uint64_t roots_len) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        std::lock_guard<std::mutex> lock(impl->_mutex);

        auto it = impl->_twiddleCache.find(roots_len);
        if (it != impl->_twiddleCache.end()) {
            return (__bridge void*)(it->second);
        }

        size_t bytes = roots_len * sizeof(uint64_t);
        id<MTLBuffer> buf =
            [impl->_device newBufferWithBytes:roots_ptr
                                       length:bytes
                                      options:MTLResourceStorageModeShared];
        if (buf == nil) {
            NSLog(@"metal_twiddle_buffer: allocation failed for %llu roots", roots_len);
            abort();
        }
        impl->_twiddleCache[roots_len] = buf;
        return (__bridge void*)(buf);
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers — internal helper macro
// ---------------------------------------------------------------------------

// Returns the optimal threadgroup size capped at the PSO's maxTotalThreadsPerThreadgroup.
static NSUInteger threadgroup_size_for(id<MTLComputePipelineState> pso, NSUInteger desired) {
    NSUInteger max_tpg = [pso maxTotalThreadsPerThreadgroup];
    return (desired < max_tpg) ? desired : max_tpg;
}

// ---------------------------------------------------------------------------
// merkle_leaves dispatch
// ---------------------------------------------------------------------------
void metal_dispatch_merkle_leaves(MetalCtxHandle ctx,
                                   MetalBufHandle in_buf,
                                   MetalBufHandle tree_buf,
                                   uint32_t ncols,
                                   uint32_t dim,
                                   uint32_t num_rows) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "merkle_leaves"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(in_buf)   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(tree_buf) offset:0 atIndex:1];
        [enc setBytes:&ncols length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&dim   length:sizeof(uint32_t) atIndex:3];

        NSUInteger tpg = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)num_rows + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_merkle_leaves_simd(MetalCtxHandle ctx,
                                        MetalBufHandle in_buf,
                                        MetalBufHandle tree_buf,
                                        uint32_t ncols,
                                        uint32_t dim,
                                        uint32_t num_rows) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "merkle_leaves_simd"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(in_buf)   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(tree_buf) offset:0 atIndex:1];
        [enc setBytes:&ncols length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&dim   length:sizeof(uint32_t) atIndex:3];

        // One simdgroup (32 threads on Apple M-series) per row.
        NSUInteger tpg    = 32;
        NSUInteger groups = (NSUInteger)num_rows;  // one group per row
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// merkle_parents dispatch
// ---------------------------------------------------------------------------
void metal_dispatch_merkle_parents(MetalCtxHandle ctx,
                                    MetalBufHandle buf,
                                    uint32_t nextIndex,
                                    uint32_t pending,
                                    uint32_t nextN) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "merkle_parents"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(buf) offset:0 atIndex:0];
        [enc setBytes:&nextIndex length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&pending   length:sizeof(uint32_t) atIndex:2];

        NSUInteger tpg = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)nextN + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_transpose_rowmajor(MetalCtxHandle ctx,
                                        MetalBufHandle src_buf,
                                        MetalBufHandle dst_buf,
                                        uint32_t num_rows,
                                        uint32_t ncols) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "transpose_rowmajor"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(src_buf) offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(dst_buf) offset:0 atIndex:1];
        [enc setBytes:&num_rows length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&ncols    length:sizeof(uint32_t) atIndex:3];

        // 32x32 tile, one thread per (col, row) pair.
        const NSUInteger TILE = 32;
        NSUInteger tg_x = (ncols    + TILE - 1) / TILE;
        NSUInteger tg_y = (num_rows + TILE - 1) / TILE;
        [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
           threadsPerThreadgroup:MTLSizeMake(TILE, TILE, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_merkle_leaves_cm(MetalCtxHandle ctx,
                                      MetalBufHandle inp_cm_buf,
                                      MetalBufHandle tree_buf,
                                      uint32_t ncols,
                                      uint32_t dim,
                                      uint32_t num_rows) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "merkle_leaves_cm"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(inp_cm_buf) offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(tree_buf)   offset:0 atIndex:1];
        [enc setBytes:&ncols    length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&dim      length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&num_rows length:sizeof(uint32_t) atIndex:4];

        NSUInteger tpg    = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)num_rows + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_merkle_leaves_tg(MetalCtxHandle ctx,
                                      MetalBufHandle in_buf,
                                      MetalBufHandle tree_buf,
                                      uint32_t ncols,
                                      uint32_t dim,
                                      uint32_t num_rows) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "merkle_leaves_tg"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(in_buf)   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(tree_buf) offset:0 atIndex:1];
        [enc setBytes:&ncols    length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&dim      length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&num_rows length:sizeof(uint32_t) atIndex:4];

        // Fixed threadgroup size 32 (one simdgroup per 32-row tile).
        NSUInteger tpg    = 32;
        NSUInteger groups = ((NSUInteger)num_rows + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_merkle_leaves_x2(MetalCtxHandle ctx,
                                      MetalBufHandle in_buf,
                                      MetalBufHandle tree_buf,
                                      uint32_t ncols,
                                      uint32_t dim,
                                      uint32_t num_rows) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "merkle_leaves_x2"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(in_buf)   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(tree_buf) offset:0 atIndex:1];
        [enc setBytes:&ncols    length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&dim      length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&num_rows length:sizeof(uint32_t) atIndex:4];

        // ceil(num_rows / 2) threads (each thread handles a pair).
        NSUInteger tpg     = threadgroup_size_for(pso, 64);
        NSUInteger threads = (NSUInteger)((num_rows + 1) / 2);
        NSUInteger groups  = (threads + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// Helper: encode a single tree's full Merkle build (leaves + all parent
// levels) into a CALLER-OWNED compute command encoder. Does NOT commit or
// wait. Barriers are within-encoder so this single tree's internal
// dependencies are respected; separate encoders can overlap across trees.
static void encode_single_tree(MetalCtxHandle ctx,
                                id<MTLCommandBuffer> cmd,
                                MetalBufHandle in_buf,
                                MetalBufHandle tree_buf,
                                uint32_t ncols,
                                uint32_t dim,
                                uint32_t num_rows) {
    GoldilocksMetalContext* impl = get_impl(ctx);
    id<MTLComputePipelineState> leaves_pso =
        (__bridge id<MTLComputePipelineState>)(
            metal_context_pipeline(ctx, "merkle_leaves"));
    id<MTLComputePipelineState> parents_pso =
        (__bridge id<MTLComputePipelineState>)(
            metal_context_pipeline(ctx, "merkle_parents"));
    (void)impl;

    // Each tree gets its OWN compute encoder. This lets Metal's automatic
    // hazard tracking allow tree N+1's encoder to run alongside tree N's
    // on the GPU (separate encoders, different buffers → no serialization).
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // -- Phase 1: leaves --
    [enc setComputePipelineState:leaves_pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)(in_buf)   offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)(tree_buf) offset:0 atIndex:1];
    [enc setBytes:&ncols length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&dim   length:sizeof(uint32_t) atIndex:3];
    NSUInteger leaves_tpg    = threadgroup_size_for(leaves_pso, 64);
    NSUInteger leaves_groups = ((NSUInteger)num_rows + leaves_tpg - 1) / leaves_tpg;
    [enc dispatchThreadgroups:MTLSizeMake(leaves_groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(leaves_tpg, 1, 1)];

    // -- Phase 2: all parent levels (batched, with barriers between) --
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:parents_pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)(tree_buf) offset:0 atIndex:0];

    NSUInteger parents_tpg = threadgroup_size_for(parents_pso, 64);
    uint32_t pending   = num_rows;
    uint32_t nextIndex = 0;
    while (pending > 1) {
        uint32_t nextN = (pending - 1) / 2 + 1;
        [enc setBytes:&nextIndex length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&pending   length:sizeof(uint32_t) atIndex:2];
        NSUInteger groups = ((NSUInteger)nextN + parents_tpg - 1) / parents_tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(parents_tpg, 1, 1)];
        nextIndex += pending * 4;
        pending   /= 2;
        if (pending > 1) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    }

    [enc endEncoding];
}

void metal_dispatch_merkletree_batch(MetalCtxHandle ctx,
                                      const MetalBufHandle* in_bufs,
                                      const MetalBufHandle* tree_bufs,
                                      uint32_t count,
                                      uint32_t ncols,
                                      uint32_t dim,
                                      uint32_t num_rows) {
    if (count == 0) return;
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLCommandBuffer> cmd = [impl->_queue commandBuffer];
        for (uint32_t i = 0; i < count; i++) {
            encode_single_tree(ctx, cmd, in_bufs[i], tree_bufs[i],
                                ncols, dim, num_rows);
        }
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_merkle_parents_all_levels(MetalCtxHandle ctx,
                                                MetalBufHandle buf,
                                                uint32_t initial_pending) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "merkle_parents"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(buf) offset:0 atIndex:0];

        uint32_t pending   = initial_pending;
        uint32_t nextIndex = 0;
        NSUInteger tpg = threadgroup_size_for(pso, 64);

        // Iterate tree levels, encoding one dispatch per level into the same
        // command buffer. Barrier between levels so each parent sees the
        // previous level's writes.
        while (pending > 1) {
            uint32_t nextN = (pending - 1) / 2 + 1;  // == ceil(pending/2)

            [enc setBytes:&nextIndex length:sizeof(uint32_t) atIndex:1];
            [enc setBytes:&pending   length:sizeof(uint32_t) atIndex:2];

            NSUInteger groups = ((NSUInteger)nextN + tpg - 1) / tpg;
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            nextIndex += pending * 4;  // HASH_SIZE = 4
            pending   /= 2;

            if (pending > 1) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// ntt_reverse_permutation dispatch
// ---------------------------------------------------------------------------
void metal_dispatch_ntt_reverse_permutation(MetalCtxHandle ctx,
                                             MetalBufHandle buf,
                                             uint32_t domainPow,
                                             uint32_t ncols) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "ntt_reverse_permutation"));

        uint32_t domain_size = 1u << domainPow;

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(buf) offset:0 atIndex:0];
        [enc setBytes:&domainPow  length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&ncols      length:sizeof(uint32_t) atIndex:2];

        NSUInteger tpg = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)domain_size + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// ntt_butterfly_phase dispatch
// ---------------------------------------------------------------------------
void metal_dispatch_ntt_rev_butterfly_s1s2(MetalCtxHandle ctx,
                                             MetalBufHandle src,
                                             MetalBufHandle dst,
                                             uint32_t domain_pow,
                                             uint32_t ncols,
                                             uint64_t I_val) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "ntt_rev_butterfly_s1s2"));

        uint32_t quarter = (1u << domain_pow) >> 2;
        uint32_t total   = quarter * ncols;

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(src) offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(dst) offset:0 atIndex:1];
        [enc setBytes:&domain_pow length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&ncols      length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&I_val      length:sizeof(uint64_t) atIndex:4];

        NSUInteger tpg    = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)total + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_ntt_rev_butterfly_s1s2s3(MetalCtxHandle ctx,
                                              MetalBufHandle src,
                                              MetalBufHandle dst,
                                              uint32_t domain_pow,
                                              uint32_t ncols,
                                              uint64_t I_val,
                                              uint64_t W8_val,
                                              uint64_t W8c_val) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "ntt_rev_butterfly_s1s2s3"));

        uint32_t eighth = (1u << domain_pow) >> 3;
        uint32_t total  = eighth * ncols;

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(src) offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(dst) offset:0 atIndex:1];
        [enc setBytes:&domain_pow length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&ncols      length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&I_val      length:sizeof(uint64_t) atIndex:4];
        [enc setBytes:&W8_val     length:sizeof(uint64_t) atIndex:5];
        [enc setBytes:&W8c_val    length:sizeof(uint64_t) atIndex:6];

        NSUInteger tpg    = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)total + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_ntt_rev_butterfly_s1(MetalCtxHandle ctx,
                                          MetalBufHandle src,
                                          MetalBufHandle dst,
                                          uint32_t domain_pow,
                                          uint32_t ncols) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "ntt_rev_butterfly_s1"));

        uint32_t half  = (1u << domain_pow) >> 1;
        uint32_t total = half * ncols;

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(src) offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(dst) offset:0 atIndex:1];
        [enc setBytes:&domain_pow length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&ncols      length:sizeof(uint32_t) atIndex:3];

        NSUInteger tpg    = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)total + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// Batched: encode ALL phases s = 1..domainPow into a single command buffer
// with one waitUntilCompleted at the end. Eliminates `log N` per-phase
// commit/wait round-trips (~150μs each) that dominated small-N runs.
// Inserts a buffer-scoped memory barrier between phase dispatches so each
// phase sees the previous phase's writes.
void metal_dispatch_ntt_butterfly_all_phases(MetalCtxHandle ctx,
                                              MetalBufHandle buf,
                                              MetalBufHandle twiddles,
                                              uint32_t ncols,
                                              uint32_t domain_size,
                                              uint32_t start_s,
                                              uint32_t domain_pow,
                                              uint32_t s_global) {
    if (start_s > domain_pow) return;
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso_r2 =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "ntt_butterfly_phase"));
        id<MTLComputePipelineState> pso_r4 =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "ntt_radix4_phase"));
        id<MTLComputePipelineState> pso_r8 =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "ntt_radix8_phase"));

        uint32_t half    = domain_size >> 1;
        uint32_t quarter = domain_size >> 2;
        uint32_t eighth  = domain_size >> 3;
        uint32_t total_r2 = half    * ncols;
        uint32_t total_r4 = quarter * ncols;
        uint32_t total_r8 = eighth  * ncols;

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)(buf)      offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(twiddles) offset:0 atIndex:1];
        [enc setBytes:&ncols       length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&domain_size length:sizeof(uint32_t) atIndex:3];

        NSUInteger tpg_r2 = threadgroup_size_for(pso_r2, 64);
        NSUInteger tpg_r4 = threadgroup_size_for(pso_r4, 64);
        NSUInteger tpg_r8 = threadgroup_size_for(pso_r8, 64);
        NSUInteger groups_r2 = ((NSUInteger)total_r2 + tpg_r2 - 1) / tpg_r2;
        NSUInteger groups_r4 = ((NSUInteger)total_r4 + tpg_r4 - 1) / tpg_r4;
        NSUInteger groups_r8 = ((NSUInteger)total_r8 + tpg_r8 - 1) / tpg_r8;

        // Radix-k wins when the per-thread work is amortized across enough
        // parallel dispatches to saturate the GPU. The crossover thresholds
        // below are empirical on Apple M4 Pro: below R4_MIN_THREADS the
        // per-thread register footprint of r4 kills occupancy; below
        // R8_MIN_THREADS the same applies to r8 (even more registers per
        // thread). Above the thresholds, fewer passes + more ILP wins.
        const uint32_t R4_MIN_THREADS = 8192;
        // R8 per-thread register pressure is ~2× r4 (8 elements live + 7
        // twiddles + intermediates). At small total-thread counts the GPU
        // can't hide the occupancy drop even when data exceeds L1/L2; the
        // crossover on M4 Pro sits near 128k threads (seen as a regression
        // at N=2^18, ncols=1 just above the previous 32k threshold).
        const uint32_t R8_MIN_THREADS = 131072;
        bool use_radix4 = (quarter * ncols >= R4_MIN_THREADS);
        bool use_radix8 = (eighth  * ncols >= R8_MIN_THREADS);

        uint32_t s = start_s;
        bool first = true;
        while (s <= domain_pow) {
            if (!first) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            first = false;

            if (use_radix8 && s + 2 <= domain_pow && eighth > 0 && s >= 1) {
                // Radix-8: combines stages s, s+1, s+2 into one pass
                [enc setComputePipelineState:pso_r8];
                uint32_t stride_s = s_global - s;
                [enc setBytes:&s         length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&stride_s  length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(groups_r8, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tpg_r8, 1, 1)];
                s += 3;
            } else if (use_radix4 && s + 1 <= domain_pow && quarter > 0) {
                // Radix-4: combines stages s and s+1 into one pass
                [enc setComputePipelineState:pso_r4];
                uint32_t stride_s1 = s_global - (s + 1);
                [enc setBytes:&s         length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&stride_s1 length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(groups_r4, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tpg_r4, 1, 1)];
                s += 2;
            } else {
                // Radix-2 tail or small-N path
                [enc setComputePipelineState:pso_r2];
                uint32_t stride_shift = s_global - s;
                [enc setBytes:&s            length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&stride_shift length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(groups_r2, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tpg_r2, 1, 1)];
                s += 1;
            }
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_ntt_butterfly_phase(MetalCtxHandle ctx,
                                         MetalBufHandle buf,
                                         MetalBufHandle twiddles,
                                         uint32_t ncols,
                                         uint32_t domain_size,
                                         uint32_t s,
                                         uint32_t roots_stride_shift) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "ntt_butterfly_phase"));

        uint32_t half  = domain_size >> 1;
        uint32_t total = half * ncols;  // total threads

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(buf)      offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(twiddles) offset:0 atIndex:1];
        [enc setBytes:&ncols              length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&domain_size        length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&s                  length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&roots_stride_shift length:sizeof(uint32_t) atIndex:5];

        NSUInteger tpg = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)total + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ---------------------------------------------------------------------------
// intt_scale dispatch
// ---------------------------------------------------------------------------
void metal_dispatch_intt_reorder(MetalCtxHandle ctx,
                                  MetalBufHandle buf,
                                  uint32_t domain_size,
                                  uint32_t ncols) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "intt_reorder"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(buf) offset:0 atIndex:0];
        [enc setBytes:&domain_size length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&ncols       length:sizeof(uint32_t) atIndex:2];

        // Dispatch (domain_size/2) threads; each handles pair (i+1, N-(i+1)).
        NSUInteger threads = (NSUInteger)(domain_size / 2);
        NSUInteger tpg = threadgroup_size_for(pso, 64);
        NSUInteger groups = (threads + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_intt_reorder_scale(MetalCtxHandle ctx,
                                        MetalBufHandle buf,
                                        uint32_t domain_size,
                                        uint32_t ncols,
                                        uint64_t inv_n) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "intt_reorder_scale"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(buf) offset:0 atIndex:0];
        [enc setBytes:&domain_size length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&ncols       length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&inv_n       length:sizeof(uint64_t) atIndex:3];

        // Dispatch (N/2 + 1) × ncols threads: one thread per (pair_key, col).
        // pair_idx == 0 is the fixed point at row 0; pair_idx in [1, N/2)
        // are swap-pairs; pair_idx == N/2 is the second fixed point (N even).
        NSUInteger threads = (NSUInteger)(domain_size / 2 + 1) * ncols;
        NSUInteger tpg     = threadgroup_size_for(pso, 64);
        NSUInteger groups  = (threads + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_intt_reorder_coset_scale(MetalCtxHandle ctx,
                                              MetalBufHandle buf,
                                              MetalBufHandle r_inv,
                                              uint32_t domain_size,
                                              uint32_t ncols) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "intt_reorder_coset_scale"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(buf)   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)(r_inv) offset:0 atIndex:1];
        [enc setBytes:&domain_size length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&ncols       length:sizeof(uint32_t) atIndex:3];

        NSUInteger threads = (NSUInteger)(domain_size / 2 + 1) * ncols;
        NSUInteger tpg     = threadgroup_size_for(pso, 64);
        NSUInteger groups  = (threads + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_dispatch_intt_scale(MetalCtxHandle ctx,
                                MetalBufHandle buf,
                                uint64_t inv_n,
                                uint32_t count) {
    @autoreleasepool {
        GoldilocksMetalContext* impl = get_impl(ctx);
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)(
                metal_context_pipeline(ctx, "intt_scale"));

        id<MTLCommandBuffer>        cmd = [impl->_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)(buf) offset:0 atIndex:0];
        [enc setBytes:&inv_n  length:sizeof(uint64_t) atIndex:1];
        [enc setBytes:&count  length:sizeof(uint32_t) atIndex:2];

        NSUInteger tpg = threadgroup_size_for(pso, 64);
        NSUInteger groups = ((NSUInteger)count + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

#endif // GOLDILOCKS_HAS_METAL
