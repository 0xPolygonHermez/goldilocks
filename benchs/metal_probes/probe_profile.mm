// probe_profile.mm — deep profiling of the merkle_leaves kernel.
//
// Reports, for each kernel & size:
//   - PSO maxTotalThreadsPerThreadgroup (proxy for register pressure)
//   - threadExecutionWidth (SIMD width for scheduling)
//   - staticThreadgroupMemoryLength
//   - CPU wall-clock per iteration
//   - GPU execution time per iteration (GPUEndTime - GPUStartTime)
//   - Derived: scheduling/dispatch overhead = wall - gpu
//
// Lets us answer:
//   * Is the kernel compute-bound or dispatch-bound at size X?
//   * Is register pressure limiting occupancy?
//   * Is there measurable variance from Metal's runtime scheduler?

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "../../src/platform.hpp"
#include "../../src/goldilocks_base_field.hpp"

using Clock = std::chrono::high_resolution_clock;
using ms_t  = std::chrono::duration<double, std::milli>;

#ifndef MTL_KERNEL_DIR
#error "MTL_KERNEL_DIR must be defined"
#endif

static std::string slurp(const std::string& p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::string strip_inc(const std::string& src) {
    std::stringstream in(src); std::stringstream out; std::string ln;
    while (std::getline(in, ln)) {
        auto p = ln.find_first_not_of(" \t");
        if (p != std::string::npos && ln[p] == '#' &&
            ln.find("include", p) != std::string::npos) continue;
        out << ln << "\n";
    }
    return out.str();
}

static id<MTLLibrary> compile_library(id<MTLDevice> dev) {
    std::string d = MTL_KERNEL_DIR;
    std::string src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") +
                      strip_inc(slurp(d + "/field.metal")) +
                      slurp(d + "/constants.metal.inc") +
                      strip_inc(slurp(d + "/poseidon.metal"));
    NSString* ns_src = [NSString stringWithUTF8String:src.c_str()];
    NSError* err = nil;
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.languageVersion = MTLLanguageVersion3_0;
    id<MTLLibrary> lib = [dev newLibraryWithSource:ns_src options:opt error:&err];
    if (!lib) { NSLog(@"compile fail: %@", err); abort(); }
    return lib;
}

static void describe_pso(id<MTLComputePipelineState> pso, const char* name) {
    printf("PSO[%s]:\n", name);
    printf("  maxTotalThreadsPerThreadgroup = %lu\n",
           (unsigned long)[pso maxTotalThreadsPerThreadgroup]);
    printf("  threadExecutionWidth          = %lu\n",
           (unsigned long)[pso threadExecutionWidth]);
    printf("  staticThreadgroupMemoryLength = %lu B\n",
           (unsigned long)[pso staticThreadgroupMemoryLength]);
}

static void time_kernel(id<MTLDevice> dev,
                        id<MTLCommandQueue> q,
                        id<MTLComputePipelineState> pso,
                        const char* name,
                        id<MTLBuffer> in_buf,
                        id<MTLBuffer> tree_buf,
                        uint32_t ncols,
                        uint32_t dim,
                        uint32_t nrows,
                        NSUInteger tpg_desired,
                        NSUInteger groups,
                        int iters) {
    // Warm-up
    @autoreleasepool {
        id<MTLCommandBuffer>        c = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:in_buf   offset:0 atIndex:0];
        [e setBuffer:tree_buf offset:0 atIndex:1];
        [e setBytes:&ncols length:sizeof(uint32_t) atIndex:2];
        [e setBytes:&dim   length:sizeof(uint32_t) atIndex:3];
        NSUInteger tpg = (tpg_desired < [pso maxTotalThreadsPerThreadgroup])
                       ? tpg_desired : [pso maxTotalThreadsPerThreadgroup];
        [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [e endEncoding];
        [c commit];
        [c waitUntilCompleted];
    }

    double wall_sum = 0, gpu_sum = 0;
    for (int it = 0; it < iters; it++) {
        @autoreleasepool {
            auto t0 = Clock::now();
            id<MTLCommandBuffer>        c = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:in_buf   offset:0 atIndex:0];
            [e setBuffer:tree_buf offset:0 atIndex:1];
            [e setBytes:&ncols length:sizeof(uint32_t) atIndex:2];
            [e setBytes:&dim   length:sizeof(uint32_t) atIndex:3];
            NSUInteger tpg = (tpg_desired < [pso maxTotalThreadsPerThreadgroup])
                           ? tpg_desired : [pso maxTotalThreadsPerThreadgroup];
            [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [e endEncoding];
            [c commit];
            [c waitUntilCompleted];
            auto t1 = Clock::now();
            wall_sum += ms_t(t1 - t0).count();
            double gpu_ms = ([c GPUEndTime] - [c GPUStartTime]) * 1000.0;
            gpu_sum += gpu_ms;
        }
    }
    double wall = wall_sum / iters;
    double gpu  = gpu_sum  / iters;
    printf("  %-22s  wall=%8.3f ms  gpu=%8.3f ms  overhead=%7.3f ms  (nrows=%u, tpg=%lu, groups=%lu)\n",
           name, wall, gpu, wall - gpu, nrows,
           (unsigned long)tpg_desired, (unsigned long)groups);
}

int main() {
    @autoreleasepool {
        id<MTLDevice>       dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q   = [dev newCommandQueue];
        id<MTLLibrary>      lib = compile_library(dev);

        printf("Device: %s\n", [[dev name] UTF8String]);
        printf("maxBufferLength: %llu GiB\n",
               (unsigned long long)[dev maxBufferLength] / (1024ULL*1024*1024));
        printf("\n");

        NSError* err = nil;
        id<MTLFunction> fn_row  = [lib newFunctionWithName:@"merkle_leaves"];
        id<MTLFunction> fn_coop = [lib newFunctionWithName:@"merkle_leaves_simd"];
        id<MTLComputePipelineState> pso_row  =
            [dev newComputePipelineStateWithFunction:fn_row  error:&err];
        id<MTLComputePipelineState> pso_coop =
            [dev newComputePipelineStateWithFunction:fn_coop error:&err];

        describe_pso(pso_row,  "merkle_leaves");
        printf("\n");
        describe_pso(pso_coop, "merkle_leaves_simd");
        printf("\n");

        // Input: Fibonacci 128 × N
        struct Case { uint64_t nrows; int iters; };
        Case cases[] = {
            { 64,     30 },
            { 4096,   15 },
            { 65536,   8 },
            { 262144,  4 },
        };

        for (auto& c : cases) {
            uint64_t ncols = 128;
            std::vector<uint64_t> inp(ncols * c.nrows);
            for (uint64_t i = 0; i < inp.size(); i++) inp[i] = i + 1;
            std::vector<uint64_t> tree(c.nrows * 4, 0);

            id<MTLBuffer> in_buf = [dev newBufferWithBytes:inp.data()
                                                     length:inp.size()*sizeof(uint64_t)
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> tree_buf = [dev newBufferWithBytes:tree.data()
                                                       length:tree.size()*sizeof(uint64_t)
                                                      options:MTLResourceStorageModeShared];

            printf("=== nrows=%llu, ncols=%llu ===\n",
                   (unsigned long long)c.nrows, (unsigned long long)ncols);

            // Row-per-thread: try threadgroup sizes 32, 64, 128, 256
            for (NSUInteger tpg : { (NSUInteger)32, (NSUInteger)64,
                                    (NSUInteger)128, (NSUInteger)256 }) {
                NSUInteger max_tpg = [pso_row maxTotalThreadsPerThreadgroup];
                if (tpg > max_tpg) continue;
                NSUInteger groups = ((NSUInteger)c.nrows + tpg - 1) / tpg;
                char label[64];
                snprintf(label, sizeof(label), "row  tpg=%3lu", (unsigned long)tpg);
                time_kernel(dev, q, pso_row, label, in_buf, tree_buf,
                            (uint32_t)ncols, 1, (uint32_t)c.nrows,
                            tpg, groups, c.iters);
            }

            // Cooperative: fixed 32 threads/group = 1 simdgroup/hash
            time_kernel(dev, q, pso_coop, "coop tpg= 32",
                        in_buf, tree_buf,
                        (uint32_t)ncols, 1, (uint32_t)c.nrows,
                        32, c.nrows, c.iters);
            printf("\n");
        }
    }
    return 0;
}
