// Probe 2: Goldilocks field arithmetic bit-exact verification
//
// Dispatches a Metal kernel that runs goldilocks_add, goldilocks_sub,
// goldilocks_mul on ~10 000 seeded random pairs. Results are compared
// bit-exact against the CPU Goldilocks::add/sub/mul oracle.
//
// The MSL kernel source (field.metal) is compiled at runtime via
// newLibraryWithSource:options:error: — no precompiled .metallib needed.
//
// Build:
//   clang++ -std=c++17 -O0 -fobjc-arc -ObjC++ -framework Metal -framework Foundation \
//     -I../../src -DMTL_KERNEL_DIR="../../src/metal/kernels" \
//     probe_field.mm -o probe_field
// Run:
//   ./probe_field
// Exit 0 on full bit-exact match; exit 1 on first mismatch.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>

// CPU oracle — pull in Goldilocks header (portable non-SIMD path)
#define GOLDILOCKS_ARCH_GENERIC 1
#include "goldilocks_base_field.hpp"
#include "goldilocks_base_field_scalar.hpp"

// ----- RNG -------------------------------------------------------------------
static std::mt19937_64 rng(0xDEADBEEF12345678ULL);

static constexpr uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

// Generate a value in [0, p) using the full 64-bit range mod p,
// or force a specific edge case.
static uint64_t rand_field() {
    uint64_t v = rng();
    return v % GOLDILOCKS_P;
}

// Generate an edge-case value: near 0, near p, near 2^32, or random.
static uint64_t rand_field_edge() {
    uint64_t pick = rng() % 8;
    switch (pick) {
        case 0: return 0;
        case 1: return 1;
        case 2: return GOLDILOCKS_P - 1;
        case 3: return GOLDILOCKS_P - 2;
        case 4: return 0xFFFFFFFFULL;       // CQ = 2^32 - 1
        case 5: return 0x100000000ULL;      // 2^32
        case 6: return (GOLDILOCKS_P >> 1); // near mid-field
        default: return rand_field();
    }
}

// ----- Read kernel source from file ------------------------------------------
static std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "Cannot open kernel file: %s\n", path);
        return "";
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ----- Kernel source (inline + from file) ------------------------------------
// The test kernel runs 3 ops (add, sub, mul) per pair and writes results.
// Output buffer layout: for each pair i: [add_result, sub_result, mul_result]
static const char* KERNEL_SOURCE_HEADER = R"(
#include <metal_stdlib>
using namespace metal;
)";

static const char* KERNEL_BODY = R"(
kernel void field_test(
    device const ulong* a_buf [[ buffer(0) ]],
    device const ulong* b_buf [[ buffer(1) ]],
    device       ulong* out   [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    ulong a = a_buf[tid];
    ulong b = b_buf[tid];
    out[tid * 3 + 0] = gl_add(a, b);
    out[tid * 3 + 1] = gl_sub(a, b);
    out[tid * 3 + 2] = gl_mul(a, b);
}
)";

// ----- Main ------------------------------------------------------------------
int main() {
    // 1. Get Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("probe_field: no Metal device\n");
        return 1;
    }
    printf("probe_field: device = %s\n", [[device name] UTF8String]);

    // 2. Generate test pairs
    static constexpr int N = 10000;
    std::vector<uint64_t> a_vals(N), b_vals(N);
    // First 64 pairs are edge cases, rest are random
    for (int i = 0; i < N; i++) {
        if (i < 64) {
            a_vals[i] = rand_field_edge();
            b_vals[i] = rand_field_edge();
        } else {
            a_vals[i] = rand_field();
            b_vals[i] = rand_field();
        }
    }

    // 3. CPU oracle reference
    std::vector<uint64_t> cpu_add(N), cpu_sub(N), cpu_mul(N);
    for (int i = 0; i < N; i++) {
        Goldilocks::Element ea{a_vals[i]}, eb{b_vals[i]};
        cpu_add[i] = Goldilocks::add(ea, eb).fe;
        cpu_sub[i] = Goldilocks::sub(ea, eb).fe;
        // CPU mul always canonicalizes; GPU mul is lazy ([0,2p)).
        // We compare after canonicalizing GPU result.
        cpu_mul[i] = Goldilocks::mul(ea, eb).fe;
    }

    // 4. Build Metal library from source
    // Load field.metal source
    std::string kernel_dir = MTL_KERNEL_DIR;
    std::string field_src = read_file((kernel_dir + "/field.metal").c_str());
    if (field_src.empty()) return 1;

    std::string full_src = std::string(KERNEL_SOURCE_HEADER)
                         + field_src
                         + std::string(KERNEL_BODY);

    NSString* src_ns = [NSString stringWithUTF8String:full_src.c_str()];
    NSError* err = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:src_ns
                                                  options:nil
                                                    error:&err];
    if (!library) {
        printf("probe_field: compile error:\n%s\n",
               [[err localizedDescription] UTF8String]);
        return 1;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"field_test"];
    if (!fn) {
        printf("probe_field: kernel function 'field_test' not found\n");
        return 1;
    }

    NSError* psoErr = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&psoErr];
    if (!pso) {
        printf("probe_field: PSO error: %s\n",
               [[psoErr localizedDescription] UTF8String]);
        return 1;
    }

    // 5. Allocate MTL buffers
    id<MTLBuffer> a_buf = [device newBufferWithBytes:a_vals.data()
                                              length:N * sizeof(uint64_t)
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_buf = [device newBufferWithBytes:b_vals.data()
                                              length:N * sizeof(uint64_t)
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [device newBufferWithLength:N * 3 * sizeof(uint64_t)
                                                options:MTLResourceStorageModeShared];

    // 6. Dispatch kernel
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmd  = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:a_buf   offset:0 atIndex:0];
    [enc setBuffer:b_buf   offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];

    NSUInteger tgSize = MIN((NSUInteger)N, pso.maxTotalThreadsPerThreadgroup);
    MTLSize grid      = MTLSizeMake(N, 1, 1);
    MTLSize tg        = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // 7. Compare results
    const uint64_t* out = (const uint64_t*)[out_buf contents];
    static constexpr uint64_t GL_PRIME = 0xFFFFFFFF00000001ULL;

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        uint64_t gpu_add = out[i * 3 + 0];
        uint64_t gpu_sub = out[i * 3 + 1];
        uint64_t gpu_mul_lazy = out[i * 3 + 2];
        // Canonicalize lazy mul before comparing
        uint64_t gpu_mul = (gpu_mul_lazy >= GL_PRIME) ? (gpu_mul_lazy - GL_PRIME) : gpu_mul_lazy;

        bool ok = (gpu_add == cpu_add[i]) && (gpu_sub == cpu_sub[i]) && (gpu_mul == cpu_mul[i]);
        if (!ok && mismatches == 0) {
            printf("MISMATCH at i=%d:\n", i);
            printf("  a=0x%016llx  b=0x%016llx\n",
                   (unsigned long long)a_vals[i], (unsigned long long)b_vals[i]);
            printf("  add:  gpu=0x%016llx cpu=0x%016llx  %s\n",
                   (unsigned long long)gpu_add, (unsigned long long)cpu_add[i],
                   gpu_add == cpu_add[i] ? "OK" : "FAIL");
            printf("  sub:  gpu=0x%016llx cpu=0x%016llx  %s\n",
                   (unsigned long long)gpu_sub, (unsigned long long)cpu_sub[i],
                   gpu_sub == cpu_sub[i] ? "OK" : "FAIL");
            printf("  mul:  gpu(lazy)=0x%016llx gpu(canon)=0x%016llx cpu=0x%016llx  %s\n",
                   (unsigned long long)gpu_mul_lazy, (unsigned long long)gpu_mul,
                   (unsigned long long)cpu_mul[i],
                   gpu_mul == cpu_mul[i] ? "OK" : "FAIL");
        }
        if (!ok) mismatches++;
    }

    if (mismatches == 0) {
        printf("probe_field: PASS — all %d pairs bit-exact (add/sub/mul)\n", N);
        return 0;
    } else {
        printf("probe_field: FAIL — %d/%d mismatches\n", mismatches, N);
        return 1;
    }
}
