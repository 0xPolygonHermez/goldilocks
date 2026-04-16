// probe_merkle.mm — Probe 4: end-to-end Merkle oracle
//
// Builds the canonical 128-col × 64-row Fibonacci input used in
// tests/tests.cpp::merkletree_seq, runs it through merkletree_metal on the
// Apple Silicon GPU, and compares the root (tree[numElementsTree-4..numElementsTree-1])
// against the hardcoded bit-exact oracle values.
//
// Expected roots (from tests/tests.cpp:merkletree_seq):
//   root[0] = 0x918F7CD0C3E8701F
//   root[1] = 0x83A130E00F961B02
//   root[2] = 0x6921497B364123F8
//   root[3] = 0xBD2B98A57B748BF4
//
// Library loading strategy (tried in order):
//   1. goldilocks.metallib in CWD
//   2. goldilocks.metallib adjacent to argv[0]
//   3. Runtime compilation from MTL_KERNEL_DIR (fallback — no xcrun needed)
//
// Build flags required:
//   -DGOLDILOCKS_HAS_METAL -fobjc-arc -ObjC++ -std=c++17
//   -framework Metal -framework Foundation
//   -DMTL_KERNEL_DIR="../../src/metal/kernels"

#ifdef GOLDILOCKS_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>

#include "../../src/goldilocks_base_field.hpp"
#include "../../src/poseidon_goldilocks.hpp"
#include "../../src/merklehash_goldilocks.hpp"
#include "../../src/metal/metal_context.hpp"
#include "../../src/metal/goldilocks_metal.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// Library loading:
//   1. CWD/goldilocks.metallib
//   2. argv[0]-dir/goldilocks.metallib
//   3. Runtime compile from MTL_KERNEL_DIR (field.metal + poseidon.metal)
// ---------------------------------------------------------------------------
static int load_metallib(MetalCtxHandle ctx, const char* argv0) {
    // Attempt 1: CWD.
    if (metal_context_load_library(ctx, "goldilocks.metallib") == 0) {
        printf("  library: loaded goldilocks.metallib from CWD\n");
        return 0;
    }

    // Attempt 2: adjacent to argv[0].
    {
        NSString* exe  = [NSString stringWithUTF8String:argv0];
        NSString* dir  = [exe stringByDeletingLastPathComponent];
        NSString* path = [dir stringByAppendingPathComponent:@"goldilocks.metallib"];
        if (metal_context_load_library(ctx, [path UTF8String]) == 0) {
            printf("  library: loaded goldilocks.metallib from %s\n",
                   [dir UTF8String]);
            return 0;
        }
    }

#ifdef MTL_KERNEL_DIR
    // Attempt 3: runtime compile from source directory.
    printf("  library: metallib not found; compiling from source at %s\n",
           MTL_KERNEL_DIR);
    {
        std::string field_src    = read_file(MTL_KERNEL_DIR "/field.metal");
        std::string consts_src   = read_file(MTL_KERNEL_DIR "/constants.metal.inc");
        std::string poseidon_src = read_file(MTL_KERNEL_DIR "/poseidon.metal");

        if (field_src.empty() || poseidon_src.empty()) {
            fprintf(stderr, "probe_merkle: cannot read kernel sources from %s\n",
                    MTL_KERNEL_DIR);
            return -1;
        }

        // poseidon.metal has #include "field.metal" and
        // #include "constants.metal.inc" which cannot be resolved when
        // compiling from a string (no include search path).
        // Strip those lines and prepend the file bodies directly.
        auto strip_includes = [](std::string src) -> std::string {
            std::string out;
            std::istringstream ss(src);
            std::string line;
            while (std::getline(ss, line)) {
                // Drop lines of the form: #include "..."
                if (line.find("#include \"") != std::string::npos) continue;
                // Also drop: #pragma once (only valid in header files)
                if (line.find("#pragma once") != std::string::npos) continue;
                out += line + "\n";
            }
            return out;
        };

        // Build combined source: field (stripped of pragma once) +
        // constants (stripped of pragma once) + poseidon (stripped of includes).
        std::string combined =
            strip_includes(field_src) + "\n" +
            strip_includes(consts_src) + "\n" +
            strip_includes(poseidon_src);

        if (metal_context_load_source(ctx, combined.c_str()) != 0) {
            fprintf(stderr, "probe_merkle: runtime compile failed\n");
            return -1;
        }
        printf("  library: runtime compile succeeded\n");
        return 0;
    }
#endif
    fprintf(stderr, "probe_merkle: no goldilocks.metallib found and "
                    "MTL_KERNEL_DIR not set.\n");
    return -1;
}

// ---------------------------------------------------------------------------
// Build Fibonacci input matrix: same as tests/tests.cpp::merkletree_seq
// cols[i]            = fromU64(i) + one()  for i in [0, ncols)
// cols[i + ncols]    = fromU64(i) + one()  for i in [0, ncols)
// cols[j*ncols + i]  = cols[(j-2)*ncols+i] + cols[(j-1)*ncols+i]  for j>=2
// ---------------------------------------------------------------------------
static Goldilocks::Element* build_fibonacci(uint64_t ncols, uint64_t nrows) {
    Goldilocks::Element* cols =
        (Goldilocks::Element*)malloc(ncols * nrows * sizeof(Goldilocks::Element));
    if (!cols) { perror("malloc"); abort(); }

    for (uint64_t i = 0; i < ncols; i++) {
        cols[i]         = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + ncols] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < nrows; j++) {
        for (uint64_t i = 0; i < ncols; i++) {
            cols[j * ncols + i] =
                cols[(j - 2) * ncols + i] + cols[(j - 1) * ncols + i];
        }
    }
    return cols;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    @autoreleasepool {
        printf("probe_merkle: Merkle oracle (128 cols x 64 rows Fibonacci)\n");

        // Get Metal context — aborts if no device.
        MetalCtxHandle ctx = metal_context_get();

        // Load .metallib
        const char* exe = (argc > 0) ? argv[0] : "./probe_merkle";
        if (load_metallib(ctx, exe) != 0) {
            return 1;
        }

        // -- Sub-case 1: ncols=128, nrows=64 --------------------------------
        {
            uint64_t ncols_hash = 128;
            uint64_t nrows_hash = (1 << 6);  // 64

            Goldilocks::Element* cols = build_fibonacci(ncols_hash, nrows_hash);

            uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(nrows_hash);
            Goldilocks::Element* tree =
                (Goldilocks::Element*)malloc(numElementsTree * sizeof(Goldilocks::Element));
            if (!tree) { perror("malloc"); free(cols); return 1; }

            printf("  sub-case 1: merkletree_metal(tree, input, 128, 64)\n");
            goldilocks_metal::merkletree_metal(tree, cols, ncols_hash, nrows_hash);

            // Root is at tree[numElementsTree - 4 .. numElementsTree - 1]
            Goldilocks::Element root[4];
            MerklehashGoldilocks::root(&root[0], tree, numElementsTree);

            uint64_t r0 = Goldilocks::toU64(root[0]);
            uint64_t r1 = Goldilocks::toU64(root[1]);
            uint64_t r2 = Goldilocks::toU64(root[2]);
            uint64_t r3 = Goldilocks::toU64(root[3]);

            printf("  root[0] = 0x%016llX  (expected 0x918F7CD0C3E8701F)\n", r0);
            printf("  root[1] = 0x%016llX  (expected 0x83A130E00F961B02)\n", r1);
            printf("  root[2] = 0x%016llX  (expected 0x6921497B364123F8)\n", r2);
            printf("  root[3] = 0x%016llX  (expected 0xBD2B98A57B748BF4)\n", r3);

            free(cols);
            free(tree);

            bool ok = (r0 == 0x918F7CD0C3E8701FULL) &&
                      (r1 == 0x83A130E00F961B02ULL) &&
                      (r2 == 0x6921497B364123F8ULL) &&
                      (r3 == 0xBD2B98A57B748BF4ULL);
            if (!ok) {
                printf("  FAIL: root mismatch\n");
                return 1;
            }
            printf("  PASS\n");
        }

        // -- Sub-case 2: ncols=0, nrows=64 (edge case) ----------------------
        {
            uint64_t ncols_hash = 0;
            uint64_t nrows_hash = (1 << 6);  // 64

            uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(nrows_hash);
            Goldilocks::Element* tree =
                (Goldilocks::Element*)malloc(numElementsTree * sizeof(Goldilocks::Element));
            if (!tree) { perror("malloc"); return 1; }

            printf("  sub-case 2: merkletree_metal(tree, NULL, 0, 64)\n");
            goldilocks_metal::merkletree_metal(tree, nullptr, ncols_hash, nrows_hash);

            Goldilocks::Element root[4];
            MerklehashGoldilocks::root(&root[0], tree, numElementsTree);

            uint64_t r0 = Goldilocks::toU64(root[0]);
            uint64_t r1 = Goldilocks::toU64(root[1]);
            uint64_t r2 = Goldilocks::toU64(root[2]);
            uint64_t r3 = Goldilocks::toU64(root[3]);

            printf("  root[0] = 0x%016llX  (expected 0x25225F1A5D49614A)\n", r0);
            printf("  root[1] = 0x%016llX  (expected 0x5A1D2A648EEE8F03)\n", r1);
            printf("  root[2] = 0x%016llX  (expected 0xDDA8F741C47DFB10)\n", r2);
            printf("  root[3] = 0x%016llX  (expected 0x49561260080D30C3)\n", r3);

            free(tree);

            bool ok = (r0 == 0x25225F1A5D49614AULL) &&
                      (r1 == 0x5A1D2A648EEE8F03ULL) &&
                      (r2 == 0xDDA8F741C47DFB10ULL) &&
                      (r3 == 0x49561260080D30C3ULL);
            if (!ok) {
                printf("  FAIL: root mismatch\n");
                return 1;
            }
            printf("  PASS\n");
        }

        printf("probe_merkle: all sub-cases PASSED\n");
        return 0;
    }  // @autoreleasepool
}

#else
// No Metal available at compile time — graceful skip.
#include <cstdio>
int main() {
    printf("probe_merkle: SKIPPED (GOLDILOCKS_HAS_METAL not defined)\n");
    return 0;
}
#endif // GOLDILOCKS_HAS_METAL
