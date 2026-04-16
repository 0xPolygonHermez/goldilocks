// tests_metal.cpp — Standalone gtest binary for Apple Metal GPU backend.
//
// Targets:
//   merkletree_metal   — bit-exact Merkle oracle (Fibonacci 128×64 input)
//   NTT_Metal_roundtrip — NTT_Metal followed by INTT_Metal recovers input
//
// Build via: make testmetal
// Run  via:  make runtestmetal
//            (or: ./testmetal --gtest_filter=GOLDILOCKS_TEST.*)
//
// This file does NOT include or modify tests/tests.cpp.

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>

// Platform + Goldilocks core headers
#include "platform.hpp"
#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"
#include "ntt_goldilocks.hpp"

#ifndef GOLDILOCKS_HAS_METAL
#  error "tests_metal.cpp must be compiled with -DGOLDILOCKS_HAS_METAL"
#endif

// Metal bridge headers (delivered by Parts 5 & 6)
#include "metal/metal_context.hpp"
#include "metal/goldilocks_metal.hpp"

// ============================================================================
// TEST: merkletree_metal
//
// Mirrors merkletree_cuda from tests/tests.cpp:3202-3268.
// Oracle roots are the same 128-column Fibonacci sequence (confirmed seq/avx/cuda).
// ============================================================================
TEST(GOLDILOCKS_TEST, merkletree_metal)
{
    // --- Primary case: 128 cols × 64 rows, Fibonacci fill ---
    uint64_t ncols_hash = 128;
    uint64_t nrows_hash = (1 << 6);

    Goldilocks::Element *cols = (Goldilocks::Element *)malloc(
        (uint64_t)ncols_hash * (uint64_t)nrows_hash * sizeof(Goldilocks::Element));
    ASSERT_NE(cols, nullptr);

    // Row 0: cols[i] = i + 1
    // Row 1: cols[ncols + i] = i + 1
#pragma omp parallel for
    for (uint64_t i = 0; i < ncols_hash; i++)
    {
        cols[i]              = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + ncols_hash] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    // Rows 2..63: Fibonacci recurrence
    for (uint64_t j = 2; j < nrows_hash; j++)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < ncols_hash; i++)
        {
            cols[j * ncols_hash + i] =
                cols[(j - 2) * ncols_hash + i] + cols[(j - 1) * ncols_hash + i];
        }
    }

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(nrows_hash);
    Goldilocks::Element *tree = (Goldilocks::Element *)malloc(
        numElementsTree * sizeof(Goldilocks::Element));
    ASSERT_NE(tree, nullptr);

    PoseidonGoldilocks::merkletree_metal(tree, cols, ncols_hash, nrows_hash);

    Goldilocks::Element root[4];
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    ASSERT_EQ(Goldilocks::toU64(root[0]), 0X918F7CD0C3E8701F);
    ASSERT_EQ(Goldilocks::toU64(root[1]), 0X83A130E00F961B02);
    ASSERT_EQ(Goldilocks::toU64(root[2]), 0X6921497B364123F8);
    ASSERT_EQ(Goldilocks::toU64(root[3]), 0XBD2B98A57B748BF4);

    free(cols);
    free(tree);

    // --- Edge case: 0 cols × 64 rows ---
    ncols_hash = 0;
    nrows_hash = (1 << 6);
    numElementsTree = MerklehashGoldilocks::getTreeNumElements(nrows_hash);
    tree = (Goldilocks::Element *)malloc(numElementsTree * sizeof(Goldilocks::Element));
    ASSERT_NE(tree, nullptr);
    cols = nullptr;

    PoseidonGoldilocks::merkletree_metal(tree, cols, ncols_hash, nrows_hash);
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    ASSERT_EQ(Goldilocks::toU64(root[0]), 0X25225F1A5D49614A);
    ASSERT_EQ(Goldilocks::toU64(root[1]), 0X5A1D2A648EEE8F03);
    ASSERT_EQ(Goldilocks::toU64(root[2]), 0xDDA8F741C47DFB10);
    ASSERT_EQ(Goldilocks::toU64(root[3]), 0X49561260080D30C3);

    free(tree);

    // --- Edge case: 0 cols × 131072 rows ---
    ncols_hash = 0;
    nrows_hash = (1 << 17);
    numElementsTree = MerklehashGoldilocks::getTreeNumElements(nrows_hash);
    tree = (Goldilocks::Element *)malloc(numElementsTree * sizeof(Goldilocks::Element));
    ASSERT_NE(tree, nullptr);
    cols = nullptr;

    PoseidonGoldilocks::merkletree_metal(tree, cols, ncols_hash, nrows_hash);
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    ASSERT_EQ(Goldilocks::toU64(root[0]), 0X5587AD00B6DDF0CB);
    ASSERT_EQ(Goldilocks::toU64(root[1]), 0X279949E14530C250);
    ASSERT_EQ(Goldilocks::toU64(root[2]), 0x2F8E22C79467775);
    ASSERT_EQ(Goldilocks::toU64(root[3]), 0XAA45BE01F9E1610);

    free(tree);
}

// ============================================================================
// TEST: NTT_Metal_roundtrip
//
// Verifies that NTT_Metal followed by INTT_Metal recovers the original input
// element-wise (modulo p = 2^64 - 2^32 + 1).
// Domain size: 1 << 16 (65536 elements, single column).
// ============================================================================
TEST(GOLDILOCKS_TEST, NTT_Metal_roundtrip)
{
    const uint64_t N = 1 << 16;

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(N * sizeof(Goldilocks::Element));
    Goldilocks::Element *orig = (Goldilocks::Element *)malloc(N * sizeof(Goldilocks::Element));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(orig, nullptr);

    // Fill with a simple pattern: a[i] = i + 1 (mod p)
    for (uint64_t i = 0; i < N; i++)
    {
        a[i] = Goldilocks::fromU64(i + 1);
    }
    memcpy(orig, a, N * sizeof(Goldilocks::Element));

    // NTT instance owns the twiddle tables used by the Metal bridge.
    NTT_Goldilocks ntt(N);

    // Forward NTT on Metal GPU (inverse=false)
    ntt.NTT_Metal(a, a, N, /*ncols=*/1, /*inverse=*/false);

    // Inverse NTT on Metal GPU (inverse=true)
    ntt.NTT_Metal(a, a, N, /*ncols=*/1, /*inverse=*/true);

    // Verify element-wise equality
    for (uint64_t i = 0; i < N; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(orig[i]))
            << "Mismatch at index " << i;
    }

    free(a);
    free(orig);
}

// ============================================================================
// TEST: extendPol_Metal
// ============================================================================
// Verifies that NTT_Goldilocks::extendPol_Metal produces bit-exact output
// against the reference NTT_Goldilocks::extendPol on a small Fibonacci-seeded
// input. Covers the full LDE pipeline: INTT-with-coset, zero-extension,
// forward NTT on the extended domain.
TEST(GOLDILOCKS_TEST, extendPol_Metal)
{
    const uint64_t N       = 1ULL << 10;   // 1024
    const uint64_t N_Ext   = 1ULL << 11;   // 2048 (blowup = 2)
    const uint64_t ncols   = 4;

    std::vector<Goldilocks::Element> input(N * ncols);
    std::vector<Goldilocks::Element> cpu_out(N_Ext * ncols);
    std::vector<Goldilocks::Element> gpu_out(N_Ext * ncols);
    std::vector<Goldilocks::Element> cpu_buf(N_Ext * ncols);

    // Fibonacci seed per column.
    for (uint64_t c = 0; c < ncols; c++)
        input[c] = input[ncols + c] = Goldilocks::one();
    for (uint64_t i = 2; i < N; i++) {
        for (uint64_t c = 0; c < ncols; c++) {
            input[i * ncols + c] = input[(i - 1) * ncols + c] + input[(i - 2) * ncols + c];
        }
    }

    NTT_Goldilocks ntt(N);
    ntt.extendPol(cpu_out.data(), input.data(), N_Ext, N, ncols, cpu_buf.data());
    ntt.extendPol_Metal(gpu_out.data(), input.data(), N_Ext, N, ncols);

    for (uint64_t i = 0; i < N_Ext * ncols; i++) {
        ASSERT_EQ(Goldilocks::toU64(gpu_out[i]), Goldilocks::toU64(cpu_out[i]))
            << "Mismatch at index " << i;
    }
}

// ============================================================================
// Environment: explicit Metal library load before any test runs.
//
// `newDefaultLibrary` in metal_context.mm expects a sibling `default.metallib`
// adjacent to the binary image. `make runtestmetal` copies the build artifact
// to CWD as `goldilocks.metallib` — not where `newDefaultLibrary` looks.
// This SetUpTestSuite tries explicit loads in order: CWD, argv[0]-adjacent,
// runtime source compile. Matches the three-tier pattern in probe_merkle.mm.
// ============================================================================
static const char *g_argv0 = nullptr;

class MetalEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        MetalCtxHandle ctx = metal_context_get();
        if (metal_context_load_library(ctx, "./goldilocks.metallib") == 0) return;
        if (g_argv0)
        {
            std::string p(g_argv0);
            size_t slash = p.find_last_of('/');
            std::string dir = (slash == std::string::npos) ? "." : p.substr(0, slash);
            std::string adj = dir + "/goldilocks.metallib";
            if (metal_context_load_library(ctx, adj.c_str()) == 0) return;
        }
        // If metallib is missing, first Metal dispatch inside a test will
        // print the bridge's own diagnostic and abort there. We do not mark
        // SetUp failed so the diagnostic can surface with full kernel context.
    }
};

int main(int argc, char **argv)
{
    g_argv0 = (argc > 0) ? argv[0] : nullptr;
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MetalEnvironment);
    return RUN_ALL_TESTS();
}
