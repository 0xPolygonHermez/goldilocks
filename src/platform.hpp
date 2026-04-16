#ifndef GOLDILOCKS_PLATFORM_HPP
#define GOLDILOCKS_PLATFORM_HPP

// ============================================================
// GOLDILOCKS PLATFORM DETECTION
// Single source of truth for all platform/ISA capability macros.
// ============================================================

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
#  define GOLDILOCKS_ARCH_X86_64
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  define GOLDILOCKS_ARCH_AARCH64
#endif

// ISA capability detection
#if defined(GOLDILOCKS_ARCH_X86_64) && defined(__AVX2__)
#  define GOLDILOCKS_HAS_AVX2
#endif

// Note: GOLDILOCKS_HAS_AVX512 is defined for future use. Existing code uses
// #ifdef __AVX512__ directly; migrating to this macro is a separate cleanup.
#if defined(GOLDILOCKS_ARCH_X86_64) && \
    (defined(__AVX512__) || defined(__AVX512F__))
#  define GOLDILOCKS_HAS_AVX512
#endif

#if defined(GOLDILOCKS_ARCH_AARCH64)
#  define GOLDILOCKS_HAS_NEON
#endif

#if defined(__CUDACC__) || defined(GOLDILOCKS_HAS_CUDA)
#  ifndef GOLDILOCKS_HAS_CUDA
#    define GOLDILOCKS_HAS_CUDA
#  endif
#endif

// Metal GPU (Apple Silicon, macOS only)
// Auto-enabled on __APPLE__ + __aarch64__ unless user predefines GOLDILOCKS_NO_METAL.
// To disable: add -DGOLDILOCKS_NO_METAL to your compile flags.
#if defined(__APPLE__) && defined(__aarch64__) && !defined(GOLDILOCKS_NO_METAL)
#  define GOLDILOCKS_HAS_METAL
#endif

// OS detection
#if defined(__APPLE__)
#  define GOLDILOCKS_OS_MACOS
#endif

#if defined(__linux__)
#  define GOLDILOCKS_OS_LINUX
#endif

#endif // GOLDILOCKS_PLATFORM_HPP
