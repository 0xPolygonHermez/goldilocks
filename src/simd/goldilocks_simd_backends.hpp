#ifndef GOLDILOCKS_SIMD_BACKENDS_HPP
#define GOLDILOCKS_SIMD_BACKENDS_HPP

#include "../platform.hpp"

namespace goldilocks {
namespace simd {

// Backend tag types. Empty; used as template parameters only.
struct Scalar {};
struct Neon   {};
struct Avx2   {};   // reserved for Phase 6
struct Avx512 {};   // reserved for Phase 6

template <class B> struct backend_available         { static constexpr bool value = false; };
template <>        struct backend_available<Scalar> { static constexpr bool value = true;  };
#ifdef GOLDILOCKS_HAS_NEON
template <>        struct backend_available<Neon>   { static constexpr bool value = true;  };
#endif
#ifdef GOLDILOCKS_HAS_AVX2
template <>        struct backend_available<Avx2>   { static constexpr bool value = true;  };
#endif
#ifdef GOLDILOCKS_HAS_AVX512
template <>        struct backend_available<Avx512> { static constexpr bool value = true;  };
#endif

template <class B> inline constexpr bool backend_available_v = backend_available<B>::value;

} // namespace simd
} // namespace goldilocks

#endif // GOLDILOCKS_SIMD_BACKENDS_HPP
