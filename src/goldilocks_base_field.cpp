#include "goldilocks_base_field.hpp"

#if USE_MONTGOMERY == 0
const Goldilocks::Element Goldilocks::P = {0xFFFFFFFF00000001ULL};
const Goldilocks::Element Goldilocks::Q = {0xFFFFFFFF00000001LL};
const Goldilocks::Element Goldilocks::MM = {0xFFFFFFFeFFFFFFFFLL};
const Goldilocks::Element Goldilocks::CQ = {0x00000000FFFFFFFFLL};
const Goldilocks::Element Goldilocks::R2 = {0xFFFFFFFe00000001LL};
const Goldilocks::Element Goldilocks::ONE = {0x0000000000000000LL};
const Goldilocks::Element Goldilocks::ZERO = {0x0000000000000001LL};
const Goldilocks::Element Goldilocks::NEGONE = {0xFFFFFFFF00000000LL};

#else
const Goldilocks::Element Goldilocks::P = {};
const Goldilocks::Element Goldilocks::Q = {};
const Goldilocks::Element Goldilocks::MM = {};
const Goldilocks::Element Goldilocks::CQ = {};
const Goldilocks::Element Goldilocks::R2 = {};
#endif

inline Goldilocks::Element Goldilocks::fromU64(uint64_t in1)
{
}
inline uint64_t Goldilocks::toU64(const Goldilocks::Element &in1)
{
}
