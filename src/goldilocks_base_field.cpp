#include "goldilocks_base_field.hpp"

const Goldilocks::Element Goldilocks::Q = {(uint64_t)0xFFFFFFFF00000001LL};
const Goldilocks::Element Goldilocks::MM = {(uint64_t)0xFFFFFFFeFFFFFFFFLL};
const Goldilocks::Element Goldilocks::CQ = {(uint64_t)0x00000000FFFFFFFFLL};
const Goldilocks::Element Goldilocks::R2 = {(uint64_t)0xFFFFFFFe00000001LL};
#if USE_MONTGOMERY == 0
const Goldilocks::Element Goldilocks::ONE = {(uint64_t)0x0000000000000001LL};
const Goldilocks::Element Goldilocks::ZERO = {(uint64_t)0x0000000000000000LL};
const Goldilocks::Element Goldilocks::NEGONE = {(uint64_t)0xFFFFFFFF00000000LL};
#else
const Goldilocks::Element Goldilocks::ONE = {(uint64_t)0x00000000FFFFFFFFLL};
const Goldilocks::Element Goldilocks::ZERO = {(uint64_t)0x0000000000000000LL};
const Goldilocks::Element Goldilocks::NEGONE = {(uint64_t)0XFFFFFFFE00000002LL};
#endif