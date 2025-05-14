#ifndef POSEIDON_GOLDILOCKS_NEON
#define POSEIDON_GOLDILOCKS_NEON

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"

inline void PoseidonGoldilocks::hash(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

inline void PoseidonGoldilocks::pow7_neon(uint64x2_t &st0, uint64x2_t &st1, uint64x2_t &st2, uint64x2_t &st3, uint64x2_t &st4, uint64x2_t &st5)
{
    uint64x2_t pw2_0, pw2_1, pw2_2, pw2_3, pw2_4, pw2_5;
    Goldilocks::square_neon(pw2_0, st0);
    Goldilocks::square_neon(pw2_1, st1);
    Goldilocks::square_neon(pw2_2, st2);
    Goldilocks::square_neon(pw2_3, st3);
    Goldilocks::square_neon(pw2_4, st4);
    Goldilocks::square_neon(pw2_5, st5);

    uint64x2_t pw4_0, pw4_1, pw4_2, pw4_3, pw4_4, pw4_5;
    Goldilocks::square_neon(pw4_0, pw2_0);
    Goldilocks::square_neon(pw4_1, pw2_1);
    Goldilocks::square_neon(pw4_2, pw2_2);
    Goldilocks::square_neon(pw4_3, pw2_3);
    Goldilocks::square_neon(pw4_4, pw2_4);
    Goldilocks::square_neon(pw4_5, pw2_5);

    uint64x2_t pw3_0, pw3_1, pw3_2, pw3_3, pw3_4, pw3_5;
    Goldilocks::mult_neon(pw3_0, pw2_0, st0);
    Goldilocks::mult_neon(pw3_1, pw2_1, st1);
    Goldilocks::mult_neon(pw3_2, pw2_2, st2);
    Goldilocks::mult_neon(pw3_3, pw2_3, st3);
    Goldilocks::mult_neon(pw3_4, pw2_4, st4);
    Goldilocks::mult_neon(pw3_5, pw2_5, st5);

    Goldilocks::mult_neon(st0, pw3_0, pw4_0);
    Goldilocks::mult_neon(st1, pw3_1, pw4_1);
    Goldilocks::mult_neon(st2, pw3_2, pw4_2);
    Goldilocks::mult_neon(st3, pw3_3, pw4_3);
    Goldilocks::mult_neon(st4, pw3_4, pw4_4);
    Goldilocks::mult_neon(st5, pw3_5, pw4_5);
}

inline void PoseidonGoldilocks::add_neon(uint64x2_t &st0, uint64x2_t &st1, uint64x2_t &st2, uint64x2_t &st3, uint64x2_t &st4, uint64x2_t &st5, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    uint64x2_t c0, c1, c2, c3, c4, c5;
    Goldilocks::load_neon(c0, &(C_[0]));
    Goldilocks::load_neon(c1, &(C_[2]));
    Goldilocks::load_neon(c2, &(C_[4]));
    Goldilocks::load_neon(c3, &(C_[6]));
    Goldilocks::load_neon(c4, &(C_[8]));
    Goldilocks::load_neon(c5, &(C_[10]));
    Goldilocks::add_neon(st0, st0, c0);
    Goldilocks::add_neon(st1, st1, c1);
    Goldilocks::add_neon(st2, st2, c2);
    Goldilocks::add_neon(st3, st3, c3);
    Goldilocks::add_neon(st4, st4, c4);
    Goldilocks::add_neon(st5, st5, c5);
}

inline void PoseidonGoldilocks::add_neon_small(uint64x2_t &st0, uint64x2_t &st1, uint64x2_t &st2, uint64x2_t &st3, uint64x2_t &st4, uint64x2_t &st5, const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    uint64x2_t c0, c1, c2, c3, c4, c5;
    Goldilocks::load_neon(c0, &(C_small[0]));
    Goldilocks::load_neon(c1, &(C_small[2]));
    Goldilocks::load_neon(c2, &(C_small[4]));
    Goldilocks::load_neon(c3, &(C_small[6]));
    Goldilocks::load_neon(c4, &(C_small[8]));
    Goldilocks::load_neon(c5, &(C_small[10]));

    Goldilocks::add_neon(st0, st0, c0);
    Goldilocks::add_neon(st1, st1, c1);
    Goldilocks::add_neon(st2, st2, c2);
    Goldilocks::add_neon(st3, st3, c3);
    Goldilocks::add_neon(st4, st4, c4);
    Goldilocks::add_neon(st5, st5, c5);
}

#endif      // POSEIDON_GOLDILOCKS_NEON