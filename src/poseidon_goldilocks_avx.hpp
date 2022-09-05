#ifndef POSEIDON_GOLDILOCKS_AVX
#define POSEIDON_GOLDILOCKS_AVX

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

inline void PoseidonGoldilocks::pow7(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
};

inline void PoseidonGoldilocks::pow7_(Goldilocks::Element *x)
{
    Goldilocks::Element x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
};

inline void PoseidonGoldilocks::pow7_avx(__m256i &st0, __m256i &st1, __m256i &st2)
{
    __m256i pw2_0, pw2_1, pw2_2;
    Goldilocks::square_avx(pw2_0, st0);
    Goldilocks::square_avx(pw2_1, st1);
    Goldilocks::square_avx(pw2_2, st2);
    __m256i pw4_0, pw4_1, pw4_2;
    Goldilocks::square_avx(pw4_0, pw2_0);
    Goldilocks::square_avx(pw4_1, pw2_1);
    Goldilocks::square_avx(pw4_2, pw2_2);
    __m256i pw3_0, pw3_1, pw3_2;
    Goldilocks::mult_avx(pw3_0, pw2_0, st0);
    Goldilocks::mult_avx(pw3_1, pw2_1, st1);
    Goldilocks::mult_avx(pw3_2, pw2_2, st2);

    Goldilocks::mult_avx(st0, pw3_0, pw4_0);
    Goldilocks::mult_avx(st1, pw3_1, pw4_1);
    Goldilocks::mult_avx(st2, pw3_2, pw4_2);
};

inline void PoseidonGoldilocks::add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i] + C[i];
    }
}
inline void PoseidonGoldilocks::add_avx(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    __m256i c0, c1, c2;
    Goldilocks::load(c0, &(C_[0]));
    Goldilocks::load(c1, &(C_[4]));
    Goldilocks::load(c2, &(C_[8]));
    Goldilocks::add_avx(st0, st0, c0);
    Goldilocks::add_avx(st1, st1, c1);
    Goldilocks::add_avx(st2, st2, c2);
}
inline void PoseidonGoldilocks::prod_(Goldilocks::Element *x, const Goldilocks::Element alpha, const Goldilocks::Element C[SPONGE_WIDTH])
{
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = alpha * C[i];
    }
}

inline void PoseidonGoldilocks::pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{
    Goldilocks::Element x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
        x[i] = x[i] + C[i];
    }
};

inline Goldilocks::Element PoseidonGoldilocks::dot_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{
    Goldilocks::Element s0 = x[0] * C[0];
    for (int i = 1; i < SPONGE_WIDTH; i++)
    {
        s0 = s0 + x[i] * C[i];
    }
    return s0;
};

// rick: check transpose access to matrix
inline void PoseidonGoldilocks::mvp_(Goldilocks::Element *state, const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH])
{
    Goldilocks::Element old_state[SPONGE_WIDTH];
    std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {

        state[i] = mat[0][i] * old_state[0];
        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (mat[j][i] * old_state[j]);
        }
    }
};

#endif