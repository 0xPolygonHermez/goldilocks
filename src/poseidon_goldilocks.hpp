#ifndef POSEIDON_GOLDILOCKS
#define POSEIDON_GOLDILOCKS

#include "poseidon_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"
#include <immintrin.h>

#define RATE 8
#define CAPACITY 4
#define HASH_SIZE 4
#define SPONGE_WIDTH (RATE + CAPACITY)
#define HALF_N_FULL_ROUNDS 4
#define N_FULL_ROUNDS_TOTAL (2 * HALF_N_FULL_ROUNDS)
#define N_PARTIAL_ROUNDS 22
#define N_ROUNDS (N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS)

class PoseidonGoldilocks
{

private:
    inline void static pow7(Goldilocks::Element &x)
    {
        Goldilocks::Element x2 = x * x;
        Goldilocks::Element x3 = x * x2;
        Goldilocks::Element x4 = x2 * x2;
        x = x3 * x4;
    };

    inline void static pow7(Goldilocks::Element *x, int ncols)
    {
        Goldilocks::Element x2[SPONGE_WIDTH * ncols], x3[SPONGE_WIDTH * ncols], x4[SPONGE_WIDTH * ncols];
        for (int k = 0; k < SPONGE_WIDTH * ncols; ++k)
        {
            x2[k] = x[k] * x[k];
        }
        for (int k = 0; k < SPONGE_WIDTH * ncols; ++k)
        {
            x3[k] = x[k] * x2[k];
            x4[k] = x2[k] * x2[k];
            x[k] = x3[k] * x4[k];
        }
    };

    inline void static add(Goldilocks::Element *x, int ncols, const Goldilocks::Element C[SPONGE_WIDTH])
    {
        for (int i = 0; i < SPONGE_WIDTH; ++i)
        {
            int offset = i * ncols;
            //#pragma omp simd
            for (int k = 0; k < ncols; ++k)
            {
                x[offset + k] = x[offset + k] + C[i];
            }
        }
    };
    // Rick: I assume ncols is multiple of 4
    inline void static add_(Goldilocks::Element *x, int ncols, const Goldilocks::Element C[SPONGE_WIDTH])
    {

        for (int i = 0; i < SPONGE_WIDTH; ++i)
        {
            int offset = i * ncols;
            __m256i c_ = _mm256_set_epi64x(C[i].fe, C[i].fe, C[i].fe, C[i].fe);
            for (int k = 0; k < ncols; k += 4)
            {
                __m256i x_ = _mm256_loadu_si256((__m256i *)&(x[offset + k]));
                x_ = _mm256_add_epi64(x_, c_);
            }
        }
    };

    inline void static pow7add(Goldilocks::Element *x, int ncols, const Goldilocks::Element C[SPONGE_WIDTH])
    {
        Goldilocks::Element x2[SPONGE_WIDTH * ncols], x3[SPONGE_WIDTH * ncols], x4[SPONGE_WIDTH * ncols];
        for (int k = 0; k < SPONGE_WIDTH * ncols; ++k)
        {
            x2[k] = x[k] * x[k];
        }
        for (int k = 0; k < SPONGE_WIDTH * ncols; ++k)
        {
            x3[k] = x[k] * x2[k];
            x4[k] = x2[k] * x2[k];
            x[k] = x3[k] * x4[k];
        }
        for (int i = 0; i < SPONGE_WIDTH; ++i)
        {
            int offset = i * ncols;
            for (int k = 0; k < ncols; ++k)
            {
                x[offset + k] = x[offset + k] + C[i];
            }
        }
    };

    // rick: check transpose access to matrix
    inline void static mvp(Goldilocks::Element *state, const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH], int ncols)
    {
        Goldilocks::Element old_state[SPONGE_WIDTH * ncols];
        const int length = SPONGE_WIDTH * ncols * sizeof(Goldilocks::Element);

        std::memcpy(old_state, state, length);
        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            int offseti = i * ncols;
            for (int k = 0; k < ncols; ++k)
            {
                state[offseti + k] = mat[0][i] * old_state[k];
            }
            for (int j = 1; j < SPONGE_WIDTH; j++)
            {
                int offsetj = j * ncols;
                for (int k = 0; k < ncols; ++k)
                {
                    state[offseti + k] = state[offseti + k] + (mat[j][i] * old_state[offsetj + k]);
                }
            }
        }
    };

public:
    void static hash_full_result(Goldilocks::Element (&state)[SPONGE_WIDTH], Goldilocks::Element const (&input)[SPONGE_WIDTH]);
    void static hash_full_result_block(Goldilocks::Element *, const Goldilocks::Element *, int ncols);
    template <int NCOLS>
    void static hash_full_result_block2(Goldilocks::Element *, const Goldilocks::Element *);
    void static hash(Goldilocks::Element (&state)[CAPACITY], const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    void static linear_hash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t dim = 1);
};

template <int NCOLS>
void PoseidonGoldilocks::hash_full_result_block2(Goldilocks::Element *state, const Goldilocks::Element *input)
{

    const int length = SPONGE_WIDTH * NCOLS * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        int offset = i * NCOLS;
        const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[i];
        //#pragma omp simd
        for (int k = 0; k < NCOLS; ++k)
        {
            state[offset + k] = state[offset + k] + C_;
        }
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            int offset = i * NCOLS;
            const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH + i];
            //#pragma omp simd
            for (int k = 0; k < NCOLS; ++k)
            {
                pow7(state[offset + k]);
                state[offset + k] = state[offset + k] + C_;
            }
        }
        Goldilocks::Element old_state[SPONGE_WIDTH * NCOLS];
        std::memcpy(old_state, state, length);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            int offseti = i * NCOLS;
            //#pragma omp simd
            for (int k = 0; k < NCOLS; ++k)
            {
                state[offseti + k] = Goldilocks::zero();
            }
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                const Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
                int offsetj = j * NCOLS;
                //#pragma omp simd
                for (int k = 0; k < NCOLS; ++k)
                {
                    state[offseti + k] = state[offseti + k] + mji * old_state[offsetj + k];
                }
            }
        }
    }

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        int offset = i * NCOLS;
        const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[i + (HALF_N_FULL_ROUNDS * SPONGE_WIDTH)];
        //#pragma omp simd
        for (int k = 0; k < NCOLS; ++k)
        {
            pow7(state[offset + k]);
            state[offset + k] = state[offset + k] + C_;
        }
    }

    Goldilocks::Element old_state[SPONGE_WIDTH * NCOLS];
    std::memcpy(old_state, state, length);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        int offseti = i * NCOLS;
        //#pragma omp simd
        for (int k = 0; k < NCOLS; ++k)
        {
            state[offseti + k] = Goldilocks::zero();
        }
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            int offsetj = j * NCOLS;
            Goldilocks::Element pji = PoseidonGoldilocksConstants::P[j][i];
            //#pragma omp simd
            for (int k = 0; k < NCOLS; ++k)
            {
                state[offseti + k] = state[offseti + k] + (pji * old_state[offsetj + k]);
            }
        }
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        Goldilocks::Element s0[NCOLS];
        const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        const Goldilocks::Element S_ = PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
        //#pragma omp simd
        for (int k = 0; k < NCOLS; ++k)
        {
            pow7(state[k]);
            state[k] = state[k] + C_;
            s0[k] = state[k] * S_;
        }

        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            int offsetj = j * NCOLS;
            const Goldilocks::Element S1_ = PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + j];
            const Goldilocks::Element S2_ = PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH + j - 1];
            //#pragma omp simd
            for (int k = 0; k < NCOLS; ++k)
            {
                s0[k] = s0[k] + state[offsetj + k] * S1_;
                state[offsetj + k] = state[offsetj + k] + state[k] * S2_;
            }
        }
        //#pragma omp simd
        for (int k = 0; k < NCOLS; ++k)
        {
            state[k] = s0[k];
        }
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            int offsetj = j * NCOLS;
            const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[j + (HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH];
            //#pragma omp simd
            for (int k = 0; k < NCOLS; ++k)
            {
                pow7(state[offsetj + k]);
                state[offsetj + k] = state[offsetj + k] + C_;
            }
        }

        Goldilocks::Element old_state[SPONGE_WIDTH * NCOLS];
        std::memcpy(old_state, state, length);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            int offseti = i * NCOLS;
            //#pragma omp simd
            for (int k = 0; k < NCOLS; ++k)
            {
                state[offseti + k] = Goldilocks::zero();
            }
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                int offsetj = j * NCOLS;
                Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
                //#pragma omp simd
                for (int k = 0; k < NCOLS; ++k)
                {
                    state[offseti + k] = state[offseti + k] + (mji * old_state[offsetj + k]);
                }
            }
        }
    }

    for (int j = 0; j < SPONGE_WIDTH; j++)
    {
        int offsetj = j * NCOLS;
        //#pragma omp simd
        for (int k = 0; k < NCOLS; ++k)
        {
            pow7(state[offsetj + k]);
        }
    }
    std::memcpy(old_state, state, length);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        int offseti = i * NCOLS;
        //#pragma omp simd
        for (int k = 0; k < NCOLS; ++k)
        {
            state[offseti + k] = Goldilocks::zero();
        }
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            int offsetj = j * NCOLS;
            Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
            //#pragma omp simd
            for (int k = 0; k < NCOLS; ++k)
            {
                state[offseti + k] = state[offseti + k] + (mji * old_state[offsetj + k]);
            }
        }
    }
}

#endif