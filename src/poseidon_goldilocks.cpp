#include "poseidon_goldilocks.hpp"
#include <math.h> /* floor */

void PoseidonGoldilocks::hash(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}
void PoseidonGoldilocks::hash_full_result(Goldilocks::Element (&state)[SPONGE_WIDTH], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{

    std::memcpy(state, input, SPONGE_WIDTH * sizeof(Goldilocks::Element));
    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] + PoseidonGoldilocksConstants::C[i];
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            pow7(state[j]);
            state[j] = state[j] + PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH + j];
        }

        Goldilocks::Element old_state[SPONGE_WIDTH];
        std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            state[i] = Goldilocks::zero();
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
                mji = mji * old_state[j];
                state[i] = state[i] + mji;
            }
        }
    }

    for (int j = 0; j < SPONGE_WIDTH; j++)
    {
        pow7(state[j]);
        state[j] = state[j] + PoseidonGoldilocksConstants::C[j + (HALF_N_FULL_ROUNDS * SPONGE_WIDTH)];
    }

    Goldilocks::Element old_state[SPONGE_WIDTH];
    std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = Goldilocks::zero();
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (PoseidonGoldilocksConstants::P[j][i] * old_state[j]);
        }
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(state[0]);
        state[0] = state[0] + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        Goldilocks::Element s0 = state[0] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];

        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            s0 = s0 + (state[j] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + j]);
            state[j] = state[j] + (state[0] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH + j - 1]);
        }
        state[0] = s0;
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            pow7(state[j]);
            state[j] = state[j] + PoseidonGoldilocksConstants::C[j + (HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH];
        }

        Goldilocks::Element old_state[SPONGE_WIDTH];
        std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            state[i] = Goldilocks::zero();
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                state[i] = state[i] + (old_state[j] * PoseidonGoldilocksConstants::M[j][i]);
            }
        }
    }

    for (int j = 0; j < SPONGE_WIDTH; j++)
    {
        pow7(state[j]);
    }

    std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = Goldilocks::zero();
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (old_state[j] * PoseidonGoldilocksConstants::M[j][i]);
        }
    }
}

void PoseidonGoldilocks::hash_full_result_block(Goldilocks::Element *state, const Goldilocks::Element *input, int ncols)
{

    const int length = SPONGE_WIDTH * ncols * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        int offset = i * ncols;
        const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[i];
#pragma omp simd
        for (int k = 0; k < ncols; ++k)
        {
            state[offset + k] = state[offset + k] + C_;
        }
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            int offset = i * ncols;
            const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH + i];
#pragma omp simd
            for (int k = 0; k < ncols; ++k)
            {
                pow7(state[offset + k]);
                state[offset + k] = state[offset + k] + C_;
            }
        }
        Goldilocks::Element old_state[SPONGE_WIDTH * ncols];
        std::memcpy(old_state, state, length);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            int offseti = i * ncols;
#pragma omp simd
            for (int k = 0; k < ncols; ++k)
            {
                state[offseti + k] = Goldilocks::zero();
            }
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                const Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
                int offsetj = j * ncols;
#pragma omp simd
                for (int k = 0; k < ncols; ++k)
                {
                    state[offseti + k] = state[offseti + k] + mji * old_state[offsetj + k];
                }
            }
        }
    }

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        int offset = i * ncols;
        const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[i + (HALF_N_FULL_ROUNDS * SPONGE_WIDTH)];
#pragma omp simd
        for (int k = 0; k < ncols; ++k)
        {
            pow7(state[offset + k]);
            state[offset + k] = state[offset + k] + C_;
        }
    }

    Goldilocks::Element old_state[SPONGE_WIDTH * ncols];
    std::memcpy(old_state, state, length);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        int offseti = i * ncols;
#pragma omp simd
        for (int k = 0; k < ncols; ++k)
        {
            state[offseti + k] = Goldilocks::zero();
        }
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            int offsetj = j * ncols;
            Goldilocks::Element pji = PoseidonGoldilocksConstants::P[j][i];
#pragma omp simd
            for (int k = 0; k < ncols; ++k)
            {
                state[offseti + k] = state[offseti + k] + (pji * old_state[offsetj + k]);
            }
        }
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        Goldilocks::Element s0[ncols];
        const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        const Goldilocks::Element S_ = PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
#pragma omp simd
        for (int k = 0; k < ncols; ++k)
        {
            pow7(state[k]);
            state[k] = state[k] + C_;
            s0[k] = state[k] * S_;
        }

        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            int offsetj = j * ncols;
            const Goldilocks::Element S1_ = PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + j];
            const Goldilocks::Element S2_ = PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH + j - 1];
#pragma omp simd
            for (int k = 0; k < ncols; ++k)
            {
                s0[k] = s0[k] + state[offsetj + k] * S1_;
                state[offsetj + k] = state[offsetj + k] + state[k] * S2_;
            }
        }
#pragma omp simd
        for (int k = 0; k < ncols; ++k)
        {
            state[k] = s0[k];
        }
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            int offsetj = j * ncols;
            const Goldilocks::Element C_ = PoseidonGoldilocksConstants::C[j + (HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH];
#pragma omp simd
            for (int k = 0; k < ncols; ++k)
            {
                pow7(state[offsetj + k]);
                state[offsetj + k] = state[offsetj + k] + C_;
            }
        }

        Goldilocks::Element old_state[SPONGE_WIDTH * ncols];
        std::memcpy(old_state, state, length);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            int offseti = i * ncols;
#pragma omp simd
            for (int k = 0; k < ncols; ++k)
            {
                state[offseti + k] = Goldilocks::zero();
            }
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                int offsetj = j * ncols;
                Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
#pragma omp simd
                for (int k = 0; k < ncols; ++k)
                {
                    state[offseti + k] = state[offseti + k] + (mji * old_state[offsetj + k]);
                }
            }
        }
    }

    for (int j = 0; j < SPONGE_WIDTH; j++)
    {
        int offsetj = j * ncols;
#pragma omp simd
        for (int k = 0; k < ncols; ++k)
        {
            pow7(state[offsetj + k]);
        }
    }
    std::memcpy(old_state, state, length);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        int offseti = i * ncols;
#pragma omp simd
        for (int k = 0; k < ncols; ++k)
        {
            state[offseti + k] = Goldilocks::zero();
        }
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            int offsetj = j * ncols;
            Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
#pragma omp simd
            for (int k = 0; k < ncols; ++k)
            {
                state[offseti + k] = state[offseti + k] + (mji * old_state[offsetj + k]);
            }
        }
    }
}

void PoseidonGoldilocks::linear_hash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + RATE, 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        else
        {
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;

        memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));

        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));

        hash_full_result(state, state);

        remaining -= n;
    }
    std::memcpy(output, state, CAPACITY * sizeof(uint64_t));
}

void PoseidonGoldilocks::merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows)
{

#pragma omp parallel for
    for (uint64_t i = 0; i < num_rows; i++)
    {
        Goldilocks::Element intermediate[num_cols];
        std::memcpy(&intermediate[0], &input[i * num_cols], num_cols * sizeof(Goldilocks::Element));
        linear_hash(&tree[i * CAPACITY], intermediate, num_cols);
    }

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;

    Goldilocks::Element *cursor = tree;
    while (pending > 1)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
            std::memcpy(pol_input, &cursor[nextIndex + i * RATE], RATE * sizeof(Goldilocks::Element));
            hash((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + i) * CAPACITY], pol_input);
        }
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
}
