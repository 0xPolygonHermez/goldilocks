#include "poseidon_goldilocks.hpp"
#include <math.h> /* floor */
#include "merklehash_goldilocks.hpp"

void PoseidonGoldilocks::hash_seq(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result_seq(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}
void PoseidonGoldilocks::hash(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}
void PoseidonGoldilocks::hash_full_result_seq_old(Goldilocks::Element (&state)[SPONGE_WIDTH], Goldilocks::Element const (&input)[SPONGE_WIDTH])
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
void PoseidonGoldilocks::hash_full_result_seq(Goldilocks::Element *state, const Goldilocks::Element *input)
{
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);

    add_(state, &(PoseidonGoldilocksConstants::C[0]));
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH]));
        mvp_(state, PoseidonGoldilocksConstants::M);
    }
    pow7add_(state, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    mvp_(state, PoseidonGoldilocksConstants::P);

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(state[0]);
        state[0] = state[0] + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        Goldilocks::Element s0 = dot_(state, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r]));
        Goldilocks::Element W_[SPONGE_WIDTH];
        prod_(W_, state[0], &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        add_(state, W_);
        state[0] = s0;
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        mvp_(state, PoseidonGoldilocksConstants::M);
    }
    pow7_(&(state[0]));
    mvp_(state, PoseidonGoldilocksConstants::M);
}

void PoseidonGoldilocks::hash_full_result(Goldilocks::Element *state, const Goldilocks::Element *input)
{

    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);
    __m256i st0, st1, st2;
    Goldilocks::load(st0, &(state[0]));
    Goldilocks::load(st1, &(state[4]));
    Goldilocks::load(st2, &(state[8]));
    add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_avx(st0, st1, st2);
        add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH]));
        Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));
    }
    pow7_avx(st0, st1, st2);
    add_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    Goldilocks::mmult_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::P_[0]));

    Goldilocks::store(&(state[0]), st0);
    Goldilocks::Element s0 = state[0];

    Goldilocks::Element state0;
    __m256i mask = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0);
    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        state0 = s0;
        pow7(state0);
        state0 = state0 + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        s0 = state0 * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
        st0 = _mm256_and_si256(st0, mask);
        s0 = s0 + Goldilocks::dot_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r]));
        __m256i scalar1 = _mm256_set1_epi64x(state0.fe);
        __m256i w0, w1, w2, s0, s1, s2;
        Goldilocks::load(s0, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        Goldilocks::load(s1, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 4]));
        Goldilocks::load(s2, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 8]));
        Goldilocks::mult_avx(w0, scalar1, s0);
        Goldilocks::mult_avx(w1, scalar1, s1);
        Goldilocks::mult_avx(w2, scalar1, s2);
        Goldilocks::add_avx(st0, st0, w0);
        Goldilocks::add_avx(st1, st1, w1);
        Goldilocks::add_avx(st2, st2, w2);
        state0 = state0 + PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
    }
    Goldilocks::store(&(state[0]), st0);
    state[0] = s0;
    Goldilocks::load(st0, &(state[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_avx(st0, st1, st2);
        add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));
    }
    pow7_avx(st0, st1, st2);
    Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_[0]));

    Goldilocks::store(&(state[0]), st0);
    Goldilocks::store(&(state[4]), st1);
    Goldilocks::store(&(state[8]), st2);
}

void PoseidonGoldilocks::linear_hash_seq(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    if (size <= CAPACITY)
    {
        std::memcpy(output, input, size * sizeof(Goldilocks::Element));
        return; // no need to hash
    }
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

        hash_full_result_seq(state, state);

        remaining -= n;
    }
    if (size > 0)
    {
        std::memcpy(output, state, CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, CAPACITY * sizeof(Goldilocks::Element));
    }
}
void PoseidonGoldilocks::linear_hash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    if (size <= CAPACITY)
    {
        std::memcpy(output, input, size * sizeof(Goldilocks::Element));
        std::memset(&output[size], 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        return; // no need to hash
    }
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
    if (size > 0)
    {
        std::memcpy(output, state, CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, CAPACITY * sizeof(Goldilocks::Element));
    }
}
void PoseidonGoldilocks::merkletree_seq(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    tree[0] = Goldilocks::fromU64(num_cols * dim);
    tree[1] = Goldilocks::fromU64(num_rows);
    int numThreads = omp_get_max_threads() / 2;
    Goldilocks::parcpy(&tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE], input, dim * num_cols * num_rows, numThreads);
    Goldilocks::Element *cursor = &tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE + num_cols * num_rows * dim];
    memset(cursor, 0, num_rows * CAPACITY * sizeof(Goldilocks::Element));

#pragma omp parallel for
    for (uint64_t i = 0; i < num_rows; i++)
    {
        Goldilocks::Element intermediate[num_cols * dim];
        std::memcpy(&intermediate[0], &input[i * num_cols * dim], dim * num_cols * sizeof(Goldilocks::Element));
        linear_hash_seq(&cursor[i * CAPACITY], intermediate, num_cols * dim);
    }

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
            std::memcpy(pol_input, &cursor[nextIndex + i * RATE], RATE * sizeof(Goldilocks::Element));
            hash_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + i) * CAPACITY], pol_input);
        }
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
}
void PoseidonGoldilocks::merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    tree[0] = Goldilocks::fromU64(num_cols * dim);
    tree[1] = Goldilocks::fromU64(num_rows);
    int numThreads = omp_get_max_threads() / 2;
    Goldilocks::parcpy(&tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE], input, dim * num_cols * num_rows, numThreads);
    Goldilocks::Element *cursor = &tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE + num_cols * num_rows * dim];
    memset(cursor, 0, num_rows * CAPACITY * sizeof(Goldilocks::Element));

#pragma omp parallel for
    for (uint64_t i = 0; i < num_rows; i++)
    {
        Goldilocks::Element intermediate[num_cols * dim];
        std::memcpy(&intermediate[0], &input[i * num_cols * dim], dim * num_cols * sizeof(Goldilocks::Element));
        linear_hash(&cursor[i * CAPACITY], intermediate, num_cols * dim);
    }

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;

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

void PoseidonGoldilocks::merkletree_batch(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t batch_size, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    tree[0] = Goldilocks::fromU64(num_cols * dim);
    tree[1] = Goldilocks::fromU64(num_rows);
    int numThreads = omp_get_max_threads() / 2;
    Goldilocks::parcpy(&tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE], input, dim * num_cols * num_rows, numThreads);
    Goldilocks::Element *cursor = &tree[MERKLEHASHGOLDILOCKS_HEADER_SIZE + num_cols * num_rows * dim];
    uint64_t nbatches = 1;
    if (num_cols > 0)
    {
        nbatches = (num_cols + batch_size - 1) / batch_size;
    }
    uint64_t nlastb = num_cols - (nbatches - 1) * batch_size;

#pragma omp parallel for
    for (uint64_t i = 0; i < num_rows; i++)
    {
        Goldilocks::Element buff0[nbatches * CAPACITY];
        for (uint64_t j = 0; j < nbatches; j++)
        {
            uint64_t nn = batch_size;
            if (j == nbatches - 1)
                nn = nlastb;
            Goldilocks::Element buff1[batch_size * dim];
            std::memcpy(&buff1[0], &input[i * num_cols * dim + j * batch_size * dim], dim * nn * sizeof(Goldilocks::Element));
            linear_hash(&buff0[j * CAPACITY], buff1, nn * dim);
        }
        linear_hash(&cursor[i * CAPACITY], buff0, nbatches * CAPACITY);
    }

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;

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