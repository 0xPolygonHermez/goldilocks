#ifdef __USE_NEON__

#include "poseidon_goldilocks.hpp"
#include <math.h> /* floor */
#include "merklehash_goldilocks.hpp"

void PoseidonGoldilocks::hash_full_result(Goldilocks::Element *state, const Goldilocks::Element *input)
{
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);
    uint64x2_t st0, st1, st2, st3, st4, st5;
    Goldilocks::load_neon(st0, &(state[0]));
    Goldilocks::load_neon(st1, &(state[2]));
    Goldilocks::load_neon(st2, &(state[4]));
    Goldilocks::load_neon(st3, &(state[6]));
    Goldilocks::load_neon(st4, &(state[8]));
    Goldilocks::load_neon(st5, &(state[10]));
    add_neon_small(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::C[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_neon(st0, st1, st2, st3, st4, st5);
        add_neon_small(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH]));
        Goldilocks::mmult_neon_8(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::M_[0]));
    }
    pow7_neon(st0, st1, st2, st3, st4, st5);
    add_neon(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    Goldilocks::mmult_neon(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::P_[0]));

    Goldilocks::store_neon(&(state[0]), st0);
    Goldilocks::Element state0_ = state[0];
    Goldilocks::Element state0;

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        state0 = state0_;
        pow7(state0);
        state0 = state0 + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        state0_ = state0 * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];
        st0 = vsetq_lane_u64(0, st0, 0);
        state0_ = state0_ + Goldilocks::dot_neon(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r]));
        uint64x2_t scalar1 = vdupq_n_u64(state0.fe);
        uint64x2_t w0, w1, w2, w3, w4, w5, s0, s1, s2, s3, s4, s5;
        Goldilocks::load_neon(s0, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        Goldilocks::load_neon(s1, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 2]));
        Goldilocks::load_neon(s2, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 4]));
        Goldilocks::load_neon(s3, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 6]));
        Goldilocks::load_neon(s4, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 8]));
        Goldilocks::load_neon(s5, &(PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 10]));
        Goldilocks::mult_neon(w0, scalar1, s0);
        Goldilocks::mult_neon(w1, scalar1, s1);
        Goldilocks::mult_neon(w2, scalar1, s2);
        Goldilocks::mult_neon(w3, scalar1, s3);
        Goldilocks::mult_neon(w4, scalar1, s4);
        Goldilocks::mult_neon(w5, scalar1, s5);
        Goldilocks::add_neon(st0, st0, w0);
        Goldilocks::add_neon(st1, st1, w1);
        Goldilocks::add_neon(st2, st2, w2);
        Goldilocks::add_neon(st3, st3, w3);
        Goldilocks::add_neon(st4, st4, w4);
        Goldilocks::add_neon(st5, st5, w5);
        state0 = state0 + PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
    }
    Goldilocks::store_neon(&(state[0]), st0);
    state[0] = state0_;
    Goldilocks::load_neon(st0, &(state[0]));

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_neon(st0, st1, st2, st3, st4, st5);
        add_neon_small(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        Goldilocks::mmult_neon_8(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::M_[0]));
    }
    pow7_neon(st0, st1, st2, st3, st4, st5);
    Goldilocks::mmult_neon_8(st0, st1, st2, st3, st4, st5, &(PoseidonGoldilocksConstants::M_[0]));

    Goldilocks::store_neon(&(state[0]), st0);
    Goldilocks::store_neon(&(state[2]), st1);
    Goldilocks::store_neon(&(state[4]), st2);
    Goldilocks::store_neon(&(state[6]), st3);
    Goldilocks::store_neon(&(state[8]), st4);
    Goldilocks::store_neon(&(state[10]), st5);
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
void PoseidonGoldilocks::merkletree_neon(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    // memset(cursor, 0, num_rows * CAPACITY * sizeof(Goldilocks::Element));
#ifndef __NO_OMP__
    if (nThreads == 0)
        nThreads = omp_get_max_threads();
#endif

#pragma omp parallel for num_threads(nThreads)
    for (uint64_t i = 0; i < num_rows; i++)
    {
        linear_hash(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;
    while (pending > 1)
    {
#pragma omp parallel for num_threads(nThreads)
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
void PoseidonGoldilocks::merkletree_batch_neon(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t batch_size, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    uint64_t nbatches = 1;
    if (num_cols > 0)
    {
        nbatches = (num_cols + batch_size - 1) / batch_size;
    }
    uint64_t nlastb = num_cols - (nbatches - 1) * batch_size;

#ifndef __NO_OMP__
    if (nThreads == 0)
        nThreads = omp_get_max_threads();
#endif

#pragma omp parallel for num_threads(nThreads)
    for (uint64_t i = 0; i < num_rows; i++)
    {
        Goldilocks::Element buff0[nbatches * CAPACITY];
        for (uint64_t j = 0; j < nbatches; j++)
        {
            uint64_t nn = batch_size;
            if (j == nbatches - 1)
                nn = nlastb;
            linear_hash(&buff0[j * CAPACITY], &input[i * num_cols * dim + j * batch_size * dim], nn * dim);
        }
        linear_hash(&cursor[i * CAPACITY], buff0, nbatches * CAPACITY);
    }

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
#pragma omp parallel for num_threads(nThreads)
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
#endif  //  __USE_NEON__