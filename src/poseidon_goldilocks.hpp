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
    inline void static pow7(Goldilocks::Element &x);
    inline void static pow7_(Goldilocks::Element *x);
    inline void static pow7_avx(__m256i &st0, __m256i &st1, __m256i &st2);
    inline void static add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static add_avx(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static mvp_(Goldilocks::Element *state, const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH]);
    inline Goldilocks::Element static dot_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static prod_(Goldilocks::Element *x, const Goldilocks::Element alpha, const Goldilocks::Element C[SPONGE_WIDTH]);

public:
    void static hash_full_result(Goldilocks::Element (&state)[SPONGE_WIDTH], Goldilocks::Element const (&input)[SPONGE_WIDTH]);
    void static hash_full_result_(Goldilocks::Element *, const Goldilocks::Element *);
    void static hash_full_result_avx(Goldilocks::Element *, const Goldilocks::Element *);
    void static hash(Goldilocks::Element (&state)[CAPACITY], const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    void static linear_hash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    void static merkletree(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t dim = 1);
};

#include "poseidon_goldilocks_avx.hpp"
#endif