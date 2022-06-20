#include <gtest/gtest.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"

#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

typedef Goldilocks::Element Element;

TEST(GOLDILOCKS_TEST, one)
{
    uint64_t a = 1;
    uint64_t b = 1 + GOLDILOCKS_PRIME;
    std::string c = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1

    Element ina1 = Goldilocks::fromU64(a);
    Element ina2 = Goldilocks::fromS32(a);
    Element ina3 = Goldilocks::fromString(std::to_string(a));
    Element inb1 = Goldilocks::fromU64(b);
    Element inc1 = Goldilocks::fromString(c);

    ASSERT_EQ(Goldilocks::toU64(ina1), a);
    ASSERT_EQ(Goldilocks::toU64(ina2), a);
    ASSERT_EQ(Goldilocks::toU64(ina3), a);
    ASSERT_EQ(Goldilocks::toU64(inb1), a);
    ASSERT_EQ(Goldilocks::toU64(inc1), a);

    Element a1 = Goldilocks::fromU64(Goldilocks::from_montgomery(0xFFFFFFFF00000000));
    Element a2 = Goldilocks::fromU64(Goldilocks::from_montgomery(0xFFFFFFFF));
    Element b1 = (a1 + a2);
    Element b2 = (b1 + b1);
    std::cout << Goldilocks::toString(b1, 16) << std::endl;
    std::cout << Goldilocks::toString(b2, 16) << std::endl;

    ASSERT_EQ(Goldilocks::toU64(b2), 0x200000002);
}

TEST(GOLDILOCKS_TEST, add)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Element inE1 = Goldilocks::fromU64(in1);
    Element inE2 = Goldilocks::fromS32(in2);
    Element inE3 = Goldilocks::fromString(in3);
    Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2), in1 + in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2 + inE3), in1 + in2 + 1);
    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2 + inE3 + inE4), 1);
}

TEST(GOLDILOCKS_TEST, sub)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Element inE1 = Goldilocks::fromU64(in1);
    Element inE2 = Goldilocks::fromS32(in2);
    Element inE3 = Goldilocks::fromString(in3);
    Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2), GOLDILOCKS_PRIME + in1 - in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2 - inE3), GOLDILOCKS_PRIME + in1 - in2 - 1);
    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2 - inE3 - inE4), 5);
}

TEST(GOLDILOCKS_TEST, mul)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Element inE1 = Goldilocks::fromU64(in1);
    Element inE2 = Goldilocks::fromS32(in2);
    Element inE3 = Goldilocks::fromString(in3);
    Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2), in1 * in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2 * inE3), in1 * in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2 * inE3 * inE4), 0XFFFFFFFEFFFFFEBDLL);
}

TEST(GOLDILOCKS_TEST, div)
{
    uint64_t in1 = 10;
    int32_t in2 = 5;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;
    int32_t in5 = 3;
    int32_t in6 = 2;

    Element inE1 = Goldilocks::fromU64(in1);
    Element inE2 = Goldilocks::fromS32(in2);
    Element inE3 = Goldilocks::fromString(in3);
    Element inE4 = Goldilocks::fromS32(in4);
    Element inE5 = Goldilocks::fromS32(in4);
    Element inE6 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2), in1 / in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2 / inE3), in1 / in2); // 10 / 2 / ( 0 + 1 ) = 10 / 2
    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2 / inE3 / inE4), 0X2AAAAAAA80000000);
    ASSERT_EQ(Goldilocks::toU64(inE5 / inE6), 1);
    ASSERT_EQ(Goldilocks::toU64(Goldilocks::one() / inE6), 0X1555555540000000);
}

TEST(GOLDILOCKS_TEST, inv)
{
    uint64_t in1 = 5;
    std::string in2 = "18446744069414584326"; // 0xFFFFFFFF00000001n + 5n

    Element input1 = Goldilocks::one();
    Element inv1 = Goldilocks::inv(input1);
    Element res1 = input1 * inv1;

    Element input5 = Goldilocks::fromU64(in1);
    Element inv5 = Goldilocks::inv(input5);
    Element res5 = input5 * inv5;

    ASSERT_EQ(res1, Goldilocks::one());
    ASSERT_EQ(res5, Goldilocks::one());

    Element inE1 = Goldilocks::fromString(std::to_string(in1));
    Element inE1_plus_p = Goldilocks::fromString(in2);

    ASSERT_EQ(Goldilocks::inv(inE1_plus_p) * inE1, Goldilocks::one());
    ASSERT_EQ(Goldilocks::inv(inE1), Goldilocks::inv(inE1_plus_p));
}

TEST(GOLDILOCKS_TEST, poseidon_full)
{

    Goldilocks::Element fibonacci[SPONGE_WIDTH];
    Goldilocks::Element result[SPONGE_WIDTH];

    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();

    for (uint64_t i = 2; i < SPONGE_WIDTH; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }

    PoseidonGoldilocks::hash_full_result(result, fibonacci);

    ASSERT_EQ(Goldilocks::toU64(result[0]), 0X3095570037F4605D);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0X3D561B5EF1BC8B58);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0X8129DB5EC75C3226);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0X8EC2B67AFB6B87ED);
    ASSERT_EQ(Goldilocks::toU64(result[4]), 0XFC591F17D0FAB161);
    ASSERT_EQ(Goldilocks::toU64(result[5]), 0X1D2B045CC2FEA1AD);
    ASSERT_EQ(Goldilocks::toU64(result[6]), 0X8A4E3B0CB12D4527);
    ASSERT_EQ(Goldilocks::toU64(result[7]), 0XFF217A756AE2211);
    ASSERT_EQ(Goldilocks::toU64(result[8]), 0X78F6E79CFC407293);
    ASSERT_EQ(Goldilocks::toU64(result[9]), 0X3DE827E086AE61C9);
    ASSERT_EQ(Goldilocks::toU64(result[10]), 0X921456F6D2D11E27);
    ASSERT_EQ(Goldilocks::toU64(result[11]), 0XF58A41D4028C66A5);

    Goldilocks::Element zero[SPONGE_WIDTH] = {Goldilocks::zero()};
    Goldilocks::Element result0[SPONGE_WIDTH];

    PoseidonGoldilocks::hash_full_result(result0, zero);

    ASSERT_EQ(Goldilocks::toU64(result0[0]), 0X3C18A9786CB0B359);
    ASSERT_EQ(Goldilocks::toU64(result0[1]), 0XC4055E3364A246C3);
    ASSERT_EQ(Goldilocks::toU64(result0[2]), 0X7953DB0AB48808F4);
    ASSERT_EQ(Goldilocks::toU64(result0[3]), 0XC71603F33A1144CA);
    ASSERT_EQ(Goldilocks::toU64(result0[4]), 0XD7709673896996DC);
    ASSERT_EQ(Goldilocks::toU64(result0[5]), 0X46A84E87642F44ED);
    ASSERT_EQ(Goldilocks::toU64(result0[6]), 0XD032648251EE0B3C);
    ASSERT_EQ(Goldilocks::toU64(result0[7]), 0X1C687363B207DF62);
    ASSERT_EQ(Goldilocks::toU64(result0[8]), 0XDF8565563E8045FE);
    ASSERT_EQ(Goldilocks::toU64(result0[9]), 0X40F5B37FF4254DAE);
    ASSERT_EQ(Goldilocks::toU64(result0[10]), 0XD070F637B431067C);
    ASSERT_EQ(Goldilocks::toU64(result0[11]), 0X1792B1C4342109D7);
}

TEST(GOLDILOCKS_TEST, poseidon)
{

    Goldilocks::Element fibonacci[SPONGE_WIDTH];
    Goldilocks::Element result[CAPACITY];

    fibonacci[0] = Goldilocks::zero();
    fibonacci[1] = Goldilocks::one();

    for (uint64_t i = 2; i < SPONGE_WIDTH; i++)
    {
        fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
    }

    PoseidonGoldilocks::hash(result, fibonacci);

    ASSERT_EQ(Goldilocks::toU64(result[0]), 0X3095570037F4605D);
    ASSERT_EQ(Goldilocks::toU64(result[1]), 0X3D561B5EF1BC8B58);
    ASSERT_EQ(Goldilocks::toU64(result[2]), 0X8129DB5EC75C3226);
    ASSERT_EQ(Goldilocks::toU64(result[3]), 0X8EC2B67AFB6B87ED);

    Goldilocks::Element zero[SPONGE_WIDTH] = {Goldilocks::zero()};
    Goldilocks::Element result0[CAPACITY];

    PoseidonGoldilocks::hash(result0, zero);

    ASSERT_EQ(Goldilocks::toU64(result0[0]), 0X3C18A9786CB0B359);
    ASSERT_EQ(Goldilocks::toU64(result0[1]), 0XC4055E3364A246C3);
    ASSERT_EQ(Goldilocks::toU64(result0[2]), 0X7953DB0AB48808F4);
    ASSERT_EQ(Goldilocks::toU64(result0[3]), 0XC71603F33A1144CA);
}
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

// Build command: g++ tests/tests.cpp src/goldilocks_base_field.cpp -lgtest -lgmp -o test && ./test