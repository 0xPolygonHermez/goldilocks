// main.cpp

#include <gtest/gtest.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"

typedef Goldilocks::Element Element;

TEST(GOLDILOCKS_TEST, add)
{
    int32_t a = -55;
    Element e1 = Goldilocks::fromS32(a);

    Element e2 = Goldilocks::fromU64(55);
    Element e4 = Goldilocks::fromU64(55);

    ASSERT_EQ(Goldilocks::toU64(e4), 55);

    uint64_t res_1 = Goldilocks::toU64(e1 + e2);
    uint64_t res_2 = Goldilocks::toU64(e1 - e2);
    uint64_t res_3 = Goldilocks::toU64(e1 * e2);
    uint64_t res_4 = Goldilocks::toU64(e1 / e2);
    Element as = (e1 + e2);

    Element c = Goldilocks::fromString("-627710173100217585286392776928019145829365870197997568000");

    std::cout << Goldilocks::toString(e1) << " " << a << " " << Goldilocks::toString(c) << "\n";

    ASSERT_EQ(res_1, res_1);
    ASSERT_EQ(as, Goldilocks::zero());

    ASSERT_EQ(res_1, 0);
    ASSERT_EQ(res_2, 0);
    ASSERT_EQ(res_3, 25);
    ASSERT_EQ(res_4, 1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

// Build command: g++ tests/tests.cpp src/goldilocks_base_field.cpp -lgtest -o test && ./test