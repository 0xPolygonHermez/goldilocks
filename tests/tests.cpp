// main.cpp

#include <gtest/gtest.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"

/*
TEST(GOLDILOCKS_TEST, gl_add)
{
    ASSERT_EQ(Goldilocks::gl_add(0, 0), 0);
    ASSERT_EQ(Goldilocks::gl_add(1, 1), 2);
    ASSERT_EQ(Goldilocks::gl_add(0, 1), 1);
    ASSERT_EQ(Goldilocks::gl_add(0xFFFFFFFFFFFFFFFFULL, 1), 0xFFFFFFFF); // 18446744073709551616 % 18446744069414584321
    ASSERT_EQ(Goldilocks::gl_add(Goldilocks::P, Goldilocks::P), Goldilocks::P);
    ASSERT_EQ(Goldilocks::gl_add(0xFFFFFFFF88265000ULL, 0xFFFFFFFF0F903000ULL), 0xFFFFFFFF97B67FFF);
}

TEST(GOLDILOCKS_TEST, add_gl)
{
    ASSERT_EQ(Goldilocks::add_gl(0, 0), 0);
    ASSERT_EQ(Goldilocks::add_gl(1, 1), 2);
    ASSERT_EQ(Goldilocks::add_gl(0, 1), 1);
    ASSERT_EQ(Goldilocks::add_gl(0xFFFFFFFFFFFFFFFFULL, 1), 0xFFFFFFFF); // 18446744073709551616 % 18446744069414584321
    ASSERT_EQ(Goldilocks::add_gl(Goldilocks::P, Goldilocks::P), Goldilocks::P);
    ASSERT_EQ(Goldilocks::add_gl(0xFFFFFFFF88265000ULL, 0xFFFFFFFF0F903000ULL), 0xFFFFFFFF97B67FFF);
}

TEST(GOLDILOCKS_TEST, gl_add_2)
{
    ASSERT_EQ(Goldilocks::gl_add_2(0, 0), 0);
    ASSERT_EQ(Goldilocks::gl_add_2(1, 1), 2);
    ASSERT_EQ(Goldilocks::gl_add_2(0, 1), 1);
    ASSERT_EQ(Goldilocks::gl_add_2(0xFFFFFFFFFFFFFFFFULL, 1), 0xFFFFFFFF); // 18446744073709551616 % 18446744069414584321
    ASSERT_EQ(Goldilocks::gl_add_2(Goldilocks::P, Goldilocks::P), Goldilocks::P);
    ASSERT_EQ(Goldilocks::gl_add_2(0xFFFFFFFF88265000ULL, 0xFFFFFFFF0F903000ULL), 0xFFFFFFFF97B67FFF);
}
*/
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

// Build command: g++ main.cpp -lgtest