#include <iostream>

#include "goldilocks_base_field.hpp"

int main(int argc, char **argv)
{

    Goldilocks::Element a = Goldilocks::fromU64(0xFFFFFFFF00000005ULL);
    uint64_t b = Goldilocks::toU64(a);
    Goldilocks::Element c = Goldilocks::fromString("6277101731002175852863927769280199145829365870197997568000");

    std::cout << Goldilocks::toString(a) << " " << b << " " << Goldilocks::toString(c) << "\n";

    Goldilocks::Element input1 = Goldilocks::one();
    Goldilocks::Element inv1 = Goldilocks::inv(input1);
    Goldilocks::Element res = input1 * inv1;

    return 0;
}