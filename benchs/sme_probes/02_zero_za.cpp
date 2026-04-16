// Probe 2: enter streaming mode and execute svzero_za.
// Passes iff the kernel allows access to the ZA tile register file.
//
// Build: clang++ -std=c++17 -O0 -march=armv9-a+sme 02_zero_za.cpp -o 02_zero_za
// Run:   ./02_zero_za        or       sudo ./02_zero_za
#include <cstdio>
#include <arm_sme.h>

__arm_locally_streaming __arm_new("za")
static void test_zero_za() {
    svzero_za();
    __asm__ volatile("" ::: "memory");
}

int main() {
    printf("about to svzero_za()...\n");
    fflush(stdout);
    test_zero_za();
    printf("svzero_za: ok\n");
    return 0;
}
