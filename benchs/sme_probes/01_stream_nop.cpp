// Probe 1: enter streaming mode, execute a single NOP, exit.
// Passes iff macOS allows a userspace process to enter streaming-SVE mode at all.
//
// Build: clang++ -std=c++17 -O0 -march=armv9-a+sme 01_stream_nop.cpp -o 01_stream_nop
// Run:   ./01_stream_nop        or       sudo ./01_stream_nop
// Expected (SME blocked):  SIGILL, exit 132
// Expected (SME allowed):  "streaming nop: ok", exit 0
#include <cstdio>

__arm_locally_streaming
static int streaming_nop() {
    __asm__ volatile("nop" ::: "memory");
    return 1;
}

int main() {
    printf("about to enter streaming mode...\n");
    fflush(stdout);
    int r = streaming_nop();
    printf("streaming nop: ok (r=%d)\n", r);
    return 0;
}
