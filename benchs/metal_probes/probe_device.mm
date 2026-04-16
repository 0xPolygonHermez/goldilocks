// Probe 1: obtain the system-default Metal device and print its name.
// Passes iff a Metal-capable GPU is present and MTLCreateSystemDefaultDevice() returns non-nil.
//
// Build: clang++ -std=c++17 -O0 -fobjc-arc -ObjC++ -framework Metal -framework Foundation \
//               probe_device.mm -o probe_device
// Run:   ./probe_device
// Expected (Metal available):  "Metal device: Apple M<n> GPU", exit 0
// Expected (no Metal device):  "Metal device: nil - no Metal-capable GPU found", exit 1
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        printf("Metal device: nil - no Metal-capable GPU found\n");
        return 1;
    }
    printf("Metal device: %s\n", [[device name] UTF8String]);
    printf("probe_device: ok\n");
    return 0;
}
