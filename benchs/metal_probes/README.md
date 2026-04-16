# metal_probes — Apple Metal GPU sandbox

Standalone probes for the Goldilocks Metal GPU backend. Each probe is a bare
`main()` that exercises one capability and exits 0 on success or non-zero /
crashes on failure. No gtest, no `src/` headers.

## Convention

| Probe file          | Probe | What it tests                          |
|---------------------|-------|----------------------------------------|
| `probe_device.mm`   | 1     | `MTLCreateSystemDefaultDevice()` works |
| `probe_field.mm`    | 2     | Field arithmetic kernel (Part 4)       |
| `probe_poseidon.mm` | 3     | Poseidon hash kernel (Part 4)          |
| `probe_merkle.mm`   | 4     | Merkle oracle end-to-end (Part 5)      |

Source files use `.mm` (Obj-C++) so that Metal/Foundation headers can be
imported with `#import`. Compiled with `-fobjc-arc -ObjC++`.

## Build & run

```sh
cd benchs/metal_probes
make          # build all probes
make run      # build + run all probes
make run-sudo # same, under sudo (rarely needed for Metal)
make clean    # remove binaries
```

## Expected output (Apple Silicon Mac with Metal)

```
=== probe 1: Metal device ===
Metal device: Apple M<n> GPU
probe_device: ok
exit=0
```

## Adding a new probe (for future parts)

1. Create `probe_<name>.mm` in this directory.
2. Add a build rule to `Makefile` following the existing pattern.
3. Add the binary to the `all` target and both `run` / `run-sudo` blocks.
4. If the probe needs a `.metal` shader: compile with
   `xcrun metal -c foo.metal -o foo.air && xcrun metallib foo.air -o foo.metallib`,
   then embed or load at runtime.
