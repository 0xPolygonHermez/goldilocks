# Apple Matrix Extensions (AMX / SME) — Negative Result

**Date:** 2026-04-15
**System:** Apple M4 Pro, macOS 26.3.1 (build 25D2128), Apple clang 17

## Question

Can we use Apple's matrix coprocessor (AMX on M1-M3, SME on M4+) to accelerate
the Goldilocks Poseidon MDS matrix-vector product? The MDS uses small (≤41)
matrix entries and 64-bit state elements, which would fit the `FEAT_SME_I16I64`
instruction (i16×i16→i64 outer product) available on M4.

## Probes

### 1. Hardware capability check

```
$ sysctl hw.optional | grep -iE "sme|sve|i8mm|amx"
hw.optional.arm.FEAT_I8MM: 1
hw.optional.arm.FEAT_SME: 1
hw.optional.arm.FEAT_SME2: 1
hw.optional.arm.SME_I16I32: 1
hw.optional.arm.FEAT_SME_I16I64: 1     ← this is what we want
hw.optional.arm.sme_max_svl_b: 64      ← 512-bit streaming vector
```

The **hardware exposes** SME with i16→i64 integer outer products at 512-bit
vector length (8×8 i64 ZA tile, 16×16 i32 tile, or 32×32 i16 tile). Ideal for
Goldilocks.

### 2. Compiler support

```
clang++ -march=armv9-a+sme+sme-i16i64 -mcpu=apple-m4 ... ✓
__ARM_FEATURE_SME: defined
```

Apple clang 17 supports SME intrinsics (`arm_sme.h`) and the streaming function
attribute (`__arm_locally_streaming`, `__arm_new("za")`).

### 3. Runtime accessibility

```cpp
__arm_locally_streaming
static int stream_test() {
    __asm__ volatile ("nop");   // force real streaming-mode entry
    return 1;
}
int main() { stream_test(); }
```

Result: **SIGILL (exit 132)** when the streaming function actually executes.
Even `svzero_za()` — the most trivial SME op — crashes. Same for the
reverse-engineered AMX opcodes.

Apple clearly intends the hardware to be reserved for internal use at present:
- SME instructions trap from userspace.
- AMX is removed on M4 Pro (M1-M3 only).
- Apple's Accelerate framework (BLAS/LAPACK) is the only way to leverage the
  matrix coprocessor. But BLAS has no 64-bit-integer matmul routine.

### 4. Via Apple Accelerate

`cblas_sgemm` works (confirms the underlying hardware is used internally by
the kernel/framework boundary). But no `cblas_lgemm` for i64 × i64 exists.
Accelerate's `BNNS` ML primitives are i8/i16 quantized and can't represent
Goldilocks elements.

### 5. Sudo / root — still blocked

```
$ sudo ./01_stream_nop
about to enter streaming mode...
zsh: illegal hardware instruction  sudo ./01_stream_nop

$ sudo ./02_zero_za
about to svzero_za()...
zsh: illegal hardware instruction  sudo ./02_zero_za

$ sudo ./03_i16_i64_matmul
probe 3: SME i16×i16→i64 outer-product timing
zsh: illegal hardware instruction  sudo ./03_i16_i64_matmul
```

Running as root (uid 0) does **not** unlock SME. The block is enforced at
the kernel/entitlement layer, not by process privileges. This matches
Apple's pattern for reserving new ARM features — for example, PAC and BTI
were similarly gated behind code-signing entitlements during their
introduction.

Possible paths (none immediately actionable):

1. **Entitlement discovery**: find the plist key Apple uses internally. No
   public documentation as of macOS 26.3.1.
2. **Beta macOS**: maybe macOS 27 exposes SME. Untested.
3. **Asahi Linux on M4**: Linux exposes SME via `prctl(PR_SME_SET_VL)`;
   running the workload on Asahi would work, but breaks our "macOS
   primary" target.

## Conclusion

**Apple matrix extensions cannot be used for Goldilocks NEON Poseidon on macOS
26.3.1 / M4 Pro** because userspace direct access is blocked. This may change
in a future macOS release, but as of this date:

- Direct SME instructions: **SIGILL in userspace**.
- Direct AMX instructions: **SIGILL on M4**.
- Accelerate BLAS: **no 64-bit integer matmul routine**.

## Implications

The Phase 1l result (M4 Pro NEON at 1.36-1.65× Xeon D-2141I AVX2) is the
current ceiling for a portable ARM64 implementation of Goldilocks Poseidon
on macOS. Further gains would require either:

1. **Apple exposing SME to userspace** in a future macOS release. Then the
   `mul_small` inside `mvp_neon` could be replaced with an i16-limb
   decomposition + SME outer product + reassembly. Estimated upside: 1.5-2×
   on `mvp_neon`, which is ~50% of hash runtime. End-to-end: ~1.25-1.5×.
2. **Algorithmic change**: Poseidon2 or smaller-round-count variants.
3. **Metal GPU**: orders of magnitude for large merkle trees, out of scope
   here.

## References

- ARM SME architecture reference:
  https://developer.arm.com/documentation/ddi0602/2024-03/SME-Instructions
- corsix/amx (reverse-engineered M1-M3 AMX):
  https://github.com/corsix/amx
- Apple Accelerate framework:
  https://developer.apple.com/documentation/accelerate

## No commit needed

No source changes result from this probe. Test files left in `/tmp/`; no
repository artifacts were modified.
