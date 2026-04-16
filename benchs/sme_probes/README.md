# Apple SME userspace probes

Three staged probes to determine whether SME is accessible from a userspace
process on your macOS + M4-class hardware. Current (macOS 26.3.1, M4 Pro)
result without sudo: **all three SIGILL**. Test with sudo:

```bash
cd benchs/sme_probes
make             # builds all 3 probes
make run         # runs them without sudo (baseline — expected SIGILL)
make run-sudo    # runs them WITH sudo (this is what you want to try)
```

## What each probe tests

| File | What it does | Fails with |
|------|-------------|-----------|
| `01_stream_nop.cpp` | Enters streaming mode via `__arm_locally_streaming`, executes a single `nop`, exits. | SIGILL = macOS blocks streaming-mode entry |
| `02_zero_za.cpp` | Same as 1, plus `svzero_za()` (clear the SME ZA tile). | SIGILL = ZA tile unavailable |
| `03_i16_i64_matmul.cpp` | Full real op: `svmopa_za64_s16_m` in a timing loop. Reports ns/op. | SIGILL = SME ops blocked; success with timing = we can use it |

## Interpreting results

### If probe 1 passes but 2 fails
Streaming SVE instructions work, but the kernel isn't allocating the ZA tile
for user processes. SME is partially usable: fine for SSVE (vector) ops, but
no matrix accumulator. Goldilocks optimization still blocked because we need
ZA for matrix outer products.

### If probes 1 & 2 pass but 3 hangs / times out
Something about the specific i16→i64 variant. Likely fine for other variants
(i8→i32, f32→f32).

### If all three pass with good timing
We can proceed with the SME-based `mvp_neon` port. Ballpark target: each
`svmopa_za64_s16_m` executes one 32×32 i16 outer product in a few cycles,
which should give us a meaningful speedup over the current NEON `mul_small`
(which does 12 i64 mul+reduce per row, 144 per mvp call).

### If sudo still SIGILLs
macOS isn't gating SME behind a permission — it's gating behind an
undocumented entitlement or boot-time configuration. A further check would
be to:
- Look for a sysctl flag that enables SME.
- Check whether SME is restricted to performance cores (disable e-cores via
  QoS class).
- Try on a newer macOS beta.

## Why we care

Goldilocks Poseidon's MDS matrix mul is ~50% of hash runtime in our current
Phase 1l NEON implementation. SME `FEAT_SME_I16I64` executes 256 i16×i16
products per instruction with 64-bit accumulation — on paper that could
replace our current 144 scalar `mul_small` instructions per mvp.

If SME becomes usable, estimated end-to-end gain: ~1.2-1.5× on top of current
NEON. See `../apple-matrix-extensions-negative-result.md` for the full
feasibility analysis.
