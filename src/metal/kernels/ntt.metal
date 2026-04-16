// ntt.metal — NTT kernels for Goldilocks field
//
// Implements:
//   kernel ntt_reverse_permutation  — bit-reverse permutation (in-place)
//   kernel ntt_butterfly_phase      — one Cooley-Tukey phase (in-place)
//   kernel intt_scale               — multiply each element by 1/N (inverse NTT scaling)
//
// Buffer layout: row-major, stride = ncols.
//   element[row][col] = buf[row * ncols + col]
//
// CPU reference: BR() at ntt_goldilocks.cpp:6-13
//   Butterfly math: t = w * a[off1]; a[off2] = u + t; a[off1] = u - t
//   where u = a[off2].

#include <metal_stdlib>
#include "field.metal"
using namespace metal;

// reverse_bits32: full 32-bit bit-reversal of x
// Then shift right by (32 - domainPow) to get the log2(N)-bit reversal.
// Matches BR() in ntt_goldilocks.cpp:6-13 exactly.
inline uint reverse_bits32(uint x) {
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00u) >> 8)  | ((x & 0x00FF00FFu) << 8);
    x = ((x & 0xF0F0F0F0u) >> 4)  | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x & 0xCCCCCCCCu) >> 2)  | ((x & 0x33333333u) << 2);
    x = ((x & 0xAAAAAAAAu) >> 1)  | ((x & 0x55555555u) << 1);
    return x;
}

// ---- kernel: ntt_reverse_permutation ---------------------------------------
// One thread per (element-pair, col) — but for simplicity, dispatch one thread
// per flat index [0, domain_size * ncols) and let threads with src < dst swap.
//
// tid = thread index in [0, domain_size)  (one per row, all cols handled)
// domainPow = log2(domain_size)
kernel void ntt_reverse_permutation(
    device ulong*   buf       [[ buffer(0) ]],
    constant uint&  domainPow [[ buffer(1) ]],
    constant uint&  ncols     [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint domain_size = 1u << domainPow;
    if (tid >= domain_size) return;

    uint rev = reverse_bits32(tid) >> (32 - domainPow);
    if (rev <= tid) return;  // only process each pair once

    uint src = tid  * ncols;
    uint dst = rev  * ncols;
    for (uint c = 0; c < ncols; c++) {
        ulong tmp     = buf[src + c];
        buf[src + c]  = buf[dst + c];
        buf[dst + c]  = tmp;
    }
}

// ---- kernel: ntt_butterfly_phase -------------------------------------------
// One thread per (butterfly pair index, col).
// Implements one level s of the DIT Cooley-Tukey NTT.
//
// Parameters:
//   buf         = data buffer [domain_size * ncols]
//   twiddles    = precomputed twiddle factors, twiddles[j] for j in [0, mdiv2)
//   ncols       = number of columns (stride)
//   domain_size = 2^domainPow
//   s           = current NTT level in [1, domainPow]
//
// Thread grid: dispatch (domain_size/2 * ncols) threads.
// tid maps to: pair_idx = tid / ncols, col = tid % ncols
//
// Butterfly:
//   m     = 2^s;  mdiv2 = m/2
//   Group g = pair_idx / mdiv2
//   Intra-group offset j = pair_idx % mdiv2
//   off2 = g*m + j          (upper element in butterfly)
//   off1 = g*m + j + mdiv2  (lower element)
//   w  = twiddles[j]
//   u  = buf[off2 * ncols + col]
//   t  = gl_mul(w, buf[off1 * ncols + col])
//   buf[off2 * ncols + col] = gl_add(u, t)   -- no lazy needed, gl_add canonicalizes
//   buf[off1 * ncols + col] = gl_sub(u, t)
// The kernel reads from a single, instance-wide twiddles buffer that holds
// the full roots[0 .. 2^s_global) array. For phase s, the twiddle at logical
// index j is roots[j << (s_global - s)] (matches CPU root(s, j) at
// ntt_goldilocks.hpp:169-172). `roots_stride_shift` is the per-call
// stride = s_global - s. This removes the per-phase twiddle buffer
// allocation + upload that dominated small-N and multi-column runs.
kernel void ntt_butterfly_phase(
    device ulong*         buf                  [[ buffer(0) ]],
    device const ulong*   twiddles             [[ buffer(1) ]],
    constant uint&        ncols                [[ buffer(2) ]],
    constant uint&        domain_size          [[ buffer(3) ]],
    constant uint&        s                    [[ buffer(4) ]],
    constant uint&        roots_stride_shift   [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint half_n = domain_size >> 1;
    uint total = half_n * ncols;
    if (tid >= total) return;

    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;

    uint m     = 1u << s;
    uint mdiv2 = m >> 1;
    uint g     = pair_idx / mdiv2;
    uint j     = pair_idx % mdiv2;

    uint off2 = (g * m + j) * ncols + col;
    uint off1 = (g * m + j + mdiv2) * ncols + col;

    ulong w = twiddles[j << roots_stride_shift];
    ulong u = buf[off2];
    ulong t = gl_mul(w, buf[off1]);

    buf[off2] = gl_add(u, t);
    buf[off1] = gl_sub(u, t);
}

// ---- kernel: ntt_rev_butterfly_s1 -----------------------------------------
// Fused kernel: reads input in natural order, writes bit-reverse-permuted
// AND phase-1 butterfly output. Replaces two separate passes
// (ntt_reverse_permutation + ntt_butterfly_phase with s=1) with a single
// pass, saving one full N-element read+write cycle plus one commit.
//
// Why this works: at phase s=1, mdiv2=1, so each butterfly pair uses
// twiddle index j=0, and root(1, 0) = roots[0] = 1 — the twiddle
// multiplication is a no-op. The s=1 butterfly reduces to just
// gl_add/gl_sub of two bit-reversed-position inputs.
//
// Requires src ≠ dst (out-of-place). Caller guarantees via
// the usual memcpy(dst, src) path, but reads from src directly here
// to save one memcpy pass.
//
// Dispatch: (domain_size / 2) × ncols threads. Thread tid handles one
// (butterfly-pair, column) pair.
kernel void ntt_rev_butterfly_s1(
    device const ulong* src        [[ buffer(0) ]],  // input (natural order)
    device       ulong* dst        [[ buffer(1) ]],  // output (perm + s=1 butterfly)
    constant     uint&  domain_pow [[ buffer(2) ]],
    constant     uint&  ncols      [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint half_n = (1u << domain_pow) >> 1;
    uint total  = half_n * ncols;
    if (tid >= total) return;

    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;

    uint shift = 32u - domain_pow;
    uint src_a = reverse_bits32(pair_idx * 2u)      >> shift;
    uint src_b = reverse_bits32(pair_idx * 2u + 1u) >> shift;

    ulong u = src[src_a * ncols + col];
    ulong t = src[src_b * ncols + col];

    dst[(pair_idx * 2u)      * ncols + col] = gl_add(u, t);
    dst[(pair_idx * 2u + 1u) * ncols + col] = gl_sub(u, t);
}

// ---- kernel: ntt_rev_butterfly_s1s2 ---------------------------------------
// Fused even further: reads input in natural order, writes
// (reverse permutation ∘ s=1 butterfly ∘ s=2 butterfly) in ONE pass. This is
// an algorithmic equivalent to running ntt_reverse_permutation +
// ntt_butterfly_phase(s=1) + ntt_butterfly_phase(s=2) — three passes collapsed
// into one, saving two full read+write cycles.
//
// At s=1, twiddles are always 1 (ω_2^0). At s=2, the two butterflies per
// group-of-4 use twiddles ω_4^0 = 1 and ω_4^1 = I (primitive 4th root of
// unity). So the only twiddle multiplication needed in the combined kernel
// is a single multiply-by-I per quad-butterfly.
//
// Requires src ≠ dst and domain_pow ≥ 2. Caller falls back to the
// s=1-only fused kernel for domain_pow == 1 or to the legacy path for
// in-place (src == dst).
//
// Dispatch: (domain_size / 4) × ncols threads. Thread tid handles one
// (quad-butterfly, column) pair.
kernel void ntt_rev_butterfly_s1s2(
    device const ulong* src        [[ buffer(0) ]],
    device       ulong* dst        [[ buffer(1) ]],
    constant     uint&  domain_pow [[ buffer(2) ]],
    constant     uint&  ncols      [[ buffer(3) ]],
    constant     ulong& I_val      [[ buffer(4) ]],  // = ω_4 (primitive 4th root of unity)
    uint tid [[ thread_position_in_grid ]]
) {
    uint quarter = (1u << domain_pow) >> 2;
    uint total   = quarter * ncols;
    if (tid >= total) return;

    uint group_idx = tid / ncols;
    uint col       = tid % ncols;

    // Group of 4 at natural positions (group_idx*4 + 0..3). Source positions
    // (before rev+butterflies) are the bit-reversed indices.
    uint shift = 32u - domain_pow;
    uint base_nat = group_idx * 4u;
    uint ra = reverse_bits32(base_nat + 0u) >> shift;
    uint rb = reverse_bits32(base_nat + 1u) >> shift;
    uint rc = reverse_bits32(base_nat + 2u) >> shift;
    uint rd = reverse_bits32(base_nat + 3u) >> shift;

    ulong x_a = src[ra * ncols + col];
    ulong x_b = src[rb * ncols + col];
    ulong x_c = src[rc * ncols + col];
    ulong x_d = src[rd * ncols + col];

    // Stage s=1 (within-pair): twiddle = 1, no mul.
    ulong y0 = gl_add(x_a, x_b);     // = rev(4g+0) + rev(4g+1)
    ulong y1 = gl_sub(x_a, x_b);
    ulong y2 = gl_add(x_c, x_d);
    ulong y3 = gl_sub(x_c, x_d);

    // Stage s=2: butterflies (y0, y2) with twiddle 1, (y1, y3) with twiddle I.
    ulong yI = gl_mul(y3, I_val);

    dst[(base_nat + 0u) * ncols + col] = gl_add(y0, y2);
    dst[(base_nat + 1u) * ncols + col] = gl_add(y1, yI);
    dst[(base_nat + 2u) * ncols + col] = gl_sub(y0, y2);
    dst[(base_nat + 3u) * ncols + col] = gl_sub(y1, yI);
}

// ---- kernel: ntt_rev_butterfly_s1s2s3 -------------------------------------
// Fuses (reverse permutation ∘ s=1 ∘ s=2 ∘ s=3) into ONE pass. Extension of
// ntt_rev_butterfly_s1s2 to 8-wide. Saves one additional full read+write
// pass versus s1s2 + the s=3 phase being handled by ntt_radix8_phase.
//
// Twiddles at the first three stages are all tiny fixed roots — no table
// lookup needed:
//   s=1: ω_2^0 = 1     (no mul)
//   s=2: ω_4^0 = 1,  ω_4^1 = I
//   s=3: ω_8^0 = 1,  ω_8^1 = W8,  ω_8^2 = I,  ω_8^3 = W8_cubed
//
// Requires src ≠ dst and domain_pow ≥ 3. Caller falls back to s1s2 for
// domain_pow == 2 and to s=1-only fused for domain_pow == 1.
//
// Dispatch: (domain_size / 8) × ncols threads.
kernel void ntt_rev_butterfly_s1s2s3(
    device const ulong* src          [[ buffer(0) ]],
    device       ulong* dst          [[ buffer(1) ]],
    constant     uint&  domain_pow   [[ buffer(2) ]],
    constant     uint&  ncols        [[ buffer(3) ]],
    constant     ulong& I_val        [[ buffer(4) ]],  // ω_4
    constant     ulong& W8_val       [[ buffer(5) ]],  // ω_8
    constant     ulong& W8c_val      [[ buffer(6) ]],  // ω_8^3
    uint tid [[ thread_position_in_grid ]]
) {
    uint eighth = (1u << domain_pow) >> 3u;
    uint total  = eighth * ncols;
    if (tid >= total) return;

    uint group_idx = tid / ncols;
    uint col       = tid % ncols;

    uint shift = 32u - domain_pow;
    uint base_nat = group_idx * 8u;

    uint ra = reverse_bits32(base_nat + 0u) >> shift;
    uint rb = reverse_bits32(base_nat + 1u) >> shift;
    uint rc = reverse_bits32(base_nat + 2u) >> shift;
    uint rd = reverse_bits32(base_nat + 3u) >> shift;
    uint re = reverse_bits32(base_nat + 4u) >> shift;
    uint rf = reverse_bits32(base_nat + 5u) >> shift;
    uint rg = reverse_bits32(base_nat + 6u) >> shift;
    uint rh = reverse_bits32(base_nat + 7u) >> shift;

    ulong x_a = src[ra * ncols + col];
    ulong x_b = src[rb * ncols + col];
    ulong x_c = src[rc * ncols + col];
    ulong x_d = src[rd * ncols + col];
    ulong x_e = src[re * ncols + col];
    ulong x_f = src[rf * ncols + col];
    ulong x_g = src[rg * ncols + col];
    ulong x_h = src[rh * ncols + col];

    // Stage s=1: 4 pair butterflies, twiddle = 1
    ulong y0 = gl_add(x_a, x_b);
    ulong y1 = gl_sub(x_a, x_b);
    ulong y2 = gl_add(x_c, x_d);
    ulong y3 = gl_sub(x_c, x_d);
    ulong y4 = gl_add(x_e, x_f);
    ulong y5 = gl_sub(x_e, x_f);
    ulong y6 = gl_add(x_g, x_h);
    ulong y7 = gl_sub(x_g, x_h);

    // Stage s=2: (y0,y2),(y1,y3),(y4,y6),(y5,y7). Twiddles {1, I, 1, I}.
    ulong y3I = gl_mul(y3, I_val);
    ulong y7I = gl_mul(y7, I_val);
    ulong z0 = gl_add(y0, y2);
    ulong z2 = gl_sub(y0, y2);
    ulong z1 = gl_add(y1, y3I);
    ulong z3 = gl_sub(y1, y3I);
    ulong z4 = gl_add(y4, y6);
    ulong z6 = gl_sub(y4, y6);
    ulong z5 = gl_add(y5, y7I);
    ulong z7 = gl_sub(y5, y7I);

    // Stage s=3: (z0,z4),(z1,z5),(z2,z6),(z3,z7).
    // Twiddles: ω_8^0=1, ω_8^1=W8, ω_8^2=I, ω_8^3=W8c.
    ulong z5w = gl_mul(z5, W8_val);
    ulong z6I = gl_mul(z6, I_val);
    ulong z7c = gl_mul(z7, W8c_val);

    dst[(base_nat + 0u) * ncols + col] = gl_add(z0, z4);
    dst[(base_nat + 1u) * ncols + col] = gl_add(z1, z5w);
    dst[(base_nat + 2u) * ncols + col] = gl_add(z2, z6I);
    dst[(base_nat + 3u) * ncols + col] = gl_add(z3, z7c);
    dst[(base_nat + 4u) * ncols + col] = gl_sub(z0, z4);
    dst[(base_nat + 5u) * ncols + col] = gl_sub(z1, z5w);
    dst[(base_nat + 6u) * ncols + col] = gl_sub(z2, z6I);
    dst[(base_nat + 7u) * ncols + col] = gl_sub(z3, z7c);
}

// ---- kernel: ntt_radix4_phase ---------------------------------------------
// Combines two consecutive radix-2 DIT stages (s and s+1) into ONE pass over
// the data. Produces the same output as running ntt_butterfly_phase twice at
// s and s+1. Saves half the memory read+write traffic of the butterfly loop.
//
// Derivation: for quad-butterfly index q within combined group of size
// M = 2^(s+1):
//   x_A0 = buf[g*M + q]            (sub-group A, pre-s)
//   x_A1 = buf[g*M + q + M/4]
//   x_B0 = buf[g*M + q + M/2]      (sub-group B)
//   x_B1 = buf[g*M + q + 3M/4]
//
// Stage s butterflies (within sub-groups A and B), twiddle w_s = ω_{M/2}^q:
//   y_A0 = x_A0 + w_s*x_A1,  y_A1 = x_A0 - w_s*x_A1
//   y_B0 = x_B0 + w_s*x_B1,  y_B1 = x_B0 - w_s*x_B1
//
// Stage s+1 outer butterflies, twiddles w_q = ω_M^q, w_q_I = ω_M^(q+M/4):
//   z_0 = y_A0 + w_q*y_B0     → goes to buf[g*M + q]
//   z_1 = y_A1 + w_q_I*y_B1   → goes to buf[g*M + q + M/4]
//   z_2 = y_A0 - w_q*y_B0     → goes to buf[g*M + q + M/2]
//   z_3 = y_A1 - w_q_I*y_B1   → goes to buf[g*M + q + 3M/4]
//
// Twiddle reads (stride_s1 = s_global - (s+1)):
//   w_s   = twiddles[q << (stride_s1 + 1)]     // = ω_{M/2}^q = ω_M^(2q)
//   w_q   = twiddles[q << stride_s1]           // = ω_M^q
//   w_q_I = twiddles[(q + M/4) << stride_s1]   // = ω_M^q * I (precomputed in table)
//
// Dispatch: (domain_size/4) * ncols threads. Thread tid handles one
// (quad-butterfly, column) pair.
kernel void ntt_radix4_phase(
    device ulong*         buf            [[ buffer(0) ]],
    device const ulong*   twiddles       [[ buffer(1) ]],
    constant uint&        ncols          [[ buffer(2) ]],
    constant uint&        domain_size    [[ buffer(3) ]],
    constant uint&        s              [[ buffer(4) ]],
    constant uint&        stride_s1      [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint quarter = domain_size >> 2u;
    uint total = quarter * ncols;
    if (tid >= total) return;

    uint idx = tid / ncols;
    uint col = tid % ncols;

    uint M       = 1u << (s + 1u);    // combined group size
    uint M_div_4 = M >> 2u;
    uint M_div_2 = M >> 1u;

    uint g = idx / M_div_4;
    uint q = idx % M_div_4;

    // Position offsets within the buffer (scaled by ncols for stride)
    uint base_elem = g * M + q;
    uint offA0 = base_elem * ncols + col;
    uint offA1 = (base_elem + M_div_4) * ncols + col;
    uint offB0 = (base_elem + M_div_2) * ncols + col;
    uint offB1 = (base_elem + M_div_2 + M_div_4) * ncols + col;

    ulong x_A0 = buf[offA0];
    ulong x_A1 = buf[offA1];
    ulong x_B0 = buf[offB0];
    ulong x_B1 = buf[offB1];

    ulong w_s   = twiddles[q << (stride_s1 + 1u)];
    ulong w_q   = twiddles[q << stride_s1];
    ulong w_q_I = twiddles[(q + M_div_4) << stride_s1];

    // Stage s
    ulong t_A1 = gl_mul(x_A1, w_s);
    ulong t_B1 = gl_mul(x_B1, w_s);
    ulong y_A0 = gl_add(x_A0, t_A1);
    ulong y_A1 = gl_sub(x_A0, t_A1);
    ulong y_B0 = gl_add(x_B0, t_B1);
    ulong y_B1 = gl_sub(x_B0, t_B1);

    // Stage s+1
    ulong t_B0  = gl_mul(y_B0, w_q);
    ulong t_B1p = gl_mul(y_B1, w_q_I);
    buf[offA0] = gl_add(y_A0, t_B0);
    buf[offA1] = gl_add(y_A1, t_B1p);
    buf[offB0] = gl_sub(y_A0, t_B0);
    buf[offB1] = gl_sub(y_A1, t_B1p);
}

// ---- kernel: ntt_radix8_phase ---------------------------------------------
// Combines THREE consecutive radix-2 DIT stages (s, s+1, s+2) into ONE pass
// over the data. Produces the same output as running ntt_butterfly_phase
// three times, cutting memory traffic by 3× relative to per-stage dispatches
// and by 1.5× relative to radix-4.
//
// Derivation (DIT, 8 elements strided by M/8 within a combined group of M):
//   M = 2^(s+2)   — combined group size
//   S8 = M / 8   — stride between the 8 elements this thread processes
//   For group g and intra-group position q ∈ [0, S8):
//     x[i] = buf[g*M + q + i*S8],  i = 0..7
//
// Stage s   : 4 pair-butterflies (x0,x1),(x2,x3),(x4,x5),(x6,x7)
//             twiddle w_a = ω_{M_a}^q  (M_a = 2^s, same twiddle for all 4)
// Stage s+1 : 4 pair-butterflies (y0,y2),(y1,y3),(y4,y6),(y5,y7)
//             twiddles w_{b0} = ω_{M_b}^q , w_{b1} = ω_{M_b}^{q + M_a/2}
//             (M_b = 2^(s+1))
// Stage s+2 : 4 pair-butterflies (z0,z4),(z1,z5),(z2,z6),(z3,z7)
//             twiddles w_{c,i} = ω_{M_c}^{q + i*M_a/2}  i = 0..3
//             (M_c = 2^(s+2))
//
// Twiddle reads (stride_s = s_global - s for the current `s`):
//   w_a    = twiddles[q << stride_s]                        // ω_{M_a}^q
//   w_{bi} = twiddles[(q + i*M_a/2) << (stride_s - 1)]      // ω_{M_b}^{·}
//   w_{ci} = twiddles[(q + i*M_a/2) << (stride_s - 2)]      // ω_{M_c}^{·}
//
// Dispatch: (domain_size/8) * ncols threads.
kernel void ntt_radix8_phase(
    device ulong*         buf            [[ buffer(0) ]],
    device const ulong*   twiddles       [[ buffer(1) ]],
    constant uint&        ncols          [[ buffer(2) ]],
    constant uint&        domain_size    [[ buffer(3) ]],
    constant uint&        s              [[ buffer(4) ]],
    constant uint&        stride_s       [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint eighth = domain_size >> 3u;
    uint total  = eighth * ncols;
    if (tid >= total) return;

    uint idx = tid / ncols;
    uint col = tid % ncols;

    uint M    = 1u << (s + 2u);   // combined group size
    uint S8   = M >> 3u;          // M/8 — stride within group
    uint Ma_2 = 1u << (s - 1u);   // M_a/2 = 2^(s-1)

    uint g = idx / S8;
    uint q = idx % S8;

    uint base = (g * M + q) * ncols + col;
    uint step = S8 * ncols;

    ulong x0 = buf[base            ];
    ulong x1 = buf[base + 1u*step  ];
    ulong x2 = buf[base + 2u*step  ];
    ulong x3 = buf[base + 3u*step  ];
    ulong x4 = buf[base + 4u*step  ];
    ulong x5 = buf[base + 5u*step  ];
    ulong x6 = buf[base + 6u*step  ];
    ulong x7 = buf[base + 7u*step  ];

    // Stage s: 4 butterflies with shared twiddle w_a
    ulong w_a = twiddles[q << stride_s];
    ulong t1 = gl_mul(x1, w_a);
    ulong t3 = gl_mul(x3, w_a);
    ulong t5 = gl_mul(x5, w_a);
    ulong t7 = gl_mul(x7, w_a);
    ulong y0 = gl_add(x0, t1);
    ulong y1 = gl_sub(x0, t1);
    ulong y2 = gl_add(x2, t3);
    ulong y3 = gl_sub(x2, t3);
    ulong y4 = gl_add(x4, t5);
    ulong y5 = gl_sub(x4, t5);
    ulong y6 = gl_add(x6, t7);
    ulong y7 = gl_sub(x6, t7);

    // Stage s+1: twiddles w_{b0}, w_{b1}
    uint  stride_b = stride_s - 1u;
    ulong w_b0 = twiddles[q << stride_b];
    ulong w_b1 = twiddles[(q + Ma_2) << stride_b];
    ulong u2 = gl_mul(y2, w_b0);
    ulong u3 = gl_mul(y3, w_b1);
    ulong u6 = gl_mul(y6, w_b0);
    ulong u7 = gl_mul(y7, w_b1);
    ulong z0 = gl_add(y0, u2);
    ulong z2 = gl_sub(y0, u2);
    ulong z1 = gl_add(y1, u3);
    ulong z3 = gl_sub(y1, u3);
    ulong z4 = gl_add(y4, u6);
    ulong z6 = gl_sub(y4, u6);
    ulong z5 = gl_add(y5, u7);
    ulong z7 = gl_sub(y5, u7);

    // Stage s+2: twiddles w_{c0..c3}
    uint  stride_c = stride_s - 2u;
    ulong w_c0 = twiddles[q << stride_c];
    ulong w_c1 = twiddles[(q + Ma_2)      << stride_c];
    ulong w_c2 = twiddles[(q + 2u*Ma_2)   << stride_c];
    ulong w_c3 = twiddles[(q + 3u*Ma_2)   << stride_c];
    ulong v4 = gl_mul(z4, w_c0);
    ulong v5 = gl_mul(z5, w_c1);
    ulong v6 = gl_mul(z6, w_c2);
    ulong v7 = gl_mul(z7, w_c3);

    buf[base            ] = gl_add(z0, v4);
    buf[base + 1u*step  ] = gl_add(z1, v5);
    buf[base + 2u*step  ] = gl_add(z2, v6);
    buf[base + 3u*step  ] = gl_add(z3, v7);
    buf[base + 4u*step  ] = gl_sub(z0, v4);
    buf[base + 5u*step  ] = gl_sub(z1, v5);
    buf[base + 6u*step  ] = gl_sub(z2, v6);
    buf[base + 7u*step  ] = gl_sub(z3, v7);
}

// ---- kernel: intt_reorder_scale -------------------------------------------
// Fused INTT finalization: combines the (N-i) % N reorder and the 1/N scale
// into one in-place pass. Saves one full read+write pass over the buffer
// and one command buffer commit+wait versus running intt_reorder followed
// by intt_scale.
//
// Parallelization: one thread per (pair_key, col) — dispatch
// (N/2 + 1) × ncols threads. Each thread handles EXACTLY one pair-swap or
// one fixed-point scale for a single column. This replaces the previous
// (N/2 + 1)-thread kernel whose each thread looped over ncols columns
// serially — at ncols=128 that was a 128× parallelism deficit.
//
// Mapping:
//   pair_idx = tid / ncols     (pair key in [0, N/2])
//   col      = tid % ncols
//   pair_idx == 0                 → position (0, col) is a fixed point.
//   pair_idx in [1, N/2)          → swap (pair_idx, col) with (N-pair_idx, col).
//   pair_idx == N/2 (N even)      → fixed point (N - N/2 == N/2). Scale only.
kernel void intt_reorder_scale(
    device ulong*    buf           [[ buffer(0) ]],
    constant uint&   domain_size   [[ buffer(1) ]],
    constant uint&   ncols         [[ buffer(2) ]],
    constant ulong&  inv_n         [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;

    uint N  = domain_size;
    uint hi = N - pair_idx;  // intt_idx(pair_idx, N) when pair_idx != 0

    if (pair_idx == 0) {
        // Row 0 is a fixed point (intt_idx(0,N)=0). Scale in place.
        uint idx = col;
        buf[idx] = gl_canonicalize(gl_mul(buf[idx], inv_n));
        return;
    }
    if (pair_idx >= hi) {
        // pair_idx == hi → (N even, pair_idx == N/2), fixed point; scale.
        // pair_idx > hi → out of range (dispatch rounded up); no-op.
        if (pair_idx == hi) {
            uint idx = pair_idx * ncols + col;
            buf[idx] = gl_canonicalize(gl_mul(buf[idx], inv_n));
        }
        return;
    }

    // Normal pair: read-then-write to swap (pair_idx, hi) with scale fused in.
    uint lo_idx = pair_idx * ncols + col;
    uint hi_idx = hi       * ncols + col;
    ulong a = buf[lo_idx];
    ulong b = buf[hi_idx];
    buf[lo_idx] = gl_canonicalize(gl_mul(b, inv_n));
    buf[hi_idx] = gl_canonicalize(gl_mul(a, inv_n));
}

// ---- kernel: intt_reorder_coset_scale -------------------------------------
// Same as intt_reorder_scale but applies a PER-ELEMENT coset multiplier
// r_inv[dsty] (where dsty = intt_idx(src_idx, N) = (N - src_idx) % N) instead
// of the scalar 1/N. This is the LDE-side variant that fuses the reorder
// step with the coset shift that `extendPol` applies via the CPU's
// `extend=true` path (ntt_goldilocks.cpp:171-182).
//
// Table `r_inv` MUST be exactly N ulongs long, with r_inv[i] = shift^i / N
// matching NTT_Goldilocks::computeR (ntt_goldilocks.hpp:153-165).
//
// Dispatch: (N/2 + 1) × ncols threads (same as intt_reorder_scale).
kernel void intt_reorder_coset_scale(
    device ulong*            buf           [[ buffer(0) ]],
    device const ulong*      r_inv         [[ buffer(1) ]],
    constant uint&           domain_size   [[ buffer(2) ]],
    constant uint&           ncols         [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;

    uint N  = domain_size;
    uint hi = N - pair_idx;

    if (pair_idx == 0) {
        uint idx = col;
        buf[idx] = gl_canonicalize(gl_mul(buf[idx], r_inv[0]));
        return;
    }
    if (pair_idx >= hi) {
        if (pair_idx == hi) {
            uint idx = pair_idx * ncols + col;
            buf[idx] = gl_canonicalize(gl_mul(buf[idx], r_inv[pair_idx]));
        }
        return;
    }

    // Pair (pair_idx, hi): read both, swap with per-destination coset scale.
    //   output[pair_idx] = input[hi]       * r_inv[pair_idx]
    //   output[hi]       = input[pair_idx] * r_inv[hi]
    uint lo_idx = pair_idx * ncols + col;
    uint hi_idx = hi       * ncols + col;
    ulong a = buf[lo_idx];
    ulong b = buf[hi_idx];
    buf[lo_idx] = gl_canonicalize(gl_mul(b, r_inv[pair_idx]));
    buf[hi_idx] = gl_canonicalize(gl_mul(a, r_inv[hi]));
}

// ---- kernel: intt_reorder --------------------------------------------------
// Inverse-NTT index permutation: out[(N - i) % N] = in[i].
// CPU reference: NTT_iters uses intt_idx(i,N) = (N - i) % N before INTT scale
// (ntt_goldilocks.cpp:175,188; ntt_goldilocks.hpp:38-46).
//
// Implemented in-place as a swap between index i and (N - i) for i in [1, N/2),
// once per row. Element 0 is fixed (N % N = 0). If N is even, element N/2 is
// also fixed (N - N/2 = N/2). Dispatch N/2 threads; thread tid handles pair
// (tid+1, N - (tid+1)) for tid in [0, N/2 - 1).
kernel void intt_reorder(
    device ulong*    buf           [[ buffer(0) ]],
    constant uint&   domain_size   [[ buffer(1) ]],
    constant uint&   ncols         [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint i = tid + 1;
    uint lo = i;
    uint hi = domain_size - i;
    if (lo >= hi) return;  // only swap once per pair; stop at the middle
    for (uint c = 0; c < ncols; c++) {
        ulong a = buf[lo * ncols + c];
        ulong b = buf[hi * ncols + c];
        buf[lo * ncols + c] = b;
        buf[hi * ncols + c] = a;
    }
}

// ---- kernel: intt_scale ----------------------------------------------------
// One thread per flat element index [0, domain_size * ncols).
// Multiplies each element by the precomputed scalar inv_n = 1/domain_size mod p.
// Used as the final step of the inverse NTT (after intt_reorder).
kernel void intt_scale(
    device ulong*       buf     [[ buffer(0) ]],
    constant ulong&     inv_n   [[ buffer(1) ]],
    constant uint&      count   [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= count) return;
    buf[tid] = gl_canonicalize(gl_mul(buf[tid], inv_n));
}
