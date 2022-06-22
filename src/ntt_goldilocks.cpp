#include "ntt_goldilocks.hpp"

static inline u_int64_t BR(u_int64_t x, u_int64_t domainPow)
{
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    return (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) >> (32 - domainPow);
}

void NTT_Goldilocks::NTT(Goldilocks::Element *_a, u_int64_t n)
{
    Goldilocks::Element *aux_a = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * n);
    Goldilocks::Element *a = _a;
    Goldilocks::Element *a2 = aux_a;
    Goldilocks::Element *tmp;

    reversePermutation(a2, a, n);

    tmp = a2;
    a2 = a;
    a = tmp;

    u_int64_t domainPow = log2(n);
    assert(((u_int64_t)1 << domainPow) == n);
    u_int64_t maxBatchPow = s / 4;
    u_int64_t batchSize = 1 << maxBatchPow;
    u_int64_t nBatches = n / batchSize;
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow)
    {
        u_int64_t sInc = s + maxBatchPow <= domainPow ? maxBatchPow : domainPow - s + 1;
        omp_set_dynamic(0);
        omp_set_num_threads(nThreads);
#pragma omp parallel for
        for (u_int64_t b = 0; b < nBatches; b++)
        {
            u_int64_t rs = s - 1;
            uint64_t re = domainPow - 1;
            uint64_t rb = 1 << rs;
            uint64_t rm = (1 << (re - rs)) - 1;
            for (u_int64_t si = 0; si < sInc; si++)
            {
                u_int64_t m = 1 << (s + si);
                u_int64_t mdiv2 = m >> 1;
                u_int64_t mdiv2i = 1 << si;
                u_int64_t mi = mdiv2i * 2;
                for (u_int64_t i = 0; i < (batchSize >> 1); i++)
                {
                    Goldilocks::Element t;
                    Goldilocks::Element u;
                    u_int64_t ki = b * batchSize + (i / mdiv2i) * mi;
                    u_int64_t ji = i % mdiv2i;

                    u_int64_t j = (b * batchSize / 2 + i);
                    j = (j & rm) * rb + (j >> (re - rs));
                    j = j % mdiv2;

                    // t = root(s + si, j) * a[ki + ji + mdiv2i];
                    Goldilocks::mul(t, root(s + si, j), a[ki + ji + mdiv2i]);
                    u = a[ki + ji];

                    Goldilocks::add(a[ki + ji], t, u);
                    // result[ki + ji] = t + u;
                    Goldilocks::sub(a[ki + ji + mdiv2i], u, t);
                    // result[ki + ji + mdiv2i] = u - t;
                }
            }
        }
        shuffle(a2, a, n, sInc);
        tmp = a2;
        a2 = a;
        a = tmp;
    }
    if (a != _a)
    {
        std::memcpy(_a, a, n * sizeof(uint64_t));
    }
    free(aux_a);
}

void NTT_Goldilocks::reversePermutation(Goldilocks::Element *result, Goldilocks::Element *a, u_int64_t size)
{
    uint32_t domainSize = log2(size);
#pragma omp parallel for
    for (u_int64_t i = 0; i < size; i++)
    {
        u_int64_t r;
        r = BR(i, domainSize);
        result[i] = a[r];
    }
}

void NTT_Goldilocks::shuffle(Goldilocks::Element *result, Goldilocks::Element *src, uint64_t size, uint64_t s)
{
    uint64_t srcRowSize = 1 << s;

    uint64_t srcX = 0;
    uint64_t srcWidth = 1 << s;
    uint64_t srcY = 0;
    uint64_t srcHeight = size / srcRowSize;

    uint64_t dstRowSize = size / srcRowSize;
    uint64_t dstX = 0;
    uint64_t dstY = 0;

#pragma omp parallel
#pragma omp single
    traspose(result, src, srcRowSize, srcX, srcWidth, srcY, srcHeight, dstRowSize, dstX, dstY);
#pragma omp taskwait
}

void NTT_Goldilocks::traspose(
    Goldilocks::Element *dst,
    Goldilocks::Element *src,
    uint64_t srcRowSize,
    uint64_t srcX,
    uint64_t srcWidth,
    uint64_t srcY,
    uint64_t srcHeight,
    uint64_t dstRowSize,
    uint64_t dstX,
    uint64_t dstY)
{
    if ((srcWidth == 1) || (srcHeight == 1) || (srcWidth * srcHeight < CACHESIZE))
    {
#pragma omp task
        {
            for (uint64_t x = 0; x < srcWidth; x++)
            {
                for (uint64_t y = 0; y < srcHeight; y++)
                {
                    dst[(dstY + +x) * dstRowSize + (dstX + y)] = src[(srcY + +y) * srcRowSize + (srcX + x)];
                }
            }
        }
        return;
    }
    if (srcWidth > srcHeight)
    {
        traspose(dst, src, srcRowSize, srcX, srcWidth / 2, srcY, srcHeight, dstRowSize, dstX, dstY);
        traspose(dst, src, srcRowSize, srcX + srcWidth / 2, srcWidth / 2, srcY, srcHeight, dstRowSize, dstX, dstY + srcWidth / 2);
    }
    else
    {
        traspose(dst, src, srcRowSize, srcX, srcWidth, srcY, srcHeight / 2, dstRowSize, dstX, dstY);
        traspose(dst, src, srcRowSize, srcX, srcWidth, srcY + srcHeight / 2, srcHeight / 2, dstRowSize, dstX + srcHeight / 2, dstY);
    }
}

void NTT_Goldilocks::INTT(Goldilocks::Element *a, u_int64_t size)
{
    NTT_Goldilocks::NTT(a, size);
    u_int64_t domainPow = NTT_Goldilocks::log2(size);
    u_int64_t nDiv2 = size >> 1;
#pragma omp parallel for num_threads(nThreads)
    for (u_int64_t i = 1; i < nDiv2; i++)
    {
        Goldilocks::Element tmp;
        u_int64_t r = size - i;
        tmp = a[i];
        a[i] = a[r] * powTwoInv[domainPow];
        a[r] = tmp * powTwoInv[domainPow];
    }
    a[0] = a[0] * powTwoInv[domainPow];
    a[size >> 1] = a[size >> 1] * powTwoInv[domainPow];
}