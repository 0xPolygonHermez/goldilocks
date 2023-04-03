#ifndef GOLDILOCKS_BATCH
#define GOLDILOCKS_BATCH
#include "goldilocks_base_field.hpp"

#define BATCH_SIZE_ 4

inline void Goldilocks::copy_batch(Element *dst, const Element &src)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        dst[i].fe = src.fe;
    }
}
inline void Goldilocks::copy_batch(Element *dst, const Element *src)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        dst[i].fe = src[i].fe;
    }
}
inline void Goldilocks::copy_batch(Element *dst, const Element *src, uint64_t stride)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        dst[i].fe = src[i * stride].fe;
    }
}
inline void Goldilocks::copy_batch(Element *dst, const Element *src, uint64_t stride[4])
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        dst[i].fe = src[stride[i]].fe;
    }
}
inline void Goldilocks::copy_batch(Element *dst, uint64_t stride, const Element *src)
{
    dst[0] = src[0];
    dst[stride] = src[1];
    dst[2 * stride] = src[2];
    dst[3 * stride] = src[3];
}
inline void Goldilocks::copy_batch(Element *dst, uint64_t stride[4], const Element *src)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        dst[stride[i]].fe = src[i].fe;
    }
}

inline void Goldilocks::add_batch(Element *result, const Element *in1, const Element *in2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        add(result[i], in1[i], in2[i]);
    }
}
inline void Goldilocks::add_batch(Element *result, const Element *in1, const Element *in2, uint64_t offset2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        add(result[i], in1[i], in2[i * offset2]);
    }
}
inline void Goldilocks::add_batch(Element *result, const Element *in1, const Element in2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        add(result[i], in1[i], in2);
    }
}
inline void Goldilocks::add_batch(Element *result, const Element *in1, const Element in2, uint64_t offset1)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        add(result[i], in1[i * offset1], in2);
    }
}
inline void Goldilocks::add_batch(Element *result, const Element *in1, const Element *in2, uint64_t offset1, uint64_t offset2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        add(result[i], in1[i * offset1], in2[i * offset2]);
    }
}
inline void Goldilocks::add_batch(Element *result, const Element *in1, const Element *in2, const uint64_t offsets1[4], const uint64_t offsets2[4])
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        add(result[i], in1[offsets1[i]], in2[offsets2[i]]);
    }
}
inline void Goldilocks::add_batch(Element *result, const Element *in1, const Element in2, const uint64_t offsets1[4])
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        add(result[i], in1[offsets1[i]], in2);
    }
}

inline void Goldilocks::sub_batch(Element *result, const Element *in1, const Element *in2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1[i], in2[i]);
    }
}
inline void Goldilocks::sub_batch(Element *result, const Element *in1, const Element *in2, uint64_t offset1, uint64_t offset2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1[i * offset1], in2[i * offset2]);
    }
}
inline void Goldilocks::sub_batch(Element *result, const Element *in1, const Element in2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1[i], in2);
    }
}
inline void Goldilocks::sub_batch(Element *result, const Element in1, const Element *in2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1, in2[i]);
    }
}
inline void Goldilocks::sub_batch(Element *result, const Element *in1, const Element in2, uint64_t offset1)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1[i * offset1], in2);
    }
}
inline void Goldilocks::sub_batch(Element *result, const Element in1, const Element *in2, uint64_t offset2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1, in2[i * offset2]);
    }
}
inline void Goldilocks::sub_batch(Element *result, const Element *in1, const Element *in2, const uint64_t offsets1[4], const uint64_t offsets2[4])
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1[offsets1[i]], in2[offsets2[i]]);
    }
}
inline void Goldilocks::sub_batch(Element *result, const Element in1, const Element *in2, const uint64_t offsets2[4])
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1, in2[offsets2[i]]);
    }
}
inline void Goldilocks::sub_batch(Element *result, const Element *in1, const Element in2, const uint64_t offsets1[4])
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        sub(result[i], in1[offsets1[i]], in2);
    }
}

inline void Goldilocks::mul_batch(Element *result, const Element *in1, const Element *in2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        mul(result[i], in1[i], in2[i]);
    }
}
inline void Goldilocks::mul_batch(Element *result, const Element in1, const Element *in2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        mul(result[i], in1, in2[i]);
    }
}
inline void Goldilocks::mul_batch(Element *result, const Element *in1, const Element *in2, uint64_t offset1, uint64_t offset2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        mul(result[i], in1[i * offset1], in2[i * offset2]);
    }
}
inline void Goldilocks::mul_batch(Element *result, const Element in1, const Element *in2, uint64_t offset2)
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        mul(result[i], in1, in2[i * offset2]);
    }
}
inline void Goldilocks::mul_batch(Element *result, const Element *in1, const Element *in2, const uint64_t offsets1[4], const uint64_t offsets2[4])
{
    for (uint64_t i = 0; i < BATCH_SIZE_; ++i)
    {
        mul(result[i], in1[offsets1[i]], in2[offsets2[i]]);
    }
}
#endif