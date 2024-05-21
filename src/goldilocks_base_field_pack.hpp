#ifndef GOLDILOCKS_PACK
#include "goldilocks_base_field.hpp"
#include <cassert>
/*
    Implementations for expressions:
*/

    inline void Goldilocks::copy_pack(Element *dst, const Element &src, uint64_t size){
        for (uint64_t i = 0; i < size; ++i)
        {
            dst[i].fe = src.fe;
        }

    }
    inline void Goldilocks::copy_pack(Element *dst, const Element *src, uint64_t size){
        for (uint64_t i = 0; i < size; ++i)
        {
            dst[i].fe = src[i].fe;
        }
    }
  
    inline void Goldilocks::op_pack(uint64_t op, Element *c, const Element *a, const Element *b, uint64_t size){

        switch (op)
        {
        case 0:
            for (uint64_t i = 0; i < size; ++i)
            {
                add(c[i], a[i], b[i]);
            }
            break;
        case 1:
            for (uint64_t i = 0; i < size; ++i)
            {
                sub(c[i], a[i], b[i]);
            }
            break;
        case 2:
            for (uint64_t i = 0; i < size; ++i)
            {
                mul(c[i], a[i], b[i]);
            }
            break;
        case 3:
            for (uint64_t i = 0; i < size; ++i)
            {
                sub(c[i], b[i], a[i]);
            }
            break;
        default:
            assert(0);
            break;
        }
    }

#endif