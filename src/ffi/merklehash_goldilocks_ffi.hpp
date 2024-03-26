#ifndef GOLDILOCKS_FFI_HPP
#define GOLDILOCKS_FFI_HPP
    #include "merklehash_goldilocks.hpp"

    // MERKLEHASH
    inline void root(Goldilocks::Element *root, Goldilocks::Element *tree, uint64_t numElementsTree) {
        MerklehashGoldilocks::root(root, tree, numElementsTree);
    }
    
    inline void root(Goldilocks::Element (&root)[HASH_SIZE], Goldilocks::Element *tree, uint64_t numElementsTree) {
        MerklehashGoldilocks::root(root, tree, numElementsTree);
    }

    inline uint64_t getTreeNumElements(uint64_t degree) {
        return MerklehashGoldilocks::getTreeNumElements(degree);
    }

#endif