#pragma once
#include <functional>
#include <stdint.h>
#include <algorithm>

namespace train_llm {
const int BLOCK_SIZE = 256;
constexpr int thread_wrap_size = 32;
const int MAX_BLOCK_SIZE = 1024;  //A100
constexpr uint32_t dim_block_size[] = {32, 16, 8, 4, 1};

inline uint32_t kernel_block_size(uint32_t dim) {
    return *lower_bound(dim_block_size, dim_block_size + 4, MAX_BLOCK_SIZE / dim, std::greater<uint32_t>());
}

}  // namespace train_llm