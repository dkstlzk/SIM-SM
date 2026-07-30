#pragma once

#include "architecture/warp.hpp"
#include <vector>
#include <cstddef>

namespace sim_sm {

class ThreadBlock {
public:
    ThreadBlock(size_t block_id);

    void add_warp(const Warp& warp);
    const std::vector<Warp>& get_warps() const;
    size_t get_block_id() const;

private:
    size_t block_id_;
    std::vector<Warp> warps_;
};

} // namespace sim_sm
