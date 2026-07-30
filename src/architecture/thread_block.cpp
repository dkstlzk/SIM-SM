#include "architecture/thread_block.hpp"

namespace sim_sm {

ThreadBlock::ThreadBlock(size_t block_id) : block_id_(block_id) {}

void ThreadBlock::add_warp(const Warp& warp) {
    warps_.push_back(warp);
}

const std::vector<Warp>& ThreadBlock::get_warps() const {
    return warps_;
}

size_t ThreadBlock::get_block_id() const {
    return block_id_;
}

} // namespace sim_sm
