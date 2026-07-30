#include "architecture/grid.hpp"

namespace sim_sm {

void Grid::add_block(const ThreadBlock& block) {
    blocks_.push_back(block);
}

const std::vector<ThreadBlock>& Grid::get_blocks() const {
    return blocks_;
}

} // namespace sim_sm
