#pragma once

#include "architecture/thread_block.hpp"
#include <vector>

namespace sim_sm {

class Grid {
public:
    void add_block(const ThreadBlock& block);
    const std::vector<ThreadBlock>& get_blocks() const;

private:
    std::vector<ThreadBlock> blocks_;
};

} // namespace sim_sm
