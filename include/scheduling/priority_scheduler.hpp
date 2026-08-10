#pragma once

#include "scheduling/warp_scheduler.hpp"
#include <cstddef>

namespace sim_sm {

class PriorityScheduler : public WarpScheduler {
public:
    Warp* select_warp(std::vector<Warp>& warps) override;
    std::string name() const override { return "Priority"; }

private:
    size_t next_warp_index_{0};
};

} // namespace sim_sm
