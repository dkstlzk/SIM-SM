#pragma once

#include "scheduling/warp_scheduler.hpp"

namespace sim_sm {

class GreedyScheduler : public WarpScheduler {
public:
    Warp* select_warp(std::vector<Warp>& warps) override;
    std::string name() const override { return "Greedy"; }
};

} // namespace sim_sm
