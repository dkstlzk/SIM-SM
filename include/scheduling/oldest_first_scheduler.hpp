#pragma once

#include "scheduling/warp_scheduler.hpp"

namespace sim_sm {

class OldestFirstScheduler : public WarpScheduler {
public:
    Warp* select_warp(std::vector<Warp>& warps) override;
    std::string name() const override { return "OldestFirst"; }
};

} // namespace sim_sm
