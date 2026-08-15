#pragma once

#include "scheduling/warp_scheduler.hpp"

namespace sim_sm {

class GTOScheduler : public WarpScheduler {
public:
    Warp* select_warp(std::vector<Warp>& warps) override;
    std::string name() const override { return "GTO"; }

private:
    size_t last_issued_warp_id_{static_cast<size_t>(-1)};
};

} // namespace sim_sm
