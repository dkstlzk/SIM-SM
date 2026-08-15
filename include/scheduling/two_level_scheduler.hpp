#pragma once

#include "scheduling/warp_scheduler.hpp"
#include <unordered_set>
#include <vector>

namespace sim_sm {

class TwoLevelScheduler : public WarpScheduler {
public:
    explicit TwoLevelScheduler(size_t active_set_size)
        : active_set_size_(active_set_size) {}

    Warp* select_warp(std::vector<Warp>& warps) override;
    std::string name() const override { return "TwoLevel"; }

private:
    size_t active_set_size_;
    std::vector<size_t> active_warp_ids_;
    size_t rr_index_{0};
    
    void update_active_set(const std::vector<Warp>& warps);
};

} // namespace sim_sm
