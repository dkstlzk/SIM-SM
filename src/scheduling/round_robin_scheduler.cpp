#include "scheduling/round_robin_scheduler.hpp"

namespace sim_sm {

Warp* RoundRobinScheduler::select_warp(std::vector<Warp>& warps) {
    if (warps.empty()) return nullptr;

    size_t num_warps = warps.size();
    if (num_warps == 0) return nullptr;

    for (size_t i = 0; i < num_warps; ++i) {
        size_t idx = (next_warp_index_ + i) % num_warps;
        if (warps[idx].get_state() == WarpState::Ready) {
            next_warp_index_ = (idx + 1) % num_warps;
            return &warps[idx];
        }
    }

    return nullptr;
}

} // namespace sim_sm
