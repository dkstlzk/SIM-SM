#include "scheduling/round_robin_scheduler.hpp"

namespace sim_sm {

Warp* RoundRobinScheduler::select_warp(std::vector<Warp>& warps) {
    if (warps.empty()) return nullptr;

    size_t num_warps = warps.size();
    size_t start_idx = (last_issued_warp_id_ + 1) % num_warps;

    for (size_t i = 0; i < num_warps; ++i) {
        size_t idx = (start_idx + i) % num_warps;
        if (warps[idx].get_state() == WarpState::Ready) {
            last_issued_warp_id_ = idx;
            return &warps[idx];
        }
    }

    return nullptr;
}

} // namespace sim_sm
