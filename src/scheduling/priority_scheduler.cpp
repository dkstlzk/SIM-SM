#include "scheduling/priority_scheduler.hpp"
#include <limits>

namespace sim_sm {

Warp* PriorityScheduler::select_warp(std::vector<Warp>& warps) {
    if (warps.empty()) return nullptr;

    size_t num_warps = warps.size();
    if (num_warps == 0) return nullptr;

    Warp* selected = nullptr;
    int max_prio = std::numeric_limits<int>::lowest();
    size_t best_idx = 0;

    for (size_t i = 0; i < num_warps; ++i) {
        size_t idx = (next_warp_index_ + i) % num_warps;
        if (warps[idx].get_state() == WarpState::Ready) {
            if (warps[idx].get_priority() > max_prio) {
                max_prio = warps[idx].get_priority();
                selected = &warps[idx];
                best_idx = idx;
            }
        }
    }

    if (selected) {
        next_warp_index_ = (best_idx + 1) % num_warps;
    }

    return selected;
}

} // namespace sim_sm
