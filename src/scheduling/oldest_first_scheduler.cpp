#include "scheduling/oldest_first_scheduler.hpp"

namespace sim_sm {

Warp* OldestFirstScheduler::select_warp(std::vector<Warp>& warps) {
    Warp* selected = nullptr;

    for (auto& warp : warps) {
        if (warp.get_state() == WarpState::Ready) {
            if (!selected) {
                selected = &warp;
            } else {
                if (warp.get_ready_since_cycle() < selected->get_ready_since_cycle()) {
                    selected = &warp;
                } else if (warp.get_ready_since_cycle() == selected->get_ready_since_cycle()) {
                    if (warp.get_warp_id() < selected->get_warp_id()) {
                        selected = &warp;
                    }
                }
            }
        }
    }

    return selected;
}

} // namespace sim_sm
