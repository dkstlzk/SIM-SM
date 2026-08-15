#include "scheduling/gto_scheduler.hpp"

namespace sim_sm {

Warp* GTOScheduler::select_warp(std::vector<Warp>& warps) {
    Warp* selected = nullptr;

    // First try to schedule the last issued warp if it's still Ready
    if (last_issued_warp_id_ != static_cast<size_t>(-1)) {
        for (auto& warp : warps) {
            if (warp.get_warp_id() == last_issued_warp_id_ && warp.get_state() == WarpState::Ready) {
                selected = &warp;
                break;
            }
        }
    }

    // Fall back to oldest ready warp
    if (!selected) {
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
    }

    if (selected) {
        last_issued_warp_id_ = selected->get_warp_id();
    } else {
        last_issued_warp_id_ = static_cast<size_t>(-1);
    }

    return selected;
}

} // namespace sim_sm
