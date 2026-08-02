#include "scheduling/greedy_scheduler.hpp"

namespace sim_sm {

Warp* GreedyScheduler::select_warp(std::vector<Warp>& warps) {
    for (auto& warp : warps) {
        if (warp.get_state() == WarpState::Ready) {
            return &warp;
        }
    }
    return nullptr;
}

} // namespace sim_sm
