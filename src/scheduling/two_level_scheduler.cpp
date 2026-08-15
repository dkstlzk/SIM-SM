#include "scheduling/two_level_scheduler.hpp"
#include <algorithm>

namespace sim_sm {

void TwoLevelScheduler::update_active_set(const std::vector<Warp>& warps) {
    // 1. Remove completed warps
    active_warp_ids_.erase(
        std::remove_if(active_warp_ids_.begin(), active_warp_ids_.end(),
            [&warps](size_t id) {
                for (const auto& w : warps) {
                    if (w.get_warp_id() == id) {
                        return w.get_state() == WarpState::Completed;
                    }
                }
                return true; // if not found, assume completed/removed
            }),
        active_warp_ids_.end()
    );

    // 2. Admit oldest available warps to fill the active set
    if (active_warp_ids_.size() < active_set_size_) {
        // Collect candidate warps
        std::vector<const Warp*> candidates;
        for (const auto& w : warps) {
            // Is it already in the active set?
            bool in_active = std::find(active_warp_ids_.begin(), active_warp_ids_.end(), w.get_warp_id()) != active_warp_ids_.end();
            if (!in_active && w.get_state() == WarpState::Ready) {
                candidates.push_back(&w);
            }
        }

        // Sort candidates by oldest (minimum ready_since_cycle_), then by warp_id
        std::sort(candidates.begin(), candidates.end(), [](const Warp* a, const Warp* b) {
            if (a->get_ready_since_cycle() != b->get_ready_since_cycle()) {
                return a->get_ready_since_cycle() < b->get_ready_since_cycle();
            }
            return a->get_warp_id() < b->get_warp_id();
        });

        // Add to active set
        for (size_t i = 0; i < candidates.size() && active_warp_ids_.size() < active_set_size_; ++i) {
            active_warp_ids_.push_back(candidates[i]->get_warp_id());
        }
    }
}

Warp* TwoLevelScheduler::select_warp(std::vector<Warp>& warps) {
    update_active_set(warps);

    if (active_warp_ids_.empty()) {
        return nullptr;
    }

    // Round-robin selection within the active set
    for (size_t i = 0; i < active_warp_ids_.size(); ++i) {
        size_t id = active_warp_ids_[(rr_index_ + i) % active_warp_ids_.size()];
        for (auto& w : warps) {
            if (w.get_warp_id() == id && w.get_state() == WarpState::Ready) {
                rr_index_ = (rr_index_ + i + 1) % active_warp_ids_.size();
                return &w;
            }
        }
    }

    return nullptr;
}

} // namespace sim_sm
