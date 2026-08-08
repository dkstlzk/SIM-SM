#include "memory/lru_policy.hpp"
#include <algorithm>
#include <stdexcept>

namespace sim_sm {

LRUPolicy::LRUPolicy(size_t associativity) : associativity_(associativity) {
    // Initialize LRU list with ways 0 to associativity-1
    for (size_t i = 0; i < associativity_; ++i) {
        lru_list_.push_back(i);
    }
}

size_t LRUPolicy::choose_victim(const std::vector<CacheLine>& lines) {
    // Return the least recently used way (at the back of the list)
    if (lru_list_.empty()) {
        throw std::runtime_error("LRU list is empty");
    }

    // Check for an invalid line first to fill before evicting valid data
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].valid) {
            return i;
        }
    }

    return lru_list_.back();
}

void LRUPolicy::on_access(size_t way) {
    auto it = std::find(lru_list_.begin(), lru_list_.end(), way);
    if (it != lru_list_.end()) {
        lru_list_.erase(it);
        lru_list_.push_front(way);
    }
}

} // namespace sim_sm
