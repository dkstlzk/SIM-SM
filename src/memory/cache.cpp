#include "memory/cache.hpp"
#include "memory/lru_policy.hpp"
#include <stdexcept>

namespace sim_sm {

Cache::Cache(size_t num_sets, size_t associativity, size_t line_size, const std::string& policy_name)
    : num_sets_(num_sets), associativity_(associativity), line_size_(line_size) {

    if (num_sets_ == 0 || associativity_ == 0 || line_size_ == 0) {
        throw std::invalid_argument("Cache dimensions must be strictly positive.");
    }

    sets_.resize(num_sets_);
    for (auto& set : sets_) {
        set.resize(associativity_);
    }

    for (size_t i = 0; i < num_sets_; ++i) {
        if (policy_name == "LRU") {
            policies_.push_back(std::make_unique<LRUPolicy>(associativity_));
        } else {
            throw std::invalid_argument("Unsupported replacement policy: " + policy_name);
        }
    }
}

bool Cache::access(size_t address) {
    stats_.accesses++;

    size_t line_address = address / line_size_;
    size_t set_index = line_address % num_sets_;
    uint64_t tag = line_address / num_sets_;

    auto& set = sets_[set_index];
    auto& policy = policies_[set_index];

    // Check for hit
    for (size_t way = 0; way < associativity_; ++way) {
        if (set[way].valid && set[way].tag == tag) {
            stats_.hits++;
            policy->on_access(way);
            return true;
        }
    }

    // Miss
    stats_.misses++;

    // Choose victim and replace
    size_t victim_way = policy->choose_victim(set);
    if (set[victim_way].valid) {
        stats_.evictions++;
    }

    set[victim_way].valid = true;
    set[victim_way].tag = tag;
    policy->on_access(victim_way);

    return false;
}

} // namespace sim_sm
