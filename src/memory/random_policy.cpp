#include "memory/random_policy.hpp"
#include <stdexcept>

namespace sim_sm {

RandomPolicy::RandomPolicy(unsigned int seed) : rng_(seed) {}

size_t RandomPolicy::choose_victim(const std::vector<CacheLine>& lines) {
    if (lines.empty()) {
        throw std::runtime_error("Cache set is empty");
    }

    std::vector<size_t> valid_ways;
    valid_ways.reserve(lines.size());

    for (size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].valid) {
            return i;
        }
        valid_ways.push_back(i);
    }

    // All are valid, pick uniformly random
    std::uniform_int_distribution<size_t> dist(0, valid_ways.size() - 1);
    return valid_ways[dist(rng_)];
}

void RandomPolicy::on_access(size_t /*way*/) {
    // Random policy has no state to update on hit/miss
}

} // namespace sim_sm
