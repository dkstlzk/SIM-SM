#pragma once

#include "replacement_policy.hpp"
#include <random>
#include <vector>

namespace sim_sm {

class RandomPolicy : public ReplacementPolicy {
public:
    explicit RandomPolicy(unsigned int seed = 42);

    size_t choose_victim(const std::vector<CacheLine>& lines) override;
    void on_access(size_t way) override;

private:
    std::mt19937 rng_;
};

} // namespace sim_sm
