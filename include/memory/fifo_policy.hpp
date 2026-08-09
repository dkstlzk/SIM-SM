#pragma once

#include "replacement_policy.hpp"
#include <queue>
#include <vector>

namespace sim_sm {

class FIFOPolicy : public ReplacementPolicy {
public:
    explicit FIFOPolicy(size_t associativity);

    size_t choose_victim(const std::vector<CacheLine>& lines) override;
    void on_access(size_t way) override;

private:
    size_t associativity_;
    std::queue<size_t> fifo_queue_;
    std::vector<bool> in_queue_;
};

} // namespace sim_sm
